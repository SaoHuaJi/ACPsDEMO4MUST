"""Leader 侧 LangChain ReAct 智能体定义。

本模块负责创建直接面对用户的 Leader ReAct 智能体。Leader 本身也由
LangChain 创建，但它的工具不是普通业务工具，而是 ACPs 协作工具：
    1. `list_partner_acs`：读取已知 Partner 的 ACS 能力描述。
    2. `call_partner`：通过 Partner RPC 地址调用某个 Partner 执行任务。

整体职责：
    1. 判断用户输入是简单对话，还是需要外部 Partner 协作的任务。
    2. 在需要协作时，先获取所有已知 Partner ACS。
    3. 根据 ACS 中的 description / capabilities 选择合适 Partner。
    4. 通过 ACPs Direct RPC 调用 Partner，并整合 Partner 返回结果。

运行位置：
    Leader 机器。

说明：
    ACPs 协议状态机由 `acps_client.py` 里的代码处理；Leader Agent 只需要把
    `call_partner` 当成工具使用，不需要在 prompt 中学习或手写协议细节。
"""

import json
import httpx
import re

from langchain.agents import create_agent
from langchain.tools import tool

from config_loader import get_llm_model, get_partner_acs_urls
from acps_client import (
    call_partner_rpc_sync,
    extract_jsonrpc_endpoint,
    fetch_acs_sync,
)
from acps_sdk.adp import DiscoveryRequest, DiscoveryResponse, ADPError


@tool
def discover_partner(
    query: str,
    limit: int = 5,
    type: str = "explicit",
    discover_url = "https://www.ioa.pub/discovery/acps-adp-v2/discover"
) -> str:
    """
    根据自然语言描述的能力需求，从公网 ACPs 发现服务中搜索匹配的 Partner 智能体。
    当本地已知 Partner 无法满足需求，需要寻找具备特定能力（如：查询天气、规划旅游行程、识别视频等）的新智能体协作时调用此工具。
    
    Args:
        query: 描述所需智能体能力的自然语言查询，例如："查询指定地点的当前天气"。
        limit: 最大返回的候选 Partner 数量，默认为 5。
        type: 查询类型，默认为 "explicit"，表示明确的能力需求。
        discover_url: 公网 ACPs 发现服务的 URL，默认为 "https://www.ioa.pub/discovery/acps-adp-v2/discover"。

    Returns:
        str: str: 包含匹配 Partner 核心信息（AIC、名称、技能、通信端点）的 JSON 字符串。
    """
    request = DiscoveryRequest(type=type, query=query, limit=limit)
    payload = request.to_dict() if hasattr(request, "to_dict") else request.model_dump(exclude_none=True)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(discover_url, json=payload, headers=headers)
            print(f"\n\npayload is {payload}\n\nresponse is {response.json()}\n\n")
            if response.status_code == 404:
                response = client.post(f"{discover_url}/discover", json=payload, headers=headers)
            response.raise_for_status()
            raw_json = response.json()

        discovery_response = DiscoveryResponse.from_dict(raw_json)
        if discovery_response.is_error():
            return f"发现服务返回协议错误: {discovery_response.error.message}"
        # 提取并精简数据 (防止完整 ACS 撑爆 LLM 上下文)
        # 根据实际 ADP 返回结构，完整的 ACS 信息存储在 result.acsMap 字典中
        result_obj = discovery_response.result if discovery_response.result else None

        # 尝试从 SDK 对象获取 acsMap，如果获取不到则直接从 raw_json 提取兜底
        acs_map = getattr(result_obj, 'acsMap', None)
        if not acs_map and isinstance(raw_json, dict):
            acs_map = raw_json.get('result', {}).get('acsMap', {})
        if not acs_map:
            return f"未找到匹配 '{query}' 的 Partner 智能体。"

        discovered_partners = []
        # acs_map 的值即为完整的 ACS 字典或 SDK 对象
        for aic, acs in acs_map.items():
            # 统一转为 dict 处理，兼容 SDK 对象和纯字典
            if hasattr(acs, 'to_dict'):
                acs_dict = acs.to_dict()
            elif isinstance(acs, dict):
                acs_dict = acs
            else:
                continue

            skills = [s.get('name', '') for s in (acs_dict.get('skills') or [])]
            endpoints = [ep.get('url', '') for ep in (acs_dict.get('endPoints') or [])]
            partner_info = {
                "aic": acs_dict.get('aic', aic),
                "name": acs_dict.get('name', "Unknown"),
                "description": acs_dict.get('description', "No description"),
                "skills": skills,
                "endpoints": endpoints
            }
            discovered_partners.append(partner_info)

        if not discovered_partners:
            return f"未找到匹配 '{query}' 的 Partner 智能体。"

        return json.dumps(discovered_partners, ensure_ascii=False, indent=2)

    except httpx.HTTPStatusError as e:
        return f"请求发现服务失败 (HTTP {e.response.status_code}): {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"网络连接发现服务失败: {str(e)}"
    except ADPError as e:
        return f"ADP 协议解析错误: {str(e)}"
    except Exception as e:
        # 带上异常类名，便于以后再出问题时快速定位
        return f"发生未知错误: {type(e).__name__}: {str(e)}"


@tool
def list_partner_acs() -> str:
    """获取 Leader 已知的所有 Partner ACS。

    该工具供 Leader ReAct 智能体调用。Leader 在面对明确任务需求时，应先
    调用该工具查看当前局域网中有哪些已知 Partner 可提供服务，以及它们分别
    声明了什么能力。

    Args:
        无。

    Returns:
        str: JSON 字符串。每个元素包含：
            - `acs_url`: Partner ACS 地址。
            - `rpc_url`: 从 ACS 中解析出的 Partner JSONRPC 通信地址。
            - `acs`: 成功读取并通过 acps_sdk.acs 校验后的规范 ACS 内容。
            - `error`: 读取失败时的错误信息。
    """
    results = []

    # 从演示版注册表中读取已知 Partner ACS 地址。
    for acs_url in get_partner_acs_urls():
        try:
            # 访问 Partner ACS 地址，读取能力描述。
            acs = fetch_acs_sync(acs_url)
            rpc_url = extract_jsonrpc_endpoint(acs)

            # 记录成功读取到的规范 ACS。
            # 同时保留 rpc_url，便于后续 call_partner 工具直接调用 Partner。
            results.append(
                {
                    "acs_url": acs_url,
                    "rpc_url": rpc_url,
                    "acs": acs.to_dict(),
                }
            )
        except Exception as exc:
            # 单个 Partner ACS 读取失败不影响其他 Partner。
            # 这里把错误返回给 Leader Agent，让它知道该 Partner 当前不可用。
            results.append(
                {
                    "acs_url": acs_url,
                    "error": str(exc),
                }
            )

    # 返回格式化 JSON，便于模型阅读和选择 Partner。
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def call_partner(rpc_url: str, task: str) -> str:
    """通过某个 Partner 的 RPC 地址调用该 Partner 执行任务。

    Leader ReAct 智能体在读取 ACS 并判断某个 Partner 适合执行任务后，会调用
    本工具。`rpc_url` 应直接来自 `list_partner_acs` 返回结果中的同名字段。
    本工具不会再次读取 ACS，而是直接使用 ACPs SDK 的 `AipRpcClient` 发起
    Direct RPC 任务。

    Args:
        rpc_url: 要调用的 Partner JSONRPC 地址，来自 `list_partner_acs` 返回结果。
        task: 下发给 Partner 的自然语言任务描述，应尽量清楚、具体、可执行。

    Returns:
        str: Partner 返回的任务结果或状态说明。
    """
    # ACS 已经在 list_partner_acs 阶段读取和解析。这里直接根据 RPC 地址派发任务。
    return call_partner_rpc_sync(rpc_url, task)


def build_leader_agent():
    """创建 Leader 侧 LangChain ReAct 智能体。

    Leader 的核心能力不是直接调用业务工具，而是判断是否需要 ACPs 协作，并在
    需要时选择合适 Partner。这里把读取 ACS 和调用 Partner 都包装成 LangChain
    工具，供 ReAct 智能体按需调用。

    Args:
        无。

    Returns:
        Runnable: LangChain 创建的 Leader 智能体对象。该对象支持 `invoke()`，
            输入为包含 `messages` 的字典，输出为包含对话消息历史的结果字典。
    """
    # 注册 Leader 可用工具。
    # `list_partner_acs` 是协作前的信息获取工具；`call_partner` 是真正执行
    # ACPs Direct RPC 协作的工具。
    tools = [
        list_partner_acs,
        call_partner,
        discover_partner
    ]

    # 定义 Leader 的协作决策规则。
    # 这里强调：只有需要外部能力时才读取 ACS 并调用 Partner；简单对话可以直接回答。
    system_prompt = (
        "你是一个直接面对用户的 ACPs Leader ReAct 智能体。"
        "你的目标是判断用户请求是否需要调用局域网中的 Partner 智能体协作。"
        "\n\n"
        "决策与执行规则："
        "1. 如果用户只是打招呼、闲聊、询问你是谁、简单解释概念，可以直接回答。"
        "2. 如果用户提出明确任务，且这个任务适合由外部 Partner 执行，你必须先调用 list_partner_acs 查看本地已知 Partner。"
        "3. 如果本地没有合适的 Partner，调用 discover_partner 去公网寻找新的 Partner。"
        "4. 根据 Partner 的 description 和 skills 判断是否适合调用。"
        "5. 决定调用时，使用 call_partner 工具，把任务清楚地交给该 Partner。"
        "   - 如果是本地 Partner，使用 list_partner_acs 返回的 rpc_url。"
        "   - 如果是公网 Partner，使用 discover_partner 返回的 endpoints 列表中的第一个 URL 作为 rpc_url。"
        "6. 收到 Partner 结果后，向用户总结最终答案。"
        "7. 如果没有合适 Partner，你可以自己回答，或说明当前没有合适 Partner。"
        "\n\n"
        "注意："
        "不要把 ACPs 协议细节暴露给用户。不要虚构 Partner 能力。"
        "不要在没查看 ACS 的情况下声称某个 Partner 能执行任务。"
    )

    # 从 Leader 本地配置文件读取 LLM 配置并创建模型对象。
    model = get_llm_model()

    # 创建 LangChain Leader Agent。
    # 模型参数来自 `leader/config/llm.json`，避免依赖终端 export 或业务代码硬编码。
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )


def run_leader(user_input: str) -> str:
    """运行 Leader 智能体处理用户输入。

    Args:
        user_input: 用户输入的自然语言消息。可能是简单对话，也可能是需要 Partner
            执行的任务需求。

    Returns:
        str: Leader 智能体给用户的最终回答。
    """
    # 创建 Leader ReAct 智能体。
    # 演示版每次调用创建一次；如需长期运行，可以在服务启动时创建并复用。
    agent = build_leader_agent()

    # 把用户输入转换成 LangChain 消息格式并执行。
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )

    # 提取最后一条消息作为 Leader 的最终回复。
    last_message = result["messages"][-1]

    # 兼容不同消息对象结构。
    return getattr(last_message, "content", str(last_message))
