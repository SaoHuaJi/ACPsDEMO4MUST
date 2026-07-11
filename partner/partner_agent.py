"""Partner 侧 LangChain ReAct 智能体定义。

本模块只负责创建和运行 Partner 内部的 LangChain 智能体，不直接处理
ACPs/AIP 通信协议。ACPs 通信外壳由同目录下的 `server.py` 负责。

整体职责：
    1. 定义 Partner 可调用的天气查询工具和 Coze 今日热点查询工具。
    2. 创建能够自主选择工具的 LangChain ReAct 智能体。
    3. 暴露 `run_partner_task()` 作为 Partner 服务端处理任务时调用的统一入口。

运行位置：
    Partner 机器。

典型调用链：
    Leader -> ACPs RPC -> partner/server.py -> run_partner_task() -> LangChain Agent
    -> 工具 -> 返回文本结果。
"""

from urllib.parse import quote

import httpx
from langchain.agents import create_agent
from langchain.tools import tool

from config_loader import get_coze_config, get_llm_model


WEATHER_BASE_URL = "https://wttr.in"
WEATHER_LANGUAGE = "zh"
WEATHER_TIMEOUT_SECONDS = 10.0


def _get_text_value(items: list[dict], key: str) -> str:
    """从 wttr.in 字段列表中提取文本值。

    Args:
        items (list[dict]): wttr.in 返回的字段列表。
        key (str): 需要读取的字段名。

    Returns:
        str: 提取到的文本值；当字段不存在时返回空字符串。
    """
    # wttr.in 的部分字段用 `[{"value": "..."}]` 表示，这里统一读取。
    if not items:
        return ""

    first_item = items[0]
    if not isinstance(first_item, dict):
        return ""

    value = first_item.get(key)
    return str(value) if value is not None else ""


def _format_weather_response(location: str, data: dict) -> str:
    """格式化天气查询结果。

    Args:
        location (str): 用户请求查询的地点。
        data (dict): wttr.in 返回的天气 JSON 数据。

    Returns:
        str: 面向 Leader 和用户阅读的天气摘要文本。

    Raises:
        ValueError: 当天气响应缺少当前天气数据时抛出。
    """
    # 提取当前天气数组的第一项，wttr.in 会把实时天气放在这里。
    current_conditions = data.get("current_condition")
    if not isinstance(current_conditions, list) or not current_conditions:
        raise ValueError("Weather response does not contain current conditions.")

    current = current_conditions[0]
    if not isinstance(current, dict):
        raise ValueError("Weather response current condition is invalid.")

    # 提取地点显示名；如果天气服务没有返回，则使用用户原始地点。
    nearest_area = data.get("nearest_area")
    area_name = location
    if isinstance(nearest_area, list) and nearest_area:
        area = nearest_area[0]
        if isinstance(area, dict):
            area_name = _get_text_value(area.get("areaName", []), "value") or location

    # 提取天气描述和关键观测指标，组成稳定的摘要文本。
    description = _get_text_value(current.get("lang_zh", []), "value")
    if not description:
        description = _get_text_value(current.get("weatherDesc", []), "value")

    return (
        f"地点: {area_name}\n"
        f"天气: {description or '未知'}\n"
        f"温度: {current.get('temp_C', '未知')}°C\n"
        f"体感温度: {current.get('FeelsLikeC', '未知')}°C\n"
        f"湿度: {current.get('humidity', '未知')}%\n"
        f"风速: {current.get('windspeedKmph', '未知')} km/h\n"
        f"观测时间: {current.get('localObsDateTime', '未知')}"
    )


@tool
def query_weather(location: str) -> str:
    """联网查询指定地点的当前天气。

    该函数会被注册为 LangChain 工具。Partner 的 ReAct 智能体在收到天气查询
    任务时，应调用该工具查询外部天气服务，并返回当前天气摘要。

    Args:
        location (str): 需要查询天气的地点，可以是城市、地标、邮编或经纬度。

    Returns:
        str: 指定地点的当前天气摘要；查询失败时返回失败原因说明。
    """
    # 清洗地点输入，避免空地点请求外部服务。
    cleaned_location = location.strip()
    if not cleaned_location:
        return "天气查询失败：缺少要查询的地点。"

    # 使用内置天气服务参数构造 wttr.in JSON 查询 URL。
    encoded_location = quote(cleaned_location)
    url = f"{WEATHER_BASE_URL}/{encoded_location}"
    params = {
        "format": "j1",
        "lang": WEATHER_LANGUAGE,
    }

    try:
        # 发起联网查询。关闭 trust_env，避免终端代理变量影响演示工具行为。
        with httpx.Client(
            timeout=WEATHER_TIMEOUT_SECONDS,
            trust_env=False,
        ) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        # 解析并格式化天气服务响应。
        return _format_weather_response(cleaned_location, data)
    except Exception as exc:
        # 将外部服务异常转换为工具结果，方便 Leader 明确看到失败原因。
        return f"天气查询失败：{exc}"


@tool
def call_coze_agent(task: str) -> str:
    """调用 Partner 本地配置的 Coze 今日热点查询智能体。

    该函数会被注册为 LangChain 工具。Partner 的 ReAct 智能体在收到今日
    热点、新闻、时事查询任务时，应调用该工具，并把 Leader 下发的任务文本
    转发给 Coze 智能体。

    Args:
        task (str): Leader 下发的自然语言任务描述。

    Returns:
        str: Coze 今日热点查询智能体返回的文本结果；调用失败时返回失败原因说明。
    """
    # 清洗任务输入，避免把空任务发送给外部 Coze 智能体。
    cleaned_task = task.strip()
    if not cleaned_task:
        return "Coze agent call failed: task is empty."
    try:
        # 延迟导入 cozepy。这样即使只检查 ACS 或启动配置，也不会在模块导入时因缺少 Coze 运行时依赖而失败。
        from cozepy import ChatStatus, Coze, Message, MessageType, TokenAuth

        # 从 Partner 本地配置读取 Coze 凭据和 Bot 信息。
        coze_config = get_coze_config()
        coze = Coze(
            auth=TokenAuth(token=coze_config["token"]),
            base_url=coze_config["base_url"],
        )
        # 通过 Coze Chat API 调用当前配置的 Coze Bot （今日热点查询智能体）。
        # 此处为单轮调用，若需要多轮 Coze 会话可以在 Partner 任务上下文中保存 conversation_id 后再传入。
        result = coze.chat.create_and_poll(
            bot_id=coze_config["bot_id"],
            user_id=coze_config["user_id"],
            additional_messages=[Message.build_user_question_text(cleaned_task)],
            auto_save_history=True,
        )
        # 提取 Coze 返回的 ANSWER 消息，拼成最终工具结果。
        answer_parts = [
            message.content
            for message in result.messages
            if getattr(message, "content", None)
            and getattr(message, "type", None) == MessageType.ANSWER
        ]
        answer = "".join(answer_parts).strip()
        if result.chat.status != ChatStatus.COMPLETED:
            return f"Coze agent call failed: chat status is {result.chat.status}."
        if not answer:
            return "Coze agent call failed: no assistant answer found."
        return answer
    except ImportError:
        return "Coze agent call failed: cozepy is not installed."
    except Exception as exc:
        # 将外部服务异常转换为工具结果，避免 Partner 服务因工具失败而崩溃。
        return f"Coze agent call failed: {exc}"


def build_partner_agent():
    """创建 Partner 内部的 LangChain ReAct 智能体。

    该函数只负责构造智能体对象，不执行具体任务。Partner 服务端每次收到
    ACPs `start` 或 `continue` 请求后，可以调用该函数创建智能体，再通过
    `invoke()` 执行任务。

    Args:
        无。

    Returns:
        Runnable: LangChain 创建的智能体对象。该对象支持 `invoke()`，输入为
            包含 `messages` 的字典，输出为包含对话消息历史的结果字典。
    """
    # 把天气查询工具和 Coze 今日热点查询工具注册给 LangChain Agent。
    # 模型在 ReAct 推理过程中根据 Leader 的任务内容选择合适工具。
    tools = [
        query_weather,
        call_coze_agent,
    ]

    # 定义 Partner 的行为边界。
    # 这里的 system_prompt 只约束 Partner 内部的任务执行方式，ACPs 协议状态
    # 仍由 server.py 中的代码处理，不交给模型自由发挥。
    system_prompt = (
        "你是一个 ACPs Partner 智能体。"
        "你的职责是执行 Leader 下发的天气查询和今日热点新闻查询任务。"
        "如果任务要求查询某个地点的当前天气，必须调用 query_weather。"
        "如果任务要求查询今日热点、新闻或时事，必须调用 call_coze_agent。"
        "如果任务缺少地点、时间范围或新闻主题等必要信息，要求 Leader 补充。"
        "如果任务不属于天气或今日热点新闻查询范围，明确说明无法完成。"
        "最终回答要简洁，并说明结果来自对应工具。"
    )

    # 从 Partner 本地配置文件读取 LLM 配置并创建模型对象。
    model = get_llm_model()

    # 创建 LangChain 智能体。
    # 模型参数来自 `partner/config/llm.json`，避免依赖终端 export 或业务代码硬编码。
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )


def run_partner_task(task: str) -> str:
    """运行 Partner 智能体完成 Leader 下发的任务。

    这是 Partner 服务端调用 LangChain 的统一入口。`server.py` 不关心智能体
    内部如何推理、如何选择工具，只需要把任务文本传进来，并拿到最终文本结果。

    Args:
        task: Leader 通过 ACPs Direct RPC 下发的自然语言任务描述。

    Returns:
        str: Partner 智能体执行后的最终回答文本。
    """
    # 创建 Partner 内部 ReAct 智能体。当前为每次请求创建一次的简单实现。
    # 如果需要更高性能，可以改成进程启动时创建一次并复用。
    agent = build_partner_agent()
    # 把 ACPs 传入的任务文本转换成 LangChain 消息格式。
    # LangChain Agent 的输入是对话消息列表，而不是裸字符串。
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )
    # LangChain 返回值中通常包含完整消息历史。
    # 提取最后一条 assistant 消息作为 Partner 的最终回答。
    last_message = result["messages"][-1]
    # 兼容不同消息对象的表示形式。
    # 大多数 LangChain 消息对象有 `content` 属性；没有则退化为字符串。
    return getattr(last_message, "content", str(last_message))
