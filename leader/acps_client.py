"""Leader 侧 ACPs Direct RPC 客户端封装。

本模块负责 Leader 与 Partner 之间的 ACPs 通信。它直接使用 ACPs SDK 的
`AipRpcClient` 调用 Partner 的 `/rpc` 端点；同时使用 `httpx` 读取 Partner
暴露的 `/acs` 能力描述。

整体职责：
    1. 读取 Partner ACS。
    2. 根据 ACS 中的 RPC endpoint 调用 Partner。
    3. 处理 ACPs 任务状态机中的常见状态。
    4. 从 TaskResult / Product / TextDataItem 中提取文本结果。

运行位置：
    Leader 机器。

说明：
    这是演示级封装。为了让 LangChain 同步工具能方便调用，模块中提供了
    `*_sync` 同步包装函数。若后续改用 LangGraph 或异步工具，可以直接使用
    `fetch_acs()` 和 `call_partner_rpc()` 这两个异步函数。
"""
import asyncio
from contextlib import contextmanager
import json
import os
import uuid

import httpx

from acps_sdk.acs import AgentCapabilitySpec
from acps_sdk.aip import AipRpcClient, TaskState, TextDataItem


# 演示用 Leader AIC。生产环境中应由 ACPs 身份体系或配置系统管理。
LEADER_AIC = "demo.leader.react-agent"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@contextmanager
def without_proxy_environment():
    """临时移除当前进程中的代理环境变量。

    Args:
        无。

    Yields:
        None: 在上下文内部，常见代理环境变量会被移除。

    Returns:
        None: 退出上下文时恢复原始代理环境变量。
    """
    # 保存当前代理环境变量，并在上下文期间移除它们。
    original_values = {
        key: os.environ[key]
        for key in PROXY_ENV_KEYS
        if key in os.environ
    }
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)

    try:
        yield
    finally:
        # 恢复进入上下文前的代理环境，避免影响后续非 ACPs 请求。
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(original_values)


async def fetch_acs(acs_url: str) -> AgentCapabilitySpec:
    """异步读取某个 Partner 暴露的 ACS 能力描述。

    Args:
        acs_url: Partner 的 ACS HTTP 地址，例如 `http://192.168.1.20:8011/acs`。

    Returns:
        AgentCapabilitySpec: 通过 acps_sdk.acs 解析和校验后的 Partner ACS。

    Raises:
        httpx.HTTPError: 当请求失败、超时或返回非 2xx 状态码时抛出。
        ValueError: 当响应体不是合法 JSON 或不符合 ACS 规范时抛出。
    """
    # 创建短生命周期 HTTP 客户端。
    # 演示版每次请求创建一次，代码更直观；高频调用时可以改成复用客户端。
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        # 请求 Partner ACS 地址。
        response = await client.get(acs_url)

        # 如果 HTTP 状态码不是成功状态，直接抛出异常。
        response.raise_for_status()

        # 解析 JSON，并交给 acps_sdk.acs 校验规范结构。
        return AgentCapabilitySpec.from_dict(response.json())


def extract_jsonrpc_endpoint(acs: AgentCapabilitySpec) -> str:
    """从规范 ACS 中提取 JSONRPC 端点 URL。

    Args:
        acs (AgentCapabilitySpec): 已通过 SDK 校验的 Partner ACS 对象。

    Returns:
        str: Partner 的 JSONRPC 端点 URL。

    Raises:
        ValueError: 当 ACS 中没有可用 JSONRPC 端点时抛出。
    """
    # 优先查找 transport 明确声明为 JSONRPC 的端点。
    for endpoint in acs.end_points:
        if endpoint.transport.upper() == "JSONRPC":
            return endpoint.url

    # 没有 JSONRPC 端点时明确报错，避免 Leader 调到错误协议。
    raise ValueError(f"ACS for {acs.aic} does not contain a JSONRPC endpoint.")


def extract_text_items(items) -> list[str]:
    """从 ACPs dataItems 列表中提取所有文本内容。

    Args:
        items: ACPs `dataItems` 列表，可能包含 `TextDataItem` 或其他类型的数据项。

    Returns:
        list[str]: 从所有 `TextDataItem` 中提取出的文本列表。
    """
    texts = []

    # 遍历 dataItems，仅处理演示版支持的 TextDataItem。
    for item in items or []:
        if isinstance(item, TextDataItem):
            texts.append(item.text)

    return texts


def extract_task_output(task) -> str:
    """从 ACPs TaskResult 中提取 Partner 返回的文本结果。

    Partner 的结果可能出现在两个位置：
        1. `task.products[*].dataItems`：正常任务产出物。
        2. `task.status.dataItems`：状态消息、错误消息或补充输入提示。

    本函数会同时读取这两个位置，并把文本结果合并为一个字符串。

    Args:
        task: ACPs SDK 返回的 `TaskResult` 对象。

    Returns:
        str: 合并后的文本结果。如果没有任何文本数据项，则返回空字符串。
    """
    chunks = []

    # 提取 Product 中的文本产出。
    # 这是 Partner 正常完成任务时最主要的结果来源。
    for product in task.products or []:
        chunks.extend(extract_text_items(product.dataItems))

    # 提取任务状态中携带的文本。
    # AwaitingInput、Failed 等状态通常会把说明信息放在 status.dataItems 中。
    if getattr(task, "status", None) and task.status.dataItems:
        chunks.extend(extract_text_items(task.status.dataItems))

    # 把多段文本合并为可读结果。
    return "\n".join(chunks).strip()


async def call_partner_rpc(rpc_url: str, task_text: str) -> str:
    """通过 ACPs Direct RPC 调用 Partner 执行任务。

    该函数直接使用 ACPs SDK 的 `AipRpcClient`，不手写 RPC 请求结构。
    它会创建一个演示用 sessionId 和 taskId，发送 `start_task`，根据状态必要时
    轮询 `get_task`，在 Partner 返回 `AwaitingCompletion` 后提取结果并发送
    `complete_task`。

    Args:
        rpc_url: Partner 的 ACPs RPC 地址，例如 `http://192.168.1.20:8011/rpc`。
        task_text: Leader 准备下发给 Partner 的自然语言任务描述。

    Returns:
        str: Partner 返回的文本结果，或任务未完成时的状态说明。

    Raises:
        Exception: ACPs SDK 调用、网络请求或状态处理过程中产生的异常会继续抛出。
    """
    # 为这次演示调用创建唯一会话 ID 和任务 ID。
    # Direct RPC 中 Leader 用它们追踪任务状态。
    session_id = f"session-{uuid.uuid4()}"
    task_id = f"task-{uuid.uuid4()}"

    # 创建 ACPs SDK Direct RPC 客户端。
    # `partner_url` 指向 Partner `/rpc`，`leader_id` 使用演示 AIC。
    with without_proxy_environment():
        client = AipRpcClient(
            partner_url=rpc_url,
            leader_id=LEADER_AIC,
        )
    client.http_client = httpx.AsyncClient(trust_env=False)

    try:
        # 发送 start_task，启动 Partner 任务。
        task = await client.start_task(
            session_id=session_id,
            task_id=task_id,
            user_input=task_text,
        )

        # 处理 Partner 仍在执行的状态。
        # 如果 Partner 返回 Accepted 或 Working，则 Leader 定时 get_task 轮询状态。
        while task.status.state in (TaskState.Accepted, TaskState.Working):
            await asyncio.sleep(0.5)
            task = await client.get_task(
                task_id=task_id,
                session_id=session_id,
            )

        # Partner 需要更多输入。
        # 演示版只把提示返回给 Leader Agent，由 Leader Agent 决定如何回复用户。
        if task.status.state == TaskState.AwaitingInput:
            return "Partner 需要补充信息：\n" + extract_task_output(task)

        # Partner 已完成任务，等待 Leader 确认。
        # 这是演示版最常见的成功路径：提取 Product，然后发送 complete_task。
        if task.status.state == TaskState.AwaitingCompletion:
            output = extract_task_output(task)

            await client.complete_task(
                task_id=task_id,
                session_id=session_id,
            )

            return output

        # Partner 明确没有完成任务。
        # 这些状态下直接把 Partner 的状态说明或错误信息返回给 Leader Agent。
        if task.status.state in (
            TaskState.Failed,
            TaskState.Rejected,
            TaskState.Canceled,
        ):
            return "Partner 未完成任务：\n" + extract_task_output(task)

        # 兜底返回。
        # 如果 SDK 未来增加了其他状态，至少尝试提取已有文本，避免无结果。
        return extract_task_output(task)

    finally:
        # 关闭 ACPs 客户端连接资源。
        await client.close()


def run_async(coro):
    """在同步上下文中运行异步协程。

    该函数用于把异步 ACPs 调用包装给 LangChain 同步工具使用。

    Args:
        coro: 待执行的协程对象。

    Returns:
        Any: 协程执行完成后的返回值。

    Raises:
        RuntimeError: 如果当前线程中已经存在运行中的事件循环，`asyncio.run()`
        可能抛出运行时错误。此时建议改用 LangChain/LangGraph 的异步工具。
    """
    # 使用 Python 标准库运行协程直到完成。
    return asyncio.run(coro)


def fetch_acs_sync(acs_url: str) -> AgentCapabilitySpec:
    """同步读取 Partner ACS。

    这是 `fetch_acs()` 的同步包装，便于 LangChain 同步工具直接调用。

    Args:
        acs_url: Partner ACS 地址。

    Returns:
        AgentCapabilitySpec: 通过 acps_sdk.acs 解析和校验后的 Partner ACS。
    """
    # 同步运行异步 ACS 获取函数。
    return run_async(fetch_acs(acs_url))


def call_partner_rpc_sync(rpc_url: str, task_text: str) -> str:
    """根据 Partner RPC 地址调用 Partner 执行任务。

    Args:
        rpc_url: Partner 的 ACPs RPC 地址，例如 `http://192.168.1.20:8011/rpc`。
        task_text: 要下发给 Partner 的自然语言任务。

    Returns:
        str: Partner 执行后的文本结果或状态说明。

    Raises:
        Exception: 网络请求或 ACPs 调用失败时继续抛出底层异常。
    """
    # 演示版在读取 ACS 阶段已经解析出 RPC 地址，因此这里直接调用 RPC。
    return run_async(call_partner_rpc(rpc_url, task_text))


def call_partner_by_acs_sync(acs_url: str, task_text: str) -> str:
    """根据 Partner ACS 地址调用 Partner 执行任务。

    该函数会先读取 ACS，再从 ACS 的 `endPoints` 中取得 JSONRPC 地址，
    最后使用 ACPs SDK 的 `AipRpcClient` 调用 Partner。

    Args:
        acs_url: Partner ACS 地址。
        task_text: 要下发给 Partner 的自然语言任务。

    Returns:
        str: Partner 执行后的文本结果或状态说明。

    Raises:
        ValueError: 当 ACS 中缺少 JSONRPC 端点时抛出。
        Exception: 网络请求或 ACPs 调用失败时继续抛出底层异常。
    """
    # 读取 Partner ACS。
    acs = fetch_acs_sync(acs_url)

    # 从规范 ACS 中提取 JSONRPC 地址。
    rpc_url = extract_jsonrpc_endpoint(acs)

    # 调用 Partner RPC 并返回结果。
    return run_async(call_partner_rpc(rpc_url, task_text))


def acs_to_readable_text(acs: AgentCapabilitySpec) -> str:
    """把 ACS 对象格式化为便于模型阅读的 JSON 字符串。

    Args:
        acs (AgentCapabilitySpec): 通过 SDK 校验后的 Partner ACS 对象。

    Returns:
        str: 缩进后的 JSON 字符串，保留中文字符。
    """
    # 使用 SDK 的规范序列化结果，字段名保持协议侧 camelCase。
    return json.dumps(acs.to_dict(), ensure_ascii=False, indent=2)
