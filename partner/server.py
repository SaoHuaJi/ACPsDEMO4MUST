"""Partner 侧 ACPs Direct RPC 服务端。

本模块负责把 Partner 内部的 LangChain ReAct 智能体包装成一个可以通过
ACPs/AIP Direct RPC 调用的 Partner 服务。

整体职责：
    1. 读取本地 ACS 文件，并通过 `/acs` 暴露给 Leader。
    2. 使用 ACPs SDK 的 `aip_rpc_server` 创建 `/rpc` 路由。
    3. 在 `on_start` / `on_continue` 处理函数中调用 LangChain Partner Agent。
    4. 把智能体执行结果封装为 ACPs `Product` 和 `TaskResult` 返回给 Leader。

运行位置：
    Partner 机器。

启动示例：
    uvicorn server:app --host 0.0.0.0 --port 8011

说明：
    这是演示级实现，使用 SDK 默认的内存 `TaskManager` 存储任务状态。
    如果需要生产化，应替换为持久化任务存储，并补充认证、权限、超时、审计等机制。
"""

import json
from pathlib import Path

from fastapi import FastAPI
from acps_sdk.acs import AgentCapabilitySpec

from acps_sdk.aip import (
    Product,
    TaskCommand,
    TaskResult,
    TaskState,
    TextDataItem,
)
from acps_sdk.aip.aip_rpc_server import (
    CommandHandlers,
    DefaultHandlers,
    TaskManager,
    add_aip_rpc_router,
)

from partner_agent import run_partner_task


# 定位并读取 Partner ACS 文件。
# ACS 用于描述 Partner 的身份、能力和 RPC 地址。Leader 会先读取该文件，
# 再决定是否调用该 Partner。
BASE_DIR = Path(__file__).resolve().parent
ACS_PATH = BASE_DIR / "acs.json"

with ACS_PATH.open("r", encoding="utf-8") as f:
    ACS = json.load(f)

# 使用 acps_sdk.acs 校验 ACS 是否符合 AgentCapabilitySpec 规范结构。
ACS_SPEC = AgentCapabilitySpec.from_dict(ACS)

# 演示用 AIC。这里直接从规范 ACS 中读取，避免服务端代码和 ACS 文件写两份。
PARTNER_AIC = ACS_SPEC.aic

# 创建 FastAPI 应用。
# `/acs` 和 `/health` 由本文件直接定义；`/rpc` 由 ACPs SDK 注册。
app = FastAPI(title="Demo ACPs Partner Agent")


def _extract_text(command: TaskCommand) -> str:
    """从 ACPs TaskCommand 中提取第一段文本任务内容。

    ACPs 的 `TaskCommand.dataItems` 可以承载不同类型的数据项。这个演示版
    Partner 只处理 `TextDataItem`，因此会遍历 `dataItems` 并返回第一段文本。

    Args:
        command: Leader 通过 ACPs Direct RPC 发来的任务命令对象。

    Returns:
        str: 从命令中提取到的文本。如果命令中没有文本数据项，则返回空字符串。
    """
    # 遍历 dataItems，找到第一条 TextDataItem。
    for item in command.dataItems or []:
        if isinstance(item, TextDataItem):
            return item.text

    # 如果没有文本任务内容，返回空字符串，由上层处理为 AwaitingInput。
    return ""


def _with_sender(task: TaskResult) -> TaskResult:
    """为 TaskResult 补充当前 Partner 的 senderId。

    Args:
        task: 即将返回给 Leader 的任务结果对象。

    Returns:
        TaskResult: 设置了 `senderId` 的同一个任务结果对象。
    """
    # 标记该 TaskResult 是由当前 Partner 返回的。
    # 这在多智能体协作时有助于 Leader 判断结果来源。
    task.senderId = PARTNER_AIC
    return task


async def on_start(command: TaskCommand, task: TaskResult | None) -> TaskResult:
    """处理 Leader 发来的 `start` 任务命令。

    该函数是 Partner 执行新任务的核心入口。它由 ACPs SDK 的 RPC 路由在收到
    `start` 命令时自动调用。

    Args:
        command: Leader 发来的任务启动命令，包含 taskId、sessionId、dataItems 等。
        task: SDK 查询到的已有任务对象。如果同一个任务已经存在，该参数不为空。

    Returns:
        TaskResult: 当前任务状态和产出物。成功执行时返回 `AwaitingCompletion`，
        表示 Partner 已完成任务，等待 Leader 确认完成。
    """
    # 处理重复 start 请求。
    # 如果同一个 taskId 已经存在，直接返回已有任务，避免重复执行。
    if task:
        return _with_sender(task)

    # 从 TaskCommand 中提取自然语言任务文本。
    user_input = _extract_text(command).strip()

    # 如果 Leader 没有提供文本任务，则返回 AwaitingInput。
    # 这表示 Partner 需要 Leader 通过 continue 命令补充任务内容。
    if not user_input:
        new_task = TaskManager.create_task(
            command,
            initial_state=TaskState.AwaitingInput,
            data_items=[TextDataItem(text="Partner 没有收到可执行的任务内容。")],
        )
        return _with_sender(new_task)

    try:
        # 调用 Partner 内部 LangChain ReAct 智能体执行任务。
        # ACPs 通信到这里已经完成协议解析，模型只需要面对普通自然语言任务。
        answer = run_partner_task(user_input)

        # 创建 ACPs 任务记录，并将状态设为 AwaitingCompletion。
        # 该状态表示 Partner 已经生成结果，等待 Leader 发送 complete 命令确认。
        new_task = TaskManager.create_task(
            command,
            initial_state=TaskState.AwaitingCompletion,
        )

        # 把 LangChain Agent 的文本结果封装为 ACPs Product。
        # Product 是 Partner 返回给 Leader 的任务产出物；这里使用 TextDataItem 承载文本。
        TaskManager.set_products(
            new_task.taskId,
            [
                Product(
                    id=f"product-{new_task.taskId}",
                    name="partner-task-result",
                    dataItems=[
                        TextDataItem(text=answer)
                    ],
                )
            ],
        )

        # 读取更新后的任务对象并返回。
        updated_task = TaskManager.get_task(new_task.taskId) or new_task
        return _with_sender(updated_task)

    except Exception as exc:
        # 把执行异常转换为 ACPs Failed 状态。
        # 演示版直接把异常文本返回给 Leader，便于调试。
        failed_task = TaskManager.create_task(
            command,
            initial_state=TaskState.Failed,
            data_items=[
                TextDataItem(text=f"Partner 执行任务失败：{exc}")
            ],
        )
        return _with_sender(failed_task)


async def on_continue(command: TaskCommand, task: TaskResult) -> TaskResult:
    """处理 Leader 发来的 `continue` 任务命令。

    当 Partner 之前返回 `AwaitingInput`，或者 Leader 希望补充新的任务信息时，
    可以发送 `continue` 命令。这个演示版会把补充文本当作新任务重新执行，并
    覆盖当前任务的产出物。

    Args:
        command: Leader 发来的继续任务命令，通常包含补充文本。
        task: SDK 根据 taskId 找到的已有任务对象。

    Returns:
        TaskResult: 更新后的任务状态和产出物。成功时返回 `AwaitingCompletion`。
    """
    # 提取 Leader 补充的文本内容。
    user_input = _extract_text(command).strip()

    # 如果没有补充内容，则不改变当前任务，直接返回。
    if not user_input:
        return _with_sender(task)

    try:
        # 调用 Partner 内部 LangChain Agent 执行补充后的任务。
        answer = run_partner_task(user_input)

        # 设置或覆盖当前任务的产出物。
        TaskManager.set_products(
            task.taskId,
            [
                Product(
                    id=f"product-{task.taskId}",
                    name="partner-task-result-after-continue",
                    dataItems=[
                        TextDataItem(text=answer)
                    ],
                )
            ],
        )

        # 将任务状态更新为 AwaitingCompletion，等待 Leader 确认完成。
        updated_task = TaskManager.update_task_status(
            task.taskId,
            TaskState.AwaitingCompletion,
        )

        return _with_sender(updated_task)

    except Exception as exc:
        # 继续执行失败时，将任务状态更新为 Failed。
        updated_task = TaskManager.update_task_status(
            task.taskId,
            TaskState.Failed,
            data_items=[
                TextDataItem(text=f"Partner 继续执行任务失败：{exc}")
            ],
        )
        return _with_sender(updated_task)


# 把 ACPs 命令处理函数注册到 SDK 的 CommandHandlers。
# 未定制的 get/cancel/complete 使用 SDK 默认实现；start/continue 使用本文件中的业务逻辑。
handlers = CommandHandlers(
    on_start=on_start,
    on_get=DefaultHandlers.get,
    on_cancel=DefaultHandlers.cancel,
    on_complete=DefaultHandlers.complete,
    on_continue=on_continue,
)

# 通过 ACPs SDK 在 FastAPI 应用上注册 Direct RPC 路由。
# Leader 的 `AipRpcClient(partner_url="http://.../rpc")` 会访问这个路由。
add_aip_rpc_router(app, "/rpc", handlers)


@app.get("/acs")
async def get_acs() -> dict:
    """返回当前 Partner 的 ACS 能力描述。

    Leader 会先访问该接口，读取 Partner 的身份、能力描述和 RPC 端点。

    Returns:
        dict: 从 `acs.json` 读取到的 ACS 字典。
    """
    # 返回 SDK 规范序列化后的 ACS 内容，确保字段使用协议约定的 camelCase 名称。
    return ACS_SPEC.to_dict()


@app.get("/health")
async def health() -> dict[str, str]:
    """返回 Partner 服务健康状态。

    该接口不是 ACPs 必需接口，只是演示和调试时方便确认服务是否启动。

    Returns:
        dict[str, str]: 包含服务状态和当前 Partner AIC 的字典。
    """
    # 返回最小健康检查信息。
    return {
        "status": "ok",
        "aic": PARTNER_AIC,
    }

if __name__ == "__main__":
    import uvicorn

    # 直接运行 server.py 时，启动 FastAPI 内置的 Uvicorn HTTP 服务。
    uvicorn.run(app, host="0.0.0.0", port=8011)
