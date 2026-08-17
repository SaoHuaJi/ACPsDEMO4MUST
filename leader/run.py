"""Leader 命令行入口。

本模块提供一个最小命令行程序，用于在 Leader 机器上测试 ACPs + LangChain
多智能体协作演示。

运行示例：
    python run.py "你好，你是谁？"
    python run.py "请找一个合适的Partner查询北京现在的天气，并告诉我温度和天气情况。"

如果命令行传入用户输入，则程序会先执行这一轮，再进入多轮交互窗口。
如果命令行没有传入用户输入，则程序会直接进入多轮交互窗口。
"""

import sys
import re
from leader_agent import run_leader

try:
    import readline
except ImportError:
    pass


EXIT_COMMANDS = {"exit", "quit", "退出"}


def _clean_user_input(text: str) -> str:
    """清洗用户终端输入，移除可能导致 LLM API 崩溃的控制字符和非法 Unicode 字符。"""
    if not isinstance(text, str):
        text = str(text)
    
    # 1. 物理消除 surrogate 字符 (终端异常或 httpx 可能产生)
    # 使用 encode/decode 技巧比 re.sub 在 Python 3.12+ 中更可靠
    try:
        text = text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')
    except Exception:
        text = text.encode('utf-8', 'replace').decode('utf-8')
        
    # 2. 移除不可见的 ASCII 控制字符 (保留空格、换行 \n、回车 \r、制表符 \t)
    # 匹配 \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def _print_answer(user_input: str) -> None:
    """执行一轮 Leader 对话并打印回答。

    Args:
        user_input (str): 用户输入的自然语言消息。

    Returns:
        None: 该函数只产生终端输出，不返回业务结果。
    """
    # 在传给 Leader 之前，先清洗掉终端混入的非法字符
    clean_input = _clean_user_input(user_input)
    
    # 调用 Leader ReAct 智能体处理当前轮用户输入。
    answer = run_leader(clean_input)

    # 将 Leader 的最终回答打印到终端。
    print("\nLeader>")
    print(answer)


def _run_interactive_loop() -> None:
    """运行 Leader 多轮命令行交互窗口。

    Args:
        无。

    Returns:
        None: 该函数持续读取终端输入并打印回答，直到用户输入退出命令。
    """
    # 持续读取用户输入。空输入或退出命令都会结束交互窗口。
    while True:
        try:
            user_input = input("\nUser> ").strip()
        except EOFError:
            print("\nLeader session ended.")
            return

        if not user_input or user_input.lower() in EXIT_COMMANDS:
            print("Leader session ended.")
            return

        # 执行当前轮对话；单轮异常不吞掉，让调用者能看到真实错误。
        _print_answer(user_input)


def main() -> None:
    """命令行主函数。

    Args:
        无。

    Returns:
        None: 该函数负责启动 Leader 命令行对话流程，不返回业务结果。
    """
    # 如果启动时带了首轮输入，先执行这一轮，再进入多轮交互窗口。
    if len(sys.argv) > 1:
        _print_answer(" ".join(sys.argv[1:]))

    # 启动持续对话窗口，直到用户主动退出。
    _run_interactive_loop()


# 仅当该文件作为脚本直接运行时才执行 main()。
# 这样其他模块也可以安全地 import 本文件而不会立即启动命令行流程。
if __name__ == "__main__":
    main()
