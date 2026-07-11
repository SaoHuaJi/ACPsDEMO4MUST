"""Leader 侧配置读取模块。

本模块只读取 Leader 机器本地的配置文件，避免 Leader 依赖 Partner 目录或
其他部署单元的配置。
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
PARTNER_REGISTRY_PATH = CONFIG_DIR / "partners.json"
LLM_CONFIG_PATH = CONFIG_DIR / "llm.json"
HTTP_PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


def _read_json_file(path: Path) -> dict[str, Any]:
    """读取 JSON 配置文件。

    Args:
        path (Path): JSON 配置文件路径。

    Returns:
        dict[str, Any]: JSON 文件解析后的字典。

    Raises:
        FileNotFoundError: 当配置文件不存在时抛出。
        ValueError: 当配置文件顶层不是 JSON 对象时抛出。
        json.JSONDecodeError: 当配置文件不是合法 JSON 时抛出。
    """
    # 读取本地配置文件，并解析为 JSON 对象。
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # 校验顶层结构，避免后续读取字段时出现难以定位的错误。
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    return data


def _get_http_proxy_url() -> str | None:
    """获取可用于 LLM 访问的 HTTP 代理地址。

    Args:
        无。

    Returns:
        str | None: HTTP/HTTPS 代理地址；没有可用代理时返回 None。
    """
    # 只接受 HTTP/HTTPS 代理，避免 ALL_PROXY 中的 SOCKS 代理触发 socksio 依赖。
    for key in HTTP_PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()

    return None


def get_partner_acs_urls() -> list[str]:
    """获取 Leader 已知的 Partner ACS 地址列表。

    Args:
        无。

    Returns:
        list[str]: 从 `leader/config/partners.json` 读取并清洗后的 ACS 地址列表。

    Raises:
        ValueError: 当配置缺少 `partners` 字段或 ACS 地址不是字符串时抛出。
    """
    # 读取 Leader 本地维护的 Partner 注册表。
    config = _read_json_file(PARTNER_REGISTRY_PATH)
    partners = config.get("partners")

    # 校验 Partner 列表结构。
    if not isinstance(partners, list):
        raise ValueError("leader/config/partners.json must contain a partners list.")

    acs_urls: list[str] = []
    for index, item in enumerate(partners):
        if not isinstance(item, dict):
            raise ValueError(f"Partner item #{index} must be a JSON object.")

        acs_url = item.get("acs_url")
        if not isinstance(acs_url, str):
            raise ValueError(f"Partner item #{index} must contain string acs_url.")

        cleaned_url = acs_url.strip()
        if cleaned_url:
            acs_urls.append(cleaned_url)

    return acs_urls


def get_llm_model() -> ChatOpenAI:
    """读取 Leader LLM 配置并创建 LangChain 模型对象。

    Args:
        无。

    Returns:
        ChatOpenAI: 按 `leader/config/llm.json` 创建的 LangChain OpenAI 聊天模型对象。

    Raises:
        ValueError: 当配置缺少模型名或 API Key 时抛出。
    """
    # 读取 Leader 本地的 LLM 配置。
    config = _read_json_file(LLM_CONFIG_PATH)
    model = config.get("model")
    api_key = config.get("api_key")
    base_url = config.get("base_url")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("leader/config/llm.json model must be a non-empty string.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("leader/config/llm.json api_key must be a non-empty string.")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError("leader/config/llm.json base_url must be a string or null.")

    # 创建模型对象。显式传入 HTTP/HTTPS 代理并关闭 trust_env，避免 ALL_PROXY 中
    # 的 SOCKS 代理触发 socksio 依赖，同时保留公网 LLM 访问能力。
    proxy_url = _get_http_proxy_url()
    return ChatOpenAI(
        model=model.strip(),
        api_key=api_key.strip(),
        base_url=base_url.strip() if isinstance(base_url, str) and base_url.strip() else None,
        http_client=httpx.Client(proxy=proxy_url, trust_env=False),
        http_async_client=httpx.AsyncClient(proxy=proxy_url, trust_env=False),
        http_socket_options=(),
    )
