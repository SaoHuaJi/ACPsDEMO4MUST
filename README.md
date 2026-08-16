# ACPs + LangChain Leader / Partner 演示工程

本文件包演示如何用 LangChain 创建能够调用工具的 Partner 智能体，以及能够自主判断是否调用 Partner 协作的 Leader 智能体。Agent 间通信直接使用 ACPs SDK 的 Direct RPC 能力：

- Partner 侧：使用 `acps_sdk.aip.aip_rpc_server` 暴露 `/rpc`。
- Leader 侧：使用 `acps_sdk.aip.AipRpcClient` 调用 Partner `/rpc`。
- LangChain 只负责智能体内部推理、工具调用和协作决策，不手写 ACPs RPC 协议。

## 目录结构

```text
workspace/
  partner/
    config/
      llm.json             # Partner 本机 LLM 配置。
      coze.json            # Partner 本机 Coze 智能体配置。
    acs.json               # Partner 能力描述。JSON 不能写注释，说明见 acs说明.md。
    config_loader.py       # Partner 本机配置读取模块。
    partner_agent.py       # Partner 内部 LangChain ReAct 智能体与工具。
    server.py              # Partner ACPs Direct RPC 服务端。
    start.sh               # Partner 服务启动脚本。

  leader/
    config/
      llm.json             # Leader 本机 LLM 配置。
      partners.json        # Leader 已知 Partner ACS 地址列表。
    config_loader.py       # Leader 本机配置读取模块。
    acps_client.py         # Leader 侧 AipRpcClient 调用封装。
    leader_agent.py        # Leader LangChain ReAct 智能体。
    run.py                 # Leader 命令行入口。
    start.sh               # Leader 命令行启动脚本。
```

## 安装依赖

假设你已经从 https://github.com/AIP-PUB/ACPs-community 克隆 ACPs-community，则至少还需要：

```bash
pip install fastapi uvicorn httpx langchain langchain-openai cozepy
pip install -e /path/to/ACPs-community/acps-sdk
```

## Partner 配置

Partner 的 LLM 配置写在 Partner 机器的 `partner/config/llm.json`，不需要通过终端 `export`：

```json
{
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "api_key": "replace-with-your-api-key",
  "base_url": null
}
```

如果 Partner 使用 OpenAI 兼容 API，把 `base_url` 改成对应服务地址，并把 `api_key` 改成该服务的密钥。

Partner 的 `partner/acs.json` 使用 `acps_sdk.acs.AgentCapabilitySpec` 规范结构。服务启动时会用
`AgentCapabilitySpec.from_dict()` 校验该文件，Leader 读取远端 ACS 后也会用同一个 SDK 模型解析。
这意味着 ACS 需要使用规范字段，例如 `endPoints`、`skills`、`capabilities`，而不是简化版字段。

这个 Partner 同时提供两类能力：

- 使用内置天气查询工具返回指定地点的当前天气。
- 使用 Partner 本地配置的 Coze 今日热点查询智能体返回新闻热点结果。

Coze 配置写在 Partner 机器的 `partner/config/coze.json`，不需要通过终端 `export`：

```json
{
  "token": "replace-with-your-coze-token",
  "bot_id": "replace-with-your-coze-bot-id",
  "base_url": "https://api.coze.cn",
  "user_id": "acps-partner"
}
```

这条链路中 Coze 不是独立 ACPs Partner，而是当前 Partner 内部的新闻查询工具：

```text
Leader -> AIP -> Partner -> LangChain Agent -> Coze Tool -> Coze 智能体
```

一个同时提供天气查询和今日热点新闻查询的 Partner 核心 ACS 结构如下：

```json
{
  "aic": "demo.partner.weather-news-agent",
  "active": true,
  "lastModifiedTime": "2026-07-10T10:00:00+08:00",
  "protocolVersion": "02.01",
  "name": "Weather and News Partner Agent",
  "description": "一个能够查询指定地点当前天气，并通过内部 Coze 今日热点查询智能体查询新闻热点的 Partner 智能体。",
  "version": "1.0.0",
  "provider": {
    "countryCode": "CN",
    "organization": "Demo University",
    "department": "Computer Science",
    "name": "Demo Team",
    "email": "demo@example.com"
  },
  "securitySchemes": {},
  "endPoints": [
    {
      "url": "http://127.0.0.1:8011/rpc",
      "transport": "JSONRPC"
    }
  ],
  "capabilities": {
    "streaming": false,
    "notification": false,
    "messageQueue": []
  },
  "defaultInputModes": [
    "text/plain"
  ],
  "defaultOutputModes": [
    "text/plain"
  ],
  "skills": [
    {
      "id": "weather.current.lookup",
      "name": "当前天气查询",
      "description": "根据城市、地标、邮编或经纬度查询指定地点的当前天气。",
      "version": "1.0.0",
      "tags": [
        "天气",
        "实时查询",
        "地理位置"
      ],
      "examples": [
        "查询北京现在的天气"
      ],
      "inputModes": [
        "text/plain"
      ],
      "outputModes": [
        "text/plain"
      ]
    },
    {
      "id": "coze.news.hotspots.lookup",
      "name": "今日热点新闻查询",
      "description": "把 Leader 下发的今日热点、新闻或时事查询任务转发给 Partner 本地配置的 Coze 今日热点查询智能体。",
      "version": "1.0.0",
      "tags": [
        "Coze",
        "今日热点",
        "新闻查询"
      ],
      "examples": [
        "查询今天的科技新闻热点"
      ],
      "inputModes": [
        "text/plain"
      ],
      "outputModes": [
        "text/plain"
      ]
    }
  ]
}
```

其中 `endPoints` 告诉 Leader 通过什么协议和地址调用 Partner，`skills` 告诉 Leader 这个 Partner
具体能完成哪些任务，`capabilities` 描述流式、通知、消息队列等技术能力。

## Leader 配置

Leader 已知的 Partner ACS 地址写在 Leader 机器的 `leader/config/partners.json`：

```json
{
  "partners": [
    {
      "name": "local-demo-partner",
      "acs_url": "http://127.0.0.1:8011/acs"
    }
  ]
}
```

Leader 的 LLM 配置写在 Leader 机器的 `leader/config/llm.json`，不需要通过终端 `export`：

```json
{
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "api_key": "replace-with-your-api-key",
  "base_url": null
}
```

如果 Leader 使用 OpenAI 兼容 API，把 `base_url` 改成对应服务地址，并把 `api_key` 改成该服务的密钥。

## 启动 Partner

```bash
# 该 sh 脚本使用 .venv 虚拟环境下的 python，需要改成自己部署的虚拟环境
./partner/start.sh
```
或
```bash
python ./partner/server.py
```

测试 ACS：

```bash
curl http://127.0.0.1:8011/acs
```

## 启动 Leader

如果 Leader 和 Partner 在同一台机器：

```bash
# 该 sh 脚本使用 .venv 虚拟环境下的 python，需要改成自己部署的虚拟环境
./leader/start.sh
```
或
```bash
python ./leader/run.py
```

启动后会进入多轮对话窗口：

```text
User>
```

也可以在启动时传入首轮问题。Leader 会先回答这一轮，然后继续等待下一轮输入：

```bash
./leader/start.sh "请找一个合适的 Partner 查询北京现在的天气。"
./leader/start.sh "请找一个合适的 Partner 查询今天的科技新闻热点。"
```

输入 `exit`、`quit`、`退出` 或空输入可以结束 Leader 对话窗口。

如果 Leader 和 Partner 在不同机器，需要先把 `partner/acs.json` 中的 `endPoints` 改成 Partner 机器局域网 IP，例如：

```json
"endPoints": [
  {
    "url": "http://192.168.1.20:8011/rpc",
    "transport": "JSONRPC"
  }
]
```

然后在 Leader 机器修改 `leader/config/partners.json`：

```json
{
  "partners": [
    {
      "name": "remote-demo-partner",
      "acs_url": "http://192.168.1.20:8011/acs"
    }
  ]
}
```

再运行：

```bash
./leader/start.sh
```

## 演示链路

```text
用户输入
  ↓
Leader LangChain ReAct Agent
  ↓
判断是否需要 Partner
  ↓
调用 list_partner_acs 读取 Partner ACS
  ↓
根据 ACS 能力描述选择 Partner
  ↓
从 list_partner_acs 返回结果中取得该 Partner 的 rpc_url
  ↓
调用 call_partner，把 rpc_url 和任务文本传入
  ↓
AipRpcClient.start_task / get_task / complete_task
  ↓
Partner aip_rpc_server /rpc
  ↓
Partner LangChain ReAct Agent 调工具执行任务
  ↓
Partner 返回 Product / TaskResult
  ↓
Leader 整合结果回复用户
```
