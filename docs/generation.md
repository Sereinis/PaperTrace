# DeepSeek 文本生成适配层

## 当前范围

`papertrace/generation.py` 提供 `TextGenerator` 协议、`GenerationResult` 结果结构和 `DeepSeekGenerator` 官方 API 实现。该层只负责文本生成与响应边界，不读取 PDF、不执行检索，也不验证答案是否被论文证据支持。上层已提供[带引用问答与引用定位检查](evidence-qa.md)的离线模拟闭环；结构化实验抽取及线上用户密钥管理尚未实现。

生成协议使用 `generate(messages, *, max_tokens, temperature, json_mode)`；后续问答逻辑可以依赖协议，其他服务商通过单独的适配器接入。当前 DeepSeek 实现不接受自定义 URL，不把任意服务地址视作已验证兼容。

## 接口与默认配置

2026-11-02 核对官方[首次调用说明](https://api-docs.deepseek.com/zh-cn/)、[Chat Completions API](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/)和[模型价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。

- 固定地址：`POST https://api.deepseek.com/chat/completions`，HTTPS Bearer 认证，不跟随重定向。
- 默认模型：`deepseek-flash`；可显式选择 `deepseek-v4-pro`，不自动切换模型。
- 每次显式发送 `thinking={"type":"disabled"}` 和 `stream=false`，不发送工具定义；适配层暂不开放思考、工具或流式功能。
- 默认输出上限为 2048 tokens，允许 1—8192。这是客户端保护范围，不是官方模型最大输出长度。默认温度 0，允许 0—2；低温不保证远端输出确定或正确。
- 请求体最多 1 MiB，响应体最多 2 MiB；字节限制不是 token 计数。输入不自动截断，官方上下文限制仍由服务端执行。
- 默认超时 60 秒，允许大于 0 且不超过 300 秒；它是 urllib 网络操作超时，不是整个请求的严格墙钟总时限。

官方模型名可能指向更新后的版本。结果分别记录 `requested_model` 和 `response_model`，不要求返回别名与请求字符串相同，也不由模型名推断实际权重或长期可复现性。返回模型名只作为服务端声明保留，不验证服务端是否实际使用该权重。

## 配置和使用

密钥优先级为：显式 `api_key` 参数、进程环境变量 `DEEPSEEK_API_KEY`、项目根目录 `.env`。已有空值会报错，不降级到下一来源。使用已有 `python-dotenv` 依赖，固定读取项目配置，不向父目录搜索、不展开变量、不修改进程环境。

在已有本地 `.env` 中自行添加 `DEEPSEEK_API_KEY=你的密钥`，保留现有 `SILICONFLOW_API_KEY` 配置。仓库中的 `.env.example` 只含空占位项。实际 `.env` 未由本次实现读取、改写或输出；测试仅使用临时文件和虚构密钥。

以下代码会向 DeepSeek 发送消息并可能计费，仅在配置密钥且准备真实调用时运行：

```python
from papertrace.generation import DeepSeekGenerator, GenerationError

generator = DeepSeekGenerator()
try:
    result = generator.generate([
        {"role": "system", "content": "只根据提供的证据回答；缺少依据时说明。"},
        {"role": "user", "content": "证据：The model uses self-attention.\n问题：模型使用什么机制？"},
    ])
    print(result.content)
    print(result.requested_model, result.response_model, result.usage)
except GenerationError as error:
    print(error)
```

消息列表必须至少包含一条 user 消息；每条仅接受 `role` 和非空字符串 `content`，角色限 system/user/assistant。不接受图像、工具、任意附加请求字段或空白消息。完整输入先校验再发送，调用不修改输入列表。

`json_mode=True` 时消息需明确出现 JSON 格式要求，客户端发送 `response_format={"type":"json_object"}`。返回内容还会解析为 JSON 对象，拒绝重复键、NaN/Infinity 等非法常量、数组与无效 JSON；不在这一层验证业务字段、引用或事实正确性。

## 响应、失败与费用边界

只接受 `chat.completion`、恰好一条 index=0 的 assistant 结果、`finish_reason=stop` 和非空 UTF-8 正文。截断、过滤、工具调用、中断、意外思考内容和畸形响应一律报错，不返回部分答案。`stop` 只表示协议上结束，不保证语义完整或证据充分。

成功结果保存正文、请求/返回模型、服务端请求 ID、结束原因和已校验的 token 统计。用量缺失时为 `None`，不填零或估算；存在时检查基础总量和缓存字段一致性，不保留未知用量字段。不保存原始响应或思考正文，不自动写入文件。结果默认 repr 隐藏答案正文，但显式访问 `content` 仍可得到内容。

所有失败均不自动重试，包括 HTTP 429、服务端故障、超时和网络中断。失败或无效响应可能已产生费用，重跑属于一次新请求。`GenerationError` 包含安全说明、可选 HTTP 状态和 `usage`：响应可解析且用量合法时，即使正文或结束原因不合格也保留统计；网络失败、不可解析或非法统计保持 `None`。不保存原始失败输出，不回显密钥、提示词、服务端错误正文或底层异常链。TLS 校验不关闭，标准库默认环境代理设置仍可能生效。

密钥只保存在本地明文配置和客户端内存中，当前不是多用户安全存储。默认不会打印请求头、正文、完整响应或密钥；调试器、局部变量异常报告与额外 HTTP 日志不在该保证范围内。论文片段和问题会发送给 DeepSeek，生产场景的隐私、保留政策和用户密钥方案需要另行确认。

## 验证状态

```powershell
python -m pytest tests/test_generation.py -q
```

离线测试覆盖默认请求、模型配置、密钥读取优先级、输入校验、JSON 模式、完整结束检查、token 统计、禁止重定向、大小限制、连接关闭与错误脱敏。离线测试中的响应和密钥均为模拟数据，不发起网络请求。

2026-11-02，使用本地配置完成一次官方 API 连通性验证：通过适配层请求 `deepseek-flash`，关闭思考、非流式、温度 0、`max_tokens=64`，启用 JSON 模式；只发送要求返回 `{"ok": true}` 的通用消息，不包含论文或个人信息。传输入口计数为 1，无自动重试，返回模型为 `deepseek-flash`、结束原因为 `stop`，JSON 对象与预期一致。服务端报告输入 39 tokens（缓存命中 0、未命中 39）、输出 5 tokens，总计 44 tokens；未核对实际账单。单次耗时约 0.98 秒，仅为观察，不是性能基准。

验证过程未输出密钥、认证头或原始响应，未修改本地 `.env`。该结果仅证明当次连通性、基础 JSON 输出与响应校验通过，不代表论文问答质量、长期稳定性或线上用户 API 配置已经验证。
