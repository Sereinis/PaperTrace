# BGE-M3 文本向量化接口

## 当前范围

提供将字符串列表转换为稠密文本向量的 Python 适配层，支持小批量请求、输入顺序恢复、响应校验与安全错误信息。暂不实现论文向量缓存、向量索引、相似度检索或问答。

新增文件用途：`papertrace/embeddings.py` 封装接口调用；`tests/test_embeddings.py` 提供不访问网络的模拟测试。本说明是接口使用文档，不是逐行源码解析稿。

## 已核对的接口契约

核对日期：2026-10-20。来源：[硅基流动创建嵌入请求文档](https://docs.siliconflow.cn/docs/api/embeddings-post)。

- 地址：`POST https://api.siliconflow.cn/v1/embeddings`，使用 HTTPS 和 Bearer 认证；不接受自定义地址，也不跟随重定向。
- 默认模型：`BAAI/bge-m3`。官方接口同时列出 `Pro/BAAI/bge-m3`，仅在明确指定时使用，不自动切换模型或计费档位。
- 请求包含 `model`、字符串数组 `input`、`encoding_format="float"`。
- BGE-M3 单条输入上限为 8192 tokens，不允许空字符串。字符数不等于 token 数；当前未安装模型 tokenizer，不做不准确的字符换算，不截断输入，超限由服务端拒绝。
- 当前官方接口页没有明确列出 BGE-M3 同步请求的最大输入条数。客户端默认每批 8 条，允许配置 1—32 条；这是客户端保护范围，不是官方保证。遇到服务端批量或总 token 限制应减小批次或缩短输入。
- 响应中的 `data[*].index` 对应当前批次的输入索引，不能假设响应数组已经排序。
- 客户端按 BGE-M3 的 1024 维稠密输出校验，拒绝模型名不匹配、数量错误、重复/越界索引、维度错误、非有限数值和零向量。不发送 `dimensions` 参数，官方接口说明该参数仅适用于指定 Qwen 模型。

平台可用模型、额度、价格与速率限制可能变化，以调用时官方文档和账号实际状态为准。本次账号已完成小规模连通性验证，但不承诺持续可用、免费或固定价格。

## 密钥与调用方式

可在项目根目录的 `.env` 文件填写 `SILICONFLOW_API_KEY=你的密钥`，也可配置同名进程环境变量。构造器允许 `api_key` 参数，供安全运行环境或交互式输入传入。优先级为显式参数、进程环境变量、项目根目录 `.env`；已有空值会报配置错误，不静默改用其他来源。不要将真实密钥硬编码、放进命令行参数或发到聊天中；不要记录请求头、完整请求体和服务端错误正文。

使用 `python-dotenv` 读取项目根目录的 `.env`，不向父目录搜索，也不依赖终端当前目录。读取时不做变量插值、不修改进程环境；HTTP 请求仍使用 Python 标准库。更新依赖后先运行 `python -m pip install -r requirements.txt`。

`.env` 是本地明文配置，不是加密存储。当前 Git 忽略规则已排除它，不能强制添加；文件如果曾被跟踪，忽略规则不能自动移除历史内容。不要把配置内容截图、输出或复制进日志。

```python
from papertrace.embeddings import EmbeddingError, SiliconFlowEmbedder

embedder = SiliconFlowEmbedder(batch_size=8, timeout=30, max_retries=0)
try:
    vectors = embedder.embed(["The model uses self-attention."])
    print("向量数量：", len(vectors), "维度：", len(vectors[0]))
except EmbeddingError as error:
    print(error)
    print("HTTP 状态：", error.status_code, "可稍后重试：", error.retryable)
```

空列表返回空列表，不联网；单个字符串、非字符串元素或空白文本在联网前报错。输入原文不被修改，返回值为 `list[list[float]]`。全部文本先校验再分批发送；后续批次失败时抛出异常，不返回部分成功结果，但之前成功的请求可能已产生费用。

## 超时、重试与安全边界

`timeout` 是 urllib 的网络操作超时，不是整个批处理的总时限。响应体最大读取 4 MiB，异常大响应直接拒绝。

默认 `max_retries=0`，避免超时后重复请求造成额外费用。显式设置 1—3 时，只对 HTTP 429、502、503、504，以及网络中断和超时有限重试，退避间隔为 1、2、4 秒。当前未解析 Retry-After，不适用于持续高并发调度。

认证、输入、模型不存在等 HTTP 错误不会重试；TLS 证书错误不会通过关闭验证或自动重试绕过。无效 JSON 和其他响应契约错误也不重试。

`EmbeddingError` 仅提供安全消息、HTTP 状态和可重试标记，不拼接原异常或服务端正文。不要用调试器、带局部变量的异常报告或额外 HTTP 调试日志记录密钥与输入原文。

## 测试与真实连通性验证

运行离线测试：

```powershell
py -3.12 -m pytest tests/test_embeddings.py -q
```

项目本身支持 Python 3.10 及以上；此处 `py -3.12` 是 Windows 开发环境的示例，也可使用已配置项目依赖的 `python`。

离线测试使用模拟响应，检查批次、排序、配置、输入、响应、重试与脱敏，不使用真实密钥，不发出计费请求。它不能代替真实 API 验收。

需要做一次小规模真实验证时，可在仓库目录执行以下命令，通过隐藏输入读取密钥，而不将密钥字面量写进终端历史：

```powershell
py -3.12 -c "from getpass import getpass; from papertrace.embeddings import SiliconFlowEmbedder; client = SiliconFlowEmbedder(api_key=getpass('SiliconFlow API key: ')); vectors = client.embed(['The Transformer uses self-attention.', 'BERT uses masked language modeling.']); print('count:', len(vectors), 'dimensions:', [len(v) for v in vectors])"
```

该命令会向硅基流动发送两条示例文本，可能消耗额度；只打印数量和维度，不打印密钥或完整向量。预期为两条 1024 维向量。账号权限、模型身份与服务可用性应在成功后记录，不应提前声明验证通过。

验证记录（2026-10-20）：使用本地配置调用默认模型 `BAAI/bge-m3`，单次请求发送上述两条英文示例文本，关闭自动重试，成功返回两条 1024 维向量，并通过模型身份、索引与数值等响应校验。验证未输出密钥或完整向量，未保存原始请求与响应。

当前全量离线测试为 217 项通过。真实调用仅验证此次授权与接口连通性、基本响应契约，不证明语义检索质量、费用水平、长文本与大批次支持情况或长期可用性；尚无向量检索比较结果。
