# PaperTrace 内部服务联调边界

当前阶段建立 Java 业务后端与 Python 问答服务之间的最小单问题 HTTP 链路。

- `GET /healthz`：返回服务版本。
- `POST /v1/question`：请求体严格为 `query` 与 `document_id`，成功响应包含既有 `answer_question` 结果。
- Python 服务只允许监听 `127.0.0.1`、`localhost` 或 `::1`，请求体上限为 256 KiB，默认不记录访问日志。
- Python 的语料、检索器工厂和生成器由启动方显式注入；服务层不会读取密钥或隐式创建 DeepSeek 客户端。
- 失败响应不回显问题正文、候选证据、模型正文或密钥。
- 现有结果仍保留 `semantic_support=not_verified`；HTTP 联调成功不等于报告内容验收通过。

Java 侧已有可运行的 Spring Boot 业务层：`POST /api/questions` → `QuestionApplicationService` → `PaperTracePythonClient` → Python `/v1/question`。业务层校验内部响应并投影为强类型答案、结论、引用及校验 DTO，省略候选全文和生成元数据。Java 默认也只监听 `127.0.0.1`。

已在 JDK21 下完成 Maven 构建、Spring Boot 启动及 Java HTTP 集成测试，并用实际 Python 服务与模拟生成器完成跨进程 HTTP 联调，真实模型请求为零。最初 PATH 中没有 Maven，后在本机 wrapper 缓存找到可用安装，无需全局安装或修改 PATH。

## 真实论文服务启动

```powershell
python -m papertrace.service_cli `
  --chunks artifacts/attention-chunks.json `
  --vectors artifacts/retrieval-dev-v2-vectors-20261031.json `
  --query-cache artifacts/retrieval-query-cache-20261031 `
  --retrieval hybrid --port 8765
```

上述路径是本地开发产物示例，不随仓库分发；部署时需另备合法语料与标准快照。启动时完整核对快照和输入chunk元数据，允许标准快照包含其他论文，但检索仍严格在请求论文内进行。不会自动解析PDF或补生成分块向量。

查询缓存复用既有按文本和完整模型配置计算的身份，缓存未命中默认会通过硅基流动BGE-M3生成一条查询向量、无重试。`--offline-query-cache`要求所有查询精确命中，否则返回retrieval_failed，不联网。keyword模式只需 `--chunks ... --retrieval keyword`，不接受向量配置。

DeepSeek密钥只在有候选且问答预检通过后由既有适配器加载；向量密钥只在缓存未命中时由既有适配器加载。服务不接收前端URL、模型密钥或任意生成参数。`--generation-timeout`默认60秒，`--embedding-timeout`默认20秒，各自范围为大于0且不超过60秒。单次生成上限默认2048 tokens，可由启动方配置。健康检查不访问模型，未知论文在检索前返回404。

## 超时、并发与失败

Java连接超时3秒，内部请求默认120秒（`PAPERTRACE_INTERNAL_TIMEOUT_SECONDS`可配置1—300秒）。这是有限的同步HTTP等待，不是端到端严格执行时限；Java超时/客户端断开不能证明上游已取消或未计费，因此不自动重试。

Python读取请求头/体的套接字超时为5秒；拒绝重复JSON键、非JSON内容类型与传输编码，要求定长请求。问答任务单实例最多一个正在执行，忙时503，不排队生成；这不是HTTP连接线程总量限制，也不是公网抗DoS机制。

| 分类 | Java HTTP状态 | 含义 |
| --- | --- | --- |
| document_not_found | 404 | 指定论文未加载 |
| invalid_input / invalid_json | 400 | 输入不符合约定 |
| context_too_large | 422 | 证据上下文超限，未自动截断 |
| retrieval_failed | 502 | 检索或查询向量化失败 |
| generation_failed | 502 | 生成失败，可能已计费 |
| citation_invalid / answer_contract_invalid / invalid_generation_result | 502 | 引用或输出契约无效，不返回部分答案 |
| service_busy | 503 | 正在处理另一个问答任务 |
| internal_timeout | 504 | Java等待内部调用超时，用量未知 |
| internal_service_unavailable | 502 | 连接失败、未知/不合规上游错误或成功响应畸形 |

Python错误与Java成功/错误响应只传递经过校验的生成token统计，未知为null；不估算账单，不透传上游message。Java成功响应的 `result.usage` 包含 `promptTokens/completionTokens/totalTokens`。错误外层的usage只在Python能提供合法用量时存在，不能把null理解为零消费。部分低层HTTP错误仍归为内部服务不可用；错误分类不包含全部生产运维场景。

## 真实开发样例验证（2026-11-06）

实际Spring Boot jar → Python服务 → 54块Transformer论文候选 → 混合检索 → DeepSeek：已观察硬件开发题返回8块NVIDIA P100 GPU及100,000训练步，两条第7页引用身份、原文、偏移及Java投影与token统计核对通过。查询精确命中旧缓存，0次新向量、1次DeepSeek、无重试，输入2996/输出75/总3071 tokens，未核账单。未发送参考答案或金标，未重新视觉核对PDF，未获得新增用户内容确认，不作独立质量结论。

未知论文经Java返回404且不调用生成器；样例验证后两服务已关闭。完整原始问答候选不由Java公开。Python的 `serve` 仍可供嵌入式调用；正式命令入口使用 `service_cli`。详细Java构建见 [后端说明](../java-client/README.md)。线上用户API配置、鉴权、多租户、报告接口和部署仍未实现。
