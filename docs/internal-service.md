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

Java 目前将上游错误统一转换为脱敏502；错误码细分、长耗时任务、鉴权和并发限额尚未完成。Python 的 `serve` 返回 server 对象，由启动方调用 `serve_forever()` 并负责关闭；没有自动启动真实模型服务的入口。详细构建和业务响应字段见 [Java 后端说明](../java-client/README.md)。线上用户 API 配置尚未实现。
