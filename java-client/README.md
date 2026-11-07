# PaperTrace Java 业务后端

`PaperTracePythonClient` 是 JDK 21 `HttpClient` 的最小调用骨架，当前只允许访问本机 HTTP 服务：

```java
var client = new PaperTracePythonClient("http://127.0.0.1:8765");
String json = client.ask("What mechanism does the model use?", "paper-a");
```

基于 Java 21、Spring Boot 3.5.7 与 Maven，提供本机单问题业务接口，调用独立 Python HTTP 服务，不通过子进程执行 Python。不包含认证、数据库、多租户、报告 API、公网部署或线上用户 API 密钥配置。

## Spring Boot 骨架

`pom.xml` 与 `src/main/java/com/papertrace/business` 已加入 Java 业务层最小骨架：

- `POST /api/questions` 接收 `query`、`documentId`。
- `GET /api/documents` 返回 `documents[].documentId/chunkCount`，只读已加载论文目录，不返回原文、文件路径或模型配置，不触发模型。
- `GET /` 提供原生HTML/CSS/JavaScript问答页面，无额外前端框架或构建依赖。只向同源Java API发请求，不直接访问Python或模型服务。
- Controller 使用 Bean Validation 校验输入。
- Application Service 校验协议版本、请求归属、状态一致性和引用字段，投影为 `QuestionResponse`：答案、结论、引用、缺失信息和校验状态均采用强类型 DTO，省略候选全文和生成元数据。
- `PAPERTRACE_INTERNAL_URL` 仅用于配置 Python 内部地址，客户端仍会拒绝非本机 HTTP 地址。

## 构建与运行

```powershell
cd java-client
mvn test
mvn package
java -jar target/papertrace-business-0.1.0-SNAPSHOT.jar
```

需要 JDK 21、Maven 3.9+。本机 Maven 不在 PATH，但 wrapper 缓存中的 3.9.16 可显式调用；构建前用 `mvn -version` 确认 JAVA_HOME 与实际 JDK。已验证 JDK21 构建打包、Spring Boot 实际启动、Java HTTP 集成测试和实际 Python 服务跨进程模拟联调。

Java 默认监听 `127.0.0.1:8080`。请求严格接收 `query`（最长4000个Java字符单位）与 `documentId`（最长256个字符单位），拒绝空白及多余字段。成功响应使用 camelCase，如 `result.claims[].citations[].chunkId`、`result.missingInfo`、`result.validation.semanticSupport`。引用偏移保留 Python Unicode 码点口径，不是 Java UTF-16 下标。`semanticSupport` 始终为 `not_verified`，`answered` 仍为模型自评。

浏览器打开 `http://127.0.0.1:8080/` 即可使用页面；Python需另行启动。目录失败可手动刷新，空目录不能提问，等待期间禁止同页重复提交；错误/超时不自动重试。模型内容只以textContent输出，不解释HTML或Markdown；没有第三方CDN资源，不保存浏览器历史数据。详细步骤与验证边界见[页面说明](../docs/question-workspace.md)。

前端离线逻辑测试使用Node内置测试器，无需npm安装：

```powershell
node --test src/test/js/app.test.cjs
```

该测试使用最小DOM替身验证状态、纯文本渲染和请求次数，不替代浏览器视觉或移动端测试。

输入错误返回400；未知论文404、忙503、Java内部请求超时504，检索/生成/引用契约问题分别以白名单错误码返回502，上游上下文超限422。未知错误回退脱敏502，不重试、不回显上游正文。成功 `result.usage` 和错误外层 `usage` 返回合法生成token统计，未知为null，不估算账单。

内部连接超时3秒，请求默认120秒，可用 `PAPERTRACE_INTERNAL_TIMEOUT_SECONDS` 配置1—300秒。超时不证明Python或模型任务已取消，已发送请求可能计费，不能自动重试。当前仍为同步问答，异步任务与公网限流未实现。

Python服务可用 `python -m papertrace.service_cli` 独立加载论文、向量快照和查询缓存，详见 [真实服务启动与错误分类](../docs/internal-service.md)。Java不启动Python，不加载模型密钥。已通过一次真实开发题端到端冒烟：1次DeepSeek、0新向量、3071 tokens，引用和结构验证不等于总体语义验收。
