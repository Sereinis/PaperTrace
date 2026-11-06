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

输入错误返回400，内部调用失败或协议不合规返回脱敏502，不重试、不回显上游正文。当前内部请求超时10秒，适用于本地模拟联调；真实模型长耗时任务与错误码细分尚未接入。Python 服务须由启动方另行注入语料、检索器和生成器；Java 不自动启动 Python 或加载模型密钥。
