# 本地论文问答工作台

Spring Boot提供静态页面，Java负责业务接口、输入和内部响应校验，Python继续负责论文内检索、生成和引用绑定。页面不直接访问模型，不接收密钥。

## 启动

先在项目根目录启动Python（以下路径为本地开发产物，不随仓库分发）：

```powershell
python -m papertrace.service_cli `
  --chunks artifacts/attention-chunks.json `
  --vectors artifacts/retrieval-dev-v2-vectors-20261031.json `
  --query-cache artifacts/retrieval-query-cache-20261031 `
  --retrieval hybrid --port 8765
```

在另一个终端构建、启动Java：

```powershell
cd java-client
mvn package
java -jar target/papertrace-business-0.1.0-SNAPSHOT.jar
```

需要JDK21和Maven3.9+，详见[Java说明](../java-client/README.md)。打开 `http://127.0.0.1:8080/`，选择论文并输入问题。目前目录显示真实稳定ID与分块数；中文检索质量尚未验证，已验证开发问题为英文。可选上传配置见[本地PDF论文库](pdf-upload.md)，上传完成后目录自动刷新并选中新论文。

`--offline-query-cache`可禁止新查询向量请求，但不禁止DeepSeek生成。默认查询未命中时会生成向量，有候选时可能调用生成模型并计费。没有自动重试。

## 行为

- 目录由Python `/v1/documents` 经Java `/api/documents`投影，仅列出当前加载论文，不读取PDF、不触发检索或模型、不暴露路径/候选原文。目录变化需手动刷新；读取失败或空目录禁用提问。
- 提交到既有 `/api/questions`，等待期间禁用输入、论文选择及按钮。一次页面提交仅发一次请求；同页面防重复不是跨标签/用户限流。
- 答案逐条展示中文结论、英文精确原文、论文ID、页码及chunk ID；缺失项独立展示。完整、部分、证据不足区分显示，“完整”仍是模型自评。
- 原文与模型内容用纯文本节点显示，不执行HTML或Markdown。CSP限制同源资源及网络；未使用第三方CDN或浏览器持久存储，不提供历史会话。
- 显示合法生成token统计（非账单），未知或超出浏览器安全整数范围显示未知。失败只给固定分类提示，不透传上游message。超时/断连不证明上游取消或未计费，不能自动重试。
- 引用定位通过不等于语义支持验证，页面保留“语义支持：未验证”。没有PDF跳页预览、数据库、登录、多租户或用户API设置；上传是可选本地功能。

## 验证记录（2026-11-07）

Python目录测试覆盖聚合排序、结果隔离、空目录和无模型读取；Java HTTP测试覆盖目录投影/非法响应与静态资源；Node内置测试器加最小DOM替身覆盖页面状态、纯文本内容、防重复提交、缺失项、安全错误及用量。DOM替身不代替真实浏览器。

本轮全量Python1004项、Java51项、页面逻辑15项测试通过，Maven打包、Python编译、JavaScript语法与Git空白检查通过。页面逻辑测试命令：`node --test java-client/src/test/js/app.test.cjs`（项目根目录执行）。

实际浏览器→Spring Boot jar→Python→混合检索→DeepSeek限次验证已观察硬件开发题：1次生成、0次新向量，输入2996/输出74/总3070 tokens（账单未核对），无重试。页面等待时禁用输入，成功显示8块NVIDIA P100 GPU、100,000训练步、两条第7页引文与token统计；引文身份/原文偏移程序校验通过。未向模型发送参考答案或金标，未新增PDF视觉或用户答案确认，非独立质量测试。页面截图与脱敏审计保存在Git忽略的artifacts；本轮验证服务已关闭。

桌面浏览器已视觉检查；移动端含响应式CSS，但本轮未实际做手机视口验收。部分/不足/失败状态已离线测试，未额外计费生成这些状态。七字段报告内容问题仍未解决，该页面不是报告验收或公网部署。
