# 本地 PDF 上传与论文库

页面支持单用户本地流程：上传 PDF → Java 持久化文件与任务 → Python 解析、分块、向量化 → 目录可见 → 单篇证据问答。上传功能不包含登录、数据库、多租户、报告 API 或公网部署。

## 启动

在项目根目录启动 Python，使用被 Git 忽略的 artifacts 保存上传文件和缓存：

```powershell
python -m papertrace.service_cli --library-dir D:\jwPlan\PaperTrace\artifacts\local-library --port 8766
```

另一个终端启动 Java，必须指向同一绝对目录与 Python 端口：

```powershell
$env:PAPERTRACE_LIBRARY_DIR = 'D:\jwPlan\PaperTrace\artifacts\local-library'
$env:PAPERTRACE_INTERNAL_URL = 'http://127.0.0.1:8766'
java -jar java-client/target/papertrace-business-0.1.0-SNAPSHOT.jar
```

端口应选择本机空闲端口；Java 默认监听 `127.0.0.1:8080`。Python 仍支持原有 `--chunks/--vectors/--query-cache` 静态语料配置，并可同时指定论文库。上传后的论文使用混合检索；`--retrieval` 指定原静态语料模式。`--offline-query-cache` 同时作用于已上传论文的查询，不禁止新上传文件的分块向量化。

浏览器打开 `http://127.0.0.1:8080/`，选择文件并点击“上传并处理”。完成后目录自动刷新并选中新论文；刷新页面会继续读取未完成任务状态，不重新提交文件。

## 接口与状态

- Java `POST /api/uploads`：multipart `file`，返回 202 和任务；`GET /api/uploads` 读取全部任务。
- Python `POST /v1/import`：只接受 `{"document_id":"pdf_<sha256>"}`，从固定配置目录读取，不接受客户端路径。
- 任务状态为 `queued`、`processing`、`ready`、`failed`。单次只处理一个新文件，忙时拒绝另一个文件。
- 内容 SHA-256 为论文身份，原始文件名仅作纯文本显示。相同 PDF 重传复用原任务，包括失败状态，不自动重新计费或续跑；首版不提供重试/删除入口，故障恢复需要维护者检查文件和上下游状态。

目录内每篇保存 `source.pdf`、`task.json`、`processing.json`，成功后保存 `chunks.json`、`vectors.json`、`ready.json`；缓存保存在库的 `embedding-cache` 和 `query-cache`。产物使用同目录临时文件原子替换。Python 重启只加载完整 ready 清单，核验源文件和产物哈希、标准向量快照及论文归属；损坏时启动失败，不自动修复。Java 重启把 queued/processing 标记为 failed/interrupted，不自动续跑。

## 限制、费用与安全边界

仅支持文本型 PDF：文件最多 10 MiB、60 页、300,000 个清洗后字符、300 个 chunk。分块参数为 `max_chars=1200/overlap_chars=100`，页码为 PDF 页序。加密 PDF、无可提取文本的扫描件与超限文件拒绝；混合扫描页面不执行 OCR。PDF 表格和跨页文本仍受现有解析器限制。

缺失分块文本会发送到硅基流动 BGE-M3，每批最多 8 条，无自动重试；最大 300 块最多需要 38 批（重复文本和缓存可减少）。查询缓存未命中会发送查询做向量化，问答会调用 DeepSeek。密钥仅由既有 Python 适配器延迟加载，Java 和浏览器不接收或保存密钥。

`processing.json` 记录外层批次尝试数、文本尝试数和缓存统计；token 用量/账单未知，不按文本长度估算。导入等待上限 900 秒，Java 超时、页面关闭、状态读取失败不保证 Python 已取消或不会继续计费。Python 成功但 Java 状态保存失败或超时，可能出现目录已可问答而任务显示失败，需要人工核对，不自动协调重跑。

该服务只监听本机，不具备用户鉴权、PDF 解析进程隔离、解析CPU/内存硬限额或跨进程库锁；同一库只能运行一组 Java/Python 进程，不应暴露公网或处理不可信任来源的大批文件。目录哈希用于完整性检查，不是安全签名。

当前验证使用离线模拟向量与生成器，覆盖实际 PDF 解析、分块、标准快照、重启恢复、查询缓存及问答。实际 Java jar → Python 跨进程联调已验证 multipart 上传、ready、目录、重复上传复用及问答引用投影，真实 API 调用为 0；尚未完成上传流程的真实计费 API 或浏览器视觉验收。引用定位通过不等于语义支持，`semantic_support=not_verified` 不变，实验报告内容尚未验收。
