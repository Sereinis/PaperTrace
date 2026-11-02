# PaperTrace

PaperTrace 是一个面向英文计算机论文的实验信息提取与证据追踪项目，目标是将论文中的研究问题、数据集、模型、Baseline、训练设置、评价指标和实验结果整理为结构化报告，并为每条信息保留原文页码或段落证据。

## 当前状态

v0.1 PDF 解析验证已经完成。v0.2 已支持保守文本清洗、章节感知分块、chunk JSON 输出、英文关键词检索、本地余弦向量检索与 RRF 混合检索。硅基流动 BGE-M3 适配层支持可追溯向量快照及内容缓存。已完成固定 v2 开发集对比，以及新增 ResNet、U-Net、GAN 三篇论文的首次留出测试；两组数据分别记录，不混合汇总。留出测试的参考答案与论文位置经人工核对，chunk 标注仍为 AI 辅助、未经独立人工穷尽核验。带证据问答和结构化实验抽取尚未实现。

已完成的 v0.1 范围：

1. 读取可复制文本的英文 PDF；
2. 按页保存解析文本；
3. 对解析结果进行人工抽查；
4. 为后续检索和结构化抽取建立稳定数据结构。

三篇本地样本论文均已完成解析验证。详细结果见 [v0.1 验证记录](docs/v0.1-validation.md)。

## 环境要求

- Python 3.10 或更高版本；
- 首版只支持可复制文本、结构较清晰的英文 PDF。

安装运行依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需运行测试：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## 使用方法

直接在终端按页查看文本：

```powershell
python -m papertrace.cli data/papers/attention_is_all_you_need.pdf
```

将解析结果保存为 JSON：

```powershell
python -m papertrace.cli data/papers/attention_is_all_you_need.pdf --output artifacts/attention.json
```

JSON 中会保存论文文件名、总页数，以及每页的页码、原始文本、清洗后文本、两者字符数和是否包含文本。`artifacts/` 默认不会提交到 Git。

原始文本不会被覆盖，清洗后的文本用于后续分块和检索；这样可以在发现清洗规则误判时回到原始证据。

## 章节感知分块

使用 `--chunks-output` 可以从 PDF 直接生成带章节信息的 chunk JSON：

```powershell
python -m papertrace.cli data/papers/attention_is_all_you_need.pdf `
  --chunks-output artifacts/attention-chunks.json `
  --max-chars 1200 `
  --overlap-chars 100
```

输出中的每个 chunk 包含页码、章节标题与分类、稳定 ID、文本、字符数和在清洗后页面文本中的字符范围。默认文档标识使用 PDF 文件名；需要固定标识时可传入 `--document-id`。

章节识别和分块采用规则方法。首个识别标题之前的内容标为 `unknown`，没有新标题的后续页面继承上一章节；表格、公式、扫描件和跨行标题仍可能需要人工检查。

## 检索评测

`papertrace.keyword_search.KeywordRetriever` 接收 chunk 列表，使用 BM25 风格评分返回带页码、章节、分数和排名的候选证据。

新增独立命令 `python -m papertrace.eval_cli` 可以读取固定的 chunk JSON 和问题集，输出 Recall@K、Hit@K、MRR@K 及逐问题结果。当前 v2 开发集包含三篇论文的 12 个英文问题和 24 个问题—证据关联，由 AI 辅助构建，并对照解析文本及相关 PDF 页面复核，尚未经过独立人工验证。评测结果用于开发诊断，不作为泛化能力的充分证据。

完整运行命令、指标口径和失败案例见 [检索评测协议](docs/retrieval-evaluation.md)。当前评测限定为指定论文内检索，不评估最终答案正确率。

`python -m papertrace.vector_eval_cli` 在同一语料、查询和金标上比较关键词、BGE-M3 稠密向量与 RRF 混合检索，输出总体、分论文和逐问题结果。分块向量取自完整快照，查询向量按文本和模型配置缓存；缓存全部命中时无需密钥或网络。首次生成缺失向量会调用计费 API。

```powershell
python -m papertrace.vector_eval_cli --vectors artifacts/retrieval-dev-v2-vectors-20261031.json --dataset data/eval/retrieval-dev.json --query-cache-dir artifacts/retrieval-query-cache-20261031 --output artifacts/retrieval-comparison-v2-new.json --ks 1 3 5
```

2026-10-31 的固定 v2 开发集结果：关键词 / 向量 Recall@5 为 0.8528 / 0.9250，MRR@5 为 0.9167 / 0.8958；第一版 RRF 混合检索 Recall@5 为 0.9389、MRR@5 为 0.9583。该混合结果仅来自已观察的开发集，尚不足以证明泛化能力；完整口径和退化案例见评测协议。

2026-11-02 的首次留出测试包含 12 个英文问题、130 个候选分块、38 个问题—证据关联，问题与金标在查看排名前冻结：

| 方法 | Recall@5 | MRR@5 | Hit@5 |
| --- | ---: | ---: | ---: |
| 关键词 | 0.7306 | 0.7667 | 11/12 |
| BGE-M3 向量 | 0.7417 | 0.8333 | 11/12 |
| RRF 混合 | 0.7660 | 0.8403 | 12/12 |

这些是按问题宏平均的指标；部分直接证据也计相关，Hit@5 不代表完整回答。混合在该样本总体指标最高，但累计命中证据少于纯向量，也有逐题退化。冻结数据与不含原文的逐问排名摘要分别位于 [测试集](data/eval/retrieval-independent-v1.json) 和 [结果摘要](data/eval/retrieval-independent-v1-results.json)；标注、复现与失败案例见 [首次留出测试记录](docs/independent-test-v1.md)。结果现已观察，后续用于回归检查和误差分析，不再属于未见测试。

## 当前限制

- 暂不处理扫描版 PDF 和 OCR；
- 复杂公式、表格和双栏阅读顺序可能存在解析误差；
- 检索对比仅覆盖小规模英文开发集和首次留出测试；完整 chunk 金标未经独立人工穷尽核验，没有无答案问题，尚未验证跨论文检索、最终答案或结构化实验字段抽取。

## BGE-M3 文本向量化

`papertrace.embeddings.SiliconFlowEmbedder` 调用硅基流动文本向量化接口，将字符串列表转换为按输入顺序排列的 1024 维稠密向量。`papertrace.embedding_cache.embed_chunks` 将向量与当前分块元数据绑定，支持按内容复用本地缓存。`papertrace.vector_search.VectorRetriever` 会将查询文本向量化，在本地计算余弦相似度并返回带来源元数据的 Top-K 分块；固定开发集对比入口为 `papertrace.vector_eval_cli`。

密钥可填写在项目根目录已被 Git 忽略的 `.env` 中：`SILICONFLOW_API_KEY=你的密钥`。模块依次使用显式 `api_key` 参数、进程环境变量、项目根目录 `.env`，不会覆盖已有环境变量。不要写入源码或提交 Git。安装依赖后即可读取配置，不自动保存密钥。

```python
from papertrace.embeddings import SiliconFlowEmbedder

embedder = SiliconFlowEmbedder()
vectors = embedder.embed([
    "The Transformer uses self-attention.",
    "BERT uses masked language modeling.",
])
print(len(vectors), len(vectors[0]))
```

接口契约、输入限制、安全配置和小规模连通性验证方法见 [向量化接口说明](docs/embeddings.md)。2026-10-20 真实调用返回两条 1024 维向量；这仅验证该次接口连通性与响应结构，不代表检索质量或长期服务稳定性。

对现有论文 chunk JSON 生成向量快照：

```powershell
python -m papertrace.embed_cli --chunks artifacts/attention-chunks.json --cache-dir artifacts/embedding-cache --output artifacts/attention-vectors.json
```

首次运行会发送缺失文本到硅基流动，可能消耗额度；相同模型与文本再次运行复用缓存。命令拒绝覆盖已有输出，复跑使用新的输出文件名。本项目将缓存和完整向量快照保存在被 Git 忽略的 `artifacts/` 下。缓存身份、失败恢复与命令参数见 [分块向量化与缓存](docs/embedding-cache.md)。

使用向量快照执行一次本地检索：

```python
from papertrace.embeddings import SiliconFlowEmbedder
from papertrace.vector_search import VectorRetriever, load_vector_snapshot

snapshot = load_vector_snapshot("artifacts/attention-vectors.json")
retriever = VectorRetriever(snapshot, SiliconFlowEmbedder())
results = retriever.search("How does self-attention work?", top_k=5)
for result in results:
    print(result["rank"], result["score"], result["page_number"], result["text"])
```

查询文本会产生一次新的向量化请求；论文分块向量从本地快照读取，不会重复生成。首次使用真实快照前，应确认其中的模型、维度和分块来源与当前实验配置一致。

检索入口仅接受 `embed_chunks` / `embed_cli` 生成的标准快照，不再接受裸记录列表或缺少配置的对象。加载和初始化都会检查版本、完整模型配置、记录数、1024 维非零有限向量，以及分块的唯一 ID、论文 ID、正整数页码、非空文本和合法 JSON 元数据。分块顺序和元数据的规范化 SHA-256 必须与 `chunks_sha256` 一致。客户端必须显式声明相同 `model`；使用 Pro 模型快照时，也需以相同模型创建客户端。

语料指纹用于发现分块内容、顺序与快照声明不一致，不是安全签名，也不能证明有限向量确实由该文本和模型生成；快照没有向量内容校验和。不应手工过滤或修改快照记录来绕过检查。底层配置之外的远端模型更新也无法仅凭模型名称识别。

内部按最大绝对值缩放后归一化，避免有限极端数值平方溢出；排序范围包含零分和负分，相似度不是相关性概率。检索结果深拷贝元数据，修改返回值不会影响后续查询。客户端错误不返回部分排名，检索层不自动重试；适配层自身是否重试仍取决于其配置。

## 页内文本分块

`papertrace.chunker.chunk_page` 可以将清洗后的单页文本转换为可追溯的文本块：

```python
from papertrace.chunker import chunk_page

chunks = chunk_page(
    text=page["cleaned_text"],
    page_number=page["page_number"],
    document_id="attention_is_all_you_need",
    max_chars=1200,
    overlap_chars=100,
)
```

每个文本块包含稳定 ID、页码、块序号、文本、字符数和原页面文本中的字符范围。分块器优先在段落、句末、换行或空格处切分；只有超长且没有合适边界的文本才按字符上限拆分。

当前分块严格限制在单页内，不恢复表格结构，也不合并跨页句子。

## 本地样本

`data/papers/` 中的 PDF 仅用于本地开发和人工测试，默认不会提交到 Git。论文来源、授权和使用范围需要在后续数据说明中单独记录。

## 开发原则

- 先完成可验证的小闭环，再增加向量数据库、大模型和 Agent；
- 真实记录实现状态和实验结果，不伪造提交历史或指标；
- 第一版只承诺英文、可复制文本、结构较清晰的 PDF；
- 核心产出是结构化实验报告、证据追踪和评测结果，而不是通用 PDF 聊天功能。
