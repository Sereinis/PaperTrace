# PaperTrace

PaperTrace 是一个面向英文计算机论文的实验信息提取与证据追踪项目，目标是将论文中的研究问题、数据集、模型、Baseline、训练设置、评价指标和实验结果整理为结构化报告，并为每条信息保留原文页码或段落证据。

## 当前状态

v0.1 PDF 解析验证已经完成。v0.2 已支持保守文本清洗、章节感知分块、chunk JSON 输出和英文关键词检索基线，并提供小规模离线检索评测。新增硅基流动 BGE-M3 文本向量化适配层，已通过离线模拟测试及两条示例文本的真实接口连通性验证。向量检索、带证据问答和结构化实验抽取尚未实现。

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

## 关键词检索评测

`papertrace.keyword_search.KeywordRetriever` 接收 chunk 列表，使用 BM25 风格评分返回带页码、章节、分数和排名的候选证据。

新增独立命令 `python -m papertrace.eval_cli` 可以读取固定的 chunk JSON 和问题集，输出 Recall@K、Hit@K、MRR@K 及逐问题结果。当前 v2 开发集包含三篇论文的 12 个英文问题和 24 个问题—证据关联，由 AI 辅助构建，并对照解析文本及相关 PDF 页面复核，尚未经过独立人工验证。评测结果用于开发诊断，不作为泛化能力的充分证据。

完整运行命令、指标口径和失败案例见 [检索评测协议](docs/retrieval-evaluation.md)。当前评测限定为指定论文内检索，不评估最终答案正确率。

## 当前限制

- 暂不处理扫描版 PDF 和 OCR；
- 复杂公式、表格和双栏阅读顺序可能存在解析误差；
- 尚未实现向量检索和结构化实验字段抽取。

## BGE-M3 文本向量化

`papertrace.embeddings.SiliconFlowEmbedder` 调用硅基流动文本向量化接口，将字符串列表转换为按输入顺序排列的 1024 维稠密向量。该模块尚未接入分块、索引或检索评测流程，不是完整的向量检索实现。

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
