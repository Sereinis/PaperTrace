# PaperTrace

PaperTrace 是一个面向英文计算机论文的实验信息提取与证据追踪项目，目标是将论文中的研究问题、数据集、模型、Baseline、训练设置、评价指标和实验结果整理为结构化报告，并为每条信息保留原文页码或段落证据。

## 当前状态

v0.1 PDF 解析验证已经完成。项目支持按页提取 PDF 文本，并可将带页码的结果保存为 JSON。

第一阶段目标是完成 v0.1：

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

## 当前限制

- 暂不处理扫描版 PDF 和 OCR；
- 复杂公式、表格和双栏阅读顺序可能存在解析误差；
- 当前只完成页级解析，尚未实现文本分块、检索和结构化实验字段抽取。

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

当前分块严格限制在单页内，不恢复表格结构，也不合并跨页句子。章节识别将在后续阶段单独实现。

## 本地样本

`data/papers/` 中的 PDF 仅用于本地开发和人工测试，默认不会提交到 Git。论文来源、授权和使用范围需要在后续数据说明中单独记录。

## 开发原则

- 先完成可验证的小闭环，再增加向量数据库、大模型和 Agent；
- 真实记录实现状态和实验结果，不伪造提交历史或指标；
- 第一版只承诺英文、可复制文本、结构较清晰的 PDF；
- 核心产出是结构化实验报告、证据追踪和评测结果，而不是通用 PDF 聊天功能。
