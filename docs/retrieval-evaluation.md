# 关键词检索评测协议

## 目标与范围

以固定的小规模开发问题集检查 BM25 风格关键词检索，记录正确证据能否进入前 K 条及其排名。当前只做指定论文内检索，不生成回答，也不评估字段抽取或答案正确率。

问题集信息见 [数据说明](../data/eval/README.md)。当前使用 v2 标注：本开发评测集由 AI 辅助构建，并对照解析文本及相关 PDF 页面复核，尚未经过独立人工验证。评测结果用于开发诊断，不作为泛化能力的充分证据；不能用该分数推断其他论文或中文查询的检索能力。

## 指标定义

对每个问题，G 为非空的已标注相关 chunk ID 集合，R_K 为排名前 K 条的 chunk：

- Recall@K = 前 K 条命中的金标数 / 全部金标数。
- Hit@K = 前 K 条有任意金标时为 1，否则为 0。
- RR@K = 前 K 条中首个金标的倒数排名，未命中为 0。
- MRR@K = 所有问题 RR@K 的平均值，不是无限制的 MRR。

总体 Recall、Hit、MRR 均按问题做宏平均，分论文也按各自问题宏平均。空结果计零，不跳过。少于 K 条时使用实际返回结果。重复 chunk ID、重复问题 ID、未知或跨论文金标、空金标和非法 K 都会报错，不静默排除问题。

例如金标为 A、B，检索顺序为 X、A、B：Recall@2 为 1/2，Hit@2 为 1，RR@2 为 1/2；Recall@3 为 1，RR@3 仍为 1/2。

## 复现

在项目根目录执行（本地需已有对应的三篇 PDF）：

```powershell
python -X utf8 -m papertrace.cli data/papers/attention_is_all_you_need.pdf --document-id attention_is_all_you_need --chunks-output artifacts/eval-attention-chunks.json --max-chars 1200 --overlap-chars 100
python -X utf8 -m papertrace.cli data/papers/bert_pretraining.pdf --document-id bert_pretraining --chunks-output artifacts/eval-bert-chunks.json --max-chars 1200 --overlap-chars 100
python -X utf8 -m papertrace.cli data/papers/lora.pdf --document-id lora --chunks-output artifacts/eval-lora-chunks.json --max-chars 1200 --overlap-chars 100
python -X utf8 -m papertrace.eval_cli --chunks artifacts/eval-attention-chunks.json artifacts/eval-bert-chunks.json artifacts/eval-lora-chunks.json --dataset data/eval/retrieval-dev.json --output artifacts/retrieval-eval-v2.json --ks 1 3 5
```

评测器核对全部论文快照指纹、证据页码和证据子串，防止分块更新后沿用旧金标。JSON 输出包含总体、分论文、逐问题指标和前 max(K) 条完整候选 chunk，同时记录 BM25 参数、数据集哈希、语料指纹及运行环境。报告和 chunk 默认保存在忽略的 `artifacts/` 中，不提交 Git。算法单元测试使用独立合成数据，另有仓库开发集的结构一致性测试，均不依赖本地 PDF；结构测试不能代替语义复核。

## v1 首轮结果（2026-10-18）

247 个 chunks（Transformer 54、BERT 85、LoRA 108），12 个问题、15 个问题—证据关联；BM25 参数保持原样：k1=1.5、b=0.75。v1 问题和金标在首次检索运行前固定，保存在提交 `bc806ff`。下表是 v1 历史结果，不是当前 v2 的结果；本地原报告为 `artifacts/retrieval-eval.json`。

| K | Recall@K | Hit@K | MRR@K |
| --- | --- | --- | --- |
| 1 | 0.6528 | 0.7500 | 0.7500 |
| 3 | 0.8472 | 0.9167 | 0.8333 |
| 5 | 0.9167 | 0.9167 | 0.8333 |

## v2 标注复核（2026-10-20）

复核采用“同一 chunk 能直接支持所问事实”的口径：保留摘要、正文、图表以及分块重叠产生的等价直接证据，不按检索排名挑选单一答案；只有背景、需要额外推算或只讨论替代方案的片段不纳入。图表判断对照 PDF 页面，`quote` 则保持为当前 chunk 文本中的精确子串，可能包含换行和解析连字。

复核已查看全部 12 问的前 5 候选，结合全文 chunk 关键词搜索，并视觉核对 Transformer 第 7、8 页，BERT 第 4、5、6、12、13、14 页，LoRA 第 1、2、4、5、10 页。该流程已经观察过 v1 排名，不是独立盲标，也不保证相关集合穷尽；没有更改检索代码、分块或语料指纹。

| 问题 ID | v2 金标数 | 复核结论与边界 |
| --- | --- | --- |
| `attention-data` | 1 | 第 7 页列出英德 4.5M、约 37000 词表，英法 36M、32000 词表；第 9 页句法分析训练集不回答翻译问题。 |
| `attention-hardware` | 1 | 第 7 页直接给出 8 张 P100 和 base 的 100000 步；不混入 big 的训练步数。 |
| `attention-optimizer` | 1 | 第 7 页直接给出 Adam、4000 步线性 warmup 和之后的逆平方根衰减；扩充定位引文。 |
| `attention-result` | 3 | 保留摘要、第 8 页表 2 和正文的 28.4 BLEU 重复证据。 |
| `bert-corpus` | 2 | 在第 5 页基础上补第 14 页相同语料与词数：BooksCorpus 800M、Wikipedia 2500M。 |
| `bert-masking` | 3 | 第 4 页之外补第 12、13 页直接表述的 15%；第 12 页仅由 1.5% 推算比例的另一片段不纳入。 |
| `bert-finetuning` | 1 | 第 6 页为 GLUE 的 batch 32、3 epochs；附录超参候选范围不是同一具体设置。 |
| `bert-metrics` | 1 | 第 6 页表注：QQP/MRPC 为 F1，STS-B 为 Spearman；不混用其他论文的指标。 |
| `lora-initialization` | 3 | 补第 1 页图 1 和第 4 页重叠 chunk，均支持 A 高斯初始化、B 为零。 |
| `lora-frozen` | 5 | 保留第 4 页公式说明，补摘要、图 1 和第 2 页两处等价表述：冻结预训练权重，训练低秩 A/B。图注“仅训练 A/B”与图中预训练权重共同支持冻结关系。 |
| `lora-ablation` | 2 | 保留第 10 页表 5 图注和正文，问题明确为作者的“总体最佳”结论，不再暗示两个任务均唯一最高。 |
| `lora-limitations` | 1 | 第 5 页说明合并权重后，不同 A/B 的任务难以在同一前向批处理；后一 chunk 仅讲不合并的替代方案，不直接说明限制。 |

共 24 个问题—证据关联，比 v1 增加 9 个。除 `lora-ablation` 外，其余 11 个查询不变。该问题 v1 原文为：

> Under the same trainable parameter budget which attention weight types perform best on WikiSQL and MultiNLI?

v2 改为：

> Under the same trainable parameter budget, which attention weight types do the authors describe as giving the best overall performance on WikiSQL and MultiNLI?

表 5 中 Wq/Wv 的 WikiSQL 为 73.7、MultiNLI 为 91.3；四种注意力权重共同适配时分别为 73.7、91.7。图注和正文仍称 Wq/Wv 总体最佳，故当前问题只检索作者所述结论，不把它等同于两列各自最大值，也不在评测中自行裁定作者结论是否成立。

## v2 结果（2026-10-20）

相同的 247 个 chunks 和 BM25 参数，12 个问题；报告为 `artifacts/retrieval-eval-v2.json`。

| K | Recall@K | Hit@K | MRR@K |
| --- | --- | --- | --- |
| 1 | 0.5167 | 0.8333 | 0.8333 |
| 3 | 0.7972 | 1.0000 | 0.9167 |
| 5 | 0.8528 | 1.0000 | 0.9167 |

Hit@3 为 1 只表示每问前 3 至少有一条已标注证据，不等于召回全部证据或答案正确率为 100%。BERT 语料、掩码比例以及 LoRA 初始化、冻结参数仍有已标注证据未进入前 5。补标增加了 Recall 的分母，不能为了提高分数而删去这些等价证据。

## 排名和失败案例

- `bert-corpus`：v1 漏标的第 14 页 `bert_pretraining_p014_c002` 已加入 v2；该问题由“初标假阴性”转为可复核的多证据问题。
- `attention-data`：金标第 7 页排第 2，第 9 页 constituency parsing 的训练数据片段排第 1。它包含 dataset、vocabulary、training 等词，但不是本问题指定的翻译训练数据。
- `bert-finetuning`：金标第 6 页排第 2，第一条为附录训练设置片段。词面相似的训练细节会竞争排名，前 1 和前 3 的结果因此不同。
- `attention-result`：3 个金标分别排第 1、2、4；前 1 已能提供正确候选证据，但 Recall@1 只有 1/3。这体现“找到一条证据”和“召回全部已标注证据”的区别。

v1 与 v2 的指标不直接作为算法优劣比较：v2 扩大了部分问题的金标集合，并澄清一个查询。算法没有改动，指标变化不是检索方法提升的证据；后续算法比较应固定 v2 查询、标注和语料。

词形差异（如 `corpora/corpus`）、连字符及 PDF 连字可能影响当前精确 token 匹配，但本轮没有做消融，不能据此断言上述排名的具体成因。上述结果也说明，在扩展模型之前必须先检查标注质量。

## 当前限制与后续入口

小样本、AI 辅助标注和复核、尚无独立人工验证、已观察检索排名、未穷尽相关片段、没有独立测试集和无答案问题；仍可能存在漏标或误标，高分不表示检索或项目已经完成。之后再扩充未参与开发的论文与改写问题，在相同评测口径下比较词形处理、向量检索或混合检索。本轮不实现这些改进。
