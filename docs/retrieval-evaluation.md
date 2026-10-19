# 关键词检索评测协议

## 目标与范围

以固定的小规模开发问题集检查 BM25 风格关键词检索，记录正确证据能否进入前 K 条及其排名。当前只做指定论文内检索，不生成回答，也不评估字段抽取或答案正确率。

问题集信息见 [数据说明](../data/eval/README.md)。当前为助手初标、待用户复核的开发集，不是独立测试集，不能用该分数推断其他论文或中文查询的检索能力。

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
python -X utf8 -m papertrace.eval_cli --chunks artifacts/eval-attention-chunks.json artifacts/eval-bert-chunks.json artifacts/eval-lora-chunks.json --dataset data/eval/retrieval-dev.json --output artifacts/retrieval-eval.json --ks 1 3 5
```

评测器核对全部论文快照指纹、证据页码和证据子串，防止分块更新后沿用旧金标。JSON 输出包含总体、分论文、逐问题指标和完整候选 chunk，同时记录 BM25 参数、数据集哈希、语料指纹及运行环境。报告和 chunk 默认保存在忽略的 `artifacts/` 中，不提交 Git。单元测试使用独立合成数据，不依赖本地 PDF。

## 首轮结果（2026-10-18）

247 个 chunks（Transformer 54、BERT 85、LoRA 108），12 个问题；BM25 参数保持原样：k1=1.5、b=0.75。问题和金标在首次检索运行前固定，没有针对分数调整算法或查询。

| K | Recall@K | Hit@K | MRR@K |
| --- | --- | --- | --- |
| 1 | 0.6528 | 0.7500 | 0.7500 |
| 3 | 0.8472 | 0.9167 | 0.8333 |
| 5 | 0.9167 | 0.9167 | 0.8333 |

## 排名和失败案例

- `bert-corpus`：询问预训练语料和词数，初版金标第 5 页未进入前 5。但核对候选原文后发现，第一名 `bert_pretraining_p014_c002` 同样明确写出 BooksCorpus (800M words) 和 Wikipedia (2,500M words)。因此这次零分是初标漏掉等价证据造成的假阴性，不能称为回答证据召回失败。本次保留冻结的 v1 标注和原始指标，后续经复核再以新版本补标，并重新评测，不与 v1 分数直接比较。
- `attention-data`：金标第 7 页排第 2，第 9 页 constituency parsing 的训练数据片段排第 1。它包含 dataset、vocabulary、training 等词，但不是本问题指定的翻译训练数据。
- `bert-finetuning`：金标第 6 页排第 2，第一条为附录训练设置片段。词面相似的训练细节会竞争排名，前 1 和前 3 的结果因此不同。
- `attention-result`：3 个金标分别排第 1、2、4；前 1 已能提供正确候选证据，但 Recall@1 只有 1/3。这体现“找到一条证据”和“召回全部已标注证据”的区别。

词形差异（如 `corpora/corpus`）、连字符及 PDF 连字可能影响当前精确 token 匹配，但本轮没有做消融，不能据此断言上述排名的具体成因。上述结果也说明，在扩展模型之前必须先检查标注质量。

## 当前限制与后续入口

小样本、助手初标、未穷尽相关片段、没有独立测试集和无答案问题；高分不表示检索或项目已经完成。用户应先复核问题和证据，尤其是漏标的等价证据和附录片段。之后再扩充未参与开发的论文与改写问题，在相同评测口径下比较词形处理、向量检索或混合检索。本轮不实现这些改进。
