# 检索评测协议与开发集对比

## 目标与范围

以固定的小规模开发问题集检查 BM25 风格关键词检索及 BGE-M3 稠密向量检索，记录已标注证据能否进入前 K 条及其排名。当前只做指定论文内检索，不生成回答，也不评估字段抽取或答案正确率。

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

## 固定 v2 的关键词与向量对比（2026-10-31）

### 实现与运行

`papertrace/vector_eval.py` 负责固定输入核验、查询向量缓存与论文内对比；`papertrace/vector_eval_cli.py` 提供报告命令。`retrieval_eval.py` 增加可注入检索器工厂，默认关键词参数和指标定义不变；统一拒绝跨论文、重复 ID、伪造元数据、非法分数或排名的返回值。对应两个新增测试文件使用模拟客户端，不读取密钥、不联网。

本轮保持 v2 查询、24个问题—证据关联、247个分块、分块顺序与元数据不变，没有根据向量排名补标或调参。候选集合先按 document_id 划分，再在各论文内独立 Top-K；不是全库检索后过滤。关键词仍为 k1=1.5、b=0.75；向量配置为 SiliconFlow `BAAI/bge-m3`、1024维、float、identity-v1，无查询指令前缀、稀疏向量或重排序，按余弦降序并在同分时保持语料顺序。

生成完整分块快照，然后对比：

```powershell
python -m papertrace.embed_cli --chunks artifacts/eval-attention-chunks.json artifacts/eval-bert-chunks.json artifacts/eval-lora-chunks.json --cache-dir artifacts/embedding-cache-smoke-20261020 --output artifacts/retrieval-dev-v2-vectors-20261031.json --batch-size 8
python -m papertrace.vector_eval_cli --vectors artifacts/retrieval-dev-v2-vectors-20261031.json --dataset data/eval/retrieval-dev.json --query-cache-dir artifacts/retrieval-query-cache-20261031 --output artifacts/retrieval-comparison-v2-20261031.json --ks 1 3 5
```

首次会向服务商发送缺失文本，可能消耗额度；默认 max_retries=0。已有输出不可覆盖，复跑需更换输出名。此次247个唯一分块复用2条旧缓存、新生成245条；12个唯一查询全部首次生成。缓存计数不是 HTTP 次数。禁止创建 API 客户端进行第二次对比时命中12条查询缓存、新生成0条，两种方法各自的总体、分论文和完整逐问题结果均与首轮完全一致。

完整报告和快照保存在被 Git 忽略的 artifacts 中，不提交论文原文、完整向量、缓存或密钥。报告含数据集身份、split、annotation_status、模型配置、各论文指纹、查询指纹、输入字节哈希、K、缓存计数、阶段耗时及 Git HEAD/dirty。新命令在任何查询请求前检查完整快照、标注和论文指纹；还检查路径冲突和明显不可用的输出父路径，但不能保证所有磁盘写入故障都能在计费前发现。

### 输入身份与复核

开发集文件 SHA-256：`5df70e2dbca17da757f67c81006dc137ec42914282edda39ca8276b1f710e54e`。

完整向量快照字节 SHA-256：`bb73e724d3b752f385eb26591fe7b9f6a9dee7fbb1f7f72baa64ca3737ef8cfb`。该哈希标识本次本地产物，重新生成包含缓存统计的快照不保证字节相同。

| 论文 | 分块数 | chunks_sha256 |
| --- | --- | --- |
| Transformer | 54 | `47bcaa05464e61fc549598cd45ffed6318377917ddac914527b7760ff9335c4d` |
| BERT | 85 | `4ca5d50726b1d0586b0b81bc529358a6019df4f0185e307ece08f80b9448acc8` |
| LoRA | 108 | `627d4377a714c002b35c85d7e883f70d791bd7fe6bd547fb34e03c8027bea8e5` |

运行时 HEAD 为 `edcc2feeb02d3aa16168115a99a5e9f81689d0bb`，dirty=true；该提交只是运行时基点，不包含本次尚未提交的评测实现。独立按报告排名重算三类指标、逐条检查候选论文归属，以及复核原始 chunk 文件和数据集字节哈希，均一致。新增38项离线测试，全量397项通过。

### 总体结果

下表均为相同12问的宏平均，关键词结果与2026-10-20固定 v2 基线一致。

| K | 关键词 Recall | 向量 Recall | 关键词 Hit | 向量 Hit | 关键词 MRR | 向量 MRR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5167 | 0.5167 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| 3 | 0.7972 | 0.7694 | 1.0000 | 0.9167 | 0.9167 | 0.8750 |
| 5 | 0.8528 | 0.9250 | 1.0000 | 1.0000 | 0.9167 | 0.8958 |

向量 Recall@5 提高0.0722，MRR@5下降0.0208；前5累计命中的问题—金标关联为关键词18/24、向量21/24。18/24和21/24是累计关联口径，不等于表中的按问题宏平均 Recall。Hit@5均为1仅表示12问各有至少一个金标，不表示完整召回或答案全部正确。

| 论文（每篇4问） | 关键词 Recall@5 | 向量 Recall@5 | 关键词 MRR@5 | 向量 MRR@5 |
| --- | --- | --- | --- | --- |
| Transformer | 1.0000 | 1.0000 | 0.8750 | 1.0000 |
| BERT | 0.7917 | 1.0000 | 0.8750 | 0.8125 |
| LoRA | 0.7667 | 0.7750 | 1.0000 | 0.8750 |

### 改善与退化案例

`attention-data`：已标注翻译训练数据由关键词第2升到向量第1；`attention-result` 的三条金标由1/2/4名变为1/2/3名。

`bert-corpus`：关键词只在前5找到第14页证据；向量在第1和第3同时找到两条金标。`bert-masking` 和 `lora-initialization` 也分别在前5补回一条已标注证据。

`bert-finetuning`：唯一金标 `bert_pretraining_p006_c001` 从关键词第2降至向量第4。向量前三条分别是GLUE任务介绍、学习率/重启设置、附录微调超参候选，均与主题有关，但不能直接给出所问的 batch 32 与3 epochs。主题相似不等于直接回答具体事实，导致向量 Hit@3从该问的1变为0。

`lora-limitations`：完整限制说明 `lora_p005_c003` 从第1降至第2；向量第1为后续不合并权重的替代方案片段 `lora_p005_c004`。两条主题接近但证据功能不同，不能为提高分数把替代方案直接标成完整限制证据。

`lora-ablation`：关键词前两名覆盖两条金标；向量仅第一名命中，另一条未进前5。向量方法并非在所有多证据问题上召回更高。`lora-frozen` 则从前5命中2/5变为3/5，仍漏2条。

本轮仅比较固定配置，不根据观察到的向量排名改标，也未实现融合、重排序或分块优化。未标注候选不自动等同于无关；上述案例仅在当前证据口径内分析，没有开展新一轮穷尽语义标注。

### 成本与时间边界

首轮本地观测：关键词校验与检索约0.020秒、查询向量化及缓存约0.988秒、向量检索器构建/校验与检索约0.143秒。不包含前置分块向量生成、顶层快照读取/校验和最终写报告；两个检索阶段包含的校验工作不同，不能据此当作公平速度基准或线上延迟承诺。本适配层未保留 token usage 和实际账单，不能报告精确费用。

复用的2条分块向量于2026-10-20生成，其余分块和查询于2026-10-31生成。托管模型未提供可固定的权重版本，相同模型名不保证跨日期底层权重完全相同；这是本轮复用旧缓存的实验限制。已有本地快照和查询缓存可复现本次排名，不代表未来重新请求必然得到相同向量。

## 当前限制与后续入口

小样本、AI 辅助标注和复核、尚无独立人工验证、已观察检索排名、未穷尽相关片段、没有独立测试集和无答案问题；仍可能存在漏标或误标，高分不表示检索或项目已经完成。本轮已完成固定 v2 的向量对比，不足以得出泛化或统计显著性结论。后续可在另行确定范围后扩充未参与开发的论文与改写问题，或研究词形处理、混合检索和重排序；本轮不实现这些改进，也不生成问答。
