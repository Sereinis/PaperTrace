# 首次留出测试 v1：冻结、复现与结果

## 范围与版本

2026-11-02，在未参与项目开发和调参的 ResNet、U-Net、GAN 三篇论文上完成首次留出测试：29 页、130 个原始分块、12 个英文问题，每篇 4 题，38 个问题—分块关联。检索范围限定为对应论文内，不是全库检索。

问题在证据标注前固定，金标在首次查看排名前冻结。标注为 AI 辅助，中文参考答案与论文位置经人工核对；完整 chunk 相关性及穷尽性未经独立人工或专家验证。项目层面的留出也不意味着经典论文未出现在向量模型预训练数据中。

运行代码为 `5dc2f3fe52e20f1cab7ebda96b28d64dedeb088d`，运行时跟踪文件干净。BM25 风格参数 k1=1.5、b=0.75；向量使用硅基流动 `BAAI/bge-m3`、1024 维、输入文本不变；混合使用现有两路 Top-5 结果进行 RRF 融合，rrf_k=60。分块 max_chars=1200、overlap_chars=100，没有改代码或调参。

### 公开文件

- [固定测试集](../data/eval/retrieval-independent-v1.json)：来源指纹、问题、参考答案、相关 ID、页码及精确引文。
- [结果摘要](../data/eval/retrieval-independent-v1-results.json)：保留原始数值精度的指标、分论文结果、逐问 Top-5 ID/分数、运行配置与校验记录。不包含完整 chunk 文本或向量。

测试集是原冻结数据的精简发布版，保留全部英文问题、相关集合、页码和论文指纹。对于同一分块中的多处证据，`quote` 保存主要定位摘录，`additional_quotes` 保存其余摘录，省去两者之间与核验无关的原文。所有摘录均来自原冻结引文，不改变相关性判定或首次评测结果。

原冻结文件 SHA-256 为 `149584e8cd95ff6eb0f7c2fbd0ca02f56ce9671a33c80d11d87c245f4a8ffce3`；发布版包含摘录和元数据整理，字节哈希不同，记录于结果摘要的 `published_dataset_sha256`。原冻结文件保留不变。

原本地完整报告 SHA-256 为 `8e03fc927f75b8c88114364320b1d79147f235e8c0eeb27689ebbbb6acc999e7`；该报告含候选原文，未纳入公开文件。发布摘要保留逐题排名，以便不调用 API 重算指标；它不是重新评测。

`.gitattributes` 仅为这两个发布 JSON 固定 LF 换行，避免 Windows checkout 自动转换换行后导致字节哈希变化；不改变其他文件的换行配置。

## 证据口径与冻结前调整

分块直接支持至少一个所问实质要点即可计为二元相关；完整和部分证据分别保留覆盖标签。等价重复证据按相同口径保留；背景、其他实验、无法独立辨认的残句、需额外代入推导的片段不纳入。金标取舍与候选范围分开记录，候选语料始终保留全部 130 个分块。

冻结前为统一口径作出以下 4 处调整，均先于首次排名；原问题和答案不变：

| 问题 | 分块 | 取舍与原因 |
| --- | --- | --- |
| `resnet-residual-mapping` | `resnet_p004_c003` | 纳入部分证据，直接说明投影快捷分支匹配维度，与已有分支变体口径一致。 |
| `gan-two-models` | `gan_p003_c001` | 纳入部分证据，价值函数直接表达对抗目标；同一分块可同时支持多个问题。 |
| `gan-global-optimum` | `gan_p004_c001` | 排除仅供推导的固定 G 最优 D 公式，未直接给出全局分布关系或 1/2。 |
| `gan-global-optimum` | `gan_p004_c002` | 同上，为重叠保留的公式；需额外代入才能得到所问结论。 |

ResNet 瓶颈题另保留正文 `resnet_p006_c004`，图表提取乱序的 `resnet_p005_c001`、`resnet_p006_c002` 不纳入该题金标。ResNet-152 结果题同时覆盖两套设置：表 3 的 10-crop 21.43%/5.71%，及表 4 的单模型 19.38%/4.49%；两表在同一分块，以两段摘录分别定位。

| 论文 | 页数 | 分块数 | PDF SHA-256 | 有序 chunk 指纹 |
| --- | ---: | ---: | --- | --- |
| ResNet | 12 | 67 | `1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a` | `8afcc0d6b953c1f339a9811f57dfa107f649ff708a9c80989c9f66c8d2bf2d5f` |
| U-Net | 8 | 29 | `a3172b2124f38e260dc2c7ed968d87c31bc94dbc19a42a7ab3dcbd7534319c44` | `22add682fc320851d6edb75104ef2318ef19da944d47d85bfd23fe9360941ff5` |
| GAN | 9 | 34 | `ff5819e3a7b713c3bd3107b7de3d51fe0a347aa5d8444f0efdcf2345ef0a8b63` | `e438722dc99f9abfbd04d5a80a73a418667e9224b0e21dd86c8e9317f1d26191` |

## 首次结果

Recall、MRR 和 Hit 均按问题宏平均，累计证据命中数单独统计。

| 方法 | Recall@5 | MRR@5 | Hit@5 | 累计命中关联 |
| --- | ---: | ---: | ---: | ---: |
| 关键词 | 0.7306 | 0.7667 | 11/12 | 19/38 |
| BGE-M3 向量 | 0.7417 | 0.8333 | 11/12 | 25/38 |
| RRF 混合 | 0.7660 | 0.8403 | 12/12 | 22/38 |

在上表的 K=5 设置下，混合宏平均指标最高，但纯向量累计命中的证据最多。部分题金标超过 5 个，因此当前宏平均 Recall@5 的理论上限为 0.940972。

### 改善与失败案例

- U-Net 路径信息交换题：完整结构证据 `unet_p004_c002` 在关键词第 2、混合第 4，纯向量未进 Top-5。纯向量召回的摘要或残句与主题相关，但未直接完整说明裁剪与拼接机制。
- GAN 角色题：关键词前 5 没有命中冻结金标；向量命中 4/8、首次第 1；混合仅命中 1/8、首次第 3。混合并非逐题更优。
- GAN minimax 题：完整式（1）所在 `gan_p003_c001` 仅向量进入前 5（第 5）。关键词和混合命中部分证据，因此 Hit@5 仍为 1，但没有取回完整原始公式。
- GAN 全局最优题：完整定理证据 `gan_p005_c001` 在关键词第 1、混合第 3，向量未进入前 5。混合第 1 仅直接支持分布相等，因此 MRR 也可能由部分证据贡献。

Hit/MRR 不评估答案完整性，Recall 也只是已标注分块覆盖率，不是问题回答准确率。小样本、标注边界和可能漏标限制了结论；当前无无答案问题、中文查询、跨论文检索、答案生成或统计显著性验证。

## 复现与成本边界

复现使用与上表 PDF SHA-256 匹配的英文论文文件，来源链接见测试集的 `documents`。中英对照版仅用于阅读辅助，检索语料固定为英文原版。论文及摘录的权利归各自权利人，来源链接不构成再分发授权；本项目未取得全文再分发许可，不分发论文 PDF 或完整解析文本。仓库中的证据摘录仅用于标注定位，不授予论文内容的再利用许可。

以下命令在仓库根目录、已安装依赖的环境中执行。PDF 解析命令会覆盖同名 JSON；向量化和评测命令则拒绝覆盖已有输出。示例使用单独的复现目录，与首次实验产物分开：

```powershell
python -X utf8 -m papertrace.cli data/papers/resnet.pdf --document-id resnet --chunks-output artifacts/independent-test-reproduction/resnet-chunks.json --max-chars 1200 --overlap-chars 100
python -X utf8 -m papertrace.cli data/papers/unet.pdf --document-id unet --chunks-output artifacts/independent-test-reproduction/unet-chunks.json --max-chars 1200 --overlap-chars 100
python -X utf8 -m papertrace.cli data/papers/gan.pdf --document-id gan --chunks-output artifacts/independent-test-reproduction/gan-chunks.json --max-chars 1200 --overlap-chars 100
```

复现使用与测试集匹配的三个 chunk 快照，以有序 chunk 指纹校验语料一致性。不同 PDF、依赖版本或解析行为可能产生不同快照。分块向量化入口仅校验输入结构，测试集指纹与证据一致性由评测入口校验；因此完整复现流程在向量化前先比较 chunk 指纹，指纹不一致时保留原测试集并检查输入差异。

```powershell
@'
import json
from pathlib import Path
from papertrace.retrieval_eval import corpus_fingerprint

dataset = json.loads(Path("data/eval/retrieval-independent-v1.json").read_bytes())
for document in dataset["documents"]:
    path = Path("artifacts/independent-test-reproduction") / (document["document_id"] + "-chunks.json")
    chunks = json.loads(path.read_bytes())["chunks"]
    if corpus_fingerprint(chunks) != document["chunks_sha256"]:
        raise SystemExit("chunk fingerprint mismatch: " + document["document_id"])
print("All three chunk fingerprints match.")
'@ | python -X utf8 -
```

该检查不读取密钥或发起请求；三个指纹一致后，继续生成向量并运行评测：

```powershell
python -X utf8 -m papertrace.embed_cli --chunks artifacts/independent-test-reproduction/resnet-chunks.json artifacts/independent-test-reproduction/unet-chunks.json artifacts/independent-test-reproduction/gan-chunks.json --cache-dir artifacts/independent-test-reproduction/chunk-cache --output artifacts/independent-test-reproduction/vectors.json --batch-size 8
python -X utf8 -m papertrace.vector_eval_cli --vectors artifacts/independent-test-reproduction/vectors.json --dataset data/eval/retrieval-independent-v1.json --query-cache-dir artifacts/independent-test-reproduction/query-cache --output artifacts/independent-test-reproduction/comparison.json --ks 1 3 5
```

首次复现可能产生计费：本轮新生成 130 条分块和 12 条查询向量，batch_size=8、max_retries=0，按成功执行路径推算 17+2 批请求，非独立 HTTP 计数。实际 token usage 与账单未记录。托管模型权重版本不可固定，未来重请求不保证完全相同排名；复用本地原快照和缓存才可核对本轮精确结果。

本轮对分块和查询都做了禁止创建 API 客户端的离线复跑；三路完整逐问报告一致，指标独立重算通过，冻结数据与快照未变，运行时 407 项测试通过。公开结构测试不需要 PDF、密钥或网络，只检验发布数据、排名及指标一致性，不代替语义标注复核。

当前通用对比器的 `limitations` 使用固定的开发集说明，与本数据集的 `split=independent_test` 不一致；排名和指标不受影响。首次原报告保留该原文，并依据实际 split 补充正确的报告元数据；公开摘要采用本测试集的限制说明。通用 CLI 复现仍带有这一已知文案限制，评测源码在本次文档发布中保持不变。

## 后续使用约束

该集是首次实验的留出测试，结果现已观察，后续用途为回归检查和误差分析。v1 的问题、金标和首次指标保持固定；标注更正以新版本记录，并注明已观察排名。后续方法选择使用开发集，新的留出评估另行固定数据与协议。

PDF、完整 chunk、密钥、向量、缓存及含候选原文的原始报告仍在 Git 忽略范围内。公开数据仅保留核对所需的有限证据摘录，不声明全文再分发授权。
