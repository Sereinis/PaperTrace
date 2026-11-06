# PaperTrace

PaperTrace 是一个面向英文计算机论文的实验信息提取与证据追踪项目，目标是将论文中的研究问题、数据集、模型、Baseline、训练设置、评价指标和实验结果整理为结构化报告，并为每条信息保留原文页码或段落证据。

## 当前状态

v0.1 PDF 解析验证已经完成。v0.2 已支持保守文本清洗、章节感知分块、chunk JSON 输出、英文关键词检索、本地余弦向量检索与 RRF 混合检索。硅基流动 BGE-M3 适配层支持可追溯向量快照及内容缓存。已完成固定 v2 开发集对比，以及新增 ResNet、U-Net、GAN 三篇论文的首次留出测试；两组数据分别记录，不混合汇总。留出测试的参考答案与论文位置经人工核对，chunk 标注仍为 AI 辅助、未经独立人工穷尽核验。带证据问答及单实验七字段报告均已有真实开发检查；结果字段支持显式主体—成绩—条件关系图，完整报告内容和泛化质量尚未验收。

已完成的 v0.1 范围：

1. 读取可复制文本的英文 PDF；
2. 按页保存解析文本；
3. 对解析结果进行人工抽查；
4. 为后续检索和结构化抽取建立稳定数据结构。

三篇本地样本论文均已完成解析验证。详细结果见 [v0.1 验证记录](docs/v0.1-validation.md)。

## 环境要求

Java 业务后端已提供本机 `POST /api/questions`，通过 REST 调用 Python 内部服务；已完成模拟生成器的跨进程联调，不代表报告内容验收。构建与运行见 [Java 后端](java-client/README.md)，边界见 [内部服务说明](docs/internal-service.md)。

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

## DeepSeek 文本生成适配层

`papertrace.generation.DeepSeekGenerator` 已提供 DeepSeek 官方非流式文本生成接口，默认 `deepseek-flash`、显式关闭思考模式、不自动重试。支持本地 `DEEPSEEK_API_KEY` 配置、文本/JSON 对象输出校验及 token 用量记录；`TextGenerator` 协议为后续替换生成服务商保留边界。详见[生成接口说明](docs/generation.md)。

已通过离线模拟测试，并于 2026-11-02 完成一次官方 API 的最小 JSON 连通性验证及下述真实问答开发检查，问答质量尚未验收。字段报告复用此生成接口；部署后用户自填 API 配置仍为后续目标。真实 `.env` 保持本地且不提交，可参考只含空占位项的 `.env.example`。

## 单篇论文带证据问答

`papertrace.qa.answer_question` 在指定论文范围内调用现有检索器，再通过 `TextGenerator` 生成逐条中文结论。当前提示为v4，沿用v2片段选择契约：模型选择程序预编号的原文片段，程序回填精确摘录、论文、页码、chunk ID与引用字符范围；拒绝未知或重复编号及模型抄写的摘录，保留整块证据兜底。无候选时不调用生成器，支持模型判断的完整回答、部分回答和证据不足状态，不自动补检索或重试。失败诊断提供受控错误类型及可获得的合法token用量，不保存失败正文。

引用定位通过不代表证据在语义上支持结论，结果固定标明语义支持未经验证、充分性为模型自评。v1三题真实开发冒烟中，两题通过定位契约但一题引用不够自包含，另一题因摘录不匹配被拒绝；据此开发的v2片段选择协议已通过离线测试，并于2026-11-03完成同题真实回归：三题定位契约均通过，本次引用包含对应数字与前提条件。另在同一组固定候选下各执行一次证据不足/部分回答真实检查，分别返回预期状态且未补造耗电量；该检查不重新检索。上述结果不是独立测试、总体质量或稳定拒答结论。现已提供单问题 CLI，但尚未提供网页或用户 API 配置页面。用法、输出契约与验证记录见[带证据问答说明](docs/evidence-qa.md)。

单问题 CLI 示例（默认关键词检索；每次有证据的问题会调用一次 DeepSeek）：

```powershell
python -m papertrace.qa_cli `
  --chunks artifacts/attention-chunks.json `
  --document-id attention_is_all_you_need `
  --query "What mechanism replaces recurrence in the Transformer?" `
  --format text
```

需要向量或混合检索时增加 `--retrieval vector|hybrid` 和 `--vectors PATH` 标准向量快照；全部输入 chunk 必须与快照中对应记录的原文及元数据一致，然后仅在指定论文内检索。向量/混合模式会额外产生一次硅基流动查询向量请求，本入口暂不接入查询缓存；不会重新生成分块向量。混合模式可用 `--preserve-source-candidates` 保留两路各 Top-K 并集。`--format json` 输出完整可追溯结果，`--output` 可保存到新文件，CLI 不覆盖已有或并发创建的输出，不自动重试。生成沿用进程环境优先、项目 `.env` 其次的密钥读取规则；向量模式还需要 `SILICONFLOW_API_KEY`。CLI 不接受密钥命令行参数，不打印或保存密钥；关键词无候选时不创建生成客户端。

2026-11-03 已通过CLI函数入口执行一次关键词模式真实冒烟：Transformer硬件开发题返回8块P100和100,000步，2条第7页引用的原文/身份/偏移通过核验，JSON成功保存；同一结果离线渲染终端文本，无额外请求。实际1次DeepSeek、0次向量、3071 tokens，无重试。该题此前已观察，结果不代表独立质量验收；向量/混合CLI尚无真实冒烟，详情见[问答说明](docs/evidence-qa.md)。

## 单实验字段化证据报告

验收方案见[报告验收与服务接入门槛](docs/report-acceptance.md)。内部接口联调、完整报告内容验收和线上可用性分别判定；当前内容尚未验收，Spring Boot/Python REST已完成单问题模拟联调，报告接口尚未接入。

当前报告引擎增加保守字段覆盖诊断：候选中存在训练/指标类别而最终结论未覆盖，或对照结论出现明确其它语言方向时，只将过于乐观的`answered`降为`partial`并记录原因；不生成或改写事实，不把该诊断当语义验证。对v10旧报告的零调用回放将训练栏从answered降为partial，指标栏保留原partial；原报告文件未修改。

2026-11-05 已完成当前v10一次完整七字段开发运行（不是旧字段拼接）：13条查询全部命中准确缓存，0次新向量、14次生成、无重试，输入79116/输出3528/总82644 tokens，未核账单。68次候选出现、104次引用和35处上下文及结果图映射离线核验通过，但内容验收未通过：模型配置和结果条件保留，训练漏dropout/label smoothing却answered，指标BLEU随成绩整条被过滤，对照仍保留英法概括。模型自评3 answered/4 partial不作为准确率。原29项核对与问题定位见`artifacts/v10完整报告验收结论_20261105.md`，机制和结果见[完整运行说明](docs/experiment-report.md#当前v10完整七字段验收2026-11-05)；未追加重跑或开始Java集成。

2026-11-05 对照字段局部回归仍未通过内容验收：有效运行保留英德对照数值并排除英法列值，但未给出文献来源边界，且把英法专属方法列为待补项。该字段的7块候选未变，0次新向量；修正查询错误后有效抽取/复核2次，输入15406/输出998/总16404 tokens。此前首轮错误回归又产生2次生成（用量11445/592/12037），原JSON已删除但已在记录中标明，故本轮实际4次生成/总26851/1590/28441；不再追加调用。详见[对照字段说明](docs/experiment-report.md#对照字段局部回归2026-11-05)和`artifacts/对照字段结构核验_20261105.json`。

当前v10按各字段子问题Top-K的去重并集有界扩容，硬上限40块，记录扩容原因及截断成员；默认普通QA不变。2026-11-05数据集/模型局部回归中，模型候选10→13，同时展示8头/每头64维、前馈2048维、6层/512维及位置编码；数据集仍10块，展示训练量/BPE和newstest2014测试集。两项仍partial，开发集用途与架构的实验归属待核验，其余五字段未重跑，完整报告尚未验收。0次新向量、4次生成、24651 tokens，无重试；967项离线测试通过不代表语义或泛化验收。详见[有界扩容](docs/experiment-report.md#有界候选扩容与两字段开发回归v10)。

v8引入字段子问题检索：研究问题、数据集、模型、训练设置和评价指标采用每字段1—3个固定通用子问题，在单论文内检索，等权RRF融合，v10在并集模式下允许有界扩容；对照与结果保持原范围查询。各子问题可能增加查询向量请求，完整计划最多13条不同查询，可由调用方按文本缓存。非结果字段接收显式目标身份以辅助范围判断；未经语义核验的自由文本缺失说明只保留在`missing_info_audit`，最终以固定待核验提示替代，不生成新的缺失事实。完整报告仍最多14次生成，无自动重试。

一次已观察开发例的七字段修订验证改善了架构、优化器、预热、正则化和评测过程覆盖，但仍缺头数/测试划分等信息，对照还保留英法概括和范围过宽结论，未通过完整内容验收。实际新增11条查询向量（2次批请求）、14次生成，共78770生成tokens，未核账单；同时改变了查询、提示、显式身份及输出上限，不作为单因素效果结论。933项离线测试通过不代表语义支持或泛化验证。详细机制见[实验报告说明](docs/experiment-report.md#字段子问题检索与缺失隔离v8)。以下v3—v7记录保留为历史，当前行为以上述版本为准。

`papertrace.experiment_report.build_experiment_report` 已提供指定实验范围的七类字段报告：研究问题、数据集、模型、对照方法、训练设置、指标和结果。逐字段复用现有问答，保留引用与缺失项，支持字段子集；当前默认保留两路各Top-K的去重并集，并对生成结论逐条模型复核，仅保留明确通过项。默认完整报告最多14次生成调用（抽取和复核），向量检索还可能产生查询向量请求。

首版与分离查询/规则后的真实回归均未通过范围与覆盖验收。候选并集已保留关键结果表，v3三字段回归曾因答案契约不符停止，后续细分了安全错误码。首次单独结果字段回归未保留base的27.3 BLEU，且混入英法任务。v4增加引用所属原始分块上下文和逐条审计；同一已观察案例的结果字段回归保留了base/英德/newstest2014/27.3 BLEU，8条原始结论、10条引用及过滤判定可追踪，但仍缺单模型checkpoint平均和解码条件，完整验收未通过。本次0次新向量、2次生成，无重试；不能将改善单独归因于上下文修改，也不代表完整报告或独立质量验证。模型复核或引用定位通过不等于实验范围或语义正确。字段值仍是结论文本，不是结构化数值关系槽位，不恢复表格或验证跨字段一致性。用法、失败记录与费用边界见[实验报告说明](docs/experiment-report.md)。

v5明确结果与必要评测条件的关联规则：成绩可在同条引用表格和正文，保留有明确适用依据的checkpoint平均、单模型/集成与解码条件，不以字段边界一概排除；缺条件或关联不明时要求 `partial`。新增13项模拟测试，全量815项通过，检索查询、引用契约和调用步骤不变。单次结果字段真实回归仍未通过：27.3 BLEU保留，但复核误保留两条明确描述big的条件（含20 checkpoint平均），目标base条件仍缺；说明提示约束与模型复核不足以保证范围正确。调用2次生成、0次新向量、无重试，非完整报告或独立质量验收。

可选结构化结果模式通过 `result_target` 明确模型、任务、数据集、划分、单模型/集成及所需条件类别，抽取主体—成绩—条件—证据关系图。程序按显式身份与条件边筛选，并从槽位生成展示文本；错误主体不因复核keep被重新纳入。v6同一开发例真实回归（0次新向量/2次生成）同条保留27.3 BLEU、单模型、5 checkpoint/10分钟间隔、beam=4/长度惩罚0.6，未混入big条件；但model_selection类别未单独生成，程序仍判partial。本次未产生其他主体，未实际检验错误主体过滤。

v7新增一段条件证据的显式多类别 `coverage`，每类绑定本条件片段，按已接纳的主体/成绩边汇总，不按关键词推断。v8进一步明确 `model_selection`（最终采用单模型、集成或checkpoint选择）与 `checkpoint_averaging`（参数平均）的区别，并要求同一原文同时表达两者时在同一条件中同时标注。新增22项模拟测试，全量876项通过，调用步骤不变。同一开发例真实回归（0次新向量/2次生成）已生成 `model_selection` 与 `checkpoint_averaging` 双类别，状态由partial变为answered；最终保留27.3、single、5 checkpoint、10分钟间隔、beam=4和长度惩罚0.6，未混入big成绩或20 checkpoint。该结果只证明这一个提示词回归案例，模型错误标注主体、关系或类别仍可能通过，语义支持仍未验证。接口及限制见[实验报告说明](docs/experiment-report.md#可选结构化结果模式v7)。
v8之后在固定的 ResNet/U-Net/GAN 12题上完成首次答案生成：检索排名已在此前留出测试中观察，本轮只检查固定候选上的生成与引用，不是未见端到端测试。12题均通过契约，11题 `answered`、1题 `partial`，0次新向量、12次生成，总计51196 tokens；精确引用和偏移均通过离线核验。逐题检查发现 ResNet-152 题漏掉已确认的单模型 top-1 19.38%却报 `answered`，GAN 角色题也有覆盖不足但状态偏乐观；GAN minimax 题正确保留缺口并返回 `partial`。因此问答整体质量仍未验收，详情见 [冻结十二题问答核验](artifacts/冻结十二题问答核验_20261103.json) 和[实验报告说明](docs/experiment-report.md)。

随后将普通问答提示升级为 `evidence-qa-v4-coverage-json-contract`：要求先拆分问题事实点，遗漏任一核心数值、条件或关系时返回 `partial`，并强化严格 JSON 外壳。离线测试为876项。针对上一轮唯一生成契约失败的 U-Net 路径题，仅复用固定 Top-5 候选执行一次真实回归：0次新向量、1次 DeepSeek 生成、4567 tokens；返回 `answered`，覆盖上采样、对应裁剪、拼接、高分辨率特征和上下文传播，引用身份、原文子串与偏移均通过校验。该单题回归不代表十二题整体质量或提示词泛化，语义支持仍标记为未验证；记录见 `artifacts/冻结单题U-Net路径v4_20261103.json`。

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
