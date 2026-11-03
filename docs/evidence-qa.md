# 单篇论文带证据问答

## 当前范围

`papertrace.qa.answer_question` 将现有检索器和文本生成适配层连接起来：先限定论文候选，再检索，生成逐条结论和引用，最后检查引用身份与精确摘录。该接口不改变关键词、向量或 RRF 排名算法，不添加重排序、自动补检索或生成重试。

现已提供 `python -m papertrace.qa_cli` 单问题命令行入口，支持关键词、向量和混合检索，以及 text/JSON 输出；尚无网页或用户 API 配置页面。上层新增[单实验字段化证据报告](experiment-report.md)，首轮与查询/规则分离后的真实开发回归均未通过范围与覆盖验收，未实现数值关系槽位抽取。DeepSeek 适配层已完成通用 JSON 连通性验证；问答流程已完成离线模拟、三题v1真实开发冒烟及同题v2真实回归。v2本次引用与候选原文相符，但仅覆盖已观察开发样本，尚未完成生成质量验收，不将 v0.2 整体标为验收完成。

## 输入与调用顺序

输入为问题 `query`、论文身份 `document_id`、候选 `chunks`、`retriever_factory` 与 `generator`。`chunks` 可以包含多篇论文，但所有分块先检查来源、唯一 ID、页码和合法 JSON 元数据。随后只把指定论文的深拷贝交给检索器工厂；不先全库检索再过滤。

可选 `retrieval_query` 单独指定检索文本（非空UTF-8、最多4000字符），缺省仍使用 `query`。可选 `generation_instructions` 为可信应用代码提供的固定系统补充指令（同样最多4000字符），不参与检索；不能直接传入终端用户的范围、问题或论文原文作为该参数。两项均在构建检索器前检查，`None` 表示缺省。返回结果新增 `retrieval_query`，`query` 仍为给生成模型的问题；`system_prompt_sha256` 对实际完整系统提示计算。默认系统提示与片段选择协议未变，补充规则不会自动成为语义验证。

工厂接收当前论文的分块列表，返回提供 `search(query, top_k)` 的检索器。现有 `KeywordRetriever` 可以直接作为工厂。使用向量或混合检索时，工厂需按传入分块建立标准子快照，保留同一向量配置并重算有序 chunk 指纹；不能将多篇快照直接当作当前论文候选。工厂是可信的程序扩展边界，程序会检查返回来源，但不能证明自定义组件内部实际检索了什么。

可选 `preserve_source_candidates=True` 使用支持的 `search_candidates(query, top_k)` 入口，保留两路各Top-K并集（最多2K条），返回 `retrieval_policy=source_union`、`candidate_limit=2K`，仍校验权威来源与排名。无该入口时回退普通搜索并如实记录 `ranked`；默认值为False，旧问答和评测策略不变。报告层默认显式开启，并额外执行逐条范围复核；普通 `answer_question` 不增加复核生成调用。并集不增加单路深度或查询向量请求，但会增加上下文长度，超过64KB仍失败而非静默截断。

报告v3真实开发回归已执行到第三字段的抽取阶段，在答案契约校验处停止；未生成完整报告或留存部分报告，尚无报告范围复核的真实质量结论，详见报告验证记录。此失败未改变普通问答协议或既有回归结论。

后续使用安全细分诊断单独重跑结果字段，两步协议均完成，但复核后仅保留包含英法任务的表2评测说明，未保留目标base成绩，仍未通过质量验收。此次未复现旧结构错误；不能反推旧失败根因，也不能把模型过滤等同于语义验证，详见报告单字段回归记录。

报告v4仅在复核层新增该条引用的原始分块上下文与 `scope_review.claim_audit` 审计；普通问答的选中摘录、偏移、分句和提示协议未改变。复核依据可包含上下文，短引用本身仍可能不充分，不能把复核保留视为独立支持验证。审计含被过滤正文/引用，展示最终答案只使用 `qa.claims`；详见报告说明。此次修订仅离线验证。

检索结果必须具有连续排名、有限分数和与原语料完全一致的元数据。未知、跨论文、重复或内容被改写的分块会被拒绝。程序为候选分配本次局部 ID `E1`、`E2` 等；ID 只在本次回答内有效，不代替稳定 chunk ID。

有证据时调用一次生成器，温度 0、JSON 模式、默认输出上限 2048 tokens。当前协议为 `evidence-qa-v4-coverage-json-contract`：程序按保守句末边界为候选原文分配 `E1.S1` 等片段编号，不在普通换行处切断，模型只选择编号，不再抄写摘录；提示要求先拆分问题事实点，遗漏任一核心要求时返回 `partial`，并减少同一事实的重复背景，同时强化顶层 JSON 和字段集合约束。上下文只包含问题、局部编号、片段类型及原文，不发送金标、参考答案、其他论文或本地配置。禁止把问题/论文中的指令当作控制指令；这降低风险，但不是对提示注入的完整防御或证明。

片段不是经过语言学验证的完整句子。句末符号后遇空白和大写字母才分割，缩写、公式、表格及提取乱序仍可能产生不完整片段。程序保留整块 `E1.FULL` 兜底（若唯一片段已经等于整块则不重复添加），超过100个片段时仅提供整块；不截断或丢弃候选原文。提示要求每条结论的引用独立包含数字、单位、条件与关系，不能依赖另一条结论的引用；程序不将该提示当作已验证的语义保证。

没有候选时不调用生成器，返回 `insufficient`，只说明当前检索没有取得证据，不据此判断论文是否提及相关内容。

## 答案契约与检查边界

模型仅输出如下字段：

```json
{
  "status": "partial",
  "claims": [
    {
      "text": "候选证据支持的中文结论。",
      "citations": [{"segment_id": "E1.S1"}]
    }
  ],
  "missing_info": ["当前候选还不足以回答的要点。"]
}
```

| 状态 | 结构要求 | 含义 |
| --- | --- | --- |
| `answered` | 至少一条有引用结论，缺失列表为空 | 模型判断候选覆盖全部所问要点。 |
| `partial` | 同时有引用结论和缺失列表 | 模型判断只能回答部分要点。 |
| `insufficient` | 无结论，至少一条缺失说明 | 模型判断不能回答，或检索没有候选。 |

模型不得自行填写摘录、页码、论文名或 chunk ID。程序只接受本次提供的 `segment_id`，回填对应精确原文及 `document_id`、`page_number`、`chunk_id`、`evidence_id`，并复查原文范围 `[quote_start, quote_end)` 与片段一致。保留空格、换行、大小写，不做模糊匹配、文本归一化或省略号拼接。重复文本的不同位置有不同编号，按所选片段定位，不再使用首次子串搜索；该范围不是 PDF 坐标或原页字符范围。

拒绝未知字段、无引用结论、未知片段 ID、模型自行抄写的摘录、同一结论的重复相同编号、状态矛盾、重复 JSON 键和非法 JSON。任何一处失败均报错，不返回部分答案，也不自动调用模型修复。整块与局部片段可能有重叠，选择了存在的片段仍可能没有支持结论。失败请求仍可能已经计费。

定位检查不判断语义：真实摘录仍可能被绑定到错误结论。输出固定记录 `validation.semantic_support="not_verified"`；`status_basis="model_self_assessment"` 表示充分性为模型自评。没有候选时使用 `status_basis="no_retrieved_evidence"`。`citation_identity_and_quote="passed"` 只证明引用来源和子串检查通过，不代表结论正确；无候选时为 `not_applicable`。

`missing_info` 是模型对当前候选缺口的说明，不是经过独立核验的“整篇论文未报告”字段。模型未按提示使用中文、结论是否足够原子化、证据是否完整/矛盾，以及每条结论是否有充分语义支持，均不由结构校验保证。

## 输出与追溯

成功返回普通字典，当前 `schema_version=2`，包含问题、论文、候选范围、有序语料指纹、Top-K、全部本次候选原文与预编号片段、检查后的结论和引用、缺失列表、生成模型与用量等元数据。`prompt_version` 和系统提示 SHA-256 标识当前提示协议。v1 的模型输出 `evidence_id/quote` 与当前协议不兼容，不静默接受旧格式；首次v1报告保留不变。指纹用于身份复核，不是签名，也不证明 PDF 解析或来源本身正确。

输出包含论文原文和问题，默认不落盘；后续保存、公开或上传时需考虑原文版权和隐私。错误的 `QAError.diagnostics` 包含受控 `code`、可选 `http_status` 与合法 `usage`；遇到已细分的校验失败时，额外提供白名单 `validation_stage` 和 `detail_code`。不包含问题、候选、生成正文、请求ID、原始错误位置或部分答案。生成已返回合法用量、但答案结构或片段选择失败时，统计仍可保留；生成适配层解析到合法用量但拒绝正文/截断结果时，也将统计传给问答层。网络失败、无法解析响应或统计畸形时保持未知，不估算。原始失败输出仍不保存，诊断不能恢复全部失败原因，费用需结合账单。

常见错误类型为 `citation_invalid`、`answer_contract_invalid`、`generation_failed`、`retrieval_failed`、`context_too_large`；其他前置检查使用 `invalid_input`。可在捕获 `QAError` 后记录 `error.diagnostics`，但不应记录异常局部变量或组件原始响应。

细分诊断不改变上述大类或已有接受规则；没有合法阶段/细分码时不添加新键。检测到多处错误时只报告校验顺序中的第一项，不代表响应没有其他问题。

2026-11-03 在已观察检索排名的 ResNet/U-Net/GAN 冻结12题上完成首次答案生成：0次新向量、12次默认QA生成、无重试，11题返回 `answered`、1题返回 `partial`。所有引用身份、原文子串和偏移通过离线核验，但逐题对照发现 ResNet-152 漏掉已确认的单模型 top-1 19.38%却返回 `answered`，说明结构契约和模型状态不能替代完整性验收；GAN minimax 候选缺完整 Eq.1 时正确保留 `partial`。该轮不是未见端到端测试，也不产生总体准确率结论。记录见 `artifacts/冻结十二题首次问答_20261103.json` 与 `artifacts/冻结十二题问答核验_20261103.json`。

2026-11-03 根据上述完整性问题将提示升级为 `evidence-qa-v4-coverage-json-contract`：模型先拆分问题事实点，漏掉任一核心数值、条件或关系时必须返回 `partial`，并要求响应首尾为 JSON 对象且顶层字段固定。离线测试为876项。仅针对上一轮 U-Net 路径题的契约失败执行一次真实回归，复用固定候选，不重新检索：0次新向量、1次 DeepSeek 生成、输入4411、输出156、总4567 tokens；生成通过契约并返回 `answered`。答案覆盖上采样、对应裁剪、拼接、收缩路径高分辨率特征及上下文传播，4条引用的身份、精确原文子串和字符偏移均通过校验。该结果只说明此单题在固定候选上的回归表现，不代表十二题整体质量、独立端到端测试或提示词泛化；`semantic_support` 仍为 `not_verified`。记录见 `artifacts/冻结单题U-Net路径v4_20261103.json`。

| `validation_stage` | `detail_code` | 含义 |
| --- | --- | --- |
| `answer_validation` | `content_invalid` / `content_too_large` | 空、非法类型/编码、字符超限，或UTF-8字节超限 |
| `answer_validation` | `json_invalid` / `fields_invalid` | 严格JSON解析失败（含重复键/非法常量等），或顶层类型/字段集合不符 |
| `answer_validation` | `status_invalid` / `status_inconsistent` | 状态不合法，或与结论/缺失项不一致 |
| `answer_validation` | `claims_invalid` / `missing_info_invalid` / `claim_invalid` | 结论列表、缺失列表或单条结论结构/文本不符 |
| `answer_validation` | `citations_invalid` / `citation_fields_invalid` | 引用数量/列表不符，或引用字段/编号格式不符 |
| `answer_validation` | `citation_unknown` / `citation_duplicate` / `citation_quote_mismatch` | 编号不属于候选、同结论引用重复、原文范围不符 |
| `scope_review` | `result_invalid` / `metadata_invalid` / `usage_invalid` | 复核结果类型/结束/大小、元数据或用量不符 |
| `scope_review` | `json_invalid` / `fields_invalid` | 复核JSON解析失败或顶层结构不符 |
| `scope_review` | `decision_count_invalid` / `decision_invalid` | 复核判定列表数量不符，或索引/顺序/字段/文本不符 |

例如状态与内容矛盾时为 `code=answer_contract_invalid`、`validation_stage=answer_validation`、`detail_code=status_inconsistent`。报告将这些安全字段透传到 `ReportError.diagnostics.cause`，并再次按白名单净化；不直接复制自定义组件的任意诊断。错误码只能描述检测位置，不能证明生成语义错误的原因。2026-11-03 v3失败记录没有细分码或原始正文，无法追补具体根因；新诊断仅用于后续执行。

客户端限制：问题最多 4000 字符；Top-K 为 1—20；上下文 JSON 不超过 64000 UTF-8 字节；答案不超过 128000 UTF-8 字节；最多 20 条结论、每条最多 20 条引用。超限明确失败，不静默截断。字节/字符上限不是模型 token 上限，生成适配层仍执行自己的限制。

## 使用示例

### 单问题命令行

关键词检索不需要向量快照；有候选且通过问答预检后才创建 DeepSeek 客户端，按进程环境优先、项目 `.env` 其次读取 `DEEPSEEK_API_KEY`，最多调用一次，不自动重试。关键词无候选时不需要生成密钥：

```powershell
python -m papertrace.qa_cli `
  --chunks artifacts/attention-chunks.json `
  --document-id attention_is_all_you_need `
  --query "What mechanism replaces recurrence in the Transformer?"
```

可选参数：

- `--retrieval keyword|vector|hybrid`：选择检索方式，默认 `keyword`；后两者必须提供 `--vectors` 标准向量快照。
- `--top-k`：候选数量，范围 1—20；`--preserve-source-candidates` 让混合检索保留两路各 Top-K 并集。
- `--format text|json`：默认输出中文摘要，JSON 输出包含候选、精确引用、页码、偏移和用量。
- `--output PATH`：保存到新文件；CLI 拒绝覆盖已有文件。
- `--model deepseek-flash|deepseek-v4-pro` 与 `--max-tokens`：控制生成适配器参数。

向量或混合模式的完整命令示例：

```powershell
python -m papertrace.qa_cli `
  --chunks artifacts/attention-chunks.json `
  --document-id attention_is_all_you_need `
  --query "What mechanism replaces recurrence in the Transformer?" `
  --retrieval hybrid --vectors artifacts/attention-vectors.json `
  --format json --output artifacts/answer-new.json
```

所有输入分块必须在标准快照中找到相同原文和全部元数据；只同编号不算匹配。CLI 先检查全部记录，再为指定论文建立子快照，额外论文不会参与排名。向量/混合模式每次产生一次查询向量化请求，需要 `SILICONFLOW_API_KEY`，暂不使用查询缓存；分块向量不重新生成。`keyword` 模式不接受无效用途的 `--vectors`；候选并集开关仅适用于 `hybrid`。

text 输出包含问题、论文、状态、结论、英文引文、页码、chunk/segment ID、块内字符偏移、缺失项及已知用量；明确提示语义支持仍为 `not_verified`。不可信文本中的终端控制字符转为可见转义，JSON 文件保留原文本。CLI 的失败诊断保留受控类型、阶段/细分码和可获得的合法用量，不输出密钥、候选正文或模型失败正文；生成后的落盘失败也可能已经计费。

有 `--output` 时会在计费前检查目标、父路径及同目录临时文件可写性；完成后通过无覆盖硬链接发布，若目标在生成期间被其他进程创建则报错，保留对方文件并清理临时文件。不支持硬链接的文件系统无法使用文件输出，可省略 `--output` 在终端查看。该检查不保证剩余磁盘空间或之后的文件系统状态。JSON 包含候选原文，text 也包含引用；保存和公开均需考虑原文版权与隐私。

2026-11-03 对已观察的开发题 `attention-hardware` 完成一次真实 CLI 冒烟。使用原有54块 Transformer 语料，关键词 Top-5，不创建向量客户端；通过 `qa_cli.main(argv)` 执行真实参数解析、检索、生成及 JSON 文件发布，未新启子进程执行 `python -m`。实际 DeepSeek 传输入口1次、无重试、新向量请求0次，CLI退出码0。答案为一台配备8块NVIDIA P100 GPU的机器、训练100,000步，两条引用来自第7页 `attention_is_all_you_need_p007_c005`；原文、页码、身份与块内偏移核验通过，助手对照候选确认覆盖本题两个要点。JSON保存成功后直接用同一结果离线渲染终端文本，没有为展示再调用模型。输入2996、输出75、总3071 tokens，未核对账单；运行期间源码与语料哈希保持不变。本次为已观察开发题的接口冒烟，不是独立质量验收；向量/混合CLI路径仅有离线测试，`semantic_support`仍为`not_verified`。答案与核验记录位于被Git忽略的 `artifacts/CLI硬件冒烟答案_20261103.json`、`artifacts/CLI硬件冒烟核验_20261103.json`。

下面示例以现有单篇向量快照连接实际混合检索与 DeepSeek；运行可能产生一次硅基流动查询向量请求和一次 DeepSeek 生成请求，没有候选或前置校验失败时可能不调用生成器。它不是本次已执行的真实问答记录。

```python
from papertrace.embeddings import SiliconFlowEmbedder
from papertrace.generation import DeepSeekGenerator
from papertrace.hybrid_search import HybridRetriever
from papertrace.keyword_search import KeywordRetriever
from papertrace.qa import answer_question
from papertrace.retrieval_eval import corpus_fingerprint
from papertrace.vector_search import VectorRetriever, load_vector_snapshot

snapshot = load_vector_snapshot("artifacts/attention-vectors.json")
chunks = [row["chunk"] for row in snapshot["records"]]
records = {row["chunk"]["chunk_id"]: row for row in snapshot["records"]}
embedder = SiliconFlowEmbedder(model=snapshot["embedding_config"]["model"])

def retriever_factory(selected):
    """仅为当前论文建立关键词索引及合法向量子快照。"""
    subset = {
        "schema_version": 1,
        "embedding_config": snapshot["embedding_config"],
        "chunk_count": len(selected),
        "chunks_sha256": corpus_fingerprint(selected),
        "records": [records[chunk["chunk_id"]] for chunk in selected],
    }
    return HybridRetriever(KeywordRetriever(selected), VectorRetriever(subset, embedder))

result = answer_question(
    "What mechanism replaces recurrence in the Transformer?",
    "attention_is_all_you_need", chunks,
    retriever_factory=retriever_factory, generator=DeepSeekGenerator(),
)
print(result["status"])
for claim in result["claims"]:
    print(claim["text"])
    for citation in claim["citations"]:
        print(citation["chunk_id"], citation["page_number"], citation["quote"])
print(result["missing_info"])
```

快照需包含指定论文且身份与实际分块一致；示例文件名不保证现有快照完整。仅使用关键词时可传 `retriever_factory=KeywordRetriever`，不创建向量客户端，但生成仍会计费。模型生成与检索异常统一转为脱敏的 `QAError`，不自动重试；生成适配层的合法HTTP状态可通过诊断字段获取。

## 验证与剩余工作

```powershell
python -m pytest tests/test_qa.py -q
```

离线测试覆盖论文内范围、原文/页码绑定、精确引用偏移、多条结论、部分回答、无候选和模型拒答、畸形检索、来源篡改、输入隔离、生成契约错误、提示注入数据边界、长度限制和异常脱敏；另用实际 `HybridRetriever`、`VectorRetriever` 与 `DeepSeekGenerator` 配合模拟向量/HTTP 响应完成集成检查。

以上自动化测试没有调用真实 API，也没有加载或修改独立测试集、标注或首次排名。真实问答质量检查还需覆盖数值/设置区分、部分证据、无答案问题与错误支持关系。现有已观察留出测试不再作为未见答案测试，也不据其结果调提示后宣称独立验证。

### 2026-11-02：三题真实开发冒烟检查

在请求前选定开发集的 `attention-hardware`、`bert-finetuning`、`lora-limitations`。复用原247块向量快照及3条查询缓存，禁止创建新的向量客户端；论文内混合检索保持 Top-5、rrf_k=60。生成使用当时的 `evidence-qa-v1` 提示、官方 `deepseek-flash`、关闭思考、温度0、max_tokens=2048。仅发送问题和检索候选，不发送金标或参考答案。实际传输入口计数3次，无自动重试，本次首次运行期间未改提示、排序或标注；随后针对已观察开发问题作出下述v2修改。

| 开发问题 | 定位契约检查 | 对照候选原文的复核发现 |
| --- | --- | --- |
| Transformer 基础模型训练硬件与步数 | 拒绝 | 至少一条摘录不是所引候选的精确子串，未输出可接受答案。原始生成正文未保留，不能据此断定是换行、空格还是内容不一致。 |
| BERT 的 GLUE 微调批大小与 epoch 数 | 通过 | 答案32与3和原文一致，但第二条结论的摘录缺少数字3，该数字只在上一条结论的摘录中；单条引用不够自包含。 |
| 合并 LoRA 权重时跨任务批处理限制 | 通过 | 主结论与候选一致，额外的不合并权重方案也有另一候选直接支持；该额外说明不是问题必答要点。 |

这不是“2/3回答正确率”：定位契约检查与答案语义质量分别判断。BERT案例直接说明精确子串存在仍不足以保证每条结论有完整支持。复核由助手对照当前候选原文及开发标注进行，未新增PDF视觉复核或用户确认。无答案与部分证据场景本轮未做真实验证。

两次通过调用合计输入3165、输出293、总计3458 tokens；被拒绝调用的用量未从问答层保留，因此全部三次总量未知，未核对账单。失败调用可能已计费。完整本地结果、输入/源码指纹和分离的复核记录保存在Git忽略目录 `artifacts/qa-development-smoke-20261102*.json`，不包含密钥；它们包含候选原文，不纳入Git。当时失败输出的诊断可观测性有限，后续调整另行记录，不用新请求覆盖首次结果。

### 2026-11-02：v2开发修订（仅离线验证）

基于上述已观察的开发案例，将引用改为程序预编号片段选择，提示要求逐条引用自包含，并增加受控错误类型与合法失败用量保留。离线回归覆盖BERT跨行数字/单位、原文范围、重复文本位置、兜底全文、未知/重复编号及失败脱敏；现有混合检索与生成适配器模拟集成仍通过。未修改检索算法、Top-K、rrf_k或独立测试数据，未追加真实调用，不能据离线测试宣称真实Transformer/BERT问题已修复或总体质量提高。该协议的开发已经参考过首次冒烟结果。

### 2026-11-03：v2同题真实回归

保持上述三题、247块语料快照、查询缓存、Top-5及rrf_k=60不变，用v2协议各生成一次，传输入口共3次，无自动重试、新向量请求0次。请求只含问题与候选片段，不含参考答案或金标；原始v1报告、快照、数据和生成源码哈希核对未变。

三题均通过结构与引用定位检查，共4条引用的编号、原文字符范围、页码及来源复核一致。对照候选原文的复核结果如下：

| 问题 | 本次答案与引用 |
| --- | --- |
| Transformer基础模型 | 8块NVIDIA P100 GPU、100,000步，各自引用直接包含对应设置；跨行原文由程序回填。 |
| BERT GLUE微调 | 批大小32、训练3个epoch，合并为一条结论，引用同一句完整原文。 |
| LoRA合并限制 | 保留将A、B吸收进W以消除额外延迟的前提，说明不同任务输入在单次前向中批处理不简单直接，未扩大为绝对不可能。 |

复核基于候选原文，未新增PDF视觉核对或用户确认；自动化语义支持仍标记为`not_verified`。这些是已参与开发的回归样本，不作为独立测试或总体正确率；本次也未真实验证无答案、部分证据及长期稳定性。

服务端报告输入9829、输出207、合计10036 tokens，未核对账单。片段与整块兜底会重复部分输入原文，尚未进行严格成本对比。独立回归记录及复核保存在Git忽略的`artifacts/qa-v2-regression-20261103*.json`，首次v1记录未覆盖；全量638项离线测试通过。

### 2026-11-03：固定候选的证据不足与部分回答检查

在生成前固定两道新问题及验收预期，回放上一轮Transformer硬件题的同一组5个原始候选，不重新检索、不创建向量客户端。历史排名/分数仅作回放元数据，不表示新问题的检索相关性；本检查隔离生成阶段，不是新问题的端到端检索评测。预期与复核说明不发送给模型。

- 仅问基础模型训练总耗电量（kWh）：预期`insufficient`，实际返回无结论并说明候选缺少耗电证据，没有以GPU型号、时长或FLOPs推算电量。
- 同时问硬件、步数与总耗电：预期`partial`，实际回答8块P100及100,000步，两条引用包含对应设置，并将耗电列为缺失。

缺失判断仅针对给定候选，不据此断言全篇论文未报告。第一题的缺失说明含E1/E4自由文本，程序不校验`missing_info`中的引用；本次对照候选相符。两个状态及语义边界由候选对照复核，自动化语义支持仍为`not_verified`，未新增PDF核验或用户确认。

各调用一次DeepSeek，共2次、无重试、新向量请求0次；服务端报告输入5682、输出205、合计5887 tokens，未核对账单。记录及分离复核在Git忽略的`artifacts/qa-v2-boundaries-20261103*.json`，既有记录和源码哈希未变，638项离线测试通过。两个受控案例不代表总体拒答率、独立质量测试或长期稳定性。
