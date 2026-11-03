"""合成关系图测试只验证身份/边/引用的程序约束，不验证模型语义。"""

import copy
import hashlib
import json

import pytest

from papertrace import embeddings, generation
from papertrace.experiment_report import ReportError, build_experiment_report
from papertrace.generation import GenerationResult
from papertrace.result_graph import IDENTITY_KEYS


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """禁止读取真实配置、网络或向量请求。"""
    def blocked(*args, **kwargs):
        """测试遗漏模拟时立即失败。"""
        pytest.fail("unexpected external access")

    monkeypatch.setattr(generation, "_resolve_api_key", blocked)
    monkeypatch.setattr(generation, "_post_json", blocked)
    monkeypatch.setattr(embeddings, "_post_json", blocked)


@pytest.fixture
def target():
    """调用方提供合成目标，不由模型解析自然语言。"""
    return {"model": "Alpha small", "task": "classification", "dataset": "Dataset Z",
            "split": "test2030", "model_type": "single",
            "required_conditions": ["checkpoint_averaging", "decoding"]}


@pytest.fixture
def chunks():
    """提供跨块成绩、主体限定和变体独有条件。"""
    return [
        {"chunk_id": "t", "document_id": "a", "page_number": 4,
         "text": "Table 8: Single-model classification on Dataset Z test2030.\nModel Accuracy\nAlpha small 73.6\nAlpha large 82.1"},
        {"chunk_id": "c", "document_id": "a", "page_number": 5,
         "text": "Alpha small averages 3 checkpoints. Alpha large averages 9 checkpoints. Both models use beam width 6."},
    ]


@pytest.fixture
def graph(target):
    """预设结构中同时包含目标、其他变体、合法共享及错误条件边。"""
    def refs(*ids):
        """构造只含片段ID的模型输出引用。"""
        return [{"segment_id": sid} for sid in ids]

    small = {k: target[k] for k in IDENTITY_KEYS}
    large = dict(small, model="Alpha large")
    return {
        "subjects": [{"subject_id": "S1", "identity": small, "citations": refs("E1.S1", "E1.S2")},
                     {"subject_id": "S2", "identity": large, "citations": refs("E1.S1", "E1.S2")}],
        "results": [{"result_id": "R1", "subject_id": "S1", "metric": "Accuracy", "value": "73.6", "unit": None,
                     "citations": refs("E1.S1", "E1.S2")},
                    {"result_id": "R2", "subject_id": "S2", "metric": "Accuracy", "value": "82.1", "unit": None,
                     "citations": refs("E1.S1", "E1.S2")}],
        "conditions": [
            {"condition_id": "C1", "kind": "checkpoint_averaging", "value": "Alpha small averages 3 checkpoints.",
             "applies_to": ["S1"], "result_ids": ["R1"], "citations": refs("E2.S1"),
             "applicability_citations": refs("E1.S1", "E1.S2", "E2.S1")},
            {"condition_id": "C2", "kind": "checkpoint_averaging", "value": "Alpha large averages 9 checkpoints.",
             "applies_to": ["S2"], "result_ids": ["R1", "R2"], "citations": refs("E2.S2"),
             "applicability_citations": refs("E2.S2")},
            {"condition_id": "C3", "kind": "decoding", "value": "Both models use beam width 6.",
             "applies_to": ["S1", "S2"], "result_ids": ["R1", "R2"], "citations": refs("E2.S3"),
             "applicability_citations": refs("E2.S3", "E1.S1", "E1.S2")},
        ],
    }


class Retriever:
    """固定候选顺序以隔离图校验，不测检索质量。"""

    def __init__(self, chunks):
        """保存单论文候选。"""
        self.chunks = chunks

    def search(self, query, top_k):
        """返回合法原始元数据及连续排名。"""
        return [dict(c, rank=i, score=1 / i) for i, c in enumerate(self.chunks[:top_k], 1)]


class Client:
    """预设图和复核判定，不执行语义推断。"""

    def __init__(self, graph, verdict="keep"):
        """保存测试输出与实际调用。"""
        self.graph, self.verdict, self.calls = copy.deepcopy(graph), verdict, []

    def generate(self, messages, **kwargs):
        """一次图抽取加最多一次完整复核。"""
        self.calls.append(copy.deepcopy(messages))
        context = json.loads(messages[1]["content"])
        value = ({"decisions": [{"claim_index": c["claim_index"], "verdict": self.verdict,
                                  "reason": "测试预设判定。"} for c in context["claims"]]}
                 if "claims" in context else self.graph)
        content = value if isinstance(value, str) else json.dumps(value)
        return GenerationResult(content, "mock", "mock", "mock-id",
                                {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12})


def build(chunks, target, graph, verdict="keep", **kwargs):
    """调用真实报告及QA校验，模型完全模拟。"""
    client = Client(graph, verdict)
    result = build_experiment_report("a", "Alpha small on Dataset Z test2030", chunks,
                                     generator=client, retriever_factory=Retriever,
                                     fields=["results"], result_target=target, **kwargs)
    return result, client


def test_wrong_subject_cannot_be_rescued_by_model_keep(chunks, target, graph):
    """即使复核无差别keep，其他主体和跨主体条件边也不进入最终投影。"""
    before = copy.deepcopy((chunks, target, graph))
    report, client = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert len(client.calls) == report["generation"]["generation_calls"] == 2
    assert report["generation"]["total_usage"]["total_tokens"] == 24
    assert report["report_version"] == "experiment-report-v7-multikind-coverage"
    assert qa["status"] == "partial" and len(qa["claims"]) == 1
    assert "73.6" in qa["claims"][0]["text"] and "9 checkpoints" not in qa["claims"][0]["text"]
    assert "Alpha large" not in qa["claims"][0]["text"]
    structured = qa["result_graph"]
    assert [n["result_id"] for n in structured["final_results"]] == ["R1"]
    assert structured["final_results"][0]["condition_ids"] == ["C1", "C3"]
    assert structured["result_audit"][1]["target_identity_match"] is False
    context = json.loads(client.calls[1][1]["content"])
    assert len(context["claims"]) == 1
    citations = context["claims"][0]["citations"]
    assert {c["chunk_id"] for c in citations} == {"t", "c"}
    assert "E2.S2" not in [c["segment_id"] for c in citations]
    assert structured["validation"]["source_entailment"] == "not_verified"
    assert (chunks, target, graph) == before


def test_fully_linked_graph_can_be_answered_but_not_verified(chunks, target, graph):
    """预设仅含目标的完整关系允许answered，仍不证明语义或穷尽性。"""
    graph["subjects"] = graph["subjects"][:1]
    graph["results"] = graph["results"][:1]
    graph["conditions"] = [graph["conditions"][0], graph["conditions"][2]]
    graph["conditions"][1].update(applies_to=["S1"], result_ids=["R1"])
    report, client = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert qa["status"] == "answered" and qa["missing_info"] == []
    assert report["validation"]["semantic_support"] == "not_verified"
    assert qa["prompt_version"] == "result-graph-v2-multikind-coverage"
    assert qa["system_prompt_sha256"] == hashlib.sha256(client.calls[0][0]["content"].encode()).hexdigest()
    assert report["result_graph_policy_sha256"] == qa["system_prompt_sha256"]


@pytest.mark.parametrize("field,value", [("model", "Alpha large"), ("split", "dev2029"),
                                         ("dataset", "Other"), ("task", None), ("model_type", "unknown")])
def test_identity_mismatch_or_unknown_skips_review(chunks, target, graph, field, value):
    """任一身份不匹配或未知时不自动用目标补齐，不发复核请求。"""
    graph["subjects"][0]["identity"][field] = value
    report, client = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert qa["status"] == "insufficient" and qa["claims"] == []
    assert len(client.calls) == 1 and qa["result_graph"]["final_results"] == []


@pytest.mark.parametrize("field", ["applies_to", "result_ids"])
def test_missing_condition_link_is_partial(chunks, target, graph, field):
    """条件只连主体或只连成绩不能进入投影，缺口由程序生成。"""
    graph["conditions"][0][field] = []
    report, _ = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert qa["status"] == "partial"
    assert "checkpoint_averaging" in qa["missing_info"][0]
    assert qa["result_graph"]["final_results"][0]["condition_ids"] == ["C3"]


def test_review_rejection_does_not_leak_final_graph_nodes(chunks, target, graph):
    """图投影被复核移除后最终节点为空，但审计仍可追踪。"""
    report, _ = build(chunks, target, graph, verdict="uncertain")
    qa = report["sections"][0]["qa"]
    assert qa["status"] == "insufficient" and not qa["claims"]
    assert not qa["result_graph"]["final_results"]
    assert qa["result_graph"]["nodes"]["results"]


@pytest.mark.parametrize("mode,detail", [
    ("dangling_subject", "links_invalid"), ("dangling_result", "links_invalid"),
    ("unknown_segment", "references_invalid"), ("duplicate_ref", "references_invalid"),
    ("duplicate_id", "nodes_invalid"), ("extra_field", "nodes_invalid"),
    ("wrong_literal", "literal_invalid"), ("numeric_value", "literal_invalid"),
    ("unknown_kind", "nodes_invalid"), ("bad_identity", "identity_invalid"),
])
def test_invalid_graph_fails_closed_with_usage(chunks, target, graph, mode, detail):
    """非法关系图在复核前停止，保留安全用量但不返回正文。"""
    if mode == "dangling_subject":
        graph["results"][0]["subject_id"] = "missing"
    elif mode == "dangling_result":
        graph["conditions"][0]["result_ids"] = ["missing"]
    elif mode == "unknown_segment":
        graph["conditions"][0]["applicability_citations"] = [{"segment_id": "E99.S1"}]
    elif mode == "duplicate_ref":
        graph["results"][0]["citations"] *= 2
    elif mode == "duplicate_id":
        graph["subjects"].append(copy.deepcopy(graph["subjects"][0]))
    elif mode == "extra_field":
        graph["results"][0]["text"] = "private-marker"
    elif mode == "wrong_literal":
        graph["results"][0]["value"] = "999.99"
    elif mode == "numeric_value":
        graph["results"][0]["value"] = 73.6
    elif mode == "unknown_kind":
        graph["conditions"][0]["kind"] = "gpu"
    elif mode == "bad_identity":
        graph["subjects"][0]["identity"] = {}
    client = Client(graph)
    with pytest.raises(ReportError) as caught:
        build_experiment_report("a", "scope", chunks, generator=client,
                                retriever_factory=Retriever, fields=["results"], result_target=target)
    assert len(client.calls) == 1
    cause = caught.value.diagnostics["cause"]
    assert cause["validation_stage"] == "result_graph" and cause["detail_code"] == detail
    assert cause["usage"]["total_tokens"] == 12
    assert "private-marker" not in str(caught.value)


@pytest.mark.parametrize("raw", ["not json", "[]", '{"subjects":[],"subjects":[]}',
                                 '{"subjects":[],"results":[],"conditions":NaN}'])
def test_invalid_json_does_not_retry(chunks, target, raw):
    """重复键及非有限JSON不自动修复。"""
    with pytest.raises(ReportError):
        build(chunks, target, raw)


@pytest.mark.parametrize("change", [{"model": ""}, {"model_type": "unknown"},
                                    {"required_conditions": ["decoding", "decoding"]}, {"extra": 1}])
def test_invalid_target_prevents_retrieval(chunks, target, change):
    """显式目标非法时在所有外部组件之前失败。"""
    target.update(change)
    def forbidden(selected):
        """非法目标不应构建检索器。"""
        pytest.fail("unexpected retrieval")

    client = Client({})
    with pytest.raises(ReportError):
        build_experiment_report("a", "scope", chunks, generator=client,
                                retriever_factory=forbidden, fields=["results"], result_target=target)
    assert client.calls == []


def test_false_canonicalization_is_not_semantically_verified(chunks, target, graph):
    """模型把错误条件重贴目标ID且有合法引用时结构校验仍不能识别语义错误。"""
    graph["conditions"][1]["applies_to"] = ["S1"]
    report, _ = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert "9 checkpoints" in qa["claims"][0]["text"]
    assert qa["result_graph"]["validation"]["source_entailment"] == "not_verified"


def test_target_not_inserted_in_system_prompt(chunks, target, graph):
    """目标作为数据传递，检索查询仍是原文本，不插入参考数值。"""
    report, client = build(chunks, target, graph)
    sent = json.loads(client.calls[0][1]["content"])
    assert sent["target"] == {k: target[k] for k in IDENTITY_KEYS}
    assert "Alpha small" not in client.calls[0][0]["content"]
    assert "73.6" not in client.calls[0][0]["content"]
    assert report["sections"][0]["qa"]["retrieval_query"].startswith("Alpha small on Dataset Z test2030\n")


def test_no_candidates_skip_both_generations(chunks, target):
    """无候选时不调用图抽取或复核，结构明确标为空且保留目标。"""
    class Empty(Retriever):
        """模拟合法空检索。"""

        def search(self, query, top_k):
            """返回空列表而不是错误。"""
            return []

    client = Client({})
    report = build_experiment_report("a", "scope", chunks, generator=client,
                                     retriever_factory=Empty, result_target=target, fields=["results"])
    qa = report["sections"][0]["qa"]
    assert client.calls == [] and report["generation"]["generation_calls"] == 0
    assert qa["result_graph"]["status"] == "not_generated_no_evidence"
    assert qa["result_graph"]["target"] == target


def test_union_capture_does_not_repeat_retrieval(chunks, target, graph):
    """捕获并集入口只执行底层一次，不退回普通搜索或重复查询。"""
    queries = []

    class Union(Retriever):
        """提供并集方法并禁止普通入口。"""

        def search(self, query, top_k):
            """并集模式不应调用这个方法。"""
            pytest.fail("unexpected ranked search")

        def search_candidates(self, query, top_k):
            """记录一次调用并返回两块合成证据。"""
            queries.append((query, top_k))
            return Retriever.search(self, query, 2 * top_k)

    report = build_experiment_report("a", "scope", chunks, generator=Client(graph),
                                     retriever_factory=Union, result_target=target, fields=["results"], top_k=1)
    assert len(queries) == 1
    assert report["sections"][0]["qa"]["retrieval_policy"] == "source_union"
    assert len(report["sections"][0]["qa"]["evidence"]) == 2


def test_real_adapter_two_steps_with_mocked_transport(chunks, target, graph, monkeypatch):
    """真实适配器通过模拟传输执行图抽取/复核两步，用量均计入报告。"""
    requests = []

    def post(body, api_key, timeout):
        """按实际请求协议返回预设JSON，不使用真实密钥。"""
        request = json.loads(body)
        requests.append(request)
        context = json.loads(request["messages"][1]["content"])
        payload = ({"decisions": [{"claim_index": c["claim_index"], "verdict": "keep", "reason": "模拟保留。"}
                                   for c in context["claims"]]} if "claims" in context else graph)
        return json.dumps({"id": "mock", "object": "chat.completion", "model": "deepseek-flash",
                           "choices": [{"index": 0, "finish_reason": "stop", "message": {
                               "role": "assistant", "content": json.dumps(payload)}}],
                           "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}}).encode()

    monkeypatch.setattr(generation, "_post_json", post)
    monkeypatch.setattr(generation, "_resolve_api_key", lambda explicit: explicit)
    report = build_experiment_report("a", "scope", chunks,
                                     generator=generation.DeepSeekGenerator("test-key"),
                                     retriever_factory=Retriever, result_target=target, fields=["results"])
    assert len(requests) == report["generation"]["generation_calls"] == 2
    assert report["generation"]["total_usage"]["total_tokens"] == 32
    assert json.loads(requests[0]["messages"][1]["content"])["target"]["model"] == "Alpha small"
    assert "result_target" in json.loads(json.loads(requests[1]["messages"][1]["content"])["question"])


def test_target_and_final_graph_are_deep_copies(chunks, target, graph):
    """最终节点、图审计、原输入互不共享可变引用。"""
    report, _ = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    final = qa["result_graph"]["final_results"][0]
    assert final["subject"]["identity"]["model"] == "Alpha small"
    final["conditions"][0]["citations"][0]["quote"] = "changed"
    report["result_target"]["model"] = "changed"
    assert qa["result_graph"]["target"]["model"] == "Alpha small"
    assert qa["result_graph"]["nodes"]["conditions"][0]["citations"][0]["quote"] != "changed"
    assert target["model"] == "Alpha small"
    assert all(c["quote"] != "changed" for c in qa["claims"][0]["citations"])


def test_empty_graph_keeps_audit_without_review(chunks, target):
    """空模型输出是无目标成绩，不补造主体或条件。"""
    report, client = build(chunks, target, {"subjects": [], "results": [], "conditions": []})
    qa = report["sections"][0]["qa"]
    assert len(client.calls) == 1 and qa["status"] == "insufficient"
    assert qa["result_graph"]["nodes"] == {"subjects": [], "results": [], "conditions": []}


def test_target_without_results_is_invalid_before_calls(chunks, target, graph):
    """不含结果字段时不能静默忽略显式目标。"""
    client = Client(graph)
    with pytest.raises(ReportError):
        build_experiment_report("a", "scope", chunks, generator=client,
                                retriever_factory=Retriever, result_target=target, fields=["datasets"])
    assert client.calls == []


@pytest.mark.parametrize("mode", ["truncated", "invalid_metadata", "huge_output"])
def test_bad_generation_result_cannot_return_graph(chunks, target, graph, mode):
    """截断、非法元数据或超长正文不回传已准备图或半份报告。"""
    from dataclasses import replace

    class Broken(Client):
        """破坏生成结果的单个边界。"""

        def generate(self, messages, **kwargs):
            """输出本地畸形返回值，不追加请求。"""
            result = super().generate(messages, **kwargs)
            options = {"truncated": {"finish_reason": "length"},
                       "invalid_metadata": {"request_id": ""},
                       "huge_output": {"content": "x" * 128001}}
            return replace(result, **options[mode])

    client = Broken(graph)
    with pytest.raises(ReportError):
        build_experiment_report("a", "scope", chunks, generator=client,
                                retriever_factory=Retriever, result_target=target, fields=["results"])
    assert len(client.calls) == 1


def multikind_graph(graph):
    """给合成模型输出显式增加逐类别原文引用，不改写原文和条件边。"""
    graph = copy.deepcopy(graph)
    graph["subjects"] = graph["subjects"][:1]
    graph["results"] = graph["results"][:1]
    graph["conditions"] = [graph["conditions"][0], graph["conditions"][2]]
    checkpoint, decoding = graph["conditions"]
    checkpoint["coverage"] = [{"kind": kind, "citations": [{"segment_id": "E2.S1"}]}
                              for kind in ("model_selection", "checkpoint_averaging")]
    decoding.update(applies_to=["S1"], result_ids=["R1"])
    decoding["coverage"] = [{"kind": "decoding", "citations": [{"segment_id": "E2.S3"}]}]
    return graph


def test_single_condition_can_cover_two_kinds_without_duplicate_text(chunks, target, graph):
    """一段明确证据覆盖选择及平均两类，不需复制节点或重复展示原文。"""
    chunks[1]["text"] = chunks[1]["text"].replace(
        "Alpha small averages 3 checkpoints.", "Alpha small uses a single model averaging 3 checkpoints.")
    graph = multikind_graph(graph)
    graph["conditions"][0]["value"] = "Alpha small uses a single model averaging 3 checkpoints."
    target["required_conditions"] = ["model_selection", "checkpoint_averaging", "decoding"]
    original = copy.deepcopy((target, graph, chunks))
    report, client = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    final = qa["result_graph"]["final_results"][0]
    assert qa["status"] == "answered" and qa["missing_info"] == []
    assert final["missing_condition_kinds"] == [] and len(final["conditions"]) == 2
    assert [e["kind"] for e in final["condition_coverage"]] == target["required_conditions"]
    assert qa["claims"][0]["text"].count(graph["conditions"][0]["value"]) == 1
    assert len(client.calls) == report["generation"]["generation_calls"] == 2
    assert report["generation"]["total_usage"]["total_tokens"] == 24
    selection = final["condition_coverage"][0]
    assert selection["sources"][0]["condition_id"] == "C1"
    assert selection["sources"][0]["origin"] == "explicit"
    ref = selection["sources"][0]["citations"][0]
    assert ref["segment_id"] == "E2.S1"
    assert ref["quote"] == chunks[1]["text"][ref["quote_start"]:ref["quote_end"]]
    assert qa["result_graph"]["validation"]["condition_coverage"].endswith("not_verified")
    assert (target, graph, chunks) == original


def test_legacy_kind_does_not_infer_additional_coverage_from_single_word(chunks, target, graph):
    """旧kind节点即使原文包含single也仅覆盖主类别，不自动推断。"""
    chunks[1]["text"] = chunks[1]["text"].replace(
        "Alpha small averages 3 checkpoints.", "Alpha small uses a single model averaging 3 checkpoints.")
    graph["conditions"][0]["value"] = "Alpha small uses a single model averaging 3 checkpoints."
    target["required_conditions"] = ["model_selection", "checkpoint_averaging", "decoding"]
    report, _ = build(chunks, target, graph)
    final = report["sections"][0]["qa"]["result_graph"]["final_results"][0]
    assert final["missing_condition_kinds"] == ["model_selection"]
    assert final["conditions"][0]["coverage_origin"] == "legacy_primary_kind_only"
    assert [c["kind"] for c in final["condition_coverage"]] == ["checkpoint_averaging", "decoding"]


@pytest.mark.parametrize("field", ["applies_to", "result_ids"])
def test_multikind_condition_with_broken_link_covers_nothing(chunks, target, graph, field):
    """同一条件多标签仍须同时关联主体和成绩，不能只凭标签补缺。"""
    graph = multikind_graph(graph)
    graph["conditions"][0][field] = []
    target["required_conditions"] = ["model_selection", "checkpoint_averaging", "decoding"]
    report, _ = build(chunks, target, graph)
    final = report["sections"][0]["qa"]["result_graph"]["final_results"][0]
    assert final["missing_condition_kinds"] == ["model_selection", "checkpoint_averaging"]
    assert [c["kind"] for c in final["condition_coverage"]] == ["decoding"]


def test_foreign_subject_multikind_condition_cannot_close_target_gaps(chunks, target, graph):
    """其他主体的显式覆盖标签不计入目标，无差别keep也不能恢复。"""
    graph["conditions"][1]["coverage"] = [
        {"kind": k, "citations": [{"segment_id": "E2.S2"}]}
        for k in ("model_selection", "checkpoint_averaging")]
    target["required_conditions"].append("model_selection")
    report, _ = build(chunks, target, graph)
    final = report["sections"][0]["qa"]["result_graph"]["final_results"][0]
    assert final["missing_condition_kinds"] == ["model_selection"]
    assert "C2" not in [s["condition_id"] for c in final["condition_coverage"] for s in c["sources"]]


@pytest.mark.parametrize("coverage,detail", [
    ([], "coverage_invalid"), (None, "coverage_invalid"), ("model_selection", "coverage_invalid"),
    ([{"kind": "gpu", "citations": [{"segment_id": "E2.S1"}]}], "coverage_invalid"),
    ([{"kind": "model_selection", "citations": [{"segment_id": "E2.S1"}]}], "coverage_invalid"),
    ([{"kind": "checkpoint_averaging", "citations": []}], "references_invalid"),
    ([{"kind": "checkpoint_averaging", "citations": [{"segment_id": "E99.S1"}]}], "references_invalid"),
    ([{"kind": "checkpoint_averaging", "citations": [{"segment_id": "E2.S3"}]}], "coverage_invalid"),
    ([{"kind": "checkpoint_averaging", "citations": [{"segment_id": "E2.S1"}], "extra": "private-marker"}], "coverage_invalid"),
    ([{"kind": "checkpoint_averaging", "citations": [{"segment_id": "E2.S1"}]}] * 2, "coverage_invalid"),
    ([{"kind": [], "citations": [{"segment_id": "E2.S1"}]}], "coverage_invalid"),
])
def test_bad_coverage_fails_before_review(chunks, target, graph, coverage, detail):
    """拒绝空/重/未知类别、跨条件借引文和坏片段，安全用量保留。"""
    graph["conditions"][0]["coverage"] = coverage
    client = Client(graph)
    with pytest.raises(ReportError) as caught:
        build_experiment_report("a", "scope", chunks, generator=client, retriever_factory=Retriever,
                                fields=["results"], result_target=target)
    assert len(client.calls) == 1
    cause = caught.value.diagnostics["cause"]
    assert cause["validation_stage"] == "result_graph" and cause["detail_code"] == detail
    assert cause["usage"]["total_tokens"] == 12
    assert "private-marker" not in json.dumps(caught.value.diagnostics)


def test_duplicate_kind_across_conditions_is_covered_once_but_keeps_sources(chunks, target, graph):
    """多个条件支持同类只计一次，保留每个证据来源供审计。"""
    graph = multikind_graph(graph)
    duplicate = copy.deepcopy(graph["conditions"][0])
    duplicate["condition_id"] = "C4"
    graph["conditions"].append(duplicate)
    report, _ = build(chunks, target, graph)
    coverage = report["sections"][0]["qa"]["result_graph"]["final_results"][0]["condition_coverage"]
    assert [c["kind"] for c in coverage] == ["model_selection", "checkpoint_averaging", "decoding"]
    assert [s["condition_id"] for s in coverage[0]["sources"]] == ["C1", "C4"]


def test_coverage_evidence_deep_copy_and_review_removal(chunks, target, graph):
    """覆盖证据与节点及输入隔离，被复核拒绝后不进入final_results。"""
    graph = multikind_graph(graph)
    report, _ = build(chunks, target, graph)
    data = report["sections"][0]["qa"]["result_graph"]
    data["final_results"][0]["condition_coverage"][0]["sources"][0]["citations"][0]["quote"] = "changed"
    assert data["result_audit"][0]["condition_coverage"][0]["sources"][0]["citations"][0]["quote"] != "changed"
    assert data["nodes"]["conditions"][0]["coverage"][0]["citations"][0]["quote"] != "changed"
    rejected, _ = build(chunks, target, graph, verdict="uncertain")
    qa = rejected["sections"][0]["qa"]
    assert qa["status"] == "insufficient" and qa["result_graph"]["final_results"] == []
    assert qa["result_graph"]["result_audit"][0]["condition_coverage"]


def test_wrong_semantic_category_is_still_not_verified(chunks, target, graph):
    """没有单模型文字却标model_selection时结构检查不能代替语义识别。"""
    graph = multikind_graph(graph)
    target["required_conditions"] = ["model_selection", "checkpoint_averaging", "decoding"]
    report, _ = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert "single model" not in graph["conditions"][0]["value"]
    assert qa["status"] == "answered"
    assert qa["result_graph"]["validation"]["source_entailment"] == "not_verified"
    assert qa["result_graph"]["validation"]["condition_coverage"].endswith("not_verified")


def test_coverage_prompt_has_no_automatic_equivalence_or_example_answers(chunks, target, graph):
    """覆盖需显式类别及本条件证据，不注入已观察论文数值。"""
    report, client = build(chunks, target, multikind_graph(graph))
    prompt = client.calls[0][0]["content"]
    assert "coverage须含主kind" in prompt
    assert "必须在同一条件的coverage中同时列出model_selection和checkpoint_averaging" in prompt
    assert "包括选用单模型、模型集成或最佳checkpoint" in prompt
    assert "checkpoint平均后得到单个评测模型，不等同于多个模型的预测集成" in prompt
    assert "不能因为model_type已经填写single而省略" in prompt
    assert "last 5 checkpoints" not in prompt and "10-minute" not in prompt
    assert "不为满足required_conditions补标签" in prompt
    assert "不借其他条件或主体引文" in prompt
    assert "Transformer" not in prompt and "27.3" not in prompt
    assert report["result_graph_policy_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()


def test_prompt_revision_does_not_fill_missing_model_labels(chunks, target, graph):
    """规则写得更明确也不授权程序补标签，模拟遗漏仍须保留partial。"""
    graph = multikind_graph(graph)
    graph["conditions"][0]["coverage"] = graph["conditions"][0]["coverage"][1:]
    target["required_conditions"] = ["model_selection", "checkpoint_averaging", "decoding"]
    report, client = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert len(client.calls) == 2 and qa["status"] == "partial"
    assert qa["result_graph"]["final_results"][0]["missing_condition_kinds"] == ["model_selection"]


def test_multikind_full_value_preserves_qualifiers_once(chunks, target, graph):
    """完整连续原文可保留组成、数量及间隔，仍由模拟器预设而非模型验证。"""
    full = "Alpha small uses one averaged model from 3 checkpoints saved at 7-minute intervals."
    chunks[1]["text"] = chunks[1]["text"].replace("Alpha small averages 3 checkpoints.", full)
    graph = multikind_graph(graph)
    graph["conditions"][0]["value"] = full
    target["required_conditions"] = ["model_selection", "checkpoint_averaging", "decoding"]
    report, _ = build(chunks, target, graph)
    qa = report["sections"][0]["qa"]
    assert qa["status"] == "answered" and qa["claims"][0]["text"].count(full) == 1
    ref = qa["result_graph"]["final_results"][0]["condition_coverage"][0]["sources"][0]["citations"][0]
    assert ref["quote"] == full
