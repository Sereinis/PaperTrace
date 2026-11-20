"""离线验证字段条目引用、投影、复核门控及安全失败，不测试模型真实遵循率。"""

import copy
import json

import pytest

from papertrace.experiment_report import ReportError, build_experiment_report
from papertrace.field_cards import FieldCardsGenerator, cards_policy_sha256, parse_cards
from papertrace.generation import DeepSeekGenerator, GenerationResult
from papertrace.keyword_search import KeywordRetriever
from papertrace.qa import QAError


EVIDENCE = [{"segments": [{"segment_id": "E1.S1", "text": "Alpha model accuracy 90 on Test X."},
                           {"segment_id": "E2.S1", "text": "Beta Ensemble from literature."}]}]


def card(kind="single_model", value="Alpha model", sid="E1.S1"):
    """构造只有原文与类型的条目，避免引入参考答案依赖。"""
    return {"kind": kind, "value": value, "citations": [{"segment_id": sid}]}


def payload(cards=None):
    """缺省构造部分字段，不把一种方法当作完整对照清单。"""
    return {"status": "partial", "cards": cards if cards is not None else [card()],
            "missing_info": ["范围待核对。"]}


def test_exact_literals_and_typed_projection_are_not_semantic_proof():
    """多来源类型条目不包含模型自由段落，错误类型仍只能交复核识别。"""
    data = payload([card(), card("ensemble", "Beta Ensemble", "E2.S1"),
                    card("comparison_scope", "from literature", "E2.S1")])
    original = copy.deepcopy(data)
    audit, answer = parse_cards(json.dumps(data), "baselines", EVIDENCE)
    assert [c["text"] for c in answer["claims"]] == [
        "单模型对照：Alpha model", "集成对照：Beta Ensemble", "比较范围与来源：from literature"]
    audit["cards"][0]["value"] = "changed"
    assert data == original
    # 原文出现并不证明它属于某类别；不制造程序语义保证。
    _, wrong = parse_cards(json.dumps(payload([card("single_model", "90")])), "baselines", EVIDENCE)
    assert wrong["claims"][0]["text"] == "单模型对照：90"


@pytest.mark.parametrize("cards,field", [
    ([card("comparison_scope", "Test X")], "baselines"),
    ([card()], "baselines"),
    ([card("unknown_model")], "baselines"),
    ([card("metric", "accuracy")], "evaluation_metrics"),
    ([card("test_split", "Test X")], "evaluation_metrics"),
])
def test_minimum_categories_only_downgrade(cards, field):
    """最低类别不齐只降级，不从输入或其他条目补造事实。"""
    _, result = parse_cards(json.dumps({"status": "answered", "cards": cards, "missing_info": []}), field, EVIDENCE)
    assert result["status"] == "partial" and result["missing_info"]


@pytest.mark.parametrize("mutate,detail", [
    (lambda d: d.update(extra=True), "fields_invalid"),
    (lambda d: d.update(status=[]), "status_invalid"),
    (lambda d: d.update(cards="bad"), "status_invalid"),
    (lambda d: d.update(cards=d["cards"] * 21), "status_invalid"),
    (lambda d: d.update(status="answered"), "status_invalid"),
    (lambda d: d["cards"][0].update(kind="score"), "card_invalid"),
    (lambda d: d["cards"][0].update(kind=[]), "card_invalid"),
    (lambda d: d["cards"][0].update(value="invented"), "literal_invalid"),
    (lambda d: d["cards"][0].update(value="Alpha model Test X"), "literal_invalid"),
    (lambda d: d["cards"][0].update(value="Alpha" * 501), "card_invalid"),
    (lambda d: d["cards"][0].update(citations=[]), "reference_count_invalid"),
    (lambda d: d["cards"][0].update(citations=[{"segment_id": "E9.S1"}]), "reference_unknown"),
    (lambda d: d["cards"][0].update(citations=[{"segment_id": []}]), "reference_id_invalid"),
    (lambda d: d["cards"][0].update(citations=[{"segment_id": "E1.S1", "quote": "Alpha"}]), "reference_fields_invalid"),
    (lambda d: d["cards"][0]["citations"].append({"segment_id": "E1.S1"}), "reference_duplicate"),
    (lambda d: d["cards"].append(copy.deepcopy(d["cards"][0])), "card_invalid"),
])
def test_invalid_cards_fail_with_fixed_diagnostics(mutate, detail):
    """非法条目/引用/类型不静默兜底，不回显模型内容。"""
    data = payload()
    mutate(data)
    with pytest.raises(QAError) as caught:
        parse_cards(json.dumps(data), "baselines", EVIDENCE)
    assert caught.value.diagnostics["validation_stage"] == "field_cards"
    assert caught.value.diagnostics["detail_code"] == detail


@pytest.mark.parametrize("content", ["not-json", '{"status":1,"status":2}', '{"cards":NaN}', "[]", ""])
def test_strict_json(content):
    """重复键、常量、非对象及空响应均被拒绝。"""
    with pytest.raises(QAError):
        parse_cards(content, "baselines", EVIDENCE)


def test_literal_must_come_from_own_references():
    """即使另一个候选有该值，也不能借它通过本条子串检查。"""
    with pytest.raises(QAError):
        parse_cards(json.dumps(payload([card(value="Beta Ensemble")])), "baselines", EVIDENCE)


def test_evaluation_cards_keep_conditions_without_numeric_result_slot():
    """指标和过程有固定槽位，数值成绩无单独类型；不夸大为语义过滤。"""
    text = "BLEU on Test X. single model last 5 checkpoints at 10-minute intervals. beam 4 alpha 0.6. development set. input length + 50; terminate early."
    evidence = [{"segments": [{"segment_id": "E1.S1", "text": text}]}]
    values = {"metric": "BLEU", "test_split": "Test X", "model_selection": "single model",
              "checkpoint": "last 5 checkpoints at 10-minute intervals", "decoding": "beam 4 alpha 0.6",
              "selection": "development set", "termination": "input length + 50; terminate early"}
    data = {"status": "answered", "cards": [card(k, v) for k, v in values.items()], "missing_info": []}
    _, result = parse_cards(json.dumps(data), "evaluation_metrics", evidence)
    assert result["status"] == "answered" and len(result["claims"]) == 7
    assert all(v in c["text"] for v, c in zip(values.values(), result["claims"]))


def test_empty_cards_skip_review_and_no_evidence_skips_adapter():
    """两种空来源都有显式边界，不为凑齐两次调用制造请求。"""
    data = {"status": "insufficient", "cards": [], "missing_info": ["候选未覆盖。"]}
    _, result = parse_cards(json.dumps(data), "baselines", [])
    assert result == {"status": "insufficient", "claims": [], "missing_info": ["候选未覆盖。"]}

    class Empty:
        """固定空检索器，确保没有生成请求。"""

        def search(self, query, top_k):
            """不返回候选。"""
            return []

    client = Generator()
    report = build_experiment_report("a", "Alpha", [{"chunk_id": "a1", "document_id": "a", "page_number": 1, "text": "Alpha."}],
                                    retriever_factory=lambda _: Empty(), generator=client, fields=["baselines"], evidence_cards=True)
    assert client.calls == []
    assert report["sections"][0]["qa"]["field_cards"]["original"] is None


class Generator:
    """只模拟生成与复核，不创建客户端或访问网络。"""

    def __init__(self, reject=False, invalid=False):
        """保存两种失败注入和调用列表。"""
        self.reject, self.invalid, self.calls = reject, invalid, []

    def generate(self, messages, **kwargs):
        """返回原文条目或预设逐项判定，用于验证生产编排。"""
        self.calls.append((messages, kwargs))
        context = json.loads(messages[1]["content"])
        if "claims" in context:
            result = {"decisions": [{"claim_index": c["claim_index"],
                                    "verdict": "unsupported" if self.reject else "keep", "reason": "合成判定"}
                                   for c in context["claims"]]}
        else:
            result = payload([card(value="invented" if self.invalid else "Alpha model")])
        return GenerationResult(json.dumps(result), "mock", "mock", "mock",
                                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})


def build(generator):
    """调用真实报告路径，仅指定本次改动的一个字段。"""
    return build_experiment_report("a", "Alpha model on Test X", [
        {"document_id": "a", "chunk_id": "a1", "page_number": 1, "text": "Alpha model accuracy 90 on Test X."}],
        retriever_factory=KeywordRetriever, generator=generator, fields=["baselines"], evidence_cards=True)


@pytest.mark.parametrize("reject", [False, True])
def test_production_projection_review_and_audit_isolation(reject):
    """原文引用由QA回填，复核移除的条目不能从审计复活。"""
    client = Generator(reject=reject)
    report = build(client)
    answer = report["sections"][0]["qa"]
    assert report["generation"]["generation_calls"] == len(client.calls) == 2
    assert report["generation"]["total_usage"]["total_tokens"] == 24
    assert answer["field_cards"]["original"]["cards"][0]["value"] == "Alpha model"
    assert len(answer["claims"]) == len(answer["field_cards"]["final_cards"]) == (0 if reject else 1)
    assert answer["status"] == ("insufficient" if reject else "partial")
    assert answer["validation"]["semantic_support"] == "not_verified"
    assert answer["system_prompt_sha256"] == cards_policy_sha256()
    if not reject:
        assert answer["claims"][0]["citations"][0]["quote_start"] == 0
        assert answer["field_cards"]["final_cards"][0]["citations"] == answer["claims"][0]["citations"]
    assert "allowed_kinds" in json.loads(client.calls[0][0][1]["content"])


def test_real_adapter_mocked_transport_counts_and_sanitizes(monkeypatch):
    """真实适配器模拟传输检查失败用量与诊断，不读取真实密钥。"""
    from papertrace import generation
    sent = []

    def post(body, key, timeout):
        """只返回非法原文值，验证无自动重试/半份结果。"""
        sent.append(json.loads(body))
        return json.dumps({"object": "chat.completion", "id": "mock", "model": "deepseek-flash",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(payload([card(value="invented")]))}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}).encode()

    monkeypatch.setattr(generation, "_post_json", post)
    with pytest.raises(ReportError) as caught:
        build(DeepSeekGenerator("test-key"))
    assert len(sent) == 1
    assert caught.value.diagnostics["generation"]["total_usage"]["total_tokens"] == 12
    assert caught.value.diagnostics["cause"]["detail_code"] == "literal_invalid"
    assert "invented" not in json.dumps(caught.value.diagnostics)


def test_context_limit_prevents_any_generation():
    """协议转换后仍保留64KB预算，失败不调用注入客户端。"""
    client = Generator()
    adapter = FieldCardsGenerator(client, "baselines")
    with pytest.raises(QAError, match="64KB"):
        adapter.generate([{}, {"content": json.dumps({"question": "x" * 65000, "evidence": []})}])
    assert client.calls == []


@pytest.mark.parametrize("option", [None, 1, "true", []])
def test_experimental_switch_is_explicit_boolean(option):
    """非法开关在任何模型请求之前拒绝，不静默转换真假值。"""
    client = Generator()
    with pytest.raises(ReportError, match="evidence_cards"):
        build_experiment_report("a", "Alpha", [], retriever_factory=KeywordRetriever,
                                generator=client, evidence_cards=option)
    assert client.calls == []


def test_default_path_does_not_silently_enable_experiment():
    """默认路径不切换条目协议，网页继续沿用原来的生成与复核。"""
    class Ordinary:
        """明确拒绝试验协议的默认路径模拟器。"""

        def generate(self, messages, **kwargs):
            """只允许普通QA响应，返回候选不足以避免无关复核。"""
            assert "allowed_kinds" not in json.loads(messages[1]["content"])
            return GenerationResult(json.dumps({"status": "insufficient", "claims": [],
                                                "missing_info": ["合成缺口。"]}), "mock", "mock", "mock", None)

    report = build_experiment_report("a", "Alpha", [{"chunk_id": "a1", "document_id": "a", "page_number": 1, "text": "Alpha model."}],
                                    retriever_factory=KeywordRetriever, generator=Ordinary(), fields=["baselines"])
    assert report["evidence_cards"] is False and report["field_cards_policy_sha256"] is None
    assert "field_cards" not in report["sections"][0]["qa"]


@pytest.mark.parametrize("refs,detail,index", [
    ([], "reference_count_invalid", None),
    ([{"segment_id": "E1.S1"}] * 21, "reference_count_invalid", None),
    ([{"segment_id": "E1.S1"}, "secret-reference"], "reference_fields_invalid", 1),
    ([{"segment_id": "E1.S1"}, {"segment_id": ""}], "reference_id_invalid", 1),
    ([{"segment_id": "E1.S1"}, {"segment_id": "secret-reference"}], "reference_unknown", 1),
    ([{"segment_id": "E1.S1"}, {"segment_id": "E1.S1"}], "reference_duplicate", 1),
])
def test_reference_failure_location_survives_production_wrappers(refs, detail, index):
    """真实编排仅用模拟生成，传递类型和列表位置，不带模型编号或原文。"""
    class InvalidReferences:
        """故意返回第二条卡片的引用错误，不允许进入复核。"""

        def __init__(self):
            """记录调用数以确保失败不重试。"""
            self.calls = 0

        def generate(self, messages, **kwargs):
            """首卡合法、次卡非法，验证位置不总是零。"""
            self.calls += 1
            assert self.calls == 1
            data = payload([card(), card("comparison_scope", "Test X")])
            data["cards"][1]["citations"] = refs
            return GenerationResult(json.dumps(data), "mock", "mock", "mock",
                                    {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})

    client = InvalidReferences()
    with pytest.raises(ReportError) as caught:
        build(client)
    diagnostics = caught.value.diagnostics
    cause = diagnostics["cause"]
    expected = {"card_index": 1}
    if index is not None:
        expected["reference_index"] = index
    assert cause["validation_stage"] == "field_cards" and cause["detail_code"] == detail
    assert cause["validation_location"] == expected
    assert diagnostics["generation"]["generation_calls"] == client.calls == 1
    assert diagnostics["generation"]["total_usage"]["total_tokens"] == 12
    assert all(value not in json.dumps(diagnostics) for value in ("secret-reference", "Alpha model", "Test X"))


@pytest.mark.parametrize("location", [
    {"card_index": True}, {"card_index": -1}, {"card_index": 20},
    {"card_index": 0, "reference_index": "secret-reference"},
    {"reference_index": 0}, {"card_index": 0, "raw": "secret-reference"},
])
def test_diagnostic_location_rejects_untrusted_values(location):
    """位置只接受固定键及0到19的整数，不接收任意诊断对象。"""
    error = QAError("safe", validation_stage="field_cards", detail_code="reference_unknown",
                    validation_location=location)
    assert "validation_location" not in error.diagnostics


def test_diagnostic_location_is_copied_and_other_stages_ignore_it():
    """诊断持有净化副本，既有其他协议不扩展位置字段。"""
    location = {"card_index": 19, "reference_index": 19}
    error = QAError("safe", validation_stage="field_cards", detail_code="reference_unknown",
                    validation_location=location)
    location["card_index"] = "secret-reference"
    assert error.diagnostics["validation_location"]["card_index"] == 19
    other = QAError("safe", validation_stage="result_graph", detail_code="references_invalid",
                    validation_location={"card_index": 0})
    assert "validation_location" not in other.diagnostics


@pytest.mark.parametrize("root,kind", [([], "array"), ("secret-body", "string"),
                                     (7, "number"), (True, "boolean"), (None, "null")])
def test_top_level_type_diagnostic_has_no_body(root, kind):
    """非对象只记录JSON类型，不记录内容或猜测缺少字段。"""
    with pytest.raises(QAError) as caught:
        parse_cards(json.dumps(root), "baselines", EVIDENCE)
    assert caught.value.diagnostics["validation_shape"] == {
        "root_type": kind, "missing_fields": [], "recognized_other_fields": [], "unknown_field_count": 0}
    assert "secret-body" not in json.dumps(caught.value.diagnostics)


def test_top_level_missing_and_extra_fields_survive_wrappers():
    """混用claims协议时保留固定字段类别，任意未知键及值不进入报告诊断。"""
    class WrongShape:
        """模拟旧QA结构与带敏感标记的额外键，不调用真实服务。"""

        def generate(self, messages, **kwargs):
            """只返回一次非法顶层结构，用量经生产包装器保留。"""
            return GenerationResult(json.dumps({"status": "answered", "claims": ["secret-body"],
                "secret-key": "secret-body"}), "mock", "mock", "mock",
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})

    with pytest.raises(ReportError) as caught:
        build(WrongShape())
    diagnostics = caught.value.diagnostics
    assert diagnostics["cause"]["validation_shape"] == {
        "root_type": "object", "missing_fields": ["cards", "missing_info"],
        "recognized_other_fields": ["claims"], "unknown_field_count": 1}
    assert diagnostics["generation"]["generation_calls"] == 1
    assert diagnostics["generation"]["total_usage"]["total_tokens"] == 12
    assert "secret-" not in json.dumps(diagnostics)


@pytest.mark.parametrize("patch", [
    {"root_type": "secret-body"}, {"missing_fields": ["secret-key"]},
    {"recognized_other_fields": ["secret-key"]}, {"unknown_field_count": True},
    {"unknown_field_count": -1}, {"unknown_field_count": 128001},
    {"missing_fields": ["cards", "cards"]}, {"extra": "secret-body"},
    {"root_type": "array", "unknown_field_count": 1},
])
def test_unsafe_top_shape_is_dropped(patch):
    """自定义组件不能借结构诊断带入任意键名、正文或无界计数。"""
    shape = {"root_type": "object", "missing_fields": [], "recognized_other_fields": [], "unknown_field_count": 0}
    shape.update(patch)
    error = QAError("safe", validation_stage="field_cards", detail_code="fields_invalid", validation_shape=shape)
    assert "validation_shape" not in error.diagnostics


def test_shape_is_copied_and_only_allowed_for_top_field_failure():
    """仅字段条目顶层错误允许结构诊断，且保留独立净化副本。"""
    shape = {"root_type": "object", "missing_fields": ["cards"],
             "recognized_other_fields": ["claims"], "unknown_field_count": 0}
    error = QAError("safe", validation_stage="field_cards", detail_code="fields_invalid", validation_shape=shape)
    shape["missing_fields"].append("secret-key")
    assert error.diagnostics["validation_shape"]["missing_fields"] == ["cards"]
    for stage, detail in (("result_graph", "fields_invalid"), ("field_cards", "reference_unknown")):
        other = QAError("safe", validation_stage=stage, detail_code=detail,
                        validation_shape=error.diagnostics["validation_shape"])
        assert "validation_shape" not in other.diagnostics
