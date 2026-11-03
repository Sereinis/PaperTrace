"""离线验证结果条件规则的传递与多块引用；模型判定为预设，不验证语义能力。"""

import copy
import hashlib
import json

import pytest

from papertrace import embeddings, experiment_report, generation, scope_review
from papertrace.experiment_report import ReportError, build_experiment_report
from papertrace.generation import GenerationResult
from papertrace.qa import PROMPT_VERSION


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """本组测试禁止网络和真实密钥读取，所有生成均为模拟。"""
    def blocked(*args, **kwargs):
        """遗漏模拟时立即失败，不产生计费请求。"""
        pytest.fail("unexpected external access")

    monkeypatch.setattr(generation, "_resolve_api_key", blocked)
    monkeypatch.setattr(generation, "_post_json", blocked)
    monkeypatch.setattr(embeddings, "_post_json", blocked)


@pytest.fixture
def chunks():
    """合成两种模型、两个划分及独立正文条件，数值不取自开发论文。"""
    return [
        {"chunk_id": "table", "document_id": "alpha", "page_number": 4,
         "text": "Table 8: Alpha scores on Dataset Z test2030.\nModel Accuracy\nAlpha small 73.6\nAlpha large 82.1"},
        {"chunk_id": "conditions", "document_id": "alpha", "page_number": 5,
         "text": "For Dataset Z test2030, Alpha small uses a single model averaging the last 3 checkpoints saved at 7-minute intervals. "
                 "For both Alpha small and Alpha large on Dataset Z test2030, we use beam width 6 and length penalty 0.8. "
                 "For Alpha large, we average 9 checkpoints."},
        {"chunk_id": "other", "document_id": "alpha", "page_number": 6,
         "text": "On Dataset Z dev2029, Alpha small scores 79.0 accuracy. "
                 "Alpha large training uses 12 GPUs for 50 hours."},
    ]


class OrderedRetriever:
    """固定合成候选顺序以隔离抽取/复核，不测试检索质量。"""

    def __init__(self, selected):
        """仅接收报告传入的单论文权威语料。"""
        self.chunks = copy.deepcopy(selected)

    def search(self, query, top_k):
        """以固定分数返回合法候选及原始元数据。"""
        return [dict(c, rank=i, score=1.0 / i)
                for i, c in enumerate(self.chunks[:top_k], 1)]


class ScriptedGenerator:
    """返回测试预设的抽取与判定，不根据文本自动识别语义。"""

    def __init__(self, claims, verdicts, *, status="answered", missing=None):
        """保存预设结果及全部实际消息以核对协议。"""
        self.claims, self.verdicts = claims, verdicts
        self.status, self.missing = status, missing or []
        self.calls = []

    def generate(self, messages, **kwargs):
        """分别输出抽取或完整逐条判定，并提供安全用量。"""
        self.calls.append(copy.deepcopy(messages))
        context = json.loads(messages[1]["content"])
        if "claims" in context:
            payload = {"decisions": [{"claim_index": i, "verdict": v, "reason": "测试预设判断，非模型质量验证。"}
                                     for i, v in enumerate(self.verdicts)]}
        else:
            payload = {"status": self.status, "claims": self.claims, "missing_info": self.missing}
        return GenerationResult(json.dumps(payload), "mock", "mock", "mock-id",
                                {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11})


def claim(text, *segments):
    """仅提供模型允许选择的片段编号，不代写引用身份和偏移。"""
    return {"text": text, "citations": [{"segment_id": s} for s in segments]}


def build(chunks, client):
    """走完整报告抽取、引用校验及范围复核流程。"""
    return build_experiment_report("alpha", "Alpha small on Dataset Z test2030, single model", chunks,
                                   retriever_factory=OrderedRetriever, generator=client, fields=["results"])


def test_result_and_conditions_across_table_and_body_remain_one_supported_claim(chunks):
    """多块引用在同条内绑定成绩及条件，不从其他结论借引文。"""
    client = ScriptedGenerator([claim(
        "Alpha small在Dataset Z test2030准确率73.6；单模型由每7分钟保存的最后3个checkpoint平均，beam宽6、长度惩罚0.8。",
        "E1.S1", "E1.S2", "E2.S1", "E2.S2")], ["keep"])
    original = copy.deepcopy(chunks)
    report = build(chunks, client)
    qa = report["sections"][0]["qa"]
    assert len(client.calls) == report["generation"]["generation_calls"] == 2
    assert report["generation"]["total_usage"]["total_tokens"] == 22
    assert report["report_version"] == "experiment-report-v5-result-condition-policy"
    assert qa["prompt_version"] == PROMPT_VERSION == "evidence-qa-v2-segment-selection"
    assert qa["status"] == "answered" and len(qa["claims"]) == 1
    citations = qa["claims"][0]["citations"]
    assert [c["chunk_id"] for c in citations] == ["table", "table", "conditions", "conditions"]
    by_id = {c["chunk_id"]: c for c in chunks}
    for c in citations:
        assert c["quote"] == by_id[c["chunk_id"]]["text"][c["quote_start"]:c["quote_end"]]
    sent = json.loads(client.calls[1][1]["content"])["claims"][0]
    assert sent["citations"] == citations
    assert [c["chunk_id"] for c in sent["citation_context"]] == ["table", "conditions"]
    assert "dev2029" not in json.dumps(sent)
    assert report["validation"]["semantic_support"] == "not_verified"
    assert qa["validation"]["semantic_support"] == "not_verified"
    assert qa["scope_review"]["system_prompt_sha256"] == hashlib.sha256(client.calls[1][0]["content"].encode()).hexdigest()
    assert chunks == original


@pytest.mark.parametrize("missing", ["缺少checkpoint平均条件。", "解码设置适用关系不明。", "缺少单模型或集成归属。"])
def test_kept_score_with_missing_conditions_stays_partial(chunks, missing):
    """预设成绩有据但条件不全时，复核保留不升级为完整答案。"""
    client = ScriptedGenerator([claim("Alpha small测试准确率73.6。", "E1.S1", "E1.S2")], ["keep"],
                               status="partial", missing=[missing])
    qa = build(chunks, client)["sections"][0]["qa"]
    assert qa["status"] == "partial" and qa["missing_info"] == [missing]
    assert qa["scope_review"]["kept_claim_indices"] == [0]


@pytest.mark.parametrize("text,segments,verdict", [
    ("small成绩73.6采用large的9个checkpoint平均。", ["E1.S2", "E2.S3"], "out_of_scope"),
    ("以dev2029上的79.0作为test2030成绩。", ["E3.S1"], "out_of_scope"),
    ("large使用12块GPU训练50小时。", ["E3.S2"], "out_of_scope"),
    ("small也采用9个checkpoint平均。", ["E1.S2"], "unsupported"),
])
def test_rejected_condition_not_rescued_from_another_claim(chunks, text, segments, verdict):
    """预设错误条件整体过滤，其上下文不包含另一条的正文条件。"""
    client = ScriptedGenerator([
        claim("small单模型平均最后3个checkpoint。", "E2.S1"), claim(text, *segments)], ["keep", verdict])
    qa = build(chunks, client)["sections"][0]["qa"]
    assert len(qa["claims"]) == 1 and qa["status"] == "partial"
    audit = qa["scope_review"]["claim_audit"]
    assert audit[1]["verdict"] == verdict and not audit[1]["included_in_final_claims"]
    expected = list(dict.fromkeys(s.split(".")[0] for s in segments))
    assert [c["evidence_id"] for c in audit[1]["citation_context"]] == expected
    assert qa["claims"][0] == audit[0]["claim"]


def test_condition_reference_does_not_bypass_unknown_segment_rejection(chunks):
    """新规则不放宽精确引用契约，不存在的条件片段在复核前失败。"""
    client = ScriptedGenerator([claim("small准确率73.6并带解码条件。", "E1.S2", "E2.S99")], ["keep"])
    with pytest.raises(ReportError) as caught:
        build(chunks, client)
    assert len(client.calls) == 1
    assert caught.value.diagnostics["cause"]["detail_code"] == "citation_unknown"


def test_only_conditions_without_score_preserves_score_gap(chunks):
    """只有目标条件时仍为partial，不把设置当作性能成绩。"""
    client = ScriptedGenerator([claim("small单模型平均最后3个checkpoint。", "E2.S1")], ["keep"],
                               status="partial", missing=["候选未给出可关联的目标测试成绩。"])
    qa = build(chunks, client)["sections"][0]["qa"]
    assert qa["status"] == "partial" and "测试成绩" in qa["missing_info"][0]
    assert len(client.calls) == 2


def test_false_model_keep_is_not_claimed_as_semantic_verification(chunks):
    """模型误保留有合法引用的错配条件时，程序不声称已经验证语义。"""
    client = ScriptedGenerator([claim("small使用large的9个checkpoint平均。", "E2.S3")], ["keep"])
    report = build(chunks, client)
    qa = report["sections"][0]["qa"]
    assert "large" in qa["claims"][0]["text"]
    assert qa["validation"]["semantic_support"] == "not_verified"
    assert report["validation"]["experiment_scope_match"] == "not_verified"
    assert qa["scope_review"]["basis"] == "model_self_assessment_not_independent_verification"


def test_rules_define_applicability_without_adjacency_inference():
    """固定规则说明同条绑定、缺口与作用域，不把相邻当适用证明。"""
    extraction = experiment_report._EXTRACTION_RULES
    review = scope_review._REVIEW_PROMPT
    assert "多个分块的segment_id" in extraction
    assert "不能借用其他claim的引用" in extraction
    assert "返回partial并在missing_info" in extraction
    assert "不能把未完整关联的成绩报告为answered" in extraction
    assert "仅仅相邻、同篇或写着“we used”不等于明确适用" in review
    assert "复核不改写或补生成条件" in review
