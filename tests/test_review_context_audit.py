"""模拟表格标题与行分离场景，验证上下文绑定和逐条审计而非模型语义能力。"""

import copy
import json

import pytest

from papertrace import generation
from papertrace.experiment_report import build_experiment_report
from papertrace.generation import GenerationResult
from papertrace.keyword_search import KeywordRetriever
from papertrace.qa import QAError, answer_question
from papertrace.scope_review import review_field


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """禁止真实密钥解析与网络。"""
    def blocked(*args, **kwargs):
        """遗漏模拟时立即失败。"""
        pytest.fail("unexpected external access")

    monkeypatch.setattr(generation, "build_opener", blocked)
    monkeypatch.setattr(generation, "_resolve_api_key", blocked)


@pytest.fixture
def chunks():
    """使用非真实论文数值，包含表格标题、列头、变体行和另一块无关证据。"""
    return [{"chunk_id": "a1", "document_id": "a", "page_number": 2,
             "text": "Table 9: Alpha scores on Dataset X test2025.\nModel Accuracy\nAlpha small 71.2\nAlpha large 80.4"},
            {"chunk_id": "a2", "document_id": "a", "page_number": 3,
             "text": "Dataset Y has different evaluation conditions."}]


class Client:
    """模拟正确抽取与指定判定，仅验证传输和门控数据。"""

    def __init__(self, verdict="keep", citations=None):
        """保存测试指定判定，不自动执行语义识别。"""
        self.verdict, self.calls = verdict, []
        self.citations = citations or [{"segment_id": "E1.S2"}]

    def generate(self, messages, **kwargs):
        """记录上下文并返回固定答案或完整逐条判定。"""
        self.calls.append((messages, kwargs))
        context = json.loads(messages[1]["content"])
        if "claims" in context:
            payload = {"decisions": [{"claim_index": c["claim_index"], "verdict": self.verdict,
                                      "reason": "模拟判定，不证明语义。"} for c in context["claims"]]}
        else:
            payload = {"status": "answered", "missing_info": [], "claims": [{
                "text": "Alpha small在Dataset X test2025准确率为71.2。", "citations": self.citations}]}
        return GenerationResult(json.dumps(payload), "test", "test", "id",
                                {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})


def bound_qa(chunks, client):
    """通过真实QA校验产生标准来源与字符偏移。"""
    return answer_question("Alpha accuracy", "a", chunks, retriever_factory=KeywordRetriever,
                           generator=client)


def test_bound_context_includes_caption_header_and_row_without_borrowing_other_evidence(chunks):
    """选中行的完整来源包含标题和列头，不附加未引用的其他块。"""
    client = Client()
    qa = bound_qa(chunks, client)
    before = copy.deepcopy(qa)
    result = review_field(qa, experiment_scope="Alpha small Dataset X", field="results",
                          generator=client, max_tokens=256)
    sent = json.loads(client.calls[-1][0][1]["content"])["claims"][0]
    assert "test2025" not in sent["citations"][0]["quote"]
    assert len(sent["citation_context"]) == 1
    context = sent["citation_context"][0]
    assert context["quote"] == chunks[0]["text"]
    assert "test2025" in context["quote"] and "Model Accuracy" in context["quote"]
    assert context["quote_start"] == 0 and context["quote_end"] == len(chunks[0]["text"])
    assert context["kind"] == "bound_full_chunk_context"
    assert "Dataset Y" not in context["quote"]
    assert qa == before and result["claims"] == before["claims"]
    assert result["validation"]["semantic_support"] == "not_verified"


def test_caption_and_row_can_be_selected_together_and_context_deduplicated(chunks):
    """同一结论多引用保留原片段身份，上下文仅按源块去重。"""
    client = Client(citations=[{"segment_id": "E1.S1"}, {"segment_id": "E1.S2"}])
    qa = bound_qa(chunks, client)
    result = review_field(qa, experiment_scope="Alpha small", field="results", generator=client, max_tokens=256)
    assert len(result["claims"][0]["citations"]) == 2
    audit = result["scope_review"]["claim_audit"][0]
    assert len(audit["citation_context"]) == 1
    assert audit["claim"] == qa["claims"][0]
    for c in audit["claim"]["citations"]:
        assert chunks[0]["text"][c["quote_start"]:c["quote_end"]] == c["quote"]


@pytest.mark.parametrize("verdict", ["keep", "out_of_scope", "unsupported", "uncertain"])
def test_audit_preserves_original_claim_citations_and_decision(chunks, verdict):
    """审计完整保留原结论和引用，但被拒项不进入最终claims。"""
    client = Client(verdict)
    qa = bound_qa(chunks, client)
    result = review_field(qa, experiment_scope="Alpha small", field="results", generator=client, max_tokens=256)
    audit = result["scope_review"]["claim_audit"][0]
    assert audit["claim"] == qa["claims"][0] and audit["verdict"] == verdict
    assert audit["included_in_final_claims"] == (verdict == "keep")
    assert result["claims"] == (qa["claims"] if verdict == "keep" else [])
    assert result["scope_review"]["audit_basis"] == "validated_generation_claims_not_verified_facts"
    assert json.loads(json.dumps(result)) == result
    audit["claim"]["text"] = "modified"
    audit["citation_context"][0]["quote"] = "modified"
    assert qa["claims"][0]["text"] != "modified"
    assert result["claims"] == [] or result["claims"][0]["text"] != "modified"
    assert qa["evidence"][0]["text"] == chunks[0]["text"]


@pytest.mark.parametrize("key,value", [("chunk_id", "wrong"), ("page_number", 99),
                                      ("quote", "fake"), ("quote_start", -1), ("evidence_id", "E99")])
def test_tampered_context_citation_rejected_before_review(chunks, key, value):
    """上下文不能根据篡改引用身份或偏移附加，失败不调用复核器。"""
    client = Client()
    qa = bound_qa(chunks, client)
    qa["claims"][0]["citations"][0][key] = value
    with pytest.raises(QAError) as caught:
        review_field(qa, experiment_scope="Alpha", field="results", generator=client, max_tokens=256)
    assert caught.value.code == "citation_invalid" and len(client.calls) == 1


def test_report_success_contains_audit_without_extra_model_calls(chunks):
    """真实报告集成只执行抽取/复核两次，并以审计保存过滤项。"""
    client = Client("uncertain")
    report = build_experiment_report("a", "Alpha small Dataset X", chunks, retriever_factory=KeywordRetriever,
                                     generator=client, fields=["results"])
    qa = report["sections"][0]["qa"]
    assert qa["status"] == "insufficient" and qa["claims"] == []
    assert qa["scope_review"]["claim_audit"][0]["claim"]["citations"]
    assert report["generation"]["generation_calls"] == len(client.calls) == 2
    assert report["generation"]["total_usage"]["total_tokens"] == 14


def test_bound_context_budget_is_checked_before_review(chunks):
    """新增上下文仍受64KB限制，超限不截断也不调用复核模型。"""
    client = Client()
    qa = bound_qa(chunks, client)
    qa["evidence"][0]["text"] += "文" * 22000
    with pytest.raises(QAError) as caught:
        review_field(qa, experiment_scope="Alpha", field="results", generator=client, max_tokens=256)
    assert caught.value.code == "context_too_large" and len(client.calls) == 1


def test_second_citation_in_same_chunk_is_still_validated(chunks):
    """上下文去重不跳过同块后续引用的一致性校验。"""
    client = Client(citations=[{"segment_id": "E1.S1"}, {"segment_id": "E1.S2"}])
    qa = bound_qa(chunks, client)
    qa["claims"][0]["citations"][1]["quote"] = "tampered"
    with pytest.raises(QAError):
        review_field(qa, experiment_scope="Alpha", field="results", generator=client, max_tokens=256)
    assert len(client.calls) == 1
