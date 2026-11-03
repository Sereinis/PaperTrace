"""离线验证安全细分校验码及报告透传，不留存错误正文或扩大接受范围。"""

import copy
import json
import traceback

import pytest

from papertrace import generation, qa
from papertrace.experiment_report import ReportError, build_experiment_report
from papertrace.generation import GenerationResult
from papertrace.keyword_search import KeywordRetriever
from papertrace.qa import QAError, answer_question
from papertrace.scope_review import review_field


USAGE = {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """禁止读取真实密钥和意外网络传输。"""
    def blocked(*args, **kwargs):
        """任何真实配置或传输调用均使测试失败。"""
        pytest.fail("unexpected external access")

    monkeypatch.setattr(generation, "_resolve_api_key", blocked)
    monkeypatch.setattr(generation, "build_opener", blocked)


@pytest.fixture
def chunks():
    """提供一条无需真实资料的合成证据。"""
    return [{"document_id": "a", "chunk_id": "a1", "page_number": 1,
             "text": "Alpha base uses attention."}]


def valid_answer():
    """返回合法的片段选择答案。"""
    return {"status": "answered", "missing_info": [], "claims": [
        {"text": "合成结论。", "citations": [{"segment_id": "E1.S1"}]}]}


class Client:
    """按阶段返回指定模拟正文，不连接外部服务。"""

    def __init__(self, content, review_content=None):
        """保存抽取及复核响应并记录调用次数。"""
        self.content, self.review_content, self.calls = content, review_content, 0

    def generate(self, messages, **kwargs):
        """复核响应由输入结构区分，每次返回合法安全用量。"""
        self.calls += 1
        context = json.loads(messages[1]["content"])
        content = self.review_content if "claims" in context else self.content
        return GenerationResult(content, "test", "test", "mock", copy.deepcopy(USAGE))


@pytest.mark.parametrize("content,detail", [
    ("private-marker", "json_invalid"),
    ('{"status":"private-marker","status":"answered"}', "json_invalid"),
    ('{"status":NaN}', "json_invalid"),
    ("[]", "fields_invalid"), ("{}", "fields_invalid"),
    ("\ud800", "content_invalid"), (" ", "content_invalid"),
    ("x" * 128001, "content_invalid"), ("文" * 50000, "content_too_large"),
], ids=["invalid-json", "duplicate-key", "nan", "array", "missing-fields",
        "invalid-unicode", "blank", "char-limit", "byte-limit"])
def test_json_and_content_failure_details(chunks, content, detail):
    """不解析错误中的原始文本，仅返回固定类别并保留已有计费用量。"""
    client = Client(content)
    with pytest.raises(QAError) as caught:
        answer_question("attention", "a", chunks, retriever_factory=KeywordRetriever, generator=client)
    error = caught.value
    assert error.code == "answer_contract_invalid"
    assert error.diagnostics == {"code": "answer_contract_invalid", "usage": USAGE,
                                 "http_status": None, "validation_stage": "answer_validation", "detail_code": detail}
    assert client.calls == 1
    assert "private-marker" not in "".join(traceback.format_exception(error))
    assert "private-marker" not in json.dumps(error.diagnostics)
    assert not hasattr(error, "content")


@pytest.mark.parametrize("stage,detail", [
    ("private-marker", "json_invalid"), ("answer_validation", "private-marker"),
    ("answer_validation", "decision_invalid"), ("scope_review", "citation_unknown"),
    (None, "json_invalid"), ([], {}), ({}, []), ("answer_validation", None),
])
def test_non_whitelisted_or_cross_stage_details_are_dropped(stage, detail):
    """只接受阶段与细分码白名单中的合法配对，未知值不进入诊断。"""
    error = QAError("safe", validation_stage=stage, detail_code=detail)
    assert error.diagnostics == {"code": "invalid_input", "usage": None, "http_status": None}


def test_report_preserves_answer_detail_and_stops(chunks):
    """报告层透传安全细分码、用量与失败字段，不执行后续复核或字段。"""
    data = valid_answer()
    data["missing_info"] = ["private-marker"]
    client = Client(json.dumps(data))
    with pytest.raises(ReportError) as caught:
        build_experiment_report("a", "Alpha base", chunks, retriever_factory=KeywordRetriever,
                                generator=client, fields=["results", "datasets"])
    diagnostics = caught.value.diagnostics
    assert diagnostics["failed_field"] == "results" and diagnostics["completed_fields"] == 0
    assert diagnostics["cause"]["detail_code"] == "status_inconsistent"
    assert diagnostics["cause"]["validation_stage"] == "answer_validation"
    assert diagnostics["cause"]["usage"] == diagnostics["generation"]["total_usage"] == USAGE
    assert client.calls == 1
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))
    assert not hasattr(caught.value, "sections")


@pytest.mark.parametrize("content,detail", [
    ("private-marker", "json_invalid"), ('{"decisions":[],"decisions":[]}', "json_invalid"),
    ("[]", "fields_invalid"), ('{"decisions":[]}', "decision_count_invalid"),
    ('{"decisions":[{"claim_index":true,"verdict":"keep","reason":"private-marker"}]}', "decision_invalid"),
    ('{"decisions":[{"claim_index":0,"verdict":"bad","reason":"private-marker"}]}', "decision_invalid"),
    ('{"decisions":[{"claim_index":0,"verdict":"keep","reason":"\\ud800"}]}', "decision_invalid"),
])
def test_report_preserves_review_detail_and_stops(chunks, content, detail):
    """抽取完成而复核失败时用细分阶段区分，仍不回传未复核答案。"""
    client = Client(json.dumps(valid_answer()), review_content=content)
    with pytest.raises(ReportError) as caught:
        build_experiment_report("a", "Alpha base", chunks, retriever_factory=KeywordRetriever,
                                generator=client, fields=["results", "datasets"])
    diagnostics = caught.value.diagnostics
    assert diagnostics["cause"]["validation_stage"] == "scope_review"
    assert diagnostics["cause"]["detail_code"] == detail
    assert diagnostics["generation"]["total_usage"]["total_tokens"] == 20
    assert diagnostics["completed_fields"] == 0 and client.calls == 2
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))


def test_report_re_sanitizes_untrusted_component_diagnostic(chunks):
    """自定义组件即使修改异常诊断，也不能把额外内容带入报告原因字段。"""
    def factory(selected):
        """伪造一个夹带原文的异常诊断。"""
        error = QAError("private-marker", code="retrieval_failed")
        error.diagnostics.update(validation_stage="private-marker", detail_code="private-marker",
                                 raw_content="private-marker")
        raise error

    with pytest.raises(ReportError) as caught:
        build_experiment_report("a", "Alpha base", chunks, retriever_factory=factory,
                                generator=Client("unused"), fields=["results"])
    cause = caught.value.diagnostics["cause"]
    assert cause == {"code": "retrieval_failed", "usage": None, "http_status": None}
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))


def test_quote_mismatch_has_dedicated_detail():
    """内部引用范围不一致仍拒绝，并与未知或重复引用区分。"""
    evidence = [{"evidence_id": "E1", "text": "Alpha", "segments": [
        {"segment_id": "E1.S1", "quote_start": 0, "quote_end": 5, "quote": "Other"}]}]
    with pytest.raises(QAError) as caught:
        qa._parse_answer(json.dumps(valid_answer()), evidence)
    assert caught.value.code == "citation_invalid"
    assert caught.value.diagnostics["detail_code"] == "citation_quote_mismatch"


@pytest.mark.parametrize("mode,detail", [
    ("result", "result_invalid"), ("metadata", "metadata_invalid"), ("usage", "usage_invalid"),
])
def test_review_result_diagnostic_details(mode, detail):
    """复核非标准结果、元数据及用量问题分别留固定细分码。"""
    class Broken:
        """构造与语义无关的本地边界失败。"""

        def generate(self, *args, **kwargs):
            """返回注入的畸形标准结果。"""
            if mode == "result":
                return None
            return GenerationResult("{}", "test", "" if mode == "metadata" else "test",
                                    "id", {} if mode == "usage" else USAGE)

    with pytest.raises(QAError) as caught:
        review_field(valid_answer() | {"query": "synthetic"}, experiment_scope="Alpha base",
                     field="results", generator=Broken(), max_tokens=128)
    assert caught.value.diagnostics["validation_stage"] == "scope_review"
    assert caught.value.diagnostics["detail_code"] == detail
