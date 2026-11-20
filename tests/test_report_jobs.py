"""离线验证报告任务幂等、进度、恢复、HTTP及公开数据边界。"""

import copy
import json
import threading
import uuid
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from papertrace.experiment_report import ReportError
from papertrace.generation import GenerationResult
from papertrace.internal_service import QuestionService, ServiceError, serve
from papertrace.keyword_search import KeywordRetriever
from papertrace.report_jobs import FIELDS, ReportJobs, _generation, project_report


def payload():
    """固定五项目标身份，每次使用独立幂等ID。"""
    return {"request_id": str(uuid.uuid4()), "document_id": "a", "experiment_scope": "Alpha on X",
            "result_target": {"model": "Alpha", "task": "classification", "dataset": "X",
                              "split": "test", "model_type": "single", "required_conditions": []}}


def report():
    """包含私有候选和审计的合成报告，用于验证最小公开投影。"""
    qa = {"query": "test question", "document_id": "a", "status": "answered",
          "claims": [{"text": "合成结论", "citations": [{"evidence_id": "E1", "segment_id": "E1.S1",
              "chunk_id": "a1", "document_id": "a", "page_number": 1, "quote": "Alpha.",
              "quote_start": 0, "quote_end": 6, "private": "secret-audit"}]}],
          "missing_info": [], "generation": {"usage": None, "private": "secret-audit"},
          "evidence": ["secret-candidate"], "audit": "secret-audit",
          "validation": {"semantic_support": "not_verified", "citation_identity_and_quote": "passed",
                         "status_basis": "generation_and_review_model_assessment", "private": "secret-audit"}}
    return {"document_id": "a", "experiment_scope": "Alpha on X",
            "sections": [{"field": key, "label": key, "qa": copy.deepcopy(qa)} for key in FIELDS],
            "generation": {"generation_calls": 14, "unknown_usage_calls": 14,
                "known_usage_subtotal": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "total_usage": None},
            "validation": {k: "not_verified" for k in ("semantic_support", "experiment_scope_match",
                                                       "cross_field_consistency", "exhaustiveness")}}


@pytest.fixture
def service():
    """模拟生成器直接报错，确保任务测试不会意外访问真实模型。"""
    class Blocked:
        def generate(self, *args, **kwargs):
            pytest.fail("unexpected generation")
    return QuestionService([{"chunk_id": "a1", "document_id": "a", "page_number": 1, "text": "Alpha."}],
                           KeywordRetriever, Blocked())


def finish(jobs):
    """有界等待测试线程结束，不轮询真实网络。"""
    jobs._thread.join(3)
    assert not jobs._thread.is_alive()
    return jobs.list()[0]


def test_idempotency_progress_busy_and_restart(service, tmp_path):
    """同ID仅生成一次；执行中可读进度，阻止其它报告及问答。"""
    entered, release = threading.Event(), threading.Event()
    calls = []
    def builder(*args, **kwargs):
        calls.append(args)
        kwargs["progress"]("datasets", 1)
        entered.set()
        assert release.wait(3)
        return report()
    jobs = ReportJobs(service, tmp_path, builder=builder)
    request = payload()
    first = jobs.submit(request)
    assert first["status"] == "queued" and first["result"] is None
    try:
        assert entered.wait(2)
        assert jobs.list()[0]["completed_fields"] == 1
        assert jobs.submit(request)["status"] == "processing"
        with pytest.raises(ServiceError, match="request_conflict"):
            jobs.submit(request | {"experiment_scope": "different"})
        with pytest.raises(ServiceError, match="service_busy"):
            jobs.submit(payload())
        with pytest.raises(ServiceError, match="service_busy"):
            service.answer("Alpha", "a")
    finally:
        release.set()
    ready = finish(jobs)
    assert ready["status"] == "ready" and ready["completed_fields"] == 7
    assert "secret" not in json.dumps(ready)
    assert jobs.submit(request) == ready and len(calls) == 1
    ready["result"]["sections"].clear()
    assert len(jobs.list()[0]["result"]["sections"]) == 7
    assert ReportJobs(service, tmp_path).list() == jobs.list()
    assert not service._lock.locked()


@pytest.mark.parametrize("error", [RuntimeError("secret"), ReportError("secret", generation=report()["generation"])])
def test_failure_never_publishes_partial_report(service, tmp_path, error):
    """失败仅保存分类、进度及可用用量；相同任务不自动重试。"""
    def builder(*args, **kwargs):
        kwargs["progress"]("model", 2)
        raise error
    jobs = ReportJobs(service, tmp_path, builder=builder)
    request = payload()
    jobs.submit(request)
    failed = finish(jobs)
    assert failed["status"] == "failed" and failed["result"] is None
    assert failed["completed_fields"] == 2 and failed["current_field"] is None
    assert failed["error_code"] == ("field_failed" if isinstance(error, ReportError) else "report_failed")
    assert "secret" not in json.dumps(failed)
    assert jobs.submit(request) == failed
    assert not service._lock.locked()


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_restart_interrupts_without_resume(service, tmp_path, status):
    """重启只标中断，不创建线程或调用生成器。"""
    request = payload()
    task = {"request": request, "status": status, "completed_fields": 0, "current_field": None,
            "error_code": None, "generation": None, "result": None}
    (tmp_path / (request["request_id"] + ".json")).write_text(json.dumps(task))
    jobs = ReportJobs(service, tmp_path)
    assert jobs.list()[0]["error_code"] == "interrupted"
    assert jobs._thread is None
    assert jobs.submit(request)["status"] == "failed"


@pytest.mark.parametrize("mutate", [lambda p: p.update(request_id="../escape"),
    lambda p: p.update(extra="secret"), lambda p: p.update(document_id=""),
    lambda p: p["result_target"].update(required_conditions=["decoding", "decoding"]),
    lambda p: p["result_target"].update(model_type="unknown")])
def test_invalid_request_cannot_start(service, tmp_path, mutate):
    request = payload()
    mutate(request)
    jobs = ReportJobs(service, tmp_path)
    with pytest.raises(ServiceError):
        jobs.submit(request)
    assert not jobs.list() and not service._lock.locked()
    assert not list(tmp_path.iterdir())


def test_unknown_document_and_imported_selection(service, tmp_path):
    jobs = ReportJobs(service, tmp_path)
    with pytest.raises(ServiceError, match="document_not_found"):
        jobs.submit(payload() | {"document_id": "missing"})
    imported = copy.copy(service)
    imported.generator = object()
    class Library:
        def service(self, identity):
            return imported
    service.library = Library()
    seen = []
    def builder(*args, **kwargs):
        seen.append(kwargs["generator"])
        return report()
    jobs.builder = builder
    jobs.submit(payload())
    assert finish(jobs)["status"] == "ready"
    assert seen == [imported.generator]


def test_thread_start_failure_does_not_leave_queued_or_locked(service, tmp_path, monkeypatch):
    jobs = ReportJobs(service, tmp_path)
    monkeypatch.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("secret")))
    with pytest.raises(RuntimeError):
        jobs.submit(payload())
    assert jobs.list()[0]["error_code"] == "report_failed"
    assert not service._lock.locked()
    assert ReportJobs(service, tmp_path).list()[0]["status"] == "failed"


@pytest.mark.parametrize("change", [lambda r: r["sections"].reverse(),
    lambda r: r["sections"][0]["qa"].update(document_id="wrong"),
    lambda r: r["validation"].update(semantic_support="verified"),
    lambda r: r["sections"][0]["qa"]["claims"][0]["citations"][0].update(quote_end=7)])
def test_public_report_rejects_invalid_identity_or_contract(change):
    value = report()
    change(value)
    with pytest.raises((ValueError, ServiceError)):
        project_report(value)


def test_projection_is_idempotent_and_supports_no_evidence():
    value = report()
    qa = value["sections"][0]["qa"]
    qa.update(status="insufficient", claims=[], missing_info=["无候选"], generation=None)
    qa["validation"].update(status_basis="no_retrieved_evidence", citation_identity_and_quote="not_applicable")
    result = project_report(value)
    assert project_report(result) == result
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("change", [dict(generation_calls=True), dict(generation_calls=15),
    dict(unknown_usage_calls=15), dict(total_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})])
def test_invalid_usage_is_rejected(change):
    with pytest.raises(ValueError):
        _generation(report()["generation"] | change)


def test_corrupt_stored_task_is_rejected(service, tmp_path):
    identity = payload()["request_id"]
    (tmp_path / f"{identity}.json").write_text('{"status":"queued","status":"ready"}')
    with pytest.raises(ValueError):
        ReportJobs(service, tmp_path)


def test_task_limit_rejects_new_but_reuses_existing(service, tmp_path):
    jobs = ReportJobs(service, tmp_path, builder=lambda *a, **k: report())
    first = payload()
    jobs.submit(first)
    finish(jobs)
    for _ in range(49):
        item = copy.deepcopy(jobs.list()[0])
        item["request"]["request_id"] = str(uuid.uuid4())
        jobs._tasks[item["request"]["request_id"]] = item
    assert jobs.submit(first)["status"] == "ready"
    with pytest.raises(ServiceError, match="report_limit"):
        jobs.submit(payload())


def test_http_post_get_and_disabled(service, tmp_path):
    """实际本机HTTP：POST202后台运行，GET只读，未知字段拒绝。"""
    server = serve(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def send(method, data=None):
        req = Request(f"http://127.0.0.1:{server.server_port}/v1/reports", method=method,
                      data=None if data is None else json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())
    try:
        assert send("GET")[0] == 503
        assert send("POST", payload())[0] == 503
        service.reports = ReportJobs(service, tmp_path, builder=lambda *a, **k: report())
        request = payload()
        assert send("POST", request)[0] == 202
        finish(service.reports)
        status, body = send("GET")
        assert status == 200 and body["tasks"][0]["status"] == "ready"
        assert send("POST", request)[0] == 202
        assert send("POST", request | {"extra": "ignored?"})[0] == 400
        assert send("POST", request | {"experiment_scope": "other"})[0] == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_real_report_engine_background_path_uses_fourteen_mock_calls(tmp_path):
    """使用真实七字段引擎、图结果和复核，只模拟模型输出。"""
    class Generator:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            context = json.loads(messages[1]["content"])
            if "claims" in context:
                data = {"decisions": [{"claim_index": c["claim_index"], "verdict": "keep", "reason": "合成保留"}
                                      for c in context["claims"]]}
            elif "target" in context:
                target = {k: v for k, v in payload()["result_target"].items() if k != "required_conditions"}
                refs = [{"segment_id": "E1.S1"}]
                data = {"subjects": [{"subject_id": "S1", "identity": target, "citations": refs}],
                        "results": [{"result_id": "R1", "subject_id": "S1", "metric": "accuracy", "value": "90",
                                     "unit": None, "citations": refs}], "conditions": []}
            else:
                data = {"status": "answered", "claims": [{"text": "合成结论", "citations": [{"segment_id": "E1.S1"}]}], "missing_info": []}
            return GenerationResult(json.dumps(data), "test", "test", "id",
                                    {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})

    class Retriever:
        def __init__(self, chunks):
            self.chunks = chunks

        def search(self, query, top_k):
            return [dict(c, rank=i, score=1 / i) for i, c in enumerate(self.chunks[:top_k], 1)]

    generator = Generator()
    service = QuestionService([{"chunk_id": "a1", "document_id": "a", "page_number": 2,
        "text": "Alpha single model classification on X test has accuracy 90."}], Retriever, generator)
    jobs = ReportJobs(service, tmp_path)
    jobs.submit(payload())
    ready = finish(jobs)
    assert ready["status"] == "ready"
    assert generator.calls == ready["generation"]["generation_calls"] == 14
    assert ready["generation"]["total_usage"]["total_tokens"] == 168
    result = ready["result"]
    assert result["sections"][-1]["qa"]["validation"]["status_basis"] == "structured_identity_link_checks_and_model_review"
    assert "result_graph" not in json.dumps(result) and "field_retrieval" not in json.dumps(result)
    assert ReportJobs(service, tmp_path).list() == [ready]
