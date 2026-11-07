"""验证 Python 内部 HTTP 服务的协议、边界和问答联调行为。"""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from papertrace.generation import GenerationResult
from papertrace.internal_service import QuestionService, ServiceError, make_handler, serve
from papertrace.keyword_search import KeywordRetriever
from papertrace.qa import QAError


class FakeGenerator:
    """返回固定的、可被原始 chunk 支持的答案。"""

    def generate(self, messages, **kwargs):
        """返回一个最小标准生成结果。"""
        content = json.dumps({"status": "answered", "claims": [{
            "text": "模型使用注意力机制。", "citations": [{"segment_id": "E1.S1"}],
        }], "missing_info": []})
        return GenerationResult(content, "test", "test", "id", None)


@pytest.fixture
def server():
    """启动随机本机端口服务，并在测试后关闭。"""
    chunks = [{"chunk_id": "a1", "document_id": "a", "page_number": 1,
               "text": "The model uses attention."}]
    service = QuestionService(chunks, KeywordRetriever, FakeGenerator())
    httpd = serve(service, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def request(base, method, path, payload=None):
    """发送测试请求并返回状态码和 JSON。"""
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base + path, data=data, method=method,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_healthz(server):
    """健康检查返回版本且不包含模型或密钥。"""
    status, body = request(server, "GET", "/healthz")
    assert status == 200
    assert body == {"ok": True, "service_version": "papertrace-internal-question-v1"}


def test_documents_lists_loaded_metadata_only(server):
    """目录只暴露稳定论文身份和分块数，不返回候选原文或来源路径。"""
    status, body = request(server, "GET", "/v1/documents")
    assert status == 200
    assert body == {"ok": True, "service_version": "papertrace-internal-question-v1",
                    "documents": [{"document_id": "a", "chunk_count": 1}]}


def test_documents_sorted_counts_and_fresh_projection():
    """目录计数按论文聚合、排序并隔离调用者对结果的修改。"""
    chunks = [{"chunk_id": str(i), "document_id": identity, "page_number": 1, "text": "attention"}
              for i, identity in enumerate(["z", "a", "z"])]
    service = QuestionService(chunks, KeywordRetriever, FakeGenerator())
    expected = [{"document_id": "a", "chunk_count": 1}, {"document_id": "z", "chunk_count": 2}]
    assert service.documents() == expected
    service.documents()[0]["document_id"] = "changed"
    assert service.documents() == expected
    assert QuestionService([], KeywordRetriever, FakeGenerator()).documents() == []


def test_documents_does_not_retrieve_or_generate():
    """问答忙时仍能读取目录，不进入检索器或生成器。"""
    def blocked(*args, **kwargs):
        """任何检索调用都会使测试失败。"""
        pytest.fail("unexpected retrieval")

    class BlockedGenerator:
        """不允许目录请求创建答案。"""

        def generate(self, *args, **kwargs):
            """任何生成调用都会使测试失败。"""
            pytest.fail("unexpected generation")

    service = QuestionService([{"chunk_id": "a", "document_id": "paper", "page_number": 1,
                               "text": "attention"}], blocked, BlockedGenerator())
    service._lock.acquire()
    try:
        assert service.documents() == [{"document_id": "paper", "chunk_count": 1}]
    finally:
        service._lock.release()


def test_question_returns_bound_citation(server):
    """单问题接口返回既有问答结果并绑定 chunk 身份。"""
    status, body = request(server, "POST", "/v1/question",
                           {"query": "attention", "document_id": "a"})
    assert status == 200
    assert body["ok"] is True
    citation = body["result"]["claims"][0]["citations"][0]
    assert citation["chunk_id"] == "a1"
    assert body["result"]["validation"]["semantic_support"] == "not_verified"
    assert "api_key" not in json.dumps(body).lower()


@pytest.mark.parametrize("path", ["/missing", "/v1/other"])
def test_not_found(server, path):
    """未知路径返回固定 404 错误。"""
    status, body = request(server, "GET", path)
    assert status == 404 and body["error"]["code"] == "not_found"


def test_invalid_json_and_fields(server):
    """非法 JSON 和多余字段均被拒绝。"""
    req = Request(server + "/v1/question", data=b"not-json", method="POST",
                  headers={"Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(req, timeout=3)
    assert error.value.code == 400
    status, body = request(server, "POST", "/v1/question", {"query": "x"})
    assert status == 400 and body["error"]["code"] == "invalid_json"


def test_oversize_request(server):
    """超过请求体上限时不进入问答流程。"""
    data = b"{" + b"a" * (256 * 1024) + b"}"
    req = Request(server + "/v1/question", data=data, method="POST",
                  headers={"Content-Type": "application/json",
                           "Content-Length": str(len(data))})
    with pytest.raises(HTTPError) as error:
        urlopen(req, timeout=3)
    assert error.value.code == 413


def test_public_bind_is_rejected():
    """公网监听配置必须在启动前拒绝。"""
    service = QuestionService([], KeywordRetriever, FakeGenerator())
    with pytest.raises(ServiceError):
        serve(service, host="0.0.0.0")


def test_missing_document_is_safe_404(server):
    """未知论文返回可区分的404，不使用上游正文。"""
    status, body = request(server, "POST", "/v1/question", {"query": "x", "document_id": "missing"})
    assert status == 404 and body["error"]["code"] == "document_not_found"


def test_duplicate_json_keys_rejected(server):
    """重复请求字段必须在执行问答前拒绝。"""
    req = Request(server + "/v1/question", data=b'{"query":"a","query":"b","document_id":"a"}',
                  headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(HTTPError) as error:
        urlopen(req, timeout=3)
    assert error.value.code == 400


@pytest.mark.parametrize("code,status", [("retrieval_failed", 502), ("generation_failed", 502),
    ("citation_invalid", 502), ("answer_contract_invalid", 502), ("context_too_large", 422)])
def test_safe_qa_failure_categories_and_usage(code, status):
    """失败保留合法用量和分类，不回显异常正文，也不重试。"""
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    service = QuestionService([], KeywordRetriever, FakeGenerator())
    calls = []

    def fail(*args):
        """只抛出合成安全诊断以验证传输外壳。"""
        calls.append(1)
        raise QAError("secret raw error", code=code, usage=usage)

    service.answer = fail
    httpd = serve(service, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        actual, body = request(f"http://127.0.0.1:{httpd.server_port}", "POST", "/v1/question",
                               {"query": "x", "document_id": "a"})
        assert actual == status and body["error"]["code"] == code
        assert body["error"]["usage"] == usage
        assert "secret" not in json.dumps(body) and len(calls) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_inflight_request_is_not_queued():
    """服务繁忙时直接拒绝新任务，避免客户端超时后仍排队计费。"""
    service = QuestionService([{"chunk_id": "a1", "document_id": "a", "page_number": 1,
                               "text": "attention"}], KeywordRetriever, FakeGenerator())
    service._lock.acquire()
    try:
        with pytest.raises(ServiceError, match="service_busy"):
            service.answer("attention", "a")
    finally:
        service._lock.release()
