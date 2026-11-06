"""验证 Python 内部 HTTP 服务的协议、边界和问答联调行为。"""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from papertrace.generation import GenerationResult
from papertrace.internal_service import QuestionService, ServiceError, make_handler, serve
from papertrace.keyword_search import KeywordRetriever


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
