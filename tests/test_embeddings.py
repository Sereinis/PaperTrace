"""不访问网络、不使用真实密钥的向量化适配层测试。"""

import io
import json
import ssl
import traceback
from urllib.error import HTTPError, URLError

import pytest

from papertrace import embeddings
from papertrace.embeddings import EmbeddingError, SiliconFlowEmbedder


@pytest.fixture(autouse=True)
def isolate_local_config(monkeypatch, tmp_path):
    """每个测试仅使用临时配置，绝不读取开发者的真实 .env 或密钥。"""
    monkeypatch.setattr(embeddings, "_DEFAULT_ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)


def test_reads_local_env_without_changing_process_environment(monkeypatch, tmp_path):
    """支持本地 UTF-8 BOM、引号与注释，不插值也不注入环境变量。"""
    embeddings._DEFAULT_ENV_FILE.write_text(
        '# local config\nSILICONFLOW_API_KEY="test-${OTHER_KEY}" # comment\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setenv("OTHER_KEY", "must-not-expand")
    monkeypatch.chdir(tmp_path.parent)
    client = SiliconFlowEmbedder()
    assert client._api_key == "test-${OTHER_KEY}"
    assert "SILICONFLOW_API_KEY" not in embeddings.os.environ


def test_explicit_key_and_environment_take_precedence(monkeypatch):
    """已有参数或环境值时不读取配置文件；空环境值也不能静默降级。"""
    def unexpected(*args, **kwargs):
        """优先配置存在时禁止访问文件。"""
        pytest.fail("unexpected dotenv read")

    monkeypatch.setattr(embeddings, "dotenv_values", unexpected)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "env-key")
    assert SiliconFlowEmbedder()._api_key == "env-key"
    assert SiliconFlowEmbedder("explicit-key")._api_key == "explicit-key"
    monkeypatch.setenv("SILICONFLOW_API_KEY", "")
    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        SiliconFlowEmbedder()


@pytest.mark.parametrize("content", ["", "SILICONFLOW_API_KEY=\n", "SILICONFLOW_API_KEY\n"])
def test_empty_local_configuration_reports_missing_key(content):
    """空白或未填写密钥时提示配置，不尝试联网。"""
    embeddings._DEFAULT_ENV_FILE.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        SiliconFlowEmbedder()


def test_local_configuration_error_is_sanitized(monkeypatch):
    """读取配置失败时不暴露文件内容和原始异常。"""
    embeddings._DEFAULT_ENV_FILE.touch()

    def fail(*args, **kwargs):
        """模拟带私密信息的读取错误。"""
        raise OSError("private-marker")

    monkeypatch.setattr(embeddings, "dotenv_values", fail)
    with pytest.raises(ValueError) as caught:
        SiliconFlowEmbedder()
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))


def response(items=None, model=embeddings.DEFAULT_MODEL):
    """构造与 BGE-M3 接口形状一致的模拟响应。"""
    if items is None:
        items = [{"index": 0, "embedding": [1.0] * 1024}]
    return json.dumps({"model": model, "data": items}).encode()


def test_batches_and_restores_order_without_mutating_inputs(monkeypatch):
    """响应乱序时仍按输入顺序返回，跨批次不混淆局部索引。"""
    calls = []

    def post(payload, key, timeout):
        """记录请求，并故意倒序返回当前批次。"""
        calls.append((payload, key, timeout))
        return response([
            {"index": index, "embedding": [float(text)] * 1024}
            for index, text in reversed(list(enumerate(payload["input"])))
        ])

    monkeypatch.setattr(embeddings, "_post_json", post)
    texts = ["1", "2", "3"]
    result = SiliconFlowEmbedder("test-key", batch_size=2).embed(texts)
    assert [vector[0] for vector in result] == [1, 2, 3]
    assert texts == ["1", "2", "3"]
    assert [call[0]["input"] for call in calls] == [["1", "2"], ["3"]]
    assert calls[0][0]["encoding_format"] == "float"
    assert "dimensions" not in calls[0][0]
    assert calls[0][1:] == ("test-key", 30.0)


def test_environment_key_and_explicit_pro_model(monkeypatch):
    """密钥可来自环境，Pro 模型必须显式选择且不自动切换。"""
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-env-key")

    def post(payload, key, timeout):
        """检查实际使用的模型和环境密钥。"""
        assert key == "test-env-key"
        assert payload["model"] == "Pro/BAAI/bge-m3"
        return response(model=payload["model"])

    monkeypatch.setattr(embeddings, "_post_json", post)
    assert len(SiliconFlowEmbedder(model="Pro/BAAI/bge-m3").embed(["text"])[0]) == 1024


def test_missing_key_is_clear(monkeypatch):
    """未配置密钥时，在发送请求前返回配置提示。"""
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SILICONFLOW_API_KEY"):
        SiliconFlowEmbedder()


@pytest.mark.parametrize("kwargs", [
    {"api_key": ""}, {"api_key": "key\nother"}, {"api_key": "密钥"},
    {"model": "unknown"}, {"batch_size": 0}, {"batch_size": 33},
    {"batch_size": True}, {"timeout": 0}, {"timeout": float("nan")},
    {"timeout": float("inf")}, {"timeout": True}, {"timeout": "30"},
    {"max_retries": -1}, {"max_retries": 4}, {"max_retries": True},
])
def test_rejects_invalid_config(kwargs):
    """拒绝无效配置，不在错误中回显配置值。"""
    with pytest.raises(ValueError):
        SiliconFlowEmbedder(**({"api_key": "test-key"} | kwargs))


@pytest.mark.parametrize("texts", ["single", None, [""], [" \n"], [1], ["ok", ""], ["\ud800"]])
def test_validates_entire_input_before_request(monkeypatch, texts):
    """后续批次有无效文本时也不能先发送前面的批次。"""
    def unexpected(*args):
        """输入非法时不应联网。"""
        pytest.fail("unexpected network call")

    monkeypatch.setattr(embeddings, "_post_json", unexpected)
    client = SiliconFlowEmbedder("test-key", batch_size=1)
    assert client.embed([]) == []
    with pytest.raises(ValueError):
        client.embed(texts)


@pytest.mark.parametrize("raw", [
    b"not json", b"\xff", b"[]", response(model="other-model"),
    response(items=[]), response(items=[None]), response(items=[{"index": True}]),
    response(items=[{"index": -1}]), response(items=[{"index": 1}]),
    response(items=[{"embedding": [1] * 1024}]),
    response(items=[{"index": 0, "embedding": [1] * 1023}]),
    response(items=[{"index": 0, "embedding": [0] * 1024}]),
    *[response(items=[{"index": 0, "embedding": [value] * 1024}])
      for value in [True, "1", None, float("nan"), float("inf")]],
])
def test_rejects_malformed_responses(monkeypatch, raw):
    """拒绝格式、模型、索引、维度和数值异常，不接受部分结果。"""
    monkeypatch.setattr(embeddings, "_post_json", lambda *args: raw)
    with pytest.raises(EmbeddingError):
        SiliconFlowEmbedder("test-key").embed(["text"])


def test_duplicate_indexes_are_rejected(monkeypatch):
    """重复索引即使响应条数正确也不能通过。"""
    raw = response(items=[{"index": 0, "embedding": [1] * 1024}] * 2)
    monkeypatch.setattr(embeddings, "_post_json", lambda *args: raw)
    with pytest.raises(EmbeddingError, match="索引"):
        SiliconFlowEmbedder("test-key").embed(["one", "two"])


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 429, 500, 502, 503, 504])
def test_http_errors_are_sanitized_and_not_retried_by_default(monkeypatch, status):
    """不泄露服务端正文、认证信息或原异常，默认只请求一次。"""
    calls = []

    def post(*args):
        """构造含敏感测试标记的原始 HTTP 错误。"""
        calls.append(1)
        raise HTTPError("https://example.invalid", status, "private-marker", {},
                        io.BytesIO(b"private-marker"))

    monkeypatch.setattr(embeddings, "_post_json", post)
    secret = "private-marker"
    client = SiliconFlowEmbedder(secret)
    with pytest.raises(EmbeddingError) as caught:
        client.embed([secret])
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private-marker" not in rendered
    assert caught.value.status_code == status
    assert caught.value.retryable == (status in {429, 502, 503, 504})
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [
    URLError("private-marker"), TimeoutError("private-marker"),
    embeddings.http.client.RemoteDisconnected("private-marker"),
])
def test_explicit_retry_is_bounded(monkeypatch, failure):
    """显式允许重试时有限退避，耗尽后仅返回安全错误。"""
    calls, sleeps = [], []

    def post(*args):
        """始终失败，用于检查请求次数。"""
        calls.append(1)
        raise failure

    monkeypatch.setattr(embeddings, "_post_json", post)
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)
    with pytest.raises(EmbeddingError) as caught:
        SiliconFlowEmbedder("test-key", max_retries=2).embed(["text"])
    assert len(calls) == 3
    assert sleeps == [1, 2]
    assert caught.value.retryable
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_http_errors_never_retry(monkeypatch, status):
    """授权和输入等永久错误即使开启重试也立即停止。"""
    calls = []

    def post(*args):
        """模拟不可重试 HTTP 错误。"""
        calls.append(1)
        raise HTTPError("https://example.invalid", status, "error", {}, io.BytesIO())

    monkeypatch.setattr(embeddings, "_post_json", post)
    with pytest.raises(EmbeddingError):
        SiliconFlowEmbedder("test-key", max_retries=2).embed(["text"])
    assert len(calls) == 1


def test_rate_limit_can_recover_on_explicit_retry(monkeypatch):
    """限流后可在有限重试中恢复，不重复已成功的批次。"""
    calls = []

    def post(*args):
        """第一次限流，第二次返回有效向量。"""
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError("https://example.invalid", 429, "limited", {}, io.BytesIO())
        return response()

    monkeypatch.setattr(embeddings, "_post_json", post)
    monkeypatch.setattr(embeddings.time, "sleep", lambda _: None)
    assert SiliconFlowEmbedder("test-key", max_retries=1).embed(["text"])
    assert len(calls) == 2


def test_tls_errors_do_not_retry(monkeypatch):
    """证书错误不能通过重试或关闭证书验证来绕过。"""
    calls = []

    def post(*args):
        """模拟证书校验失败。"""
        calls.append(1)
        raise URLError(ssl.SSLCertVerificationError("private-marker"))

    monkeypatch.setattr(embeddings, "_post_json", post)
    with pytest.raises(EmbeddingError) as caught:
        SiliconFlowEmbedder("test-key", max_retries=2).embed(["text"])
    assert not caught.value.retryable
    assert len(calls) == 1


def test_transport_sends_expected_request_and_closes_response(monkeypatch):
    """验证固定 HTTPS 地址、请求头、UTF-8 JSON、超时和响应关闭。"""
    stream = io.BytesIO(response())

    class Opener:
        """代替网络连接，检查发出的请求。"""

        def open(self, request, timeout):
            """仅检查测试密钥，不接触真实网络。"""
            assert request.full_url == embeddings.EMBEDDINGS_URL
            assert request.method == "POST"
            assert request.get_header("Authorization") == "Bearer test-key"
            assert request.get_header("Content-type") == "application/json"
            assert json.loads(request.data)["input"] == ["中文 text"]
            assert timeout == 7
            return stream

    def opener(handler):
        """确认传输层安装了禁止重定向的处理器。"""
        assert isinstance(handler, embeddings._NoRedirect)
        assert handler.redirect_request(None, None, 302, "", {}, "https://other.invalid") is None
        return Opener()

    monkeypatch.setattr(embeddings, "build_opener", opener)
    assert SiliconFlowEmbedder("test-key", timeout=7).embed(["中文 text"])
    assert stream.closed


def test_response_size_limit(monkeypatch):
    """异常大响应在解析前被拒绝。"""
    monkeypatch.setattr(embeddings, "_MAX_RESPONSE_BYTES", 10)

    class Opener:
        """返回超过客户端上限的响应体。"""

        def open(self, *args, **kwargs):
            """构造可关闭的模拟响应。"""
            return io.BytesIO(b"x" * 11)

    monkeypatch.setattr(embeddings, "build_opener", lambda _: Opener())
    with pytest.raises(EmbeddingError, match="过大"):
        SiliconFlowEmbedder("test-key").embed(["text"])


def test_bad_response_is_not_retried(monkeypatch):
    """响应契约错误不通过重试掩盖。"""
    calls = []

    def post(*args):
        """返回无效响应，同时记录调用次数。"""
        calls.append(1)
        return b"{}"

    monkeypatch.setattr(embeddings, "_post_json", post)
    with pytest.raises(EmbeddingError):
        SiliconFlowEmbedder("test-key", max_retries=2).embed(["text"])
    assert len(calls) == 1


def test_later_batch_failure_does_not_return_partial_vectors(monkeypatch):
    """后续批次失败时整个操作报错，而非返回不完整向量列表。"""
    calls = []

    def post(*args):
        """第一批成功，第二批返回缺失数据。"""
        calls.append(1)
        return response() if len(calls) == 1 else response(items=[])

    monkeypatch.setattr(embeddings, "_post_json", post)
    with pytest.raises(EmbeddingError, match="条数"):
        SiliconFlowEmbedder("test-key", batch_size=1).embed(["one", "two"])
    assert len(calls) == 2


def test_extremely_large_number_is_rejected(monkeypatch):
    """JSON 中可解析但无法转换为有限浮点数的整数也应拒绝。"""
    raw = response(items=[{"index": 0, "embedding": [10 ** 1000] * 1024}])
    monkeypatch.setattr(embeddings, "_post_json", lambda *args: raw)
    with pytest.raises(EmbeddingError, match="有限数值"):
        SiliconFlowEmbedder("test-key").embed(["text"])
