"""仅使用虚构密钥和模拟传输验证生成层，不访问真实配置或网络。"""

import copy
import io
import json
import ssl
import traceback
from urllib.error import HTTPError, URLError

import pytest

from papertrace import generation
from papertrace.generation import DeepSeekGenerator, GenerationError


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    """隔离开发者本地配置，并在测试遗漏模拟时阻止网络连接。"""
    monkeypatch.setattr(generation, "_DEFAULT_ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def blocked(*args, **kwargs):
        """任何未显式模拟的网络连接均使测试失败。"""
        pytest.fail("unexpected network call")

    monkeypatch.setattr(generation, "build_opener", blocked)


def payload():
    """构造一份非思考模式的完整响应对象。"""
    return {
        "id": "test-request", "object": "chat.completion", "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "测试答案", "reasoning_content": None,
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25,
                  "prompt_cache_hit_tokens": 12, "prompt_cache_miss_tokens": 8},
    }


def mock_response(monkeypatch, data):
    """让传输函数返回给定 JSON，并记录唯一请求。"""
    calls = []

    def post(body, key, timeout):
        """只记录虚构测试请求，不使用网络。"""
        calls.append((json.loads(body), key, timeout))
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(generation, "_post_json", post)
    return calls


def test_default_contract_and_immutable_input(monkeypatch):
    """默认关闭思考与流式，保留完整答案和用量，不改变输入消息。"""
    calls = mock_response(monkeypatch, payload())
    messages = [{"role": "system", "content": "仅使用证据。"},
                {"role": "user", "content": "问题"}]
    original = copy.deepcopy(messages)
    result = DeepSeekGenerator("test-key").generate(messages)
    assert messages == original
    assert result.content == "测试答案"
    assert result.requested_model == result.response_model == "deepseek-flash"
    assert result.request_id == "test-request"
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 25
    assert calls == [({"model": "deepseek-flash", "messages": original,
                      "max_tokens": 2048, "temperature": 0.0, "stream": False,
                      "thinking": {"type": "disabled"}}, "test-key", 60.0)]
    assert "测试答案" not in repr(result)
    assert "test-key" not in repr(DeepSeekGenerator("test-key"))


def test_explicit_model_and_response_alias_are_recorded(monkeypatch):
    """可显式选择 Pro，返回别名按原样记录，不声称校验了服务端权重。"""
    data = payload()
    data["model"] = "DeepSeek-V4-Pro-0813"
    calls = mock_response(monkeypatch, data)
    result = DeepSeekGenerator("test-key", model="deepseek-v4-pro", timeout=7).generate(
        [{"role": "user", "content": "问题"}], max_tokens=100, temperature=0.2,
    )
    assert result.requested_model == "deepseek-v4-pro"
    assert result.response_model == "DeepSeek-V4-Pro-0813"
    assert calls[0][0]["max_tokens"] == 100
    assert calls[0][2] == 7


def test_configuration_priority(monkeypatch):
    """显式值和进程环境优先，已有空环境值不能静默降级。"""
    def forbidden(*args, **kwargs):
        """已有更高优先级值时禁止读取文件。"""
        pytest.fail("unexpected dotenv read")

    monkeypatch.setattr(generation, "dotenv_values", forbidden)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert DeepSeekGenerator()._api_key == "env-key"
    assert DeepSeekGenerator("explicit-key")._api_key == "explicit-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekGenerator()


def test_local_env_does_not_interpolate_or_change_environment(monkeypatch, tmp_path):
    """支持 BOM 与注释，不把本地配置注入环境，不展开占位变量。"""
    generation._DEFAULT_ENV_FILE.write_text(
        'DEEPSEEK_API_KEY="test-${OTHER}" # comment\n', encoding="utf-8-sig",
    )
    monkeypatch.setenv("OTHER", "private-marker")
    monkeypatch.chdir(tmp_path.parent)
    assert DeepSeekGenerator()._api_key == "test-${OTHER}"
    assert "DEEPSEEK_API_KEY" not in generation.os.environ


@pytest.mark.parametrize("content", [None, "", "DEEPSEEK_API_KEY=\n", "DEEPSEEK_API_KEY\n"])
def test_missing_key(content):
    """配置缺失或为空时明确报错，不进行请求。"""
    if content is not None:
        generation._DEFAULT_ENV_FILE.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekGenerator()


def test_config_errors_are_sanitized(monkeypatch):
    """文件错误不泄漏原始异常内容。"""
    generation._DEFAULT_ENV_FILE.touch()

    def fail(*args, **kwargs):
        """模拟包含敏感标记的读取失败。"""
        raise OSError("private-marker")

    monkeypatch.setattr(generation, "dotenv_values", fail)
    with pytest.raises(ValueError) as caught:
        DeepSeekGenerator()
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("kwargs", [
    {"api_key": ""}, {"api_key": "key\nother"}, {"api_key": "密钥"}, {"api_key": 1},
    {"model": "unknown"}, {"model": []}, {"timeout": 0}, {"timeout": -1},
    {"timeout": 301}, {"timeout": True}, {"timeout": "60"},
    {"timeout": float("nan")}, {"timeout": float("inf")},
])
def test_invalid_constructor(kwargs):
    """非法模型、密钥与超时均在联网前拒绝。"""
    with pytest.raises(ValueError):
        DeepSeekGenerator(**({"api_key": "test-key"} | kwargs))


@pytest.mark.parametrize("messages", [
    None, [], "question", [None], [{"role": [], "content": "text"}],
    [{"role": "tool", "content": "text"}], [{"role": "user", "content": " "}],
    [{"role": "user", "content": 1}], [{"role": "user"}],
    [{"role": "user", "content": "text", "name": "extra"}],
    [{"role": "system", "content": "text"}], [{"role": "user", "content": "\ud800"}],
    [{"role": "user", "content": "valid"}, {"role": "user", "content": ""}],
])
def test_invalid_messages_never_send(messages):
    """完整检查消息列表和 UTF-8，不先发送有效部分。"""
    with pytest.raises(ValueError):
        DeepSeekGenerator("test-key").generate(messages)


@pytest.mark.parametrize("kwargs", [
    {"max_tokens": 0}, {"max_tokens": 8193}, {"max_tokens": True}, {"max_tokens": 1.0},
    {"temperature": -1}, {"temperature": 3}, {"temperature": True},
    {"temperature": float("nan")}, {"temperature": float("inf")},
    {"temperature": "0"}, {"json_mode": 1}, {"json_mode": True},
])
def test_invalid_generation_options(kwargs):
    """参数错误和 JSON 模式缺少提示时不得计费。"""
    with pytest.raises(ValueError):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}], **kwargs)


def test_request_size_limit(monkeypatch):
    """客户端按字节限制请求，不以字符冒充 token，也不静默截断。"""
    monkeypatch.setattr(generation, "_MAX_REQUEST_BYTES", 20)
    with pytest.raises(ValueError, match="上限"):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])


def test_json_mode_validates_object(monkeypatch):
    """JSON 模式发送格式要求并验证结果，但不声称已验证业务字段。"""
    data = payload()
    data["choices"][0]["message"]["content"] = '{"answer":"ok"}'
    calls = mock_response(monkeypatch, data)
    result = DeepSeekGenerator("test-key").generate(
        [{"role": "user", "content": "Return JSON."}], json_mode=True,
    )
    assert json.loads(result.content) == {"answer": "ok"}
    assert calls[0][0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("content", ["[]", "null", "not json", '{"x":1,"x":2}', '{"x":NaN}'])
def test_invalid_json_mode_output(monkeypatch, content):
    """对象格式错误不会被当作合法结构化结果返回或重试。"""
    data = payload()
    data["choices"][0]["message"]["content"] = content
    calls = mock_response(monkeypatch, data)
    with pytest.raises(GenerationError):
        DeepSeekGenerator("test-key").generate(
            [{"role": "user", "content": "Return JSON."}], json_mode=True,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("raw", [b"not json", b"\xff", b"[]", b'{"x":NaN}', b'{"x":1,"x":2}'])
def test_invalid_response_json(monkeypatch, raw):
    """外层响应必须是严格 JSON，不返回解析异常中的正文。"""
    monkeypatch.setattr(generation, "_post_json", lambda *args: raw)
    with pytest.raises(GenerationError):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])


@pytest.mark.parametrize("field,value", [
    ("object", "other"), ("id", ""), ("model", None), ("choices", []),
    ("choices", [None]), ("choices", [{}, {}]), ("id", "\ud800"),
])
def test_invalid_envelope(monkeypatch, field, value):
    """缺少身份、模型或唯一结果时拒绝响应。"""
    data = payload()
    data[field] = value
    monkeypatch.setattr(generation, "_post_json", lambda *args: json.dumps(data).encode())
    with pytest.raises(GenerationError):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])


@pytest.mark.parametrize("reason", [
    "length", "content_filter", "tool_calls", "insufficient_system_resource", "aborted", None,
])
def test_incomplete_output_rejected(monkeypatch, reason):
    """截断、过滤、中断和工具结果不是完整答案，即便正文非空。"""
    data = payload()
    data["choices"][0]["finish_reason"] = reason
    mock_response(monkeypatch, data)
    with pytest.raises(GenerationError, match="完整"):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])


@pytest.mark.parametrize("field,value", [
    ("content", None), ("content", " "), ("content", []), ("content", "\ud800"),
    ("role", "user"), ("tool_calls", [{"id": "tool"}]),
    ("reasoning_content", "unexpected reasoning"),
])
def test_invalid_message(monkeypatch, field, value):
    """拒绝空内容、错误角色、非法 Unicode、工具调用及意外思考内容。"""
    data = payload()
    data["choices"][0]["message"][field] = value
    monkeypatch.setattr(generation, "_post_json", lambda *args: json.dumps(data).encode())
    with pytest.raises(GenerationError):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])


@pytest.mark.parametrize("index", [True, 1, "0", None])
def test_invalid_choice_index(monkeypatch, index):
    """仅接受整数零作为单条结果索引。"""
    data = payload()
    data["choices"][0]["index"] = index
    mock_response(monkeypatch, data)
    with pytest.raises(GenerationError):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])


@pytest.mark.parametrize("usage", [
    {}, [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
    {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
    {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
    {"prompt_tokens": 1, "completion_tokens": 1.0, "total_tokens": 2},
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
     "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 0},
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
     "prompt_cache_hit_tokens": 2},
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
     "prompt_cache_miss_tokens": True},
])
def test_invalid_usage(monkeypatch, usage):
    """拒绝负数、布尔值、非整数及总量矛盾，避免生成伪成本记录。"""
    data = payload()
    data["usage"] = usage
    mock_response(monkeypatch, data)
    with pytest.raises(GenerationError, match="token"):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])


def test_missing_usage_stays_unknown(monkeypatch):
    """服务端未提供统计时保留未知，不以零值或估计数替代。"""
    data = payload()
    del data["usage"]
    mock_response(monkeypatch, data)
    assert DeepSeekGenerator("test-key").generate(
        [{"role": "user", "content": "问题"}],
    ).usage is None


@pytest.mark.parametrize("status", [301, 400, 401, 402, 403, 404, 422, 429, 500, 503])
def test_http_failure_sanitized_and_never_retried(monkeypatch, status):
    """服务端错误只暴露状态码，关闭响应且绝不重复请求。"""
    calls = []
    stream = io.BytesIO(b"private-marker")

    def fail(*args):
        """构造含私密测试标记的 HTTP 失败。"""
        calls.append(1)
        raise HTTPError("https://private-marker.invalid", status, "private-marker", {}, stream)

    monkeypatch.setattr(generation, "_post_json", fail)
    secret = "private-marker"
    client = DeepSeekGenerator(secret)
    messages = [{"role": "user", "content": secret}]
    with pytest.raises(GenerationError) as caught:
        client.generate(messages)
    assert caught.value.status_code == status
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))
    assert len(calls) == 1 and stream.closed


@pytest.mark.parametrize("failure", [
    URLError("private-marker"), TimeoutError("private-marker"),
    URLError(ssl.SSLCertVerificationError("private-marker")),
    generation.http.client.RemoteDisconnected("private-marker"),
])
def test_network_failure_sanitized_and_not_retried(monkeypatch, failure):
    """超时、网络与证书失败不自动重试，不泄漏底层异常。"""
    calls = []

    def fail(*args):
        """模拟不确定是否已在远端执行的失败。"""
        calls.append(1)
        raise failure

    monkeypatch.setattr(generation, "_post_json", fail)
    with pytest.raises(GenerationError) as caught:
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])
    assert "private-marker" not in "".join(traceback.format_exception(caught.value))
    assert len(calls) == 1


def test_transport_fixed_url_headers_timeout_and_close(monkeypatch):
    """检查 HTTPS 地址、请求头和禁止重定向，并确认响应关闭。"""
    stream = io.BytesIO(json.dumps(payload()).encode())

    class Opener:
        """只检查请求，不连接网络。"""

        def open(self, request, timeout):
            """验证实际构造的标准库 HTTP 请求。"""
            assert request.full_url == "https://api.deepseek.com/chat/completions"
            assert request.method == "POST"
            assert request.get_header("Authorization") == "Bearer test-key"
            assert request.get_header("Content-type") == "application/json"
            assert json.loads(request.data)["messages"][0]["content"] == "中文问题"
            assert timeout == 7
            return stream

    def opener(handler):
        """确认重定向处理器拒绝创建第二个请求。"""
        assert isinstance(handler, generation._NoRedirect)
        assert handler.redirect_request(None, None, 302, "", {}, "https://other.invalid") is None
        return Opener()

    monkeypatch.setattr(generation, "build_opener", opener)
    DeepSeekGenerator("test-key", timeout=7).generate([{"role": "user", "content": "中文问题"}])
    assert stream.closed


def test_oversized_response_closed_and_rejected(monkeypatch):
    """超大响应只读上限加一字节并拒绝，不继续解析或重试。"""
    monkeypatch.setattr(generation, "_MAX_RESPONSE_BYTES", 10)
    stream = io.BytesIO(b"x" * 100)

    class Opener:
        """提供过大的模拟响应。"""

        def open(self, *args, **kwargs):
            """返回内存流以检查关闭行为。"""
            return stream

    monkeypatch.setattr(generation, "build_opener", lambda _: Opener())
    with pytest.raises(GenerationError, match="大小"):
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "问题"}])
    assert stream.closed


@pytest.mark.parametrize("mode", ["length", "content_filter", "invalid-json-content", "missing-id"])
def test_response_failure_keeps_valid_usage_without_content(monkeypatch, mode):
    """响应已带统计但正文检查失败时，安全异常保留统计而非原始输出。"""
    data = payload()
    if mode in {"length", "content_filter"}:
        data["choices"][0]["finish_reason"] = mode
    elif mode == "missing-id":
        del data["id"]
    else:
        data["choices"][0]["message"]["content"] = "private-marker"
    calls = mock_response(monkeypatch, data)
    with pytest.raises(GenerationError) as caught:
        DeepSeekGenerator("test-key").generate(
            [{"role": "user", "content": "Return JSON"}], json_mode=True,
        )
    assert caught.value.usage["total_tokens"] == 25
    assert "private-marker" not in str(caught.value)
    assert not hasattr(caught.value, "content")
    assert len(calls) == 1


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens": True}, {"total_tokens": float("nan")}, None])
def test_failure_never_preserves_invalid_usage(monkeypatch, usage):
    """不把失败响应中的未知或畸形统计当作可计费事实。"""
    data = payload()
    data["usage"] = usage
    data["choices"][0]["finish_reason"] = "length"
    mock_response(monkeypatch, data)
    with pytest.raises(GenerationError) as caught:
        DeepSeekGenerator("test-key").generate([{"role": "user", "content": "test"}])
    assert caught.value.usage is None
