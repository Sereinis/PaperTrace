"""提供可替换的生成契约及 DeepSeek 官方非流式文本适配层。"""

import http.client
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import dotenv_values


CHAT_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
_MODELS = {DEFAULT_MODEL, "deepseek-v4-pro"}
_DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class GenerationError(RuntimeError):
    """表示已脱敏的生成错误，不包含密钥、提示词或服务端正文。"""

    def __init__(
        self, message: str, *, status_code: int | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        """保存安全错误信息与已验证用量，不保留原始响应。"""
        super().__init__(message)
        self.status_code = status_code
        self.usage = safe_usage(usage)


@dataclass(frozen=True)
class GenerationResult:
    """保存完整输出与复核元数据，默认表示中隐藏生成正文。"""

    content: str = field(repr=False)
    requested_model: str
    response_model: str
    request_id: str
    usage: dict[str, int] | None
    finish_reason: str = "stop"


class TextGenerator(Protocol):
    """定义生成层的最小契约，不将后续问答逻辑绑定到服务商。"""

    def generate(
        self, messages: list[dict[str, str]], *, max_tokens: int = 2048,
        temperature: float = 0.0, json_mode: bool = False,
    ) -> GenerationResult:
        """根据文本消息生成一个完整结果，失败时抛出安全异常。"""
        ...


class DeepSeekGenerator:
    """只调用官方固定地址，关闭思考、流式和工具调用，不自动重试。"""

    def __init__(
        self, api_key: str | None = None, *, model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        """优先使用显式密钥，其次进程环境，最后读取项目本地配置。"""
        if not isinstance(model, str) or model not in _MODELS:
            raise ValueError("model 必须是 deepseek-flash 或 deepseek-v4-pro")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not 0 < timeout <= 300):
            raise ValueError("timeout 必须是 0 到 300 之间的有限正数")
        key = _resolve_api_key(api_key)
        if not isinstance(key, str) or not key.strip():
            raise ValueError("请配置 DEEPSEEK_API_KEY 或显式提供 api_key")
        key = key.strip()
        if any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise ValueError("api_key 必须是无空白的 ASCII 字符串")
        self._api_key = key
        self.model = model
        self.timeout = float(timeout)

    def generate(
        self, messages: list[dict[str, str]], *, max_tokens: int = 2048,
        temperature: float = 0.0, json_mode: bool = False,
    ) -> GenerationResult:
        """校验全部输入后仅发一次请求，拒绝截断、过滤和工具调用结果。"""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 必须是非空文本消息列表")
        clean = []
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or not isinstance(message["role"], str)
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)
                    or not message["content"].strip()):
                raise ValueError("每条消息仅接受 role 与非空 content，角色限 system/user/assistant")
            clean.append(dict(message))
        if not any(item["role"] == "user" for item in clean):
            raise ValueError("messages 至少需要一条 user 消息")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("max_tokens 必须是 1 到 8192 的整数（客户端限制）")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not 0 <= temperature <= 2):
            raise ValueError("temperature 必须是 0 到 2 之间的有限数值")
        if type(json_mode) is not bool:
            raise ValueError("json_mode 必须是布尔值")
        if json_mode and not any("json" in item["content"].lower() for item in clean):
            raise ValueError("JSON 模式需要在消息中明确要求生成 JSON")
        payload = {
            "model": self.model, "messages": clean, "max_tokens": max_tokens,
            "temperature": float(temperature), "stream": False,
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError):
            raise ValueError("消息包含非法 Unicode 或无法编码的内容") from None
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("生成请求超过客户端 1 MiB 上限；不会自动截断")
        try:
            raw = _post_json(body, self._api_key, self.timeout)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise GenerationError(
                f"生成请求失败（HTTP {status}）；请检查授权、额度、输入或服务状态",
                status_code=status,
            ) from None
        except (URLError, OSError, http.client.HTTPException):
            raise GenerationError("生成请求网络失败或超时；未自动重试，服务端可能已计费") from None
        return _parse_result(raw, self.model, json_mode)


def _resolve_api_key(api_key: str | None) -> str | None:
    """按优先级读取密钥，不修改环境、不插值、不向父目录搜索。"""
    if api_key is not None:
        return api_key
    if "DEEPSEEK_API_KEY" in os.environ:
        return os.environ["DEEPSEEK_API_KEY"]
    try:
        if not _DEFAULT_ENV_FILE.exists():
            return None
        return dotenv_values(
            _DEFAULT_ENV_FILE, encoding="utf-8-sig", interpolate=False,
        ).get("DEEPSEEK_API_KEY")
    except (OSError, UnicodeError):
        raise ValueError("无法读取项目 .env，请检查权限与 UTF-8 编码") from None


class _NoRedirect(HTTPRedirectHandler):
    """阻止重定向，避免把认证信息转发到其他地址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """拒绝创建重定向请求，由 urllib 抛出 HTTP 错误。"""
        return None


def _post_json(body: bytes, api_key: str, timeout: float) -> bytes:
    """发送固定 HTTPS 请求，限制读取大小且关闭连接，不记录正文。"""
    request = Request(
        CHAT_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise GenerationError("生成响应超过客户端大小上限")
    return raw


def _json_object(pairs: list[tuple]) -> dict:
    """拒绝重复 JSON 键，避免不同解析器对同一响应产生不同解释。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("重复 JSON 键")
        result[key] = value
    return result


def _reject_constant(value: str):
    """拒绝 JSON 标准以外的 NaN 和 Infinity 常量。"""
    raise ValueError("非法 JSON 常量")


def _load_json(value: bytes | str):
    """解析严格 UTF-8 JSON，错误不回显服务端响应内容。"""
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(text, object_pairs_hook=_json_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise GenerationError("生成响应不是有效的严格 JSON") from None


def _parse_usage(value) -> dict[str, int] | None:
    """保留已验证的 token 统计，缺失时保持未知，不估算成本。"""
    if value is None:
        return None
    required = {"prompt_tokens", "completion_tokens", "total_tokens"}
    optional = {"prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise GenerationError("生成响应的 token 用量结构无效")
    result = {}
    for name in required | (optional & value.keys()):
        count = value[name]
        if type(count) is not int or count < 0:
            raise GenerationError("生成响应的 token 用量必须为非负整数")
        result[name] = count
    if any(result[name] > result["prompt_tokens"] for name in optional & result.keys()):
        raise GenerationError("生成响应的缓存 token 用量超过输入总量")
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        raise GenerationError("生成响应的 token 总量不一致")
    if optional <= result.keys():
        if result["prompt_tokens"] != sum(result[name] for name in optional):
            raise GenerationError("生成响应的缓存 token 用量不一致")
    return result


def safe_usage(value) -> dict[str, int] | None:
    """仅保留通过既有契约检查的用量，畸形统计保持未知。"""
    try:
        return _parse_usage(value)
    except GenerationError:
        return None


def _parse_result(raw: bytes, requested_model: str, json_mode: bool) -> GenerationResult:
    """失败时尽可能保留合法用量，不附带答案、模型声明或原始响应。"""
    try:
        return _parse_result_checked(raw, requested_model, json_mode)
    except GenerationError as error:
        message = str(error)
    usage = None
    try:
        data = _load_json(raw)
        if isinstance(data, dict):
            usage = safe_usage(data.get("usage"))
    except GenerationError:
        pass
    raise GenerationError(message, usage=usage) from None


def _parse_result_checked(raw: bytes, requested_model: str, json_mode: bool) -> GenerationResult:
    """验证单条完整文本响应；记录返回模型名，不据别名推断固定权重。"""
    data = _load_json(raw)
    if not isinstance(data, dict) or data.get("object") != "chat.completion":
        raise GenerationError("生成响应类型无效")
    for name in ("id", "model"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise GenerationError("生成响应缺少模型或请求身份")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise GenerationError("生成响应必须恰好包含一条结果")
    choice = choices[0]
    if type(choice.get("index")) is not int or choice["index"] != 0:
        raise GenerationError("生成响应索引无效")
    if choice.get("finish_reason") != "stop":
        raise GenerationError("生成未完整结束：可能截断、过滤或中断；未自动重试")
    message = choice.get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or message.get("tool_calls") not in (None, [])
            or message.get("reasoning_content") not in (None, "")
            or not isinstance(message.get("content"), str) or not message["content"].strip()):
        raise GenerationError("生成响应必须包含非空答案且不能包含工具调用或思考内容")
    content = message["content"]
    try:
        content.encode("utf-8")
        data["id"].encode("utf-8")
        data["model"].encode("utf-8")
    except UnicodeError:
        raise GenerationError("生成响应包含非法 Unicode 字符") from None
    if json_mode and not isinstance(_load_json(content), dict):
        raise GenerationError("JSON 模式输出必须是对象")
    return GenerationResult(
        content=content, requested_model=requested_model, response_model=data["model"],
        request_id=data["id"], usage=_parse_usage(data.get("usage")),
    )
