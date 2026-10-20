"""通过硅基流动 API 获取 BGE-M3 的稠密文本向量。"""

import http.client
import json
import math
import os
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import dotenv_values


EMBEDDINGS_URL = "https://api.siliconflow.cn/v1/embeddings"
DEFAULT_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
_MODELS = {DEFAULT_MODEL, "Pro/BAAI/bge-m3"}
_RETRYABLE_STATUSES = {429, 502, 503, 504}
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class EmbeddingError(RuntimeError):
    """表示已脱敏的接口或响应错误，不携带原始请求及响应正文。"""

    def __init__(
        self, message: str, *, status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        """保存安全错误信息、HTTP 状态码及是否适合稍后重试。"""
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class SiliconFlowEmbedder:
    """批量请求 BGE-M3 向量，按输入顺序返回并检查响应完整性。"""

    def __init__(
        self, api_key: str | None = None, *, model: str = DEFAULT_MODEL,
        batch_size: int = 8, timeout: float = 30.0, max_retries: int = 0,
    ) -> None:
        """按显式参数、环境变量、本地配置的优先级读取密钥，不自动重试。"""
        key = _resolve_api_key(api_key)
        if not isinstance(key, str) or not key.strip():
            raise ValueError("请在环境变量或项目 .env 中配置 SILICONFLOW_API_KEY，或显式提供 api_key")
        key = key.strip()
        if any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise ValueError("api_key 必须是无空白的 ASCII 字符串")
        if not isinstance(model, str) or model not in _MODELS:
            raise ValueError("model 必须是 BAAI/bge-m3 或 Pro/BAAI/bge-m3")
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("batch_size 必须是 1 到 32 的整数（客户端限制）")
        if (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("timeout 必须是有限的正数")
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries 必须是 0 到 3 的整数")
        self._api_key = key
        self.model = model
        self.batch_size = batch_size
        self.timeout = float(timeout)
        self.max_retries = max_retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        """校验全部文本后分批请求；空列表不联网，失败不返回部分结果。"""
        if not isinstance(texts, list):
            raise ValueError("texts 必须是字符串列表，而不是单个字符串")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] 必须是非空文本")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(f"texts[{index}] 包含非法 Unicode 字符") from None
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = {"model": self.model, "input": batch, "encoding_format": "float"}
            raw = self._request(payload)
            vectors.extend(_parse_vectors(raw, len(batch), self.model))
        return vectors

    def _request(self, payload: dict) -> bytes:
        """只对限流、暂时服务故障和网络异常执行显式允许的有限重试。"""
        for attempt in range(self.max_retries + 1):
            try:
                return _post_json(payload, self._api_key, self.timeout)
            except HTTPError as exc:
                status = exc.code
                exc.close()
                retryable = status in _RETRYABLE_STATUSES
                error = EmbeddingError(
                    f"向量化请求失败（HTTP {status}）；请检查授权、输入、额度或服务状态",
                    status_code=status, retryable=retryable,
                )
            except (ssl.SSLError, URLError, OSError, http.client.HTTPException) as exc:
                reason = exc.reason if isinstance(exc, URLError) else exc
                retryable = not isinstance(reason, ssl.SSLError)
                error = EmbeddingError(
                    "向量化请求网络失败或超时；请检查网络与 TLS 配置",
                    retryable=retryable,
                )
            if not error.retryable or attempt == self.max_retries:
                raise error from None
            time.sleep(2 ** attempt)
        raise AssertionError("不可达的重试状态")


class _NoRedirect(HTTPRedirectHandler):
    """拒绝重定向，防止认证头被转发到其他地址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """交由 urllib 抛出 HTTP 错误，不创建后续请求。"""
        return None


def _resolve_api_key(api_key: str | None) -> str | None:
    """仅在参数和环境变量缺失时读取项目 .env，不修改进程环境或插值。"""
    if api_key is not None:
        return api_key
    if "SILICONFLOW_API_KEY" in os.environ:
        return os.environ["SILICONFLOW_API_KEY"]
    try:
        if not _DEFAULT_ENV_FILE.exists():
            return None
        return dotenv_values(
            _DEFAULT_ENV_FILE, encoding="utf-8-sig", interpolate=False,
        ).get("SILICONFLOW_API_KEY")
    except (OSError, UnicodeError):
        raise ValueError("无法读取项目 .env，请检查文件权限与 UTF-8 编码") from None


def _post_json(payload: dict, api_key: str, timeout: float) -> bytes:
    """向固定 HTTPS 地址发送 JSON，并限制响应体大小；不记录正文。"""
    request = Request(
        EMBEDDINGS_URL,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise EmbeddingError("向量化响应过大，已拒绝处理")
    return raw


def _parse_vectors(raw: bytes, expected_count: int, model: str) -> list[list[float]]:
    """验证模型、条数、索引和有限数值，并按响应索引恢复输入顺序。"""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise EmbeddingError("向量化响应不是有效 JSON") from None
    if not isinstance(data, dict) or data.get("model") != model:
        raise EmbeddingError("向量化响应模型不匹配或结构无效")
    items = data.get("data")
    if not isinstance(items, list) or len(items) != expected_count:
        raise EmbeddingError("向量化响应条数与输入不一致")
    ordered: dict[int, list[float]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise EmbeddingError("向量化响应条目必须是对象")
        index = item.get("index")
        if type(index) is not int or not 0 <= index < expected_count or index in ordered:
            raise EmbeddingError("向量化响应索引重复、缺失或越界")
        values = item.get("embedding")
        if not isinstance(values, list) or len(values) != EMBEDDING_DIMENSION:
            raise EmbeddingError("BGE-M3 向量维度必须为 1024")
        vector: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingError("向量化响应必须包含有限数值")
            try:
                number = float(value)
            except (OverflowError, ValueError):
                raise EmbeddingError("向量化响应必须包含有限数值") from None
            if not math.isfinite(number):
                raise EmbeddingError("向量化响应必须包含有限数值")
            vector.append(number)
        if not any(vector):
            raise EmbeddingError("向量化响应不能是零向量")
        ordered[index] = vector
    return [ordered[index] for index in range(expected_count)]
