"""启动读取真实论文快照、缓存与官方生成配置的本机问答服务。"""

import argparse
import copy
import json
import math
from pathlib import Path

from papertrace.embedding_cache import embed_chunks
from papertrace.embeddings import SiliconFlowEmbedder
from papertrace.generation import DeepSeekGenerator
from papertrace.hybrid_search import HybridRetriever
from papertrace.internal_service import QuestionService, serve
from papertrace.keyword_search import KeywordRetriever
from papertrace.qa_cli import _load_chunks
from papertrace.retrieval_eval import corpus_fingerprint
from papertrace.vector_search import VectorRetriever, load_vector_snapshot


class CachedQueryEmbedder:
    """按原始查询文本及完整模型配置复用已有缓存，缺失时最多生成一次。"""

    def __init__(self, model, directory, factory):
        """注入延迟客户端工厂，构造时不读取密钥。"""
        self.model, self.directory, self.factory = model, Path(directory), factory
        self.hits = self.generated = 0

    def embed(self, texts):
        """使用既有内容缓存格式并保留可观测的命中计数。"""
        chunks = [{"chunk_id": f"query-{i}", "document_id": "service_queries",
                   "page_number": 1, "text": text} for i, text in enumerate(texts)]
        snapshot = embed_chunks(chunks, self.directory, model=self.model, batch_size=1,
                                embedder_factory=self.factory)
        self.hits += snapshot["cache_stats"]["cache_hit_texts"]
        self.generated += snapshot["cache_stats"]["embedded_texts"]
        return [row["embedding"] for row in snapshot["records"]]


class LazyGenerator:
    """仅在检索和问答预检成功后加载既有DeepSeek配置。"""

    def __init__(self, model, timeout):
        """保存服务端模型与超时设置，不读取.env。"""
        self.model, self.timeout = model, timeout

    def generate(self, messages, **kwargs):
        """仅调用一次现有官方适配器，不自动重试。"""
        return DeepSeekGenerator(model=self.model, timeout=self.timeout).generate(messages, **kwargs)


def create_service(chunk_paths, *, vectors=None, retrieval="hybrid", query_cache=None,
                   model="deepseek-flash", top_k=5, max_tokens=2048,
                   generation_timeout=60.0, embedding_timeout=20.0,
                   offline_query_cache=False):
    """联网前校验全部语料与向量身份，创建延迟读取密钥的服务。"""
    if retrieval not in {"keyword", "vector", "hybrid"}:
        raise ValueError("检索模式无效")
    if model not in {"deepseek-flash", "deepseek-v4-pro"}:
        raise ValueError("生成模型无效")
    for timeout in (generation_timeout, embedding_timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("组件超时必须为不超过60秒的有限正数")
    paths = [Path(path).resolve() for path in chunk_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("分块路径不能为空或重复")
    chunks = _load_chunks(paths)
    if not chunks:
        raise ValueError("不能启动空语料服务")
    if retrieval == "keyword" and (vectors is not None or query_cache is not None):
        raise ValueError("关键词模式不接受向量配置")
    if retrieval != "keyword" and (vectors is None or query_cache is None):
        raise ValueError("向量模式必须提供向量快照和查询缓存目录")
    records, snapshot, client = {}, None, None
    if vectors is not None:
        snapshot = load_vector_snapshot(vectors)
        records = {row["chunk"]["chunk_id"]: row for row in snapshot["records"]}
        for chunk in chunks:
            row = records.get(chunk["chunk_id"])
            if row is None or json.dumps(row["chunk"], sort_keys=True, allow_nan=False) != json.dumps(
                    chunk, sort_keys=True, allow_nan=False):
                raise ValueError("向量快照与输入分块身份或元数据不一致")

        def embedder_factory():
            """只在缓存未命中时构造客户端，缓存离线模式禁止创建客户端。"""
            if offline_query_cache:
                raise ValueError("查询缓存未命中，离线模式禁止新向量请求")
            return SiliconFlowEmbedder(model=snapshot["embedding_config"]["model"],
                                       timeout=embedding_timeout, max_retries=0)

        client = CachedQueryEmbedder(snapshot["embedding_config"]["model"], query_cache, embedder_factory)

    def retriever_factory(selected):
        """在指定论文内创建检索器，不先全库搜索再过滤。"""
        keyword = KeywordRetriever(selected)
        if retrieval == "keyword":
            return keyword
        subset = {"schema_version": 1, "embedding_config": copy.deepcopy(snapshot["embedding_config"]),
                  "chunk_count": len(selected), "chunks_sha256": corpus_fingerprint(selected),
                  "records": [records[chunk["chunk_id"]] for chunk in selected]}
        vector = VectorRetriever(subset, client)
        return vector if retrieval == "vector" else HybridRetriever(keyword, vector)

    service = QuestionService(chunks, retriever_factory, LazyGenerator(model, generation_timeout),
                              top_k=top_k, max_tokens=max_tokens)
    service.query_embedder = client
    return service


def main(argv=None):
    """解析启动参数并运行本机服务，退出时关闭套接字，不记录配置正文。"""
    parser = argparse.ArgumentParser(description="启动PaperTrace本机问答服务")
    parser.add_argument("--chunks", type=Path, nargs="+", required=True)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--query-cache", type=Path)
    parser.add_argument("--retrieval", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--model", choices=["deepseek-flash", "deepseek-v4-pro"], default="deepseek-flash")
    parser.add_argument("--generation-timeout", type=float, default=60)
    parser.add_argument("--embedding-timeout", type=float, default=20)
    parser.add_argument("--offline-query-cache", action="store_true")
    args = parser.parse_args(argv)
    server = None
    try:
        service = create_service(args.chunks, vectors=args.vectors, retrieval=args.retrieval,
                                 query_cache=args.query_cache, model=args.model, top_k=args.top_k,
                                 max_tokens=args.max_tokens, generation_timeout=args.generation_timeout,
                                 embedding_timeout=args.embedding_timeout,
                                 offline_query_cache=args.offline_query_cache)
        server = serve(service, args.host, args.port)
        print("PaperTrace内部服务已就绪（仅本机；不自动重试）", flush=True)
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        print("服务启动失败：检查参数、语料、快照或端口；未输出配置正文", flush=True)
        return 1
    finally:
        if server is not None:
            server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
