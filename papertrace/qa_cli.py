"""提供单问题证据问答的命令行入口。"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from papertrace.embedding_cache import validate_chunks
from papertrace.embeddings import SiliconFlowEmbedder
from papertrace.generation import DeepSeekGenerator
from papertrace.hybrid_search import HybridRetriever
from papertrace.keyword_search import KeywordRetriever
from papertrace.qa import QAError, answer_question
from papertrace.retrieval_eval import corpus_fingerprint
from papertrace.vector_search import VectorRetriever, load_vector_snapshot


def _load_chunks(paths: list[Path]) -> list[dict]:
    """读取一个或多个 chunk JSON，并拒绝重复分块与解析错误。"""
    chunks: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
            raise ValueError("chunk 输入必须是包含 chunks 列表的 JSON 对象")
        if payload.get("error") is not None:
            raise ValueError("chunk 输入包含解析错误")
        for chunk in payload["chunks"]:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str):
                raise ValueError("chunk 输入包含无效分块")
            if chunk["chunk_id"] in seen:
                raise ValueError("chunk 输入包含重复 chunk_id")
            seen.add(chunk["chunk_id"])
            chunks.append(chunk)
    validate_chunks(chunks)
    return chunks


def _make_retriever_factory(mode: str, chunks: list[dict], snapshot: dict | None):
    """为当前论文建立隔离的关键词、向量或混合检索器工厂。"""
    if mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError("retrieval mode 无效")
    records = None
    if mode in {"vector", "hybrid"}:
        if snapshot is None:
            raise ValueError("vector/hybrid 模式必须提供 --vectors")
        records = {row["chunk"]["chunk_id"]: row for row in snapshot["records"]}
        for chunk in chunks:
            row = records.get(chunk["chunk_id"])
            if row is None or json.dumps(row["chunk"], sort_keys=True, allow_nan=False) != json.dumps(
                chunk, sort_keys=True, allow_nan=False,
            ):
                raise ValueError("向量快照与输入分块身份或元数据不一致")

    def factory(selected: list[dict]):
        """只为 answer_question 当前论文候选建立检索器。"""
        keyword = KeywordRetriever(selected)
        if mode == "keyword":
            return keyword
        subset = {
            "schema_version": 1,
            "embedding_config": snapshot["embedding_config"],
            "chunk_count": len(selected),
            "chunks_sha256": corpus_fingerprint(selected),
            "records": [records[chunk["chunk_id"]] for chunk in selected],
        }
        embedder = SiliconFlowEmbedder(model=snapshot["embedding_config"]["model"], max_retries=0)
        vector = VectorRetriever(subset, embedder)
        return vector if mode == "vector" else HybridRetriever(keyword, vector)

    return factory


class _LazyGenerator:
    """仅在存在候选且通过问答预检后读取生成密钥。"""

    def __init__(self, model: str) -> None:
        """保存模型身份，不创建客户端或读取配置。"""
        self.model = model

    def generate(self, messages, **kwargs):
        """创建官方适配器并执行一次请求，不增加重试。"""
        return DeepSeekGenerator(model=self.model).generate(messages, **kwargs)


def _terminal_text(value: str) -> str:
    """将不可信文本中的终端控制字符转为可见转义，保留换行和制表符。"""
    return "".join(
        f"\\u{ord(char):04x}" if (ord(char) < 32 or 127 <= ord(char) <= 159)
        and char not in "\n\t" else char for char in value
    )


def _render_text(result: dict) -> str:
    """将问答结果渲染为适合终端阅读的中文文本。"""
    lines = [f"论文：{result['document_id']}", f"问题：{result['query']}",
             f"状态：{result['status']}（候选充分性判断，非质量验收）",
             "引用定位不等于语义支持；semantic_support=not_verified。"]
    for index, claim in enumerate(result.get("claims", []), start=1):
        lines.append(f"\n结论 {index}：{claim['text']}")
        for citation in claim.get("citations", []):
            lines.append(
                f"  证据：{citation['document_id']} / 第{citation['page_number']}页"
                f" / {citation['chunk_id']} / {citation['segment_id']}"
                f" / 块内字符[{citation['quote_start']}, {citation['quote_end']})"
                f"\n  {citation['quote']}"
            )
    missing = result.get("missing_info", [])
    if missing:
        lines.append("\n缺失信息：")
        lines.extend(f"- {item}" for item in missing)
    generation = result.get("generation")
    if isinstance(generation, dict) and generation.get("usage") is not None:
        usage = generation["usage"]
        lines.append(
            f"\n用量：输入 {usage.get('prompt_tokens', '?')}，"
            f"输出 {usage.get('completion_tokens', '?')}，"
            f"总计 {usage.get('total_tokens', '?')} tokens"
        )
    elif generation is not None:
        lines.append("\n生成用量未知；不估算 token 或费用。")
    else:
        lines.append("\n未调用生成器。")
    return _terminal_text("\n".join(lines)) + "\n"


def _prepare_output(output: Path, sources: list[Path]) -> Path:
    """计费前检查路径并建立同目录临时文件，最终以无覆盖硬链接发布。"""
    if output.exists() or output.is_symlink() or output.resolve() in sources:
        raise ValueError("输出不能覆盖输入、已有文件或符号链接")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".papertrace-qa-", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    return Path(name)


def _publish_output(temporary: Path, output: Path, rendered: str) -> None:
    """完整落盘后无覆盖发布；若目标已被并发创建则明确失败。"""
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        file.write(rendered)
        file.flush()
        os.fsync(file.fileno())
    os.link(temporary, output)


def main(argv: list[str] | None = None) -> int:
    """解析单问题参数，执行一次问答并输出完整结果或人类可读摘要。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="对指定论文提出一个带证据的问题")
    parser.add_argument("--chunks", type=Path, nargs="+", required=True,
                        help="一个或多个 PDF CLI 生成的 chunk JSON")
    parser.add_argument("--document-id", required=True, help="限定检索范围的论文标识")
    parser.add_argument("--query", required=True, help="要回答的问题")
    parser.add_argument("--vectors", type=Path, help="标准向量快照；vector/hybrid 模式必需")
    parser.add_argument("--retrieval", choices=["keyword", "vector", "hybrid"],
                        default="keyword", help="检索方式，默认 keyword")
    parser.add_argument("--top-k", type=int, default=5, help="候选数量，默认5")
    parser.add_argument("--max-tokens", type=int, default=2048, help="生成上限，默认2048")
    parser.add_argument("--model", choices=["deepseek-flash", "deepseek-v4-pro"],
                        default="deepseek-flash", help="DeepSeek 模型")
    parser.add_argument("--preserve-source-candidates", action="store_true",
                        help="混合检索保留两路各Top-K并集供生成器选择")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="输出格式，默认 text")
    parser.add_argument("--output", type=Path, help="可选输出文件；不能覆盖已有文件")
    args = parser.parse_args(argv)

    temporary = None
    try:
        paths = [path.resolve() for path in args.chunks]
        if len(set(paths)) != len(paths):
            raise ValueError("不能重复传入同一个 chunk 文件")
        if args.retrieval == "keyword" and args.vectors is not None:
            raise ValueError("keyword 模式不接受 --vectors")
        if args.preserve_source_candidates and args.retrieval != "hybrid":
            raise ValueError("候选并集选项只适用于 hybrid 模式")
        if args.vectors is not None and not args.vectors.exists():
            raise ValueError("向量快照文件不存在")
        chunks = _load_chunks(paths)
        snapshot = load_vector_snapshot(args.vectors) if args.vectors is not None else None
        if args.retrieval in {"vector", "hybrid"} and snapshot is None:
            raise ValueError("vector/hybrid 模式必须提供 --vectors")
        factory = _make_retriever_factory(args.retrieval, chunks, snapshot)
        if args.output is not None:
            sources = paths + ([args.vectors.resolve()] if args.vectors is not None else [])
            temporary = _prepare_output(args.output, sources)
        result = answer_question(
            args.query, args.document_id, chunks, retriever_factory=factory,
            generator=_LazyGenerator(args.model), top_k=args.top_k,
            max_tokens=args.max_tokens,
            preserve_source_candidates=args.preserve_source_candidates,
        )
        rendered = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                    if args.format == "json" else _render_text(result))
        if args.output is not None:
            _publish_output(temporary, args.output, rendered)
            print(_terminal_text(f"问答结果已保存：{args.output}"))
        else:
            print(_terminal_text(rendered), end="")
        return 0
    except QAError as error:
        print("问答失败：" + json.dumps(error.diagnostics, ensure_ascii=False), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        print("问答失败：请检查输入、向量快照、配置和输出路径；不会自动重试，"
              "已发送的请求可能已计费",
              file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
