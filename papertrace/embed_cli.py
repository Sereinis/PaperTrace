"""读取现有 chunk JSON，生成带来源信息的向量快照与本地缓存。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from papertrace.embedding_cache import atomic_write_json, embed_chunks
from papertrace.embeddings import DEFAULT_MODEL, EmbeddingError


def main(argv: list[str] | None = None) -> int:
    """执行分块向量化；拒绝覆盖已有输出，失败时不写完整结果快照。"""
    parser = argparse.ArgumentParser(description="向量化论文分块并复用本地缓存")
    parser.add_argument("--chunks", type=Path, nargs="+", required=True,
                        help="现有 PDF CLI 生成的 chunk JSON 文件")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/embedding-cache"))
    parser.add_argument("--output", type=Path, required=True, help="新的向量快照 JSON")
    parser.add_argument("--model", choices=[DEFAULT_MODEL, "Pro/BAAI/bge-m3"], default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        sources = {path.resolve() for path in args.chunks}
        output = args.output.resolve()
        cache = args.cache_dir.resolve()
        if len(sources) != len(args.chunks):
            raise ValueError("不能重复传入同一个输入文件")
        if output in sources or output.exists():
            raise ValueError("输出不能覆盖输入或已有文件，请使用新的输出路径")
        if output.is_relative_to(cache) or any(path.is_relative_to(cache) for path in sources):
            raise ValueError("输入与输出文件必须位于缓存目录之外")
        if cache.exists() and not cache.is_dir():
            raise ValueError("cache-dir 必须是目录")
        chunks, manifests = [], []
        for path in args.chunks:
            payload_bytes = path.read_bytes()
            payload = json.loads(payload_bytes)
            if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
                raise ValueError("输入必须是包含 chunks 列表的 JSON 对象")
            if payload.get("error") is not None:
                raise ValueError("输入 chunk JSON 含解析错误")
            chunks.extend(payload["chunks"])
            manifests.append({
                "file_name": path.name,
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "metadata": {key: value for key, value in payload.items() if key != "chunks"},
            })
        # 包括源文件元数据在内的全部校验必须先于潜在计费请求。
        json.dumps(manifests, ensure_ascii=False, allow_nan=False).encode("utf-8")
        result = embed_chunks(chunks, cache, model=args.model, batch_size=args.batch_size)
        result["source_files"] = manifests
        atomic_write_json(result, output)
    except EmbeddingError as error:
        print(f"向量化失败：{error}；已完成的缓存可供下次复用", file=sys.stderr)
        return 1
    except (OSError, ValueError, RecursionError):
        # 不回显畸形 JSON 的片段或任意输入字段，避免泄露本地数据。
        print("向量化失败：请检查输入、配置、缓存完整性及输出路径；已有缓存不会自动重建",
              file=sys.stderr)
        return 1
    stats = result["cache_stats"]
    print(f"完成：{result['chunk_count']} 个分块；复用 {stats['cache_hit_texts']} 个文本向量，"
          f"新生成 {stats['embedded_texts']} 个文本向量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
