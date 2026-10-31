"""读取标准向量快照与固定开发集，输出两种检索方法的对比报告。"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from papertrace.embedding_cache import atomic_write_json
from papertrace.embeddings import EmbeddingError
from papertrace.vector_eval import compare_retrieval
from papertrace.vector_search import load_vector_snapshot


def main(argv: list[str] | None = None) -> int:
    """校验路径后比较检索，拒绝覆盖已有文件，失败时不生成部分报告。"""
    parser = argparse.ArgumentParser(description="固定开发集的关键词与向量检索对比")
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--query-cache-dir", type=Path, default=Path("artifacts/embedding-cache"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args(argv)
    try:
        sources = {args.vectors.resolve(), args.dataset.resolve()}
        output, cache = args.output.resolve(), args.query_cache_dir.resolve()
        if (len(sources) != 2 or output in sources or output.exists()
                or output.is_relative_to(cache) or any(path.is_relative_to(cache) for path in sources)
                or (cache.exists() and not cache.is_dir())):
            raise ValueError("输入、输出或缓存目录冲突")
        # 输出路径不可用时也要在潜在计费请求前失败。
        parent = output.parent
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError("输出父路径不是目录")
        vector_bytes, dataset_bytes = args.vectors.read_bytes(), args.dataset.read_bytes()
        snapshot = load_vector_snapshot(args.vectors)
        report = compare_retrieval(snapshot, json.loads(dataset_bytes), cache, ks=args.ks)
        report.update({
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "vector_snapshot_sha256": hashlib.sha256(vector_bytes).hexdigest(),
            "environment": {"python": platform.python_version()},
        })
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                   text=True, check=True).stdout.strip()
            report["code_state"] = {"head": revision, "dirty": bool(dirty)}
        except (OSError, subprocess.SubprocessError):
            report["code_state"] = {"head": None, "dirty": None}
        atomic_write_json(report, output)
    except EmbeddingError as error:
        print(f"对比失败：{error}；已生成的查询缓存可复用", file=sys.stderr)
        return 1
    except (OSError, ValueError, UnicodeError, RecursionError):
        print("对比失败：请检查快照、标注、配置、缓存及输出路径；不会自动重建缓存",
              file=sys.stderr)
        return 1
    print(f"对比完成：{report['query_count']} 问、{report['chunk_count']} 个分块，报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
