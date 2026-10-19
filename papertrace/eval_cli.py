"""读取固定 chunk 快照和问题集，输出可复现的关键词检索评测 JSON。"""

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from papertrace.parser import save_to_json
from papertrace.retrieval_eval import corpus_fingerprint, evaluate_retrieval


def main(argv: list[str] | None = None) -> int:
    """运行独立评测命令；输入、标注或写入错误返回非零退出码。"""
    parser = argparse.ArgumentParser(description="评测指定论文内的关键词检索")
    parser.add_argument("--chunks", type=Path, nargs="+", required=True,
                        help="由现有 PDF CLI 生成的 chunk JSON，可传入多个文件")
    parser.add_argument("--dataset", type=Path, required=True, help="评测问题集 JSON")
    parser.add_argument("--output", type=Path, required=True, help="评测报告 JSON")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args(argv)
    try:
        inputs = [args.dataset, *args.chunks]
        if args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("output must not overwrite an input file")
        dataset_bytes = args.dataset.read_bytes()
        dataset = json.loads(dataset_bytes)
        if (not isinstance(dataset, dict)
                or type(dataset.get("schema_version")) is not int
                or dataset["schema_version"] != 1):
            raise ValueError("dataset schema_version must be 1")
        if not isinstance(dataset.get("dataset_id"), str) or not dataset["dataset_id"].strip():
            raise ValueError("dataset_id must be a non-empty string")
        documents = dataset.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError("dataset documents must be a non-empty list")
        chunks = []
        for path in args.chunks:
            payload = json.loads(path.read_bytes())
            if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
                raise ValueError(f"invalid chunk JSON: {path}")
            if payload.get("error") is not None:
                raise ValueError(f"chunk JSON contains an error: {path}")
            chunks.extend(payload["chunks"])
        # 先由评测器验证字段，避免畸形输入在指纹检查时产生不明确异常。
        report = evaluate_retrieval(chunks, dataset.get("cases"), args.ks)
        grouped = defaultdict(list)
        for chunk in chunks:
            grouped[chunk["document_id"]].append(chunk)
        expected = {}
        for document in documents:
            if not isinstance(document, dict):
                raise ValueError("each document manifest must be an object")
            did, fingerprint = document.get("document_id"), document.get("chunks_sha256")
            if not isinstance(did, str) or not did or did in expected:
                raise ValueError("manifest document_id must be unique and non-empty")
            if (not isinstance(fingerprint, str)
                    or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None):
                raise ValueError("manifest chunks_sha256 must be a SHA-256 digest")
            expected[did] = fingerprint
        if set(expected) != set(grouped):
            raise ValueError("document manifest does not match input documents")
        actual = {did: corpus_fingerprint(items) for did, items in grouped.items()}
        if actual != expected:
            raise ValueError("chunk fingerprint mismatch; regenerate or recheck annotations")
        report.update({
            "schema_version": 1, "dataset_id": dataset["dataset_id"],
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "annotation_status": dataset.get("annotation_status", "unspecified"),
            "corpus_fingerprints": actual,
            "environment": {"python": platform.python_version(), "pypdf": version("pypdf")},
        })
        save_to_json(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"评测失败：{exc}", file=sys.stderr)
        return 1
    print(f"评测完成：{report['query_count']} 个问题，结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
