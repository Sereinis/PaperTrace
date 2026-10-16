import argparse
import sys
from pathlib import Path

from papertrace.chunker import chunk_document
from papertrace.parser import extract_pdf_pages, save_to_json


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="按页解析英文论文 PDF")
    parser.add_argument("pdf_file", help="论文 PDF 文件路径")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON 输出路径；省略时只在终端展示解析结果",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        help="章节感知 chunk JSON 输出路径；省略时不生成 chunks",
    )
    parser.add_argument(
        "--document-id",
        help="chunk 使用的文档标识；省略时使用 PDF 文件名",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="单个 chunk 的最大字符数，默认 1200",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=100,
        help="相邻 chunk 的目标重叠字符数，默认 100",
    )
    args = parser.parse_args()

    result = extract_pdf_pages(args.pdf_file)
    if result["error"] is not None:
        print(result["error"], file=sys.stderr)
        return 1

    print(f"文档总页数：{result['total_pages']}")
    if args.output:
        output_path = save_to_json(result, args.output)
        print(f"解析结果已保存：{output_path}")
    if args.chunks_output:
        document_id = args.document_id or Path(args.pdf_file).stem
        try:
            chunks = chunk_document(
                result["pages"],
                document_id=document_id,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )
        except ValueError as exc:
            print(f"chunk 参数错误：{exc}", file=sys.stderr)
            return 1
        chunk_result = {
            "file_name": result["file_name"],
            "total_pages": result["total_pages"],
            "document_id": document_id,
            "max_chars": args.max_chars,
            "overlap_chars": args.overlap_chars,
            "chunks": chunks,
            "error": None,
        }
        chunks_path = save_to_json(chunk_result, args.chunks_output)
        print(f"chunk 结果已保存：{chunks_path}")
    elif not args.output:
        for page in result["pages"]:
            print(f"===== Page {page['page_number']} =====")
            print(page["cleaned_text"] or "[当前页面未提取到文本]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
