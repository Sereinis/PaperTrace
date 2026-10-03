import argparse
import sys
from pathlib import Path

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
    args = parser.parse_args()

    result = extract_pdf_pages(args.pdf_file)
    if result["error"] is not None:
        print(result["error"], file=sys.stderr)
        return 1

    print(f"文档总页数：{result['total_pages']}")
    if args.output:
        output_path = save_to_json(result, args.output)
        print(f"解析结果已保存：{output_path}")
    else:
        for page in result["pages"]:
            print(f"===== Page {page['page_number']} =====")
            print(page["text"] or "[当前页面未提取到文本]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
