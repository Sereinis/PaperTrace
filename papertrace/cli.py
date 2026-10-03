import argparse
import sys

from papertrace.parser import extract_pdf_pages


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", help="论文pdf文件路径")
    args = parser.parse_args()

    res = extract_pdf_pages(args.pdf_file)
    # 如果有错误，直接打印退出
    if res["error"] is not None:
        print(res["error"])
        return 1

    print(f"文档总页数：{res['total_pages']}")
    pages = res["pages"]
    for idx, text in enumerate(pages, start=1):
        print(f"===== Page {idx} =====")
        print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
