import json
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from papertrace.cleaner import clean_page_text


def extract_pdf_pages(pdf_path: str | Path) -> dict:
    """读取 PDF，返回包含文件名、总页数和页级文本的字典。"""
    path = Path(pdf_path)
    result = {
        "file_name": path.name,
        "total_pages": 0,
        "pages": [],
        "error": None,
    }

    try:
        reader = PdfReader(path)
        result["total_pages"] = len(reader.pages)

        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_text = clean_page_text(raw_text, page_number)
            result["pages"].append(
                {
                    "page_number": page_number,
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_text,
                    "raw_char_count": len(raw_text),
                    "cleaned_char_count": len(cleaned_text),
                    "has_text": bool(cleaned_text),
                }
            )
    except FileNotFoundError:
        result["error"] = f"错误：文件不存在 -> {path}"
    except PdfReadError:
        result["error"] = f"错误：该文件不是合法 PDF -> {path}"
    except Exception as exc:
        result["error"] = f"未知异常：{exc}"

    return result


def save_to_json(data: dict, output_json_path: str | Path) -> Path:
    """将解析结果以 UTF-8 编码保存为 JSON，并返回输出路径。"""
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return output_path
