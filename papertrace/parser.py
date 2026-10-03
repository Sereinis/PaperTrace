from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_pdf_pages(pdf_path: str) -> dict:
    """
    读取PDF
    返回字典: {"total_pages":总页数, "pages": [每页文本], "error":错误信息}
    """
    result = {
        "total_pages": 0,
        "pages": [],
        "error": None
    }
    try:
        reader = PdfReader(pdf_path)
        # 获取总页数
        total = len(reader.pages)
        result["total_pages"] = total

        for page in reader.pages:
            text = page.extract_text()
            # 页面没有文本，填充提示字符串
            if text is None or text.strip() == "":
                text = "[当前页面未提取到文本]"
            result["pages"].append(text)

    except FileNotFoundError:
        result["error"] = f"错误：文件不存在 -> {pdf_path}"
    except PdfReadError:
        result["error"] = f"错误：该文件不是合法PDF -> {pdf_path}"
    except Exception as e:
        result["error"] = f"未知异常：{str(e)}"

    return result
