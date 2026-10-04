import json

from papertrace import parser


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, _path):
        self.pages = [FakePage(" first page "), FakePage(None)]


def test_extract_pdf_pages_keeps_page_metadata(monkeypatch):
    monkeypatch.setattr(parser, "PdfReader", FakeReader)

    result = parser.extract_pdf_pages("example.pdf")

    assert result["file_name"] == "example.pdf"
    assert result["total_pages"] == 2
    assert result["error"] is None
    assert result["pages"][0] == {
        "page_number": 1,
        "raw_text": " first page ",
        "cleaned_text": "first page",
        "raw_char_count": 12,
        "cleaned_char_count": 10,
        "has_text": True,
    }
    assert result["pages"][1]["page_number"] == 2
    assert result["pages"][1]["raw_text"] == ""
    assert result["pages"][1]["cleaned_text"] == ""
    assert result["pages"][1]["raw_char_count"] == 0
    assert result["pages"][1]["cleaned_char_count"] == 0
    assert result["pages"][1]["has_text"] is False


def test_extract_pdf_pages_applies_cleaner(monkeypatch):
    monkeypatch.setattr(parser, "PdfReader", FakeReader)
    calls = []

    def fake_cleaner(text, page_number):
        calls.append((text, page_number))
        return f"cleaned-{page_number}"

    monkeypatch.setattr(parser, "clean_page_text", fake_cleaner)

    result = parser.extract_pdf_pages("example.pdf")

    assert calls == [(" first page ", 1), ("", 2)]
    assert result["pages"][0]["cleaned_text"] == "cleaned-1"
    assert result["pages"][1]["cleaned_text"] == "cleaned-2"


def test_extract_pdf_pages_reports_missing_file(monkeypatch):
    def raise_missing(_path):
        raise FileNotFoundError

    monkeypatch.setattr(parser, "PdfReader", raise_missing)
    result = parser.extract_pdf_pages("missing.pdf")

    assert result["total_pages"] == 0
    assert "文件不存在" in result["error"]


def test_extract_pdf_pages_reports_invalid_pdf(monkeypatch):
    def raise_invalid(_path):
        raise parser.PdfReadError("invalid")

    monkeypatch.setattr(parser, "PdfReader", raise_invalid)
    result = parser.extract_pdf_pages("invalid.pdf")

    assert "不是合法 PDF" in result["error"]


def test_save_to_json_creates_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "paper.json"
    data = {"file_name": "论文.pdf", "pages": []}

    saved_path = parser.save_to_json(data, output_path)

    assert saved_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == data


def test_extract_pdf_pages_preserves_unicode_error_message(monkeypatch):
    def raise_missing(_path):
        raise FileNotFoundError

    monkeypatch.setattr(parser, "PdfReader", raise_missing)
    result = parser.extract_pdf_pages("中文论文.pdf")

    assert result["error"] == "错误：文件不存在 -> 中文论文.pdf"
