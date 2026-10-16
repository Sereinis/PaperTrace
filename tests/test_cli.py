import json

from papertrace import cli


def test_main_writes_chunk_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli,
        "extract_pdf_pages",
        lambda _path: {
            "file_name": "paper.pdf",
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "cleaned_text": "Abstract\nShort text.",
                }
            ],
            "error": None,
        },
    )
    output = tmp_path / "chunks.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "papertrace",
            "paper.pdf",
            "--chunks-output",
            str(output),
            "--max-chars",
            "20",
            "--overlap-chars",
            "0",
        ],
    )

    assert cli.main() == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["document_id"] == "paper"
    assert saved["chunks"][0]["section_key"] == "abstract"
    assert "chunk 结果已保存" in capsys.readouterr().out


def test_main_rejects_invalid_chunk_arguments(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli,
        "extract_pdf_pages",
        lambda _path: {
            "file_name": "paper.pdf",
            "total_pages": 1,
            "pages": [{"page_number": 1, "cleaned_text": "Text"}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["papertrace", "paper.pdf", "--chunks-output", str(tmp_path / "x.json"),
         "--max-chars", "10", "--overlap-chars", "10"],
    )

    assert cli.main() == 1
    assert "chunk 参数错误" in capsys.readouterr().err


def test_main_preserves_page_output_mode(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "extract_pdf_pages",
        lambda _path: {
            "file_name": "paper.pdf",
            "total_pages": 1,
            "pages": [{"page_number": 1, "cleaned_text": "Text"}],
            "error": None,
        },
    )
    monkeypatch.setattr("sys.argv", ["papertrace", "paper.pdf"])

    assert cli.main() == 0
    assert "===== Page 1 =====" in capsys.readouterr().out
