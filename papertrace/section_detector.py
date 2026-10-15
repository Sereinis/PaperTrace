"""使用保守规则识别英文论文的常见章节标题。"""

import re


_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<number>(?:[1-9]\d?(?:\.\d+)*|[A-Z](?:\.\d+)*))"
    r"\.?\s+(?P<title>.+?)\s*$"
)

# 无版面信息时，独立的 Model/Method 更可能是表头或图例，宁可漏检。
_AMBIGUOUS_UNNUMBERED = {"model", "models", "method", "methods"}

_KNOWN_SECTIONS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "related_work",
    "related works": "related_work",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "model": "method",
    "models": "method",
    "experimental setup": "experiments",
    "experiments": "experiments",
    "experiment": "experiments",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations": "limitations",
    "limitation": "limitations",
    "appendix": "appendix",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
}


def detect_sections(text: str) -> list[dict]:
    """返回文本中的章节标题、分类、编号、行号及字符位置。

    字符位置直接对应传入文本，不会因标题分类时的空格修复而改变。
    无编号标题只接受明确的常见标题；编号标题额外检查大小写及符号。
    本函数只生成候选标题，不推断跨页章节归属，也不能完全排除误检。
    """
    if not text:
        return []

    sections: list[dict] = []
    offset = 0
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        heading = _parse_heading(line)
        if heading is not None:
            start = offset + (len(line) - len(line.lstrip()))
            sections.append(
                {
                    "section_key": heading["section_key"],
                    "title": heading["title"],
                    "number": heading["number"],
                    "line_number": line_number,
                    "start_char": start,
                    "end_char": offset + len(line.rstrip()),
                }
            )
        offset += len(raw_line)
    return sections


def _parse_heading(line: str) -> dict | None:
    candidate = " ".join(line.strip().split())
    if not candidate:
        return None

    without_colon = candidate.rstrip(":").strip()
    normalized = _normalize_title(without_colon).casefold()
    key = _KNOWN_SECTIONS.get(normalized)
    if normalized in _AMBIGUOUS_UNNUMBERED:
        return None
    if key is not None:
        return {"section_key": key, "title": without_colon, "number": None}

    match = _NUMBERED_HEADING.match(candidate)
    if match is None:
        return None

    title = match.group("title").rstrip(":").strip()
    if not _looks_like_heading(title):
        return None
    number = match.group("number")
    # 单字母编号与公式变量容易混淆，要求附录根标题提供更多文字证据。
    if number.isalpha() and len(title.split()) < 2:
        if _normalize_title(title).casefold() not in _KNOWN_SECTIONS:
            return None
    key = _classify_title(title)
    return {"section_key": key, "title": title, "number": number}


def _looks_like_heading(title: str) -> bool:
    """排除正文残句、引用、数学表达式及过长的标题候选。"""
    words = title.split()
    if not 1 <= len(words) <= 12 or len(title) > 100:
        return False
    if not title[0].isascii() or not title[0].isupper():
        return False
    if title.endswith((".", "!", "-")):
        return False
    if any(symbol in title for symbol in "=∈∥⊤×+;,():"):
        return False
    if all(len(word) == 1 for word in words):
        return False
    # 允许全大写的问句标题，拒绝以问答列表形式出现的正文。
    if "?" in title and not title.isupper():
        return False
    return True


def _normalize_title(title: str) -> str:
    """仅为分类合并小型大写造成的断开词，不改写返回的标题或偏移。"""
    return re.sub(r"\b([A-Z]) +([A-Z]{2,})\b", r"\1\2", title)


def _classify_title(title: str) -> str:
    """将可识别的标题映射到稳定类别，其余保留为 other。"""
    normalized = _normalize_title(title).casefold()
    for phrase, key in _KNOWN_SECTIONS.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", normalized):
            return key
    if "ablation" in normalized:
        return "ablation"
    if "training" in normalized or "architecture" in normalized:
        return "method"
    return "other"
