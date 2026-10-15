import re

from papertrace.section_detector import detect_sections


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_NON_IDENTIFIER_CHARACTER = re.compile(r"[^a-z0-9]+")


def chunk_page(
    text: str,
    page_number: int,
    document_id: str,
    max_chars: int = 1200,
    overlap_chars: int = 100,
) -> list[dict]:
    """Split cleaned text into deterministic, page-local chunks.

    Chunks never cross a page boundary. Natural boundaries are preferred,
    while long unbroken text is split at the configured character limit.
    Character offsets refer to the supplied page text after outer whitespace
    has been removed.
    """
    _validate_arguments(page_number, document_id, max_chars, overlap_chars)

    source = text.strip()
    if not source:
        return []

    normalized_document_id = _normalize_document_id(document_id)
    chunks: list[dict] = []
    start = 0

    while start < len(source):
        start = _skip_whitespace(source, start)
        if start >= len(source):
            break

        end = _find_chunk_end(source, start, max_chars)
        content_start, content_end = _trim_bounds(source, start, end)
        if content_start == content_end:
            break

        chunk_index = len(chunks) + 1
        chunk_text = source[content_start:content_end]
        chunks.append(
            {
                "chunk_id": (
                    f"{normalized_document_id}_p{page_number:03d}"
                    f"_c{chunk_index:03d}"
                ),
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "start_char": content_start,
                "end_char": content_end,
            }
        )

        if content_end >= len(source):
            break

        next_start = _find_overlap_start(
            source, content_start, content_end, overlap_chars
        )
        if next_start <= content_start:
            next_start = content_end
        start = next_start

    return chunks


def _validate_arguments(
    page_number: int,
    document_id: str,
    max_chars: int,
    overlap_chars: int,
) -> None:
    if page_number < 1:
        raise ValueError("page_number must be at least 1")
    if not document_id.strip():
        raise ValueError("document_id cannot be empty")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")


def _normalize_document_id(document_id: str) -> str:
    normalized = _NON_IDENTIFIER_CHARACTER.sub(
        "_", document_id.strip().lower()
    ).strip("_")
    if not normalized:
        raise ValueError("document_id must contain a letter or number")
    return normalized


def _find_chunk_end(text: str, start: int, max_chars: int) -> int:
    """在长度上限内优先选择真实存在的自然边界，否则硬切。"""
    limit = min(start + max_chars, len(text))
    if limit == len(text):
        return limit

    minimum_boundary = start + max(1, max_chars // 2)
    window = text[start:limit]

    paragraph_end = window.rfind("\n\n")
    if paragraph_end >= 0 and start + paragraph_end + 2 >= minimum_boundary:
        return start + paragraph_end + 2

    sentence_ends = [match.end() for match in _SENTENCE_BOUNDARY.finditer(window)]
    if sentence_ends and start + sentence_ends[-1] >= minimum_boundary:
        return start + sentence_ends[-1]

    newline_end = window.rfind("\n")
    if newline_end >= 0 and start + newline_end + 1 >= minimum_boundary:
        return start + newline_end + 1

    whitespace_end = max(window.rfind(" "), window.rfind("\t"))
    if whitespace_end >= 0 and start + whitespace_end + 1 >= minimum_boundary:
        return start + whitespace_end + 1

    return limit


def _find_overlap_start(
    text: str,
    chunk_start: int,
    chunk_end: int,
    overlap_chars: int,
) -> int:
    if overlap_chars == 0:
        return chunk_end

    desired = max(chunk_start, chunk_end - overlap_chars)
    if desired == chunk_start:
        return chunk_end

    # Start at the next word boundary so overlap never exceeds the requested
    # size and a new chunk does not begin in the middle of a word.
    while desired < chunk_end and not text[desired].isspace():
        desired += 1
    return _skip_whitespace(text, desired)


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_document(
    pages: list[dict],
    document_id: str,
    max_chars: int = 1200,
    overlap_chars: int = 100,
) -> list[dict]:
    """按章节与页面边界分块，接收解析器返回的 pages 列表。

    每页必须包含 page_number 和 cleaned_text，页码须严格递增。
    标题归属新章节；章节内可重叠，但不跨章节或跨页重叠。
    页首沿用前页最后的标题，首次标题之前标为 unknown。
    start_char/end_char 直接对应原 cleaned_text 的左闭右开字符区间，
    与 chunk_page 去除外层空白后的坐标不同。不修改传入数据。
    规则误检或漏检会影响章节归属；不推断父章节或合并跨行标题。
    """
    _validate_arguments(1, document_id, max_chars, overlap_chars)
    normalized_id = _normalize_document_id(document_id)
    chunks: list[dict] = []
    section = {"section_key": "unknown", "title": None, "number": None}
    previous_page = 0

    for page in pages:
        page_number = page["page_number"]
        if page_number <= previous_page:
            raise ValueError("page_number must be positive and strictly increasing")
        previous_page = page_number
        text = page["cleaned_text"]
        headings = detect_sections(text)
        start = 0
        page_index = 0

        # 每次遇到标题，先处理此前的区间，再切换到新章节。
        for heading in [*headings, None]:
            end = heading["start_char"] if heading is not None else len(text)
            segment = text[start:end]
            leading_space = len(segment) - len(segment.lstrip())
            for chunk in chunk_page(
                segment, page_number, document_id, max_chars, overlap_chars
            ):
                page_index += 1
                chunk.update(
                    document_id=document_id,
                    chunk_id=f"{normalized_id}_p{page_number:03d}_c{page_index:03d}",
                    chunk_index=page_index,
                    start_char=start + leading_space + chunk["start_char"],
                    end_char=start + leading_space + chunk["end_char"],
                    section_key=section["section_key"],
                    section_title=section["title"],
                    section_number=section["number"],
                )
                chunks.append(chunk)
            if heading is not None:
                start = end
                section = heading

    return chunks
