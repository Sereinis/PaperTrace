import re


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
    limit = min(start + max_chars, len(text))
    if limit == len(text):
        return limit

    minimum_boundary = start + max(1, max_chars // 2)
    window = text[start:limit]

    paragraph_end = window.rfind("\n\n")
    if start + paragraph_end + 2 >= minimum_boundary:
        return start + paragraph_end + 2

    sentence_ends = [match.end() for match in _SENTENCE_BOUNDARY.finditer(window)]
    if sentence_ends and start + sentence_ends[-1] >= minimum_boundary:
        return start + sentence_ends[-1]

    newline_end = window.rfind("\n")
    if start + newline_end + 1 >= minimum_boundary:
        return start + newline_end + 1

    whitespace_end = max(window.rfind(" "), window.rfind("\t"))
    if start + whitespace_end + 1 >= minimum_boundary:
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
