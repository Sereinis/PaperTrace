"""使用 BM25 风格评分对文本块执行本地关键词检索。"""

import math
import re
from collections import Counter


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


class KeywordRetriever:
    """建立轻量倒排统计，并返回带页码和章节元数据的检索结果。"""

    def __init__(
        self,
        chunks: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """根据 chunk 列表构建 BM25 风格关键词索引。"""
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._chunks = [dict(chunk) for chunk in chunks]
        self._k1 = k1
        self._b = b
        self._term_counts: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._lengths: list[int] = []

        for chunk in self._chunks:
            counts = Counter(_tokenize(chunk.get("text", "")))
            self._term_counts.append(counts)
            self._lengths.append(sum(counts.values()))
            self._document_frequency.update(counts.keys())

        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """按 BM25 风格相关性返回前 top_k 个非零结果。"""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_terms = _tokenize(query)
        if not query_terms or not self._chunks:
            return []

        query_counts = Counter(query_terms)
        scored: list[tuple[float, int]] = []
        document_count = len(self._chunks)
        for index, counts in enumerate(self._term_counts):
            score = 0.0
            for term, query_frequency in query_counts.items():
                term_frequency = counts.get(term, 0)
                if term_frequency == 0:
                    continue
                document_frequency = self._document_frequency[term]
                idf = math.log(
                    1 + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = term_frequency + self._k1 * (
                    1 - self._b
                    + self._b * self._lengths[index] / self._average_length
                )
                score += idf * (
                    term_frequency * (self._k1 + 1) / denominator
                ) * query_frequency
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict] = []
        for rank, (score, index) in enumerate(scored[:top_k], start=1):
            result = dict(self._chunks[index])
            result["score"] = score
            result["rank"] = rank
            results.append(result)
        return results


def _tokenize(text: str) -> list[str]:
    """将英文关键词规范化为小写 token，忽略未匹配符号。"""
    return [match.group(0).casefold() for match in _TOKEN.finditer(text)]
