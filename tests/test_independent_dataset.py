"""离线核对发布测试集、冻结投影与结果摘要，不依赖论文或API。"""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import pytest

from papertrace.retrieval_eval import corpus_fingerprint, score_ranking


ROOT = Path(__file__).resolve().parents[1]
DATA_BYTES = (ROOT / 'data/eval/retrieval-independent-v1.json').read_bytes()
DATA = json.loads(DATA_BYTES)
REPORT = json.loads((ROOT / 'data/eval/retrieval-independent-v1-results.json').read_bytes())


def test_frozen_identity_and_review_scope():
    """固定首次留出身份，明确人工确认仅涵盖参考答案而非穷尽标注。"""
    assert DATA['dataset_id'] == REPORT['dataset_id'] == 'papertrace-retrieval-independent-v1'
    assert DATA['split'] == REPORT['split'] == 'independent_test'
    assert DATA['schema_version'] == 1
    assert DATA['annotation_status'] == 'ai_assisted_answer_human_confirmed_chunk_gold_frozen'
    assert DATA['provenance']['retrieval_rankings_observed_at_freeze'] is False
    assert DATA['provenance']['results_observed_at_publication'] is True
    assert 'not independently human-validated' in DATA['human_review_scope']
    assert hashlib.sha256(DATA_BYTES).hexdigest() == REPORT['published_dataset_sha256']
    assert DATA['provenance']['source_frozen_dataset_sha256'] == REPORT['evaluated_frozen_dataset_sha256']
    assert REPORT['code_state'] == {'head': '5dc2f3fe52e20f1cab7ebda96b28d64dedeb088d', 'dirty': False}


def test_fixed_question_and_gold_projection():
    """固定问题、金标、页码和引文投影摘要，防止发布整理后悄然改标。"""
    projection = [{key: case[key] for key in
                   ('query_id', 'document_id', 'query', 'relevant_chunk_ids', 'evidence')}
                  for case in DATA['cases']]
    assert corpus_fingerprint(projection) == '26f8df53ae77ad1bd41fe10d11bfac8ec79c5135ba44268cfec7ecfaafa76a8f'
    assert len(DATA['cases']) == len({case['query_id'] for case in DATA['cases']}) == 12
    assert Counter(case['document_id'] for case in DATA['cases']) == {'resnet': 4, 'unet': 4, 'gan': 4}
    assert sum(len(case['relevant_chunk_ids']) for case in DATA['cases']) == 38


def test_sources_and_candidate_counts():
    """核对来源指纹和候选规模，不需要下载或分发论文。"""
    assert DATA['chunking'] == {'max_chars': 1200, 'overlap_chars': 100, 'scope': 'within_document'}
    assert sum(doc['pages'] for doc in DATA['documents']) == 29
    assert sum(doc['chunk_count'] for doc in DATA['documents']) == REPORT['chunk_count'] == 130
    assert REPORT['corpus_fingerprints'] == {doc['document_id']: doc['chunks_sha256'] for doc in DATA['documents']}
    for doc in DATA['documents']:
        assert doc['source_url'].startswith('https://arxiv.org/abs/')
        for key in ('pdf_sha256', 'chunks_sha256'):
            assert re.fullmatch(r'[0-9a-f]{64}', doc[key])


@pytest.mark.parametrize('case', DATA['cases'], ids=lambda case: case['query_id'])
def test_gold_evidence_structure(case):
    """一条相关分块对应一组定位摘录；结构校验不等于原文语义核验。"""
    gold = case['relevant_chunk_ids']
    assert len(gold) == len(set(gold)) == len(case['evidence'])
    assert set(gold) == {entry['chunk_id'] for entry in case['evidence']}
    assert case['reference_answer_zh'].strip() and case['query'].strip()
    assert case['human_review_status'] == 'reference_answer_confirmed_chunk_labels_ai_assisted'
    for entry in case['evidence']:
        match = re.fullmatch(re.escape(case['document_id']) + r'_p(\d{3})_c\d{3}', entry['chunk_id'])
        assert match and type(entry['page_number']) is int
        assert entry['page_number'] == int(match.group(1)) > 0
        assert entry['quote'].strip() and entry['coverage_labels']
        additional = entry.get('additional_quotes', [])
        assert isinstance(additional, list)
        quotes = [entry['quote']] + additional
        assert all(isinstance(quote, str) and quote.strip() for quote in quotes)
        assert len(quotes) == len(set(quotes))


@pytest.mark.parametrize('method', ['keyword', 'vector', 'hybrid'])
def test_published_rankings_reproduce_metrics(method):
    """从公开Top5身份独立重算逐问与宏平均指标，不请求向量服务。"""
    report = REPORT[method]
    assert REPORT['ks'] == report['ks'] == [1, 3, 5]
    assert report['query_count'] == 12 and report['chunk_count'] == 130
    assert [row['query_id'] for row in report['queries']] == [case['query_id'] for case in DATA['cases']]
    grouped = {}
    all_metrics = []
    for case, row in zip(DATA['cases'], report['queries']):
        assert row['document_id'] == case['document_id']
        assert row['relevant_chunk_ids'] == case['relevant_chunk_ids']
        results = row['results']
        assert len(results) == 5
        assert [item['rank'] for item in results] == [1, 2, 3, 4, 5]
        assert len({item['chunk_id'] for item in results}) == 5
        for item in results:
            assert set(item) == {'chunk_id', 'rank', 'score'}
            assert item['chunk_id'].startswith(case['document_id'] + '_p')
            assert math.isfinite(item['score'])
        metrics = score_ranking([item['chunk_id'] for item in results], case['relevant_chunk_ids'])
        assert row['metrics'] == pytest.approx(metrics)
        all_metrics.append(metrics)
        grouped.setdefault(case['document_id'], []).append(metrics)
    for rows, actual in [(all_metrics, report['metrics'])] + [
        (rows, report['by_document'][did]['metrics']) for did, rows in grouped.items()
    ]:
        expected = {key.replace('rr@', 'mrr@', 1): sum(row[key] for row in rows) / len(rows)
                    for key in rows[0]}
        assert actual == pytest.approx(expected)


def test_partial_evidence_does_not_imply_complete_answer():
    """固定公式题失败边界，不把部分证据命中描述为完整公式召回。"""
    for method in ('keyword', 'hybrid'):
        row = next(q for q in REPORT[method]['queries'] if q['query_id'] == 'gan-minimax-objective')
        assert row['metrics']['hit@5'] == 1
        assert 'gan_p003_c001' not in {item['chunk_id'] for item in row['results']}
