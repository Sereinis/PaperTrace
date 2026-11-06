"""验证字段覆盖诊断的保守降级和无语义伪造边界。"""

from papertrace.field_coverage import apply_field_coverage_diagnostic, diagnose_field_coverage


def qa(status, claims, evidence):
    """构造最小字段结果，不调用模型。"""
    return {"status": status, "claims": [{"text": text, "citations": []} for text in claims],
            "missing_info": [], "evidence": [{"text": text} for text in evidence]}


def test_training_candidate_categories_downgrade_answered():
    """候选同时包含训练类别而claims漏掉类别时只降级状态。"""
    result = apply_field_coverage_diagnostic(
        qa("answered", ["使用Adam优化器。"],
           ["We used Adam optimizer with warmup steps.", "We apply dropout and label smoothing."]),
        field="training_settings", experiment_scope="single experiment")
    assert result["status"] == "partial"
    assert result["coverage_diagnostic"]["downgraded"] is True
    assert "regularization" in result["coverage_diagnostic"]["candidate_categories_not_in_claims"]


def test_single_category_does_not_overrule_synthetic_answer():
    """只有一个通用术语类别时不把候选存在误判成必漏项。"""
    result = apply_field_coverage_diagnostic(
        qa("answered", ["合成结论。"], ["The batch size is 32."]),
        field="training_settings", experiment_scope="synthetic")
    assert result["status"] == "answered"
    assert result["coverage_diagnostic"]["downgraded"] is False


def test_other_language_direction_downgrades_baseline_answered():
    """对照claim出现明确其它English-to方向时保守降级。"""
    result = apply_field_coverage_diagnostic(
        qa("answered", ["English-to-French results are reported."], ["Table reports benchmark results."]),
        field="baselines", experiment_scope="English-to-German translation")
    assert result["status"] == "partial"
    assert "claim_contains_explicit_other_language_direction" in result["coverage_diagnostic"]["blocking_reasons"]


def test_chinese_other_language_direction_is_detected():
    """中文英法方向也应被识别为目标英德范围外。"""
    result = diagnose_field_coverage(
        "baselines", qa("partial", ["Table 2 同时报告英德和英法结果。"], ["benchmark results"]),
        experiment_scope="Transformer base on WMT 2014 English-to-German translation")
    assert "claim_contains_explicit_other_language_direction" in result["blocking_reasons"]


def test_partial_status_is_not_upgraded_or_rewritten():
    """已有partial不因诊断变成answered，且输入对象保持不变。"""
    original = qa("partial", ["English-to-French results."], ["benchmark results"])
    result = apply_field_coverage_diagnostic(original, field="baselines",
                                             experiment_scope="English-to-German translation")
    assert result["status"] == "partial" and original.get("coverage_diagnostic") is None
    assert result["coverage_diagnostic"]["status_after"] == "partial"
