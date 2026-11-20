"""验证字段覆盖诊断的保守降级和无语义伪造边界。"""

import copy
import pytest

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


@pytest.mark.parametrize("candidate,claim,facet", [
    ("We use dropout and label smoothing.", "使用dropout。", "label_smoothing"),
    ("Adam optimizer and warmup learning rate schedule.", "使用Adam优化器。", "learning_rate_schedule"),
    ("Adam optimizer; batches group similar sequence lengths.", "使用Adam优化器。", "batching_by_length"),
    ("Checkpoint averaging and beam decoding, terminate early.", "使用checkpoint平均与beam解码。", "termination"),
    ("Checkpoint averaging; parameters chosen on development set.", "使用checkpoint平均。", "selection"),
])
def test_individual_facets_cannot_hide_in_same_broad_category(candidate, claim, facet):
    """同类中只覆盖一项仍需保守提醒，不把只在引文出现算作已展示。"""
    field = "evaluation_metrics" if facet in {"selection", "termination"} else "training_settings"
    original = qa("answered", [claim], [candidate])
    original["claims"][0]["citations"] = [{"quote": candidate}]
    before = copy.deepcopy(original)
    result = apply_field_coverage_diagnostic(original, field=field, experiment_scope="synthetic")
    assert result["status"] == "partial"
    assert facet in result["coverage_diagnostic"]["candidate_facets_not_in_claims"]
    assert any("细项" in gap for gap in result["missing_info"])
    assert result["claims"] == before["claims"] and original == before


@pytest.mark.parametrize("field,claims,evidence", [
    ("training_settings", "使用Adam优化器与学习率预热，并使用dropout和标签平滑。",
     "Adam optimizer, warmup learning rate, dropout and label smoothing."),
    ("evaluation_metrics", "BLEU用于测试集，checkpoint平均，束搜索，开发集选参，提前终止。",
     "BLEU test set; checkpoints averaging, beam search, development set, terminate early."),
])
def test_chinese_equivalents_do_not_cause_false_missing(field, claims, evidence):
    """支持有限中文同义词，但术语覆盖不声称事实或范围验证。"""
    result = apply_field_coverage_diagnostic(qa("answered", [claims], [evidence]), field=field, experiment_scope="synthetic")
    assert result["status"] == "answered"
    assert result["coverage_diagnostic"]["candidate_facets_not_in_claims"] == []


@pytest.mark.parametrize("task_word", ["translation", "classification", "segmentation"])
def test_generic_motivation_is_not_complete_research_question(task_word):
    """多种合成任务中只有架构动机都不能掩盖具体任务未展示。"""
    result = apply_field_coverage_diagnostic(qa("answered", ["提出不依赖循环的架构。"],
        [f"We evaluate the model on {task_word}."]), field="research_question", experiment_scope="synthetic")
    assert result["status"] == "partial"
    assert result["coverage_diagnostic"]["blocking_reasons"] == ["specific_experiment_task_not_reflected_in_claims"]
    assert any("具体实验任务" in text for text in result["missing_info"])


def test_task_diagnostic_does_not_copy_target_or_require_all_background_tasks():
    """目标输入不是证据；候选多任务背景不要求结论照单全收。"""
    complete = apply_field_coverage_diagnostic(qa("answered", ["评估图像分类。"],
        ["Classification is evaluated; segmentation is earlier work."]),
        field="research_question", experiment_scope="secret-target")
    assert complete["status"] == "answered"
    absent = apply_field_coverage_diagnostic(qa("answered", ["讨论方法动机。"], ["General model motivation."]),
        field="research_question", experiment_scope="translation target only")
    assert absent["status"] == "answered"
    assert "secret-target" not in str(complete)


def test_english_substrings_do_not_trigger_categories():
    """token/step等短词不应匹配词内子串。"""
    result = diagnose_field_coverage("training_settings", qa("answered", ["方法说明。"],
        ["A tokenization method with stepwise reasoning on machinelearning."]), experiment_scope="synthetic")
    assert result["candidate_categories"] == []


def test_diagnostic_hints_are_bounded_idempotent_and_do_not_upgrade_insufficient():
    """多次投影不重复附加提示；不足状态不升级，不泄漏原始候选。"""
    original = qa("insufficient", [], ["SECRET Adam optimizer dropout and label smoothing."])
    original["missing_info"] = [f"已有待核对项{i}" for i in range(20)]
    first = apply_field_coverage_diagnostic(original, field="training_settings", experiment_scope="synthetic")
    second = apply_field_coverage_diagnostic(first, field="training_settings", experiment_scope="synthetic")
    assert first["status"] == "insufficient" and len(first["missing_info"]) <= 20
    assert first["missing_info"] == second["missing_info"]
    assert "SECRET" not in str(first["missing_info"])
