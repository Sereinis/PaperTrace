"""对字段报告做保守的候选覆盖诊断，不生成或改写论文事实。"""

import copy
import re


COVERAGE_POLICY_VERSION = "field-coverage-diagnostic-v1"

_TRAINING_GROUPS = {
    "hardware": ("gpu", "cpu", "hardware", "machine"),
    "schedule": ("step", "epoch", "warmup", "learning rate", "learning-rate", "optimizer"),
    "batch": ("batch", "token", "sentence pair", "sequence length"),
    "regularization": ("dropout", "label smoothing", "regularization", "weight decay"),
}
_EVALUATION_GROUPS = {
    "metric": ("bleu", "accuracy", "precision", "recall", "f1", "perplexity", "metric"),
    "split": ("test", "development", "dev set", "validation", "newstest"),
    "procedure": ("beam", "checkpoint", "decoding", "inference", "length penalty", "early stop"),
}


def _texts(qa: dict) -> tuple[str, str]:
    """收集候选原文与最终结论文本，保持诊断只读。"""
    evidence = "\n".join(str(item.get("text", "")) for item in qa.get("evidence", []))
    claims = "\n".join(str(item.get("text", "")) for item in qa.get("claims", []))
    return evidence.casefold(), claims.casefold()


def _groups_in_text(text: str, groups: dict[str, tuple[str, ...]]) -> set[str]:
    """用小写通用术语定位候选中出现的字段类别，不判定语义支持。"""
    return {name for name, terms in groups.items() if any(term in text for term in terms)}


def _language_scope_mismatch(experiment_scope: str, claims: str) -> bool:
    """只识别同一 English-to-X 范式下明显不同的目标语言，不推断其它语言。"""
    scope_langs = {item.casefold() for item in re.findall(r"english\s*[-–]\s*to\s*[-–]?\s*([a-z]+)", experiment_scope,
                                                            flags=re.IGNORECASE)}
    claim_langs = {item.casefold() for item in re.findall(r"english\s*[-–]\s*to\s*[-–]?\s*([a-z]+)", claims,
                                                          flags=re.IGNORECASE)}
    if scope_langs and claim_langs and not claim_langs <= scope_langs:
        return True
    # 处理当前报告中文结论中的常见语言方向表达；不把其它语言自动当作事实。
    if "english-to-german" in experiment_scope.casefold() or "英德" in experiment_scope:
        return "英法" in claims or "英语-法语" in claims or "english-to-french" in claims
    return False


def diagnose_field_coverage(field: str, qa: dict, *, experiment_scope: str) -> dict:
    """报告候选术语与最终claims的覆盖差异，必要时给出保守降级建议。"""
    evidence, claims = _texts(qa)
    evidence_groups: set[str] = set()
    claim_groups: set[str] = set()
    if field == "training_settings":
        evidence_groups = _groups_in_text(evidence, _TRAINING_GROUPS)
        claim_groups = _groups_in_text(claims, _TRAINING_GROUPS)
    elif field == "evaluation_metrics":
        evidence_groups = _groups_in_text(evidence, _EVALUATION_GROUPS)
        claim_groups = _groups_in_text(claims, _EVALUATION_GROUPS)
    missing_from_claims = sorted(evidence_groups - claim_groups)
    reasons = []
    if len(evidence_groups) >= 2 and missing_from_claims:
        reasons.append("candidate_categories_not_reflected_in_claims")
    if field == "baselines" and _language_scope_mismatch(experiment_scope, claims):
        reasons.append("claim_contains_explicit_other_language_direction")
    downgraded = qa.get("status") == "answered" and bool(reasons)
    return {
        "version": COVERAGE_POLICY_VERSION,
        "basis": "lexical_candidate_vs_claim_diagnostic_not_semantic_verification",
        "field": field,
        "candidate_categories": sorted(evidence_groups),
        "claim_categories": sorted(claim_groups),
        "candidate_categories_not_in_claims": missing_from_claims,
        "blocking_reasons": reasons,
        "status_before": qa.get("status"),
        "status_after": "partial" if downgraded else qa.get("status"),
        "downgraded": downgraded,
    }


def apply_field_coverage_diagnostic(qa: dict, *, field: str, experiment_scope: str) -> dict:
    """附加可审计诊断；只把明显过于乐观的answered降为partial。"""
    result = copy.deepcopy(qa)
    audit = diagnose_field_coverage(field, result, experiment_scope=experiment_scope)
    result["coverage_diagnostic"] = audit
    if audit["downgraded"]:
        result["status"] = "partial"
        result["missing_info"] = list(result.get("missing_info", [])) + [
            "候选中存在字段类别但最终结论未覆盖，需核对抽取或复核是否遗漏；该诊断不证明具体事实。"
        ]
    return result
