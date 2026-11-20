"""对字段报告做保守的候选覆盖诊断，不生成或改写论文事实。"""

import copy
import re


COVERAGE_POLICY_VERSION = "field-coverage-diagnostic-v2-task-and-facets"

_TRAINING_GROUPS = {
    "hardware": ("gpu", "cpu", "hardware", "machine", "硬件", "显卡"),
    "schedule": ("step", "steps", "epoch", "epochs", "warmup", "learning rate", "learning-rate", "optimizer", "步", "轮", "优化器", "预热", "学习率"),
    "batch": ("batch", "token", "tokens", "sentence pair", "sequence length", "批次", "批量", "组批"),
    "regularization": ("dropout", "label smoothing", "regularization", "weight decay", "正则化", "标签平滑", "权重衰减"),
}
_EVALUATION_GROUPS = {
    "metric": ("bleu", "accuracy", "precision", "recall", "f1", "perplexity", "metric", "准确率", "精确率", "召回率", "困惑度"),
    "split": ("test", "development", "dev set", "validation", "newstest", "测试", "开发集", "验证集"),
    "procedure": ("beam", "checkpoint", "checkpoints", "decoding", "inference", "length penalty", "early stop", "解码", "推理", "长度惩罚", "提前终止"),
}
_TRAINING_FACETS = {
    "dropout": ("dropout",),
    "label_smoothing": ("label smoothing", "标签平滑"),
    "weight_decay": ("weight decay", "权重衰减"),
    "optimizer": ("optimizer", "adam", "sgd", "优化器"),
    "learning_rate_schedule": ("warmup", "learning rate", "learning-rate", "预热", "学习率"),
    "batching_by_length": ("similar sequence length", "similar sequence lengths", "similar length", "similar lengths", "近似长度", "相近长度", "相似长度", "长度相近", "长度相似"),
}
_EVALUATION_FACETS = {
    "checkpoint": ("checkpoint", "checkpoints", "检查点"),
    "decoding": ("beam", "decoding", "解码", "束搜索"),
    "selection": ("development set", "dev set", "validation set", "开发集", "验证集"),
    "termination": ("maximum output length", "terminate early", "early stopping", "最大输出长度", "提前终止", "提前停止"),
}
_TASK_FACETS = {
    "translation": ("translation", "translate", "translating", "翻译"),
    "classification": ("classification", "classify", "分类"),
    "segmentation": ("segmentation", "segmenting", "分割"),
    "recognition": ("recognition", "识别"),
    "generation": ("image generation", "image synthesis", "图像生成", "图像合成"),
}
_LABELS = {"hardware": "硬件", "schedule": "优化与训练计划", "batch": "批次", "regularization": "正则化",
    "metric": "指标", "split": "评测划分", "procedure": "评测过程", "dropout": "dropout",
    "label_smoothing": "标签平滑", "weight_decay": "权重衰减", "optimizer": "优化器",
    "learning_rate_schedule": "学习率调度", "batching_by_length": "按长度组批", "checkpoint": "checkpoint处理",
    "decoding": "解码", "selection": "开发/验证选参", "termination": "输出终止条件"}


def _texts(qa: dict) -> tuple[str, str]:
    """收集候选原文与最终结论文本，保持诊断只读。"""
    evidence = "\n".join(str(item.get("text", "")) for item in qa.get("evidence", []))
    claims = "\n".join(str(item.get("text", "")) for item in qa.get("claims", []))
    return evidence.casefold(), claims.casefold()


def _groups_in_text(text: str, groups: dict[str, tuple[str, ...]]) -> set[str]:
    """按中英文术语定位类别，英文有词边界，不判定语义支持。"""
    return {name for name, terms in groups.items() if any(
        (re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None
         if term.isascii() else term in text) for term in terms)}


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
    facets = {}
    candidate_tasks, claim_tasks = set(), set()
    if field == "training_settings":
        evidence_groups = _groups_in_text(evidence, _TRAINING_GROUPS)
        claim_groups = _groups_in_text(claims, _TRAINING_GROUPS)
        facets = _TRAINING_FACETS
    elif field == "evaluation_metrics":
        evidence_groups = _groups_in_text(evidence, _EVALUATION_GROUPS)
        claim_groups = _groups_in_text(claims, _EVALUATION_GROUPS)
        facets = _EVALUATION_FACETS
    evidence_facets = _groups_in_text(evidence, facets)
    claim_facets = _groups_in_text(claims, facets)
    # 细项只是候选关注点，可能来自其他配置；不据此生成具体论文事实。
    missing_facets = sorted(evidence_facets - claim_facets)
    missing_from_claims = sorted(evidence_groups - claim_groups)
    reasons = []
    if len(evidence_groups) >= 2 and missing_from_claims:
        reasons.append("candidate_categories_not_reflected_in_claims")
    if len(evidence_facets) >= 2 and missing_facets:
        reasons.append("candidate_facets_not_reflected_in_claims")
    if field == "research_question":
        candidate_tasks = _groups_in_text(evidence, _TASK_FACETS)
        claim_tasks = _groups_in_text(claims, _TASK_FACETS)
        if candidate_tasks and not candidate_tasks.intersection(claim_tasks):
            reasons.append("specific_experiment_task_not_reflected_in_claims")
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
        "candidate_facets": sorted(evidence_facets),
        "claim_facets": sorted(claim_facets),
        "candidate_facets_not_in_claims": missing_facets,
        "candidate_task_categories": sorted(candidate_tasks),
        "claim_task_categories": sorted(claim_tasks),
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
    if audit["blocking_reasons"] and result.get("status") in {"partial", "insufficient"}:
        hints = []
        if "specific_experiment_task_not_reflected_in_claims" in audit["blocking_reasons"]:
            hints.append("待核对：结论未体现候选中的具体实验任务，仅有方法动机不足以说明指定实验；该诊断不确认任务归属。")
        if "candidate_categories_not_reflected_in_claims" in audit["blocking_reasons"]:
            hints.append("待核对候选关注类别：" + "、".join(_LABELS[k] for k in audit["candidate_categories_not_in_claims"]) + "；不表示这些内容已确认适用于目标。")
        if "candidate_facets_not_reflected_in_claims" in audit["blocking_reasons"]:
            hints.append("待核对候选细项：" + "、".join(_LABELS[k] for k in audit["candidate_facets_not_in_claims"]) + "；需检查目标归属和抽取/复核遗漏，不能据此补造事实。")
        if "claim_contains_explicit_other_language_direction" in audit["blocking_reasons"]:
            hints.append("待核对：结论含与指定范围不同的语言方向，当前诊断未自动删除或改写。")
        existing = list(result.get("missing_info", []))
        added = [h for h in hints if h not in existing]
        result["missing_info"] = existing[:20 - len(added)] + added
    return result
