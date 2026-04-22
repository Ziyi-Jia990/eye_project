from __future__ import annotations

from statistics import median
from typing import Any, Mapping


# Top-level DPO score weights. The default priority follows the requirement:
# description > format > diagnosis.
DESCRIPTION_WEIGHT = 0.25
FORMAT_WEIGHT = 0.15
DIAGNOSIS_WEIGHT = 0.6

# Internal composition of the description score. These are normalized again
# inside compute_description_score, so the sum does not have to be 1.0 exactly.
# We keep the defaults neutral unless the user later wants to tune them.
DESCRIPTION_COMPONENT_WEIGHTS = {
    "description_finding_set_f1": 0.25,
    "description_location_f1": 0.25,
    "description_count_bucket_acc": 0.25,
    "description_cdr_tol_hit": 0.25,
}

# For DPO final_score, default to the diagnosis exact-set metric referenced in
# the design note. This remains configurable at call sites.
DIAGNOSIS_SCORE_FIELD = "diagnosis_exact_set_acc"

# Learning-value filter threshold confirmed by the user.
DESCRIPTION_SCORE_THRESHOLD = 0.8


def compute_description_score(
    score: Mapping[str, Any],
    component_weights: Mapping[str, float] | None = None,
) -> float:
    component_weights = component_weights or DESCRIPTION_COMPONENT_WEIGHTS
    total_weight = float(sum(component_weights.values()))
    if total_weight <= 0:
        return 0.0

    weighted_sum = 0.0
    for field_name, weight in component_weights.items():
        weighted_sum += float(score.get(field_name, 0.0)) * float(weight)
    return weighted_sum / total_weight


def compute_diagnosis_score(
    score: Mapping[str, Any],
    diagnosis_score_field: str = DIAGNOSIS_SCORE_FIELD,
) -> float:
    return float(score.get(diagnosis_score_field, score.get("diagnosis_exact_set_acc", 0.0)))


def augment_score(
    score: Mapping[str, Any],
    *,
    description_weight: float = DESCRIPTION_WEIGHT,
    format_weight: float = FORMAT_WEIGHT,
    diagnosis_weight: float = DIAGNOSIS_WEIGHT,
    component_weights: Mapping[str, float] | None = None,
    diagnosis_score_field: str = DIAGNOSIS_SCORE_FIELD,
) -> dict[str, Any]:
    enriched = dict(score)
    enriched["description_score"] = compute_description_score(
        enriched,
        component_weights=component_weights,
    )
    enriched["format_score"] = float(enriched.get("format_correct", 0.0))
    enriched["diagnosis_score"] = compute_diagnosis_score(
        enriched,
        diagnosis_score_field=diagnosis_score_field,
    )
    enriched["final_score"] = (
        description_weight * enriched["description_score"]
        + format_weight * enriched["format_score"]
        + diagnosis_weight * enriched["diagnosis_score"]
    )
    return enriched


def compute_margin(
    score_chosen: float,
    score_rejected: float,
) -> float:
    return float(score_chosen) - float(score_rejected)


def median_margin_threshold(margins: list[float]) -> float:
    if not margins:
        return 0.0
    return float(median(margins))


def has_learning_value(
    rejected_score: Mapping[str, Any],
    *,
    description_score_threshold: float = DESCRIPTION_SCORE_THRESHOLD,
) -> bool:
    return (
        float(rejected_score.get("diagnosis_exact_set_acc", 0.0)) < 1.0
        or float(rejected_score.get("format_correct", 0.0)) < 1.0
        or float(rejected_score.get("description_score", 0.0)) < description_score_threshold
    )
