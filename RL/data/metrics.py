from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

from dpo_utils import augment_score


def _load_sft_metrics_module():
    module_name = "_eye_project_sft_metrics_for_dpo"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[2] / "SFT" / "metrics.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load SFT metrics module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SFT_METRICS = _load_sft_metrics_module()


PREDICTION_ROW_FIELDNAMES = [
    *_SFT_METRICS.PREDICTION_ROW_FIELDNAMES,
    "description_score",
    "format_score",
    "diagnosis_score",
    "final_score",
]


def score_report(prediction: str, reference: str) -> dict[str, Any]:
    base_score = _SFT_METRICS.score_report(prediction, reference)
    return augment_score(base_score)


def prediction_row_metrics(score: dict[str, Any]) -> dict[str, Any]:
    row = _SFT_METRICS.prediction_row_metrics(score)
    row.update(
        {
            "description_score": score["description_score"],
            "format_score": score["format_score"],
            "diagnosis_score": score["diagnosis_score"],
            "final_score": score["final_score"],
        }
    )
    return row


def aggregate_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    score_list = list(scores)
    aggregated = _SFT_METRICS.aggregate_scores(score_list)
    if not score_list:
        aggregated.update(
            {
                "description_score": 0.0,
                "format_score": 0.0,
                "diagnosis_score": 0.0,
                "final_score": 0.0,
            }
        )
        return aggregated

    num_samples = len(score_list)
    aggregated.update(
        {
            "description_score": sum(float(score["description_score"]) for score in score_list) / num_samples,
            "format_score": sum(float(score["format_score"]) for score in score_list) / num_samples,
            "diagnosis_score": sum(float(score["diagnosis_score"]) for score in score_list) / num_samples,
            "final_score": sum(float(score["final_score"]) for score in score_list) / num_samples,
        }
    )
    return aggregated


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    _SFT_METRICS.save_metrics(metrics, output_path)


def is_format_correct(text: str) -> bool:
    return bool(_SFT_METRICS.is_format_correct(text))
