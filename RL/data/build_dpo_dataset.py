from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dpo_utils import DESCRIPTION_SCORE_THRESHOLD, has_learning_value, median_margin_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES_JSONL = PROJECT_ROOT / "RL" / "data" / "dpo_candidates_stage1_plain" / "candidates.jsonl"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "RL" / "data" / "dpo_train_stage1_plain.jsonl"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "RL" / "data" / "dpo_train_stage1_plain.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final DPO JSONL from scored candidate pairs.")
    parser.add_argument("--candidates-jsonl", type=Path, default=DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--description-score-threshold", type=float, default=DESCRIPTION_SCORE_THRESHOLD)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_records(
    records: list[dict[str, Any]],
    description_score_threshold: float,
) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
    margins = [float(record.get("margin", 0.0)) for record in records]
    margin_threshold = median_margin_threshold(margins)

    margin_kept = [
        record
        for record in records
        if float(record.get("margin", 0.0)) >= margin_threshold
    ]

    selected = [
        record
        for record in margin_kept
        if has_learning_value(
            record,
            description_score_threshold=description_score_threshold,
        )
    ]
    return selected, margin_threshold, margin_kept


def to_dpo_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "img_id": record["img_id"],
        "image_path": record["image_path"],
        "prompt": record["prompt"],
        "chosen": record["chosen"],
        "rejected": record["rejected"],
        "score_chosen": float(record["score_chosen"]),
        "score_rejected": float(record["score_rejected"]),
        "margin": float(record["margin"]),
        "format_correct": float(record.get("format_correct", 0.0)),
        "diagnosis_exact_set_acc": float(record.get("diagnosis_exact_set_acc", 0.0)),
        "description_score": float(record.get("description_score", 0.0)),
    }


def build_summary(
    all_records: list[dict[str, Any]],
    margin_kept: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    margin_threshold: float,
    description_score_threshold: float,
) -> dict[str, Any]:
    def avg_margin(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(float(item["margin"]) for item in items) / len(items)

    return {
        "num_candidates": len(all_records),
        "num_margin_kept": len(margin_kept),
        "num_selected": len(selected),
        "margin_threshold": margin_threshold,
        "description_score_threshold": description_score_threshold,
        "avg_margin_all": avg_margin(all_records),
        "avg_margin_selected": avg_margin(selected),
        "format_error_count_all": sum(1 for item in all_records if float(item.get("format_correct", 0.0)) < 1.0),
        "diagnosis_error_count_all": sum(1 for item in all_records if float(item.get("diagnosis_exact_set_acc", 0.0)) < 1.0),
        "low_description_score_count_all": sum(
            1 for item in all_records if float(item.get("description_score", 0.0)) < description_score_threshold
        ),
        "format_error_count_selected": sum(1 for item in selected if float(item.get("format_correct", 0.0)) < 1.0),
        "diagnosis_error_count_selected": sum(
            1 for item in selected if float(item.get("diagnosis_exact_set_acc", 0.0)) < 1.0
        ),
        "low_description_score_count_selected": sum(
            1 for item in selected if float(item.get("description_score", 0.0)) < description_score_threshold
        ),
    }


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.candidates_jsonl)
    selected, margin_threshold, margin_kept = select_records(
        records,
        description_score_threshold=args.description_score_threshold,
    )
    dpo_records = [to_dpo_record(record) for record in selected]
    write_jsonl(args.output_jsonl, dpo_records)

    summary = build_summary(
        all_records=records,
        margin_kept=margin_kept,
        selected=selected,
        margin_threshold=margin_threshold,
        description_score_threshold=args.description_score_threshold,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] dpo jsonl: {args.output_jsonl}")
    print(f"[done] summary: {args.summary_json}")


if __name__ == "__main__":
    main()
