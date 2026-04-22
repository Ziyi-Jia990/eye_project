from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dpo_utils import DESCRIPTION_SCORE_THRESHOLD


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = PROJECT_ROOT / "SFT"
DEFAULT_PROMPT_PATH = SFT_ROOT / "prompt.txt"
DEFAULT_TRAIN_SPLIT = SFT_ROOT / "splits_qc_clean" / "train.csv"
DEFAULT_ADAPTER_PATH = SFT_ROOT / "outputs" / "qwen_retina_lora_stage1_plain_eval_loss"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "RL" / "data" / "dpo_candidates_stage1_plain"
RUN_STATE_VERSION = 1

if str(SFT_ROOT) not in sys.path:
    sys.path.append(str(SFT_ROOT))

from generation_eval import load_model_and_processor  # noqa: E402
from metrics import aggregate_scores, score_report  # noqa: E402
from retina_sft_utils import QwenVLGenerationCollator, RetinaSFTDataset, filter_valid_rows, read_split_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and score DPO candidate pairs from the stage1 model.")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--input-split", type=Path, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing candidates.jsonl in output-dir.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Delete existing generated outputs in output-dir before starting a fresh run.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def resolve_output_paths(output_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    candidates_jsonl = output_dir / "candidates.jsonl"
    candidates_csv = output_dir / "candidates.csv"
    summary_json = output_dir / "summary.json"
    invalid_images_json = output_dir / "invalid_images.json"
    run_state_json = output_dir / "run_state.json"
    legacy_shard_dir = output_dir / "shards"
    return candidates_jsonl, candidates_csv, summary_json, invalid_images_json, run_state_json, legacy_shard_dir


def ensure_serializable_score(score: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(score)
    for key, value in list(serializable.items()):
        if isinstance(value, Path):
            serializable[key] = str(value)
    return serializable


def build_record(
    metadata: dict[str, Any],
    prediction: str,
) -> dict[str, Any]:
    ground_truth = metadata["answer"]
    rejected_score = score_report(prediction, ground_truth)
    chosen_score = score_report(ground_truth, ground_truth)

    return {
        "source_index": int(metadata["source_index"]),
        "img_id": metadata.get("img_id") or metadata.get("\ufeffimg_id", ""),
        "image_path": metadata["image_path"],
        "diagnosis": metadata.get("diagnosis", ""),
        "prompt": metadata["prompt"],
        "ground_truth": ground_truth,
        "prediction": prediction,
        "chosen": ground_truth,
        "rejected": prediction,
        "score_chosen": float(chosen_score["final_score"]),
        "score_rejected": float(rejected_score["final_score"]),
        "margin": float(chosen_score["final_score"]) - float(rejected_score["final_score"]),
        "format_correct": float(rejected_score["format_correct"]),
        "description_exact_match": float(rejected_score["description_exact_match"]),
        "description_finding_set_f1": float(rejected_score["description_finding_set_f1"]),
        "description_location_f1": float(rejected_score["description_location_f1"]),
        "description_count_bucket_acc": float(rejected_score["description_count_bucket_acc"]),
        "description_cdr_abs_error": float(rejected_score["description_cdr_abs_error"]),
        "description_cdr_tol_hit": float(rejected_score["description_cdr_tol_hit"]),
        "description_score": float(rejected_score["description_score"]),
        "diagnosis_exact_set_acc": float(rejected_score["diagnosis_exact_set_acc"]),
        "diagnosis_score": float(rejected_score["diagnosis_score"]),
        "format_score": float(rejected_score["format_score"]),
        "final_score": float(rejected_score["final_score"]),
        "chosen_metrics": ensure_serializable_score(chosen_score),
        "rejected_metrics": ensure_serializable_score(rejected_score),
    }


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        normalized_row = dict(row)
        normalized_row["source_index"] = source_index
        normalized_rows.append(normalized_row)
    return normalized_rows


def record_key_from_row(row: dict[str, Any]) -> str:
    return str(int(row["source_index"]))


def record_key_from_record(record: dict[str, Any]) -> str:
    return str(int(record["source_index"]))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse JSONL at {path}:{line_number}") from exc
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "img_id",
        "image_path",
        "diagnosis",
        "score_chosen",
        "score_rejected",
        "margin",
        "format_correct",
        "description_exact_match",
        "description_finding_set_f1",
        "description_location_f1",
        "description_count_bucket_acc",
        "description_cdr_abs_error",
        "description_cdr_tol_hit",
        "description_score",
        "diagnosis_exact_set_acc",
        "diagnosis_score",
        "format_score",
        "final_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def build_summary(
    requested_rows: int,
    valid_rows: int,
    invalid_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    rejected_scores = [record["rejected_metrics"] for record in records]
    chosen_scores = [record["chosen_metrics"] for record in records]
    rejected_aggregate = aggregate_scores(rejected_scores)
    chosen_aggregate = aggregate_scores(chosen_scores)
    margins = [float(record["margin"]) for record in records]

    return {
        "num_requested_rows": requested_rows,
        "num_valid_rows": valid_rows,
        "num_invalid_rows": len(invalid_rows),
        "num_scored_rows": len(records),
        "avg_score_chosen": sum(float(record["score_chosen"]) for record in records) / len(records) if records else 0.0,
        "avg_score_rejected": sum(float(record["score_rejected"]) for record in records) / len(records) if records else 0.0,
        "avg_margin": sum(margins) / len(margins) if margins else 0.0,
        "format_error_count": sum(1 for record in records if float(record["format_correct"]) < 1.0),
        "diagnosis_error_count": sum(1 for record in records if float(record["diagnosis_exact_set_acc"]) < 1.0),
        "low_description_score_count": sum(
            1 for record in records if float(record["description_score"]) < DESCRIPTION_SCORE_THRESHOLD
        ),
        "chosen_metrics_aggregate": chosen_aggregate,
        "rejected_metrics_aggregate": rejected_aggregate,
    }


def build_run_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": RUN_STATE_VERSION,
        "input_split": str(args.input_split.resolve()),
        "prompt_path": str(args.prompt_path.resolve()),
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "dtype": args.dtype,
        "max_image_side": args.max_image_side,
    }


def run_states_match(existing_state: dict[str, Any], current_state: dict[str, Any]) -> bool:
    return all(existing_state.get(key) == value for key, value in current_state.items())


def clear_generated_outputs(
    candidates_jsonl: Path,
    candidates_csv: Path,
    summary_json: Path,
    invalid_images_json: Path,
    run_state_json: Path,
    legacy_shard_dir: Path,
) -> None:
    for path in (candidates_jsonl, candidates_csv, summary_json, invalid_images_json, run_state_json):
        if path.exists():
            path.unlink()
    if legacy_shard_dir.exists():
        shutil.rmtree(legacy_shard_dir)


def prepare_output_dir(
    *,
    output_dir: Path,
    candidates_jsonl: Path,
    candidates_csv: Path,
    summary_json: Path,
    invalid_images_json: Path,
    run_state_json: Path,
    legacy_shard_dir: Path,
    run_state: dict[str, Any],
    resume: bool,
    overwrite_existing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_outputs = [
        path
        for path in (candidates_jsonl, candidates_csv, summary_json, invalid_images_json, run_state_json)
        if path.exists()
    ]
    if legacy_shard_dir.exists():
        existing_outputs.append(legacy_shard_dir)

    if overwrite_existing and resume:
        raise ValueError("`--resume` and `--overwrite-existing` cannot be used together.")

    if overwrite_existing:
        clear_generated_outputs(
            candidates_jsonl=candidates_jsonl,
            candidates_csv=candidates_csv,
            summary_json=summary_json,
            invalid_images_json=invalid_images_json,
            run_state_json=run_state_json,
            legacy_shard_dir=legacy_shard_dir,
        )
    elif existing_outputs and not resume:
        existing_list = ", ".join(str(path) for path in existing_outputs)
        raise RuntimeError(
            "Output directory already contains generated files. "
            "Use `--resume` to continue or `--overwrite-existing` to restart.\n"
            f"Existing paths: {existing_list}"
        )

    if run_state_json.exists():
        existing_state = json.loads(run_state_json.read_text(encoding="utf-8"))
        if not run_states_match(existing_state, run_state):
            raise RuntimeError(
                "Existing run state does not match the current command. "
                "Use a different output directory or restart with `--overwrite-existing`.\n"
                f"existing={json.dumps(existing_state, ensure_ascii=False, sort_keys=True)}\n"
                f"current={json.dumps(run_state, ensure_ascii=False, sort_keys=True)}"
            )
    else:
        run_state_json.write_text(json.dumps(run_state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_records_by_key(
    candidates_jsonl: Path,
    legacy_shard_dir: Path,
) -> dict[str, dict[str, Any]]:
    existing_records: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(candidates_jsonl):
        existing_records[record_key_from_record(record)] = record

    if legacy_shard_dir.exists():
        for shard_path in sorted(legacy_shard_dir.glob("*.jsonl")):
            for record in load_jsonl(shard_path):
                existing_records[record_key_from_record(record)] = record

    return existing_records


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    candidates_jsonl, candidates_csv, summary_json, invalid_images_json, run_state_json, legacy_shard_dir = (
        resolve_output_paths(output_dir)
    )
    run_state = build_run_state(args)

    prepare_output_dir(
        output_dir=output_dir,
        candidates_jsonl=candidates_jsonl,
        candidates_csv=candidates_csv,
        summary_json=summary_json,
        invalid_images_json=invalid_images_json,
        run_state_json=run_state_json,
        legacy_shard_dir=legacy_shard_dir,
        run_state=run_state,
        resume=args.resume,
        overwrite_existing=args.overwrite_existing,
    )

    rows = read_split_csv(args.input_split, prompt_path=args.prompt_path)
    requested_rows = len(rows)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    rows = normalize_rows(rows)
    rows, invalid_rows = filter_valid_rows(rows, split_name=args.input_split.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_images_json.write_text(json.dumps(invalid_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    existing_records = load_existing_records_by_key(candidates_jsonl, legacy_shard_dir) if args.resume else {}
    pending_rows = [row for row in rows if record_key_from_row(row) not in existing_records]
    print(f"[resume] completed={len(existing_records)} pending={len(pending_rows)} valid={len(rows)}")

    if pending_rows:
        model, processor = load_model_and_processor(
            model_name_or_path=args.model_name_or_path,
            adapter_path=str(args.adapter_path) if args.adapter_path else None,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        dataloader = DataLoader(
            RetinaSFTDataset(pending_rows),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=QwenVLGenerationCollator(processor, max_image_side=args.max_image_side),
        )

        generated_records = 0
        for batch_index, batch in enumerate(dataloader, start=1):
            metadata = batch.pop("metadata")
            prompt_length = int(batch.pop("prompt_length"))
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                )

            batch_records: list[dict[str, Any]] = []
            for item_index, generated_ids in enumerate(generated):
                new_tokens = generated_ids[prompt_length:]
                prediction = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                record = build_record(metadata[item_index], prediction)
                batch_records.append(record)
                existing_records[record_key_from_record(record)] = record

            append_jsonl(candidates_jsonl, batch_records)
            generated_records += len(batch_records)

            if batch_index % 10 == 0:
                print(f"[progress] processed {generated_records}/{len(pending_rows)} pending samples")

    records = sorted(existing_records.values(), key=lambda record: int(record["source_index"]))
    if len(records) != len(rows):
        raise RuntimeError(
            "Generated record count does not match the number of valid rows. "
            f"expected={len(rows)} collected={len(records)}"
        )

    write_jsonl(candidates_jsonl, records)
    if legacy_shard_dir.exists():
        shutil.rmtree(legacy_shard_dir)
    write_csv(candidates_csv, records)

    summary = build_summary(
        requested_rows=requested_rows if args.max_samples is None else min(requested_rows, args.max_samples),
        valid_rows=len(rows),
        invalid_rows=invalid_rows,
        records=records,
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] candidates jsonl: {candidates_jsonl}")
    print(f"[done] candidates csv: {candidates_csv}")
    print(f"[done] summary: {summary_json}")


if __name__ == "__main__":
    main()
