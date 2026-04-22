from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/mnt/hdd/jiazy/eye_project")
SFT_ROOT = PROJECT_ROOT / "SFT"
DEFAULT_OUTPUTS_DIR = SFT_ROOT / "outputs" / "qwen_retina_lora_mt_structured"
DEFAULT_SPLIT_CSV = SFT_ROOT / "splits_normdiag" / "val.csv"
DEFAULT_PROMPT_PATH = SFT_ROOT / "prompt.txt"
DEFAULT_EVAL_SCRIPT = SFT_ROOT / "evaluate_qwen_vl_sft.py"

SUMMARY_FIELDS = [
    "checkpoint_name",
    "step",
    "format_correct_rate",
    "diagnosis_exact_set_acc",
    "diagnosis_micro_f1",
    "diagnosis_macro_f1",
    "diagnosis_family_level_acc",
    "description_exact_match",
    "description_finding_set_f1",
    "description_location_f1",
    "description_count_bucket_acc",
    "description_cdr_mae",
    "description_cdr_tol_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple Qwen-VL checkpoints on one split and summarize the comparison."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Optional explicit checkpoint directories.")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--primary-metric", type=str, default="diagnosis_micro_f1")
    parser.add_argument("--secondary-metric", type=str, default="diagnosis_exact_set_acc")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            return 10**18
    return 10**18


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        checkpoints = [Path(item) for item in args.checkpoints]
    else:
        checkpoints = sorted(
            [path for path in args.checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
            key=checkpoint_step,
        )
    if not checkpoints:
        raise FileNotFoundError("No checkpoint directories were found.")
    return checkpoints


def ensure_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = args.checkpoints_dir / f"checkpoint_compare_{args.split_csv.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_eval_for_checkpoint(args: argparse.Namespace, checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    checkpoint_output_dir = output_dir / checkpoint_path.name
    split_name = args.split_csv.stem
    metrics_path = checkpoint_output_dir / f"{split_name}_metrics.json"
    if args.overwrite and checkpoint_output_dir.exists():
        for child in checkpoint_output_dir.iterdir():
            if child.is_file():
                child.unlink()

    if not metrics_path.exists() or args.overwrite:
        command = [
            args.python_executable,
            str(args.eval_script),
            "--model-name-or-path",
            args.model_name_or_path,
            "--adapter-path",
            str(checkpoint_path),
            "--split-csv",
            str(args.split_csv),
            "--prompt-path",
            str(args.prompt_path),
            "--output-dir",
            str(checkpoint_output_dir),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--num-beams",
            str(args.num_beams),
            "--dtype",
            args.dtype,
            "--max-image-side",
            str(args.max_image_side),
        ]
        if args.trust_remote_code:
            command.append("--trust-remote-code")

        print(f"[eval] {checkpoint_path.name}")
        try:
            subprocess.run(command, check=True, cwd=str(SFT_ROOT))
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Evaluation failed for {checkpoint_path}. "
                f"Command: {' '.join(command)}"
            ) from exc
    else:
        print(f"[skip] {checkpoint_path.name} already evaluated")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    row = {"checkpoint_name": checkpoint_path.name, "step": checkpoint_step(checkpoint_path)}
    for field in SUMMARY_FIELDS[2:]:
        row[field] = metrics.get(field)
    row["metrics_path"] = str(metrics_path)
    return row


def sort_rows(rows: list[dict[str, Any]], primary_metric: str, secondary_metric: str) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        primary = row.get(primary_metric)
        secondary = row.get(secondary_metric)
        desc_f1 = row.get("description_finding_set_f1")
        exact_match = row.get("description_exact_match")
        step = row.get("step", 10**18)
        return (
            -(primary if primary is not None else float("-inf")),
            -(secondary if secondary is not None else float("-inf")),
            -(desc_f1 if desc_f1 is not None else float("-inf")),
            -(exact_match if exact_match is not None else float("-inf")),
            step,
        )

    return sorted(rows, key=key)


def best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return max(rows, key=lambda row: (row.get(metric, float("-inf")), -row.get("step", 10**18)))


def build_training_hint(rows: list[dict[str, Any]], primary_metric: str) -> str:
    best = best_row(rows, primary_metric)
    latest = max(rows, key=lambda row: row["step"])
    best_value = best.get(primary_metric)
    latest_value = latest.get(primary_metric)
    if best["step"] < latest["step"] and latest_value is not None and best_value is not None and latest_value <= best_value:
        return (
            f"Best {primary_metric} appears at {best['checkpoint_name']} rather than the latest checkpoint. "
            "This suggests the training budget was already sufficient, and later epochs did not improve the main validation target."
        )
    if best["step"] == latest["step"]:
        return (
            f"The latest checkpoint also gives the best {primary_metric}. "
            "This suggests the model may still have been improving at the end, so another shorter continuation is worth testing."
        )
    return "Validation results are mixed; compare the top checkpoints on your preferred metric before extending training."


def write_summary_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*SUMMARY_FIELDS, "metrics_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in [*SUMMARY_FIELDS, "metrics_path"]})


def write_summary_md(rows: list[dict[str, Any]], md_path: Path, primary_metric: str, secondary_metric: str) -> None:
    headers = [
        "checkpoint",
        "step",
        primary_metric,
        secondary_metric,
        "description_finding_set_f1",
        "description_exact_match",
        "format_correct_rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row["checkpoint_name"]),
            str(row["step"]),
            f"{row.get(primary_metric, 0.0):.6f}",
            f"{row.get(secondary_metric, 0.0):.6f}",
            f"{row.get('description_finding_set_f1', 0.0):.6f}",
            f"{row.get('description_exact_match', 0.0):.6f}",
            f"{row.get('format_correct_rate', 0.0):.6f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_json(
    rows: list[dict[str, Any]],
    summary_path: Path,
    primary_metric: str,
    secondary_metric: str,
    split_csv: Path,
) -> None:
    summary = {
        "split_csv": str(split_csv),
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "best_by_primary_metric": best_row(rows, primary_metric),
        "best_by_secondary_metric": best_row(rows, secondary_metric),
        "training_sufficiency_hint": build_training_hint(rows, primary_metric),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoints = discover_checkpoints(args)
    output_dir = ensure_output_dir(args)

    rows = [run_eval_for_checkpoint(args, checkpoint, output_dir) for checkpoint in checkpoints]
    ranked_rows = sort_rows(rows, args.primary_metric, args.secondary_metric)

    write_summary_csv(ranked_rows, output_dir / "summary.csv")
    write_summary_md(ranked_rows, output_dir / "summary.md", args.primary_metric, args.secondary_metric)
    write_summary_json(
        ranked_rows,
        output_dir / "summary.json",
        args.primary_metric,
        args.secondary_metric,
        args.split_csv,
    )

    best = ranked_rows[0]
    print(
        json.dumps(
            {
                "best_checkpoint": best["checkpoint_name"],
                "best_step": best["step"],
                "primary_metric": args.primary_metric,
                "primary_metric_value": best.get(args.primary_metric),
                "secondary_metric": args.secondary_metric,
                "secondary_metric_value": best.get(args.secondary_metric),
                "training_sufficiency_hint": build_training_hint(ranked_rows, args.primary_metric),
                "summary_csv": str(output_dir / "summary.csv"),
                "summary_md": str(output_dir / "summary.md"),
                "summary_json": str(output_dir / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
