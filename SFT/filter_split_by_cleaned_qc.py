from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_TEST_CSV = Path("/mnt/hdd/jiazy/eye_project/SFT/splits_normdiag/test.csv")
DEFAULT_CLEANED_QC_CSV = Path("/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.qc.csv")
DEFAULT_OUTPUT_CSV = Path("/mnt/hdd/jiazy/eye_project/SFT/splits_normdiag/test.cleaned_qc_overlap.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a split CSV to samples whose img_id also appears in "
            "description.cleaned.qc.csv."
        )
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--cleaned-qc-csv", type=Path, default=DEFAULT_CLEANED_QC_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--removed-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--key-field", type=str, default="img_id")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_sidecar_paths(output_csv: Path) -> tuple[Path, Path]:
    stem = output_csv.with_suffix("")
    return Path(str(stem) + ".removed.csv"), Path(str(stem) + ".summary.json")


def main() -> None:
    args = parse_args()
    removed_csv_default, summary_json_default = default_sidecar_paths(args.output_csv)
    removed_csv = args.removed_csv or removed_csv_default
    summary_json = args.summary_json or summary_json_default

    split_rows = read_rows(args.split_csv)
    cleaned_rows = read_rows(args.cleaned_qc_csv)
    if not split_rows:
        raise ValueError(f"No rows found in split CSV: {args.split_csv}")
    if not cleaned_rows:
        raise ValueError(f"No rows found in cleaned QC CSV: {args.cleaned_qc_csv}")

    split_fields = list(split_rows[0].keys())
    cleaned_fields = set(cleaned_rows[0].keys())
    if args.key_field not in split_fields:
        raise KeyError(f"`{args.key_field}` is missing from split CSV fields: {split_fields}")
    if args.key_field not in cleaned_fields:
        raise KeyError(f"`{args.key_field}` is missing from cleaned QC CSV fields: {sorted(cleaned_fields)}")

    cleaned_ids = {row.get(args.key_field, "").strip() for row in cleaned_rows if row.get(args.key_field, "").strip()}
    kept_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []
    for row in split_rows:
        key = row.get(args.key_field, "").strip()
        if key in cleaned_ids:
            kept_rows.append(row)
        else:
            removed_rows.append(row)

    write_rows(args.output_csv, kept_rows, split_fields)
    write_rows(removed_csv, removed_rows, split_fields)

    summary = {
        "split_csv": str(args.split_csv),
        "cleaned_qc_csv": str(args.cleaned_qc_csv),
        "output_csv": str(args.output_csv),
        "removed_csv": str(removed_csv),
        "key_field": args.key_field,
        "num_split_rows": len(split_rows),
        "num_cleaned_qc_rows": len(cleaned_rows),
        "num_cleaned_qc_unique_keys": len(cleaned_ids),
        "num_kept_rows": len(kept_rows),
        "num_removed_rows": len(removed_rows),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
