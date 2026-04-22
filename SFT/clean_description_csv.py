from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from retina_sft_utils import build_image_index, resolve_image_path, validate_image_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove missing or unreadable images from a retinal report CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.invalid_images.json",
    )
    parser.add_argument("--images-root", type=str, default="")
    parser.add_argument("--recursive-images", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Number of worker threads used for image validation.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader), reader.fieldnames


def validate_rows(
    rows: list[dict[str, Any]],
    csv_dir: Path,
    image_index: dict[str, str] | None,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, str]] = []

    def check_one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, str] | None]:
        row_number, row = item
        image_path = resolve_image_path(row, csv_dir=csv_dir, image_index=image_index)
        if image_path is None:
            return row_number, row, {
                "row_number": str(row_number),
                "img_id": str(row.get("img_id", "")),
                "image_path": str(row.get("img_path", "")),
                "error": "FileNotFoundError: image path could not be resolved",
            }

        error = validate_image_file(image_path)
        if error is None:
            return row_number, row, None

        return row_number, row, {
            "row_number": str(row_number),
            "img_id": str(row.get("img_id", "")),
            "image_path": image_path,
            "error": error,
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _, row, invalid in executor.map(check_one, enumerate(rows, start=2)):
            if invalid is None:
                valid_rows.append(row)
            else:
                invalid_rows.append(invalid)

    return valid_rows, invalid_rows


def write_rows(rows: list[dict[str, Any]], fieldnames: list[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(invalid_rows: list[dict[str, str]], report_json: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(invalid_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    report_json = Path(args.report_json)

    rows, fieldnames = load_rows(input_csv)

    image_index = None
    if args.images_root:
        image_index = build_image_index(args.images_root, recursive=args.recursive_images)

    valid_rows, invalid_rows = validate_rows(
        rows=rows,
        csv_dir=input_csv.parent,
        image_index=image_index,
        workers=args.workers,
    )

    write_rows(valid_rows, fieldnames, output_csv)
    write_report(invalid_rows, report_json)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "report_json": str(report_json),
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "dropped_rows": len(invalid_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
