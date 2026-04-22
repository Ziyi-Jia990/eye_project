from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


PROJECT_ROOT = Path("/mnt/hdd/jiazy/eye_project")
SFT_ROOT = PROJECT_ROOT / "SFT"
QC_SCRIPT_DIR = PROJECT_ROOT / "diabetic-retinopathy-detection"
DEFAULT_TAILSCORE_CSV = PROJECT_ROOT / "Statistics" / "sample_tailscore.csv"
DEFAULT_TRAIN_CSV = SFT_ROOT / "splits_qc_clean" / "train.csv"
DEFAULT_QC_CHECKPOINT = (
    PROJECT_ROOT
    / "diabetic-retinopathy-detection"
    / "runs"
    / "eyeq_binary_resnet50"
    / "eyeq_crop"
    / "best.pt"
)

if str(QC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(QC_SCRIPT_DIR))

from sample_personal_qc_inference import build_qc_predictor  # noqa: E402


LOCATION_TOKENS = ("鼻下方", "鼻上方", "颞下方", "颞上方", "下方", "上方", "鼻侧", "颞侧")
CLAHE_TARGET_DISEASES = {
    "糖尿病视网膜病变轻度非增生期",
    "糖尿病视网膜病变中度非增生期",
    "糖尿病视网膜病变重度非增生期",
    "糖尿病视网膜病变增生期",
    "黄斑水肿轻度",
    "黄斑水肿中度",
    "黄斑水肿重度",
    "高血压视网膜病变轻度",
    "高血压视网膜病变中度",
    "高血压视网膜病变重度",
    "疑似青光眼",
    "分支静脉阻塞",
    "中央静脉阻塞",
    "动脉阻塞",
    "年龄相关性黄斑变性进展期",
    "黄斑中浆",
    "玻璃体浑浊",
}


@dataclass(frozen=True)
class AugmentationPolicy:
    target_total_multiplier: int
    generation_batch_size: int
    oversample_after_qc_multiplier: int


POLICIES = {
    "weak": AugmentationPolicy(target_total_multiplier=1, generation_batch_size=0, oversample_after_qc_multiplier=1),
    "medium": AugmentationPolicy(target_total_multiplier=2, generation_batch_size=1, oversample_after_qc_multiplier=1),
    "strong": AugmentationPolicy(target_total_multiplier=3, generation_batch_size=3, oversample_after_qc_multiplier=2),
    "extreme": AugmentationPolicy(target_total_multiplier=5, generation_batch_size=5, oversample_after_qc_multiplier=2),
}

# target_total_multiplier：最终希望这个原始样本在输出数据里总共变成多少份，例如原图 1 张，target_total_multiplier=3，说明最终想保留成 1 张原图 + 2 张增强图。
# generation_batch_size：每一轮先实际生成多少个增强候选图，然后统一过一次 QC。
# oversample_after_qc_multiplier：对已通过 QC 的增强样本再离线复制多少倍；2 表示 strong/extreme 会在增强后再做 1 次等量重采样。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline tail-score-augmented training CSV.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--tailscore-csv", type=Path, default=DEFAULT_TAILSCORE_CSV)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--rejected-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--augmented-image-dir", type=Path, default=None)
    parser.add_argument("--qc-checkpoint", type=Path, default=DEFAULT_QC_CHECKPOINT)
    parser.add_argument("--qc-threshold", type=float, default=0.85)
    parser.add_argument("--qc-batch-size", type=int, default=256)
    parser.add_argument("--qc-num-workers", type=int, default=8)
    parser.add_argument("--qc-device", type=str, default="cuda")
    parser.add_argument("--qc-image-size", type=int, default=224)
    parser.add_argument("--qc-apply-eyeq-preprocess", action="store_true")
    parser.add_argument("--qc-preprocess-size", type=int, default=800)
    parser.add_argument("--qc-no-amp", action="store_true")
    parser.add_argument("--clahe-prob", type=float, default=0.5)
    parser.add_argument("--max-attempt-multiplier", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-rejected-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_paths(train_csv: Path) -> tuple[Path, Path, Path, Path]:
    stem = train_csv.with_suffix("")
    output_csv = Path(str(stem) + ".tail_aug.csv")
    rejected_csv = Path(str(stem) + ".tail_aug.reject.csv")
    summary_json = Path(str(stem) + ".tail_aug.summary.json")
    image_dir = Path(str(stem) + ".tail_aug.images")
    return output_csv, rejected_csv, summary_json, image_dir


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def normalize_report(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def split_diagnosis(diagnosis_text: str) -> list[str]:
    return [item.strip() for item in (diagnosis_text or "").split("、") if item.strip()]


def has_location_tokens(report_text: str) -> bool:
    return any(token in report_text for token in LOCATION_TOKENS)


def report_hash(report_text: str) -> str:
    """
    给报告文本生成一个短哈希值, 主要用于增强图像文件命名：
    """
    return hashlib.sha1(normalize_report(report_text).encode("utf-8")).hexdigest()[:16]


def load_tailscore_maps(
    tailscore_csv: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    description_map: dict[str, dict[str, Any]] = {}
    diagnosis_map: dict[str, dict[str, Any]] = {}

    with tailscore_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc = normalize_report(row.get("description", ""))
            diseases = (row.get("diseases", "") or "").strip()
            metadata = {
                "aug_level": (row.get("aug_level", "weak") or "weak").strip(),
                "tail_score": float(row.get("TailScore", 0.0) or 0.0),
                "diseases": diseases,
            }
            if desc and desc not in description_map:
                description_map[desc] = metadata
            if diseases and diseases not in diagnosis_map:
                diagnosis_map[diseases] = metadata
    return description_map, diagnosis_map


def resolve_aug_metadata(
    row: dict[str, str],
    description_map: dict[str, dict[str, Any]],
    diagnosis_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    answer = normalize_report(row.get("answer", ""))
    diagnosis = (row.get("diagnosis", "") or "").strip()
    if answer in description_map:
        return description_map[answer]
    if diagnosis in diagnosis_map:
        return diagnosis_map[diagnosis]
    return {"aug_level": "weak", "tail_score": 0.0, "diseases": diagnosis}


def clamp_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0, 255).astype(np.uint8)


def apply_hue_shift(image: Image.Image, shift: float) -> Image.Image:
    """
    对图像做色相偏移。
    """
    if abs(shift) < 1e-8:
        return image
    hsv = np.array(image.convert("HSV"), dtype=np.uint8)
    delta = int(round(shift * 255))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + delta) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def apply_green_channel_clahe(image: Image.Image) -> Image.Image:
    """
    只对 RGB 图像的 绿色通道 做 CLAHE 增强。
    """
    import cv2  # type: ignore

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    green = rgb[..., 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rgb[..., 1] = clahe.apply(green)
    return Image.fromarray(rgb)


def random_resized_crop(
    image: Image.Image,
    rng: random.Random,
    scale_range: tuple[float, float] = (0.9, 1.0),
    ratio_range: tuple[float, float] = (0.95, 1.05),
) -> Image.Image:
    width, height = image.size
    area = width * height
    scale = rng.uniform(*scale_range)
    ratio = rng.uniform(*ratio_range)
    target_area = area * scale

    crop_w = min(width, max(1, int(round((target_area * ratio) ** 0.5))))
    crop_h = min(height, max(1, int(round((target_area / ratio) ** 0.5))))
    if crop_w >= width and crop_h >= height:
        return image

    left = 0 if width == crop_w else rng.randint(0, width - crop_w)
    top = 0 if height == crop_h else rng.randint(0, height - crop_h)
    crop = image.crop((left, top, left + crop_w, top + crop_h))
    resampling = getattr(Image, "Resampling", Image)
    return crop.resize((width, height), resampling.BICUBIC)


def apply_random_augmentation(
    image: Image.Image,
    diagnosis_labels: list[str],
    allow_geometry: bool,
    clahe_prob: float,
    rng: random.Random,
) -> tuple[Image.Image, list[str]]:
    """
    对单张图像执行一套随机增强，并返回：
    1. 增强后的图像
    2. 本次增强实际用了哪些操作的记录 operations
    """
    augmented = image.copy()
    operations: list[str] = []

    # 几何增强
    if allow_geometry:
        if rng.random() < 0.5:
            augmented = ImageOps.mirror(augmented)
            operations.append("hflip")
        if rng.random() < 0.6:
            angle = rng.uniform(-15.0, -5.0) if rng.random() < 0.5 else rng.uniform(5.0, 15.0)
            resampling = getattr(Image, "Resampling", Image)
            augmented = augmented.rotate(angle, resample=resampling.BICUBIC, fillcolor=(0, 0, 0))
            operations.append(f"rotate_{angle:.2f}")
        augmented = random_resized_crop(augmented, rng=rng)
        operations.append("crop")

    # 外观增强
    brightness_delta = rng.uniform(0.05, 0.12)
    contrast_delta = rng.uniform(0.05, 0.12)
    saturation_delta = rng.uniform(0.0, 0.08)
    hue_delta = rng.uniform(-0.02, 0.02)

    augmented = ImageEnhance.Brightness(augmented).enhance(1.0 + brightness_delta)
    operations.append(f"brightness_{brightness_delta:.3f}")
    augmented = ImageEnhance.Contrast(augmented).enhance(1.0 + contrast_delta)
    operations.append(f"contrast_{contrast_delta:.3f}")

    if saturation_delta > 0:
        augmented = ImageEnhance.Color(augmented).enhance(1.0 + saturation_delta)
        operations.append(f"saturation_{saturation_delta:.3f}")
    if abs(hue_delta) > 1e-6:
        augmented = apply_hue_shift(augmented, hue_delta)
        operations.append(f"hue_{hue_delta:.3f}")

    # CLAHE
    if any(label in CLAHE_TARGET_DISEASES for label in diagnosis_labels) and rng.random() < clahe_prob:
        augmented = apply_green_channel_clahe(augmented)
        operations.append("clahe_green")

    return augmented, operations


def build_base_rows(
    rows: list[dict[str, str]],
    description_map: dict[str, dict[str, Any]],
    diagnosis_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base_rows: list[dict[str, Any]] = []

    for row in rows:
        normalized = dict(row)
        metadata = resolve_aug_metadata(row, description_map, diagnosis_map)
        aug_level = metadata["aug_level"]
        tail_score = metadata["tail_score"]
        base_row = dict(normalized)
        base_row["is_augmented"] = "0"
        base_row["aug_level"] = aug_level
        base_row["tail_score"] = f"{tail_score:.6f}"
        base_row["aug_ops"] = ""
        base_row["source_img_id"] = normalized.get("img_id", "")
        base_rows.append(base_row)
    return base_rows


def generate_candidate_batch(
    source_image: Image.Image,
    base_row: dict[str, Any],
    source_index: int,
    attempt_start: int,
    batch_size: int,
    image_dir: Path,
    clahe_prob: float,
    seed: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    diagnosis_labels = split_diagnosis(base_row.get("diagnosis", ""))
    allow_geometry = not has_location_tokens(base_row.get("answer", ""))
    report_key = report_hash(base_row.get("answer", ""))
    tail_score = float(base_row.get("tail_score", 0.0))
    aug_level = base_row["aug_level"]

    candidates: list[dict[str, Any]] = []
    for offset in range(batch_size):
        attempt_idx = attempt_start + offset
        rng = random.Random(seed + source_index * 100000 + attempt_idx)
        augmented_image, operations = apply_random_augmentation(
            image=source_image,
            diagnosis_labels=diagnosis_labels,
            allow_geometry=allow_geometry,
            clahe_prob=clahe_prob,
            rng=rng,
        )
        new_img_id = f"{base_row['img_id']}__aug{attempt_idx + 1:03d}"
        output_path = image_dir / aug_level / f"{new_img_id}_{report_key}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not output_path.exists():
            augmented_image.save(output_path, format="JPEG", quality=95)
        candidates.append(
            {
                "source_index": source_index,
                "source_img_id": base_row["img_id"],
                "attempt_index": attempt_idx,
                "img_id": new_img_id,
                "image_path": str(output_path),
                "answer": base_row["answer"],
                "institution_name": base_row.get("institution_name", ""),
                "diagnosis": base_row.get("diagnosis", ""),
                "aug_level": aug_level,
                "tail_score": tail_score,
                "aug_ops": "|".join(operations),
                "allow_geometry": allow_geometry,
            }
        )
    return candidates


def build_augmented_dataset(
    base_rows: list[dict[str, Any]],
    image_dir: Path,
    qc_predictor,
    clahe_prob: float,
    max_attempt_multiplier: int,
    seed: int,
    keep_rejected_images: bool,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected_rows: list[dict[str, Any]] = list(base_rows)
    rejected_rows: list[dict[str, Any]] = []
    summary_counter = Counter()

    image_dir.mkdir(parents=True, exist_ok=True)

    states: dict[int, dict[str, Any]] = {}
    augmentable_indices: list[int] = []
    total_target_augmented = 0

    for source_index, base_row in enumerate(base_rows):
        policy = POLICIES.get(base_row["aug_level"], POLICIES["weak"])
        target_augmented = max(0, policy.target_total_multiplier - 1)
        if target_augmented <= 0 or policy.generation_batch_size <= 0:
            continue
        augmentable_indices.append(source_index)
        total_target_augmented += target_augmented
        states[source_index] = {
            "target_augmented": target_augmented,
            "attempts": 0,
            "max_attempts": max(target_augmented, policy.generation_batch_size) * max_attempt_multiplier,
            "accepted_rows": [],
        }

    total_augmentable_rows = len(augmentable_indices)
    active_indices = list(augmentable_indices)
    start_time = time.perf_counter()
    round_index = 0

    while active_indices:
        round_index += 1
        round_candidates: list[dict[str, Any]] = []

        for source_index in active_indices:
            state = states[source_index]
            base_row = base_rows[source_index]
            policy = POLICIES.get(base_row["aug_level"], POLICIES["weak"])
            remaining_needed = state["target_augmented"] - len(state["accepted_rows"])
            remaining_attempts = state["max_attempts"] - state["attempts"]
            if remaining_needed <= 0 or remaining_attempts <= 0:
                continue

            batch_size = min(policy.generation_batch_size, remaining_attempts)
            source_image = Image.open(base_row["image_path"]).convert("RGB")
            try:
                candidates = generate_candidate_batch(
                    source_image=source_image,
                    base_row=base_row,
                    source_index=source_index,
                    attempt_start=state["attempts"],
                    batch_size=batch_size,
                    image_dir=image_dir,
                    clahe_prob=clahe_prob,
                    seed=seed,
                    overwrite=overwrite,
                )
            finally:
                source_image.close()

            state["attempts"] += batch_size
            round_candidates.extend(candidates)
            summary_counter[f"candidate_{base_row['aug_level']}"] += len(candidates)

        if not round_candidates:
            break

        qc_predictions = qc_predictor.predict_image_paths(
            [candidate["image_path"] for candidate in round_candidates]
        )

        for candidate, prediction in zip(round_candidates, qc_predictions, strict=True):
            source_index = candidate["source_index"]
            state = states[source_index]
            base_row = base_rows[source_index]
            merged = dict(candidate)
            merged["qc_pred_label"] = prediction["pred_label"]
            merged["qc_pred_text"] = prediction["pred_text"]
            merged["qc_prob_usable"] = f"{prediction['prob_usable']:.6f}"
            merged["qc_prob_unusable"] = f"{prediction['prob_unusable']:.6f}"
            merged["qc_preprocess_status"] = prediction["preprocess_status"]

            if prediction["pred_label"] == 0 and len(state["accepted_rows"]) < state["target_augmented"]:
                state["accepted_rows"].append(
                    {
                        "img_id": merged["img_id"],
                        "image_path": merged["image_path"],
                        "answer": merged["answer"],
                        "institution_name": merged["institution_name"],
                        "diagnosis": merged["diagnosis"],
                        "is_augmented": "1",
                        "aug_level": merged["aug_level"],
                        "tail_score": f"{merged['tail_score']:.6f}",
                        "aug_ops": merged["aug_ops"],
                        "source_img_id": merged["source_img_id"],
                    }
                )
                summary_counter[f"accepted_{base_row['aug_level']}"] += 1
            else:
                rejected_rows.append(merged)
                if prediction["pred_label"] == 0:
                    summary_counter[f"unused_accepted_{base_row['aug_level']}"] += 1
                else:
                    summary_counter[f"rejected_{base_row['aug_level']}"] += 1
                if not keep_rejected_images:
                    image_path = Path(merged["image_path"])
                    if image_path.exists():
                        image_path.unlink()

        active_indices = [
            source_index
            for source_index in augmentable_indices
            if len(states[source_index]["accepted_rows"]) < states[source_index]["target_augmented"]
            and states[source_index]["attempts"] < states[source_index]["max_attempts"]
        ]

        completed_rows = total_augmentable_rows - len(active_indices)
        accepted_total = sum(len(states[idx]["accepted_rows"]) for idx in augmentable_indices)
        elapsed = time.perf_counter() - start_time
        eta_seconds = 0.0
        if completed_rows > 0 and completed_rows < total_augmentable_rows:
            eta_seconds = elapsed / completed_rows * (total_augmentable_rows - completed_rows)
        print(
            "[progress] "
            f"round={round_index} "
            f"rows={completed_rows}/{total_augmentable_rows} "
            f"accepted={accepted_total}/{total_target_augmented} "
            f"elapsed={format_seconds(elapsed)} "
            f"eta={format_seconds(eta_seconds)}"
        )

    for source_index in augmentable_indices:
        base_row = base_rows[source_index]
        policy = POLICIES.get(base_row["aug_level"], POLICIES["weak"])
        accepted_rows = states[source_index]["accepted_rows"]
        oversample_multiplier = max(1, policy.oversample_after_qc_multiplier)
        if accepted_rows and oversample_multiplier > 1:
            clones: list[dict[str, Any]] = []
            for repeat_idx in range(1, oversample_multiplier):
                for accepted_idx, accepted_row in enumerate(accepted_rows, start=1):
                    clone = dict(accepted_row)
                    clone["img_id"] = f"{accepted_row['img_id']}__os{repeat_idx:02d}_{accepted_idx:03d}"
                    clone["aug_ops"] = accepted_row["aug_ops"] + f"|oversample_copy_{repeat_idx:02d}"
                    clones.append(clone)
            selected_rows.extend(clones)
            summary_counter[f"oversampled_{base_row['aug_level']}"] += len(clones)

        selected_rows.extend(accepted_rows)
        summary_counter[f"attempts_{base_row['aug_level']}"] += states[source_index]["attempts"]
        shortfall = states[source_index]["target_augmented"] - len(accepted_rows)
        if shortfall > 0:
            summary_counter[f"shortfall_{base_row['aug_level']}"] += shortfall

    return selected_rows, rejected_rows, dict(summary_counter)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_csv_default, rejected_csv_default, summary_json_default, image_dir_default = default_output_paths(args.train_csv)
    output_csv = args.output_csv or output_csv_default
    rejected_csv = args.rejected_csv or rejected_csv_default
    summary_json = args.summary_json or summary_json_default
    image_dir = args.augmented_image_dir or image_dir_default

    rows = load_rows(args.train_csv)
    if not rows:
        raise ValueError(f"No rows found in train CSV: {args.train_csv}")
    required_fields = {"img_id", "image_path", "answer", "diagnosis"}
    missing_fields = required_fields - set(rows[0].keys())
    if missing_fields:
        raise KeyError(f"train CSV is missing required fields: {sorted(missing_fields)}")

    description_map, diagnosis_map = load_tailscore_maps(args.tailscore_csv)
    base_rows = build_base_rows(
        rows=rows,
        description_map=description_map,
        diagnosis_map=diagnosis_map,
    )
    qc_predictor = build_qc_predictor(
        checkpoint_path=args.qc_checkpoint,
        threshold=args.qc_threshold,
        batch_size=args.qc_batch_size,
        num_workers=args.qc_num_workers,
        device=args.qc_device,
        image_size=args.qc_image_size,
        apply_eyeq_preprocess=args.qc_apply_eyeq_preprocess,
        preprocess_size=args.qc_preprocess_size,
        use_amp=not args.qc_no_amp,
    )
    selected_rows, rejected_rows, summary_counter = build_augmented_dataset(
        base_rows=base_rows,
        image_dir=image_dir,
        qc_predictor=qc_predictor,
        clahe_prob=args.clahe_prob,
        max_attempt_multiplier=args.max_attempt_multiplier,
        seed=args.seed,
        keep_rejected_images=args.keep_rejected_images,
        overwrite=args.overwrite,
    )

    fieldnames = ["img_id", "image_path", "answer", "institution_name", "diagnosis", "is_augmented", "aug_level", "tail_score", "aug_ops", "source_img_id"]
    write_csv(output_csv, selected_rows, fieldnames)
    write_csv(
        rejected_csv,
        rejected_rows,
        [
            "img_id",
            "image_path",
            "answer",
            "institution_name",
            "diagnosis",
            "aug_level",
            "tail_score",
            "aug_ops",
            "source_img_id",
            "qc_pred_label",
            "qc_pred_text",
            "qc_prob_usable",
            "qc_prob_unusable",
            "qc_preprocess_status",
        ],
    )

    summary = {
        "train_csv": str(args.train_csv),
        "tailscore_csv": str(args.tailscore_csv),
        "output_csv": str(output_csv),
        "rejected_csv": str(rejected_csv),
        "augmented_image_dir": str(image_dir),
        "num_original_rows": len(rows),
        "num_output_rows": len(selected_rows),
        "num_augmented_rows": len(selected_rows) - len(rows),
        "num_rejected_candidates": len(rejected_rows),
        "max_attempt_multiplier": args.max_attempt_multiplier,
        "qc_batch_size": args.qc_batch_size,
        "qc_num_workers": args.qc_num_workers,
        "qc_use_amp": not args.qc_no_amp,
        "policy": {
            level: {
                "target_total_multiplier": policy.target_total_multiplier,
                "generation_batch_size": policy.generation_batch_size,
                "oversample_after_qc_multiplier": policy.oversample_after_qc_multiplier,
            }
            for level, policy in POLICIES.items()
        },
        "counters": summary_counter,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
