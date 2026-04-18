from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_prompt(prompt_path: str | Path) -> str:
    return Path(prompt_path).read_text(encoding="utf-8").strip()


def validate_image_file(image_path: str | Path) -> str | None:
    image_path = str(image_path)
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.convert("RGB").load()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def filter_valid_rows(
    rows: list[dict[str, Any]],
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, str]] = []

    for row in rows:
        image_path = row["image_path"]
        error = validate_image_file(image_path)
        if error is None:
            valid_rows.append(row)
            continue

        invalid_rows.append(
            {
                "split": split_name,
                "img_id": str(row.get("img_id", "")),
                "image_path": image_path,
                "error": error,
            }
        )

    return valid_rows, invalid_rows


def load_rgb_image(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def resize_image_longest_side(image: Image.Image, max_image_side: int | None) -> Image.Image:
    if max_image_side is None or max_image_side <= 0:
        return image

    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_image_side:
        return image

    scale = max_image_side / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((resized_width, resized_height), resampling.LANCZOS)


def extract_diagnosis(report_text: str) -> str:
    match = re.search(r"初步诊断\s*[：:]\s*(.+)", report_text or "")
    return match.group(1).strip() if match else "UNKNOWN"


def split_report_sections(report_text: str) -> tuple[str, str]:
    report_text = (report_text or "").replace("\r\n", "\n").replace("\r", "\n")
    desc_match = re.search(r"(描述\s*[：:].*?)(?:\n初步诊断\s*[：:]|$)", report_text, flags=re.S)
    diag_match = re.search(r"(初步诊断\s*[：:].*)", report_text, flags=re.S)

    desc_line = desc_match.group(1).strip() if desc_match else report_text.strip()
    diag_line = diag_match.group(1).strip() if diag_match else ""
    return desc_line, diag_line


def split_diagnosis_labels(diagnosis_text: str) -> list[str]:
    diagnosis_text = (diagnosis_text or "").strip()
    if not diagnosis_text:
        return []
    return [item.strip() for item in diagnosis_text.split("、") if item.strip()]


def build_diagnosis_label_vocab(rows: list[dict[str, Any]]) -> list[str]:
    labels = sorted({label for row in rows for label in split_diagnosis_labels(str(row.get("diagnosis", "")))})
    return labels


def diagnosis_to_multihot(
    diagnosis_text: str,
    label_to_idx: dict[str, int],
) -> torch.Tensor:
    target = torch.zeros(len(label_to_idx), dtype=torch.float32)
    for label in split_diagnosis_labels(diagnosis_text):
        if label in label_to_idx:
            target[label_to_idx[label]] = 1.0
    return target


def build_image_index(images_root: str | Path, recursive: bool = True) -> dict[str, str]:
    images_root = Path(images_root)
    if not images_root.exists():
        raise FileNotFoundError(f"images_root does not exist: {images_root}")

    pattern = "**/*" if recursive else "*"
    image_index: dict[str, str] = {}
    duplicate_stems = Counter()

    for path in images_root.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in image_index:
            duplicate_stems[stem] += 1
            continue
        image_index[stem] = str(path.resolve())

    if not image_index:
        raise RuntimeError(f"No image files were found under {images_root}")

    if duplicate_stems:
        preview = ", ".join(list(duplicate_stems.keys())[:10])
        print(f"[warn] duplicate image stems found, first match kept: {preview}")
    return image_index


def resolve_image_path(
    row: dict[str, Any],
    csv_dir: Path,
    image_index: dict[str, str] | None = None,
) -> str | None:
    raw_img_path = (row.get("img_path") or "").strip()
    if raw_img_path:
        candidate = Path(raw_img_path)
        if candidate.exists():
            return str(candidate.resolve())

        if not candidate.is_absolute():
            relative_candidate = (csv_dir / candidate).resolve()
            if relative_candidate.exists():
                return str(relative_candidate)

    img_id = (row.get("img_id") or "").strip()
    if image_index and img_id:
        return image_index.get(img_id)
    return None


def load_records(
    csv_path: str | Path,
    prompt_path: str | Path,
    images_root: str | Path | None = None,
    recursive_images: bool = True,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = load_prompt(prompt_path)
    csv_path = Path(csv_path)
    csv_dir = csv_path.parent

    image_index = None
    if images_root:
        image_index = build_image_index(images_root, recursive=recursive_images)

    records: list[dict[str, Any]] = []
    missing_images = 0
    empty_reports = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img_id = (row.get("img_id") or "").strip()
            description = (row.get("description") or "").strip()
            if not description:
                empty_reports += 1
                continue

            image_path = resolve_image_path(row, csv_dir=csv_dir, image_index=image_index)
            if image_path is None:
                missing_images += 1
                continue

            record = {
                "img_id": img_id,
                "image_path": image_path,
                "prompt": prompt,
                "answer": description,
                "institution_name": (row.get("institution_name") or "").strip(),
                "diagnosis": extract_diagnosis(description),
            }
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break

    stats = {
        "total_loaded": len(records),
        "missing_images": missing_images,
        "empty_reports": empty_reports,
        "num_diagnoses": len({record["diagnosis"] for record in records}),
        "using_csv_img_path": True,
    }
    return records, stats


def _allocate_counts(group_size: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if group_size <= 1:
        return group_size, 0, 0

    val_count = int(round(group_size * val_ratio)) if val_ratio > 0 else 0
    test_count = int(round(group_size * test_ratio)) if test_ratio > 0 else 0

    if group_size >= 10 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if group_size >= 10 and test_ratio > 0 and test_count == 0:
        test_count = 1

    while val_count + test_count > group_size - 1:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    train_count = group_size - val_count - test_count
    return train_count, val_count, test_count


def stratified_split(
    records: list[dict[str, Any]],
    val_ratio: float = 0.02,
    test_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1")

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["diagnosis"]].append(record)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for diagnosis, group in grouped.items():
        rng.shuffle(group)
        train_count, val_count, test_count = _allocate_counts(len(group), val_ratio, test_ratio)

        train_rows.extend(group[:train_count])
        val_rows.extend(group[train_count : train_count + val_count])
        test_rows.extend(group[train_count + val_count : train_count + val_count + test_count])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_split_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["img_id", "image_path", "answer", "institution_name", "diagnosis"]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_split_csv(split_path: str | Path, prompt_path: str | Path | None = None) -> list[dict[str, Any]]:
    with Path(split_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if prompt_path is not None:
        prompt = load_prompt(prompt_path)
        for row in rows:
            row["prompt"] = prompt
    return rows


def dump_split_summary(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def top_diagnoses(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
        return Counter(row["diagnosis"] for row in rows).most_common(20)

    summary = {
        "dataset_stats": stats,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "top_train_diagnoses": top_diagnoses(train_rows),
        "top_val_diagnoses": top_diagnoses(val_rows),
        "top_test_diagnoses": top_diagnoses(test_rows),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_messages(prompt: str, answer: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
    return messages


def apply_chat_template(processor, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }

    processor_template = getattr(processor, "chat_template", None)
    if processor_template:
        if add_generation_prompt:
            kwargs["enable_thinking"] = False
        return processor.apply_chat_template(
            messages,
            **kwargs,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    if tokenizer_template:
        if add_generation_prompt:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(
            messages,
            **kwargs,
        )

    raise ValueError(
        "Cannot render chat template because neither processor.chat_template nor tokenizer.chat_template is available."
    )


class RetinaSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class QwenVLChatCollator:
    def __init__(
        self,
        processor,
        max_image_side: int | None = None,
        diagnosis_label_to_idx: dict[str, int] | None = None,
    ):
        self.processor = processor
        self.max_image_side = max_image_side
        self.diagnosis_label_to_idx = diagnosis_label_to_idx or {}
        if getattr(self.processor.tokenizer, "pad_token", None) is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        full_texts = []
        prompt_lengths = []
        desc_lengths = []
        diag_content_start_lengths = []
        full_lengths = []
        diagnosis_targets = []

        for feature in features:
            image = load_rgb_image(feature["image_path"])
            image = resize_image_longest_side(image, self.max_image_side)
            desc_line, diag_line = split_report_sections(feature["answer"])
            desc_answer = desc_line
            full_answer = desc_line if not diag_line else f"{desc_line}\n{diag_line}"
            diagnosis_content = re.sub(r"^初步诊断\s*[：:]\s*", "", diag_line).strip()
            diag_prefix_answer = desc_line if not diag_line else f"{desc_line}\n初步诊断："

            prompt_messages = build_messages(feature["prompt"])
            desc_messages = build_messages(feature["prompt"], desc_answer)
            diag_prefix_messages = build_messages(feature["prompt"], diag_prefix_answer)
            full_messages = build_messages(feature["prompt"], full_answer)

            prompt_text = apply_chat_template(
                self.processor,
                prompt_messages,
                add_generation_prompt=True,
            )
            desc_text = apply_chat_template(
                self.processor,
                desc_messages,
                add_generation_prompt=False,
            )
            full_text = apply_chat_template(
                self.processor,
                full_messages,
                add_generation_prompt=False,
            )

            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
            desc_inputs = self.processor(
                text=[desc_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
            diag_prefix_text = apply_chat_template(
                self.processor,
                diag_prefix_messages,
                add_generation_prompt=False,
            )
            diag_prefix_inputs = self.processor(
                text=[diag_prefix_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
            full_inputs = self.processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
            prompt_lengths.append(int(prompt_inputs["input_ids"].shape[-1]))
            desc_lengths.append(int(desc_inputs["input_ids"].shape[-1]))
            diag_content_start_lengths.append(int(diag_prefix_inputs["input_ids"].shape[-1]) if diagnosis_content else int(full_inputs["input_ids"].shape[-1]))
            full_lengths.append(int(full_inputs["input_ids"].shape[-1]))
            full_texts.append(full_text)
            images.append(image)
            diagnosis_targets.append(
                diagnosis_to_multihot(str(feature.get("diagnosis", "")), self.diagnosis_label_to_idx)
            )

        batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = batch["input_ids"]
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels_desc = torch.full_like(input_ids, -100)
        labels_diag_text = torch.full_like(input_ids, -100)
        diag_token_mask = torch.zeros_like(input_ids, dtype=torch.float32)

        for row_idx, (prompt_length, desc_length, diag_content_start_length, full_length) in enumerate(
            zip(prompt_lengths, desc_lengths, diag_content_start_lengths, full_lengths, strict=True)
        ):
            labels_desc[row_idx, prompt_length:desc_length] = input_ids[row_idx, prompt_length:desc_length]
            labels_diag_text[row_idx, desc_length:full_length] = input_ids[row_idx, desc_length:full_length]
            diag_token_mask[row_idx, diag_content_start_length:full_length] = 1.0

        if pad_token_id is not None:
            pad_mask = input_ids == pad_token_id
            labels_desc[pad_mask] = -100
            labels_diag_text[pad_mask] = -100
            diag_token_mask = diag_token_mask.masked_fill(pad_mask, 0.0)

        batch["labels_desc"] = labels_desc
        batch["labels_diag_text"] = labels_diag_text
        batch["diag_token_mask"] = diag_token_mask
        batch["diagnosis_targets"] = (
            torch.stack(diagnosis_targets) if diagnosis_targets else torch.zeros((0, len(self.diagnosis_label_to_idx)))
        )
        return batch


class QwenVLGenerationCollator:
    def __init__(self, processor, max_image_side: int | None = None):
        self.processor = processor
        self.max_image_side = max_image_side
        if getattr(self.processor.tokenizer, "pad_token", None) is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        images = []
        prompt_texts = []
        metadata = []

        for feature in features:
            image = load_rgb_image(feature["image_path"])
            image = resize_image_longest_side(image, self.max_image_side)
            prompt_messages = build_messages(feature["prompt"])
            prompt_text = apply_chat_template(
                self.processor,
                prompt_messages,
                add_generation_prompt=True,
            )
            images.append(image)
            prompt_texts.append(prompt_text)
            metadata.append(feature)

        batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        # For left-padded decoder-only generation, new tokens start after the
        # full padded prompt width, not after the per-sample non-pad token count.
        batch["prompt_length"] = int(batch["input_ids"].shape[1])
        batch["metadata"] = metadata
        return batch


def masked_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != -100
    if not valid_mask.any():
        return logits.new_zeros(())

    valid_logits = shift_logits[valid_mask]
    valid_labels = shift_labels[valid_mask]
    return F.cross_entropy(valid_logits, valid_labels)


def pool_sequence_hidden_states(
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    token_mask = token_mask.to(dtype=hidden_states.dtype)
    mask_sum = token_mask.sum(dim=1, keepdim=True)
    batch_size, _, hidden_size = hidden_states.shape

    if pooling == "mean":
        safe_mask_sum = mask_sum.clamp_min(1.0)
        pooled = (hidden_states * token_mask.unsqueeze(-1)).sum(dim=1) / safe_mask_sum
        return pooled

    pooled = hidden_states.new_zeros((batch_size, hidden_size))
    for row_idx in range(batch_size):
        indices = torch.nonzero(token_mask[row_idx] > 0, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        if pooling == "first":
            pooled[row_idx] = hidden_states[row_idx, indices[0]]
        elif pooling == "last":
            pooled[row_idx] = hidden_states[row_idx, indices[-1]]
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
    return pooled


def compute_diag_metrics_from_logits(
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
) -> dict[str, float]:
    if logits is None or targets is None or logits.numel() == 0 or targets.numel() == 0:
        return {
            "diag_bce_loss": 0.0,
            "diag_exact_set_acc": 0.0,
            "diag_micro_f1": 0.0,
            "diag_macro_f1": 0.0,
        }

    targets = targets.float()
    predictions = logits > 0
    target_bools = targets > 0.5

    exact_set_acc = float((predictions == target_bools).all(dim=1).float().mean().item())
    tp = torch.logical_and(predictions, target_bools).sum().item()
    fp = torch.logical_and(predictions, ~target_bools).sum().item()
    fn = torch.logical_and(~predictions, target_bools).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    per_label_tp = torch.logical_and(predictions, target_bools).sum(dim=0).float()
    per_label_fp = torch.logical_and(predictions, ~target_bools).sum(dim=0).float()
    per_label_fn = torch.logical_and(~predictions, target_bools).sum(dim=0).float()
    per_label_precision = per_label_tp / (per_label_tp + per_label_fp).clamp_min(1.0)
    per_label_recall = per_label_tp / (per_label_tp + per_label_fn).clamp_min(1.0)
    per_label_denominator = per_label_precision + per_label_recall
    per_label_f1 = torch.where(
        per_label_denominator > 0,
        2 * per_label_precision * per_label_recall / per_label_denominator,
        torch.zeros_like(per_label_denominator),
    )
    supported_labels = target_bools.sum(dim=0) > 0
    macro_f1 = (
        float(per_label_f1[supported_labels].mean().item())
        if supported_labels.any()
        else 0.0
    )
    diag_bce_loss = float(F.binary_cross_entropy_with_logits(logits, targets).item())

    return {
        "diag_bce_loss": diag_bce_loss,
        "diag_exact_set_acc": exact_set_acc,
        "diag_micro_f1": micro_f1,
        "diag_macro_f1": macro_f1,
    }
