from __future__ import annotations

import csv
import importlib.metadata as importlib_metadata
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import transformers
from peft import PeftModel
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.utils.hub import PushToHubMixin

from metrics import (
    PREDICTION_ROW_FIELDNAMES,
    aggregate_scores,
    prediction_row_metrics,
    save_metrics,
    score_report,
)
from retina_sft_utils import QwenVLGenerationCollator, RetinaSFTDataset, filter_valid_rows, read_split_csv


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "auto": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def resolve_processor_source(model_name_or_path: str, adapter_path: str | None) -> str:
    if not adapter_path:
        return model_name_or_path

    adapter_dir = Path(adapter_path)
    candidate_dirs = [adapter_dir, *adapter_dir.parents]
    processor_markers = (
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    for directory in candidate_dirs:
        if any((directory / marker).exists() for marker in processor_markers):
            return str(directory)
    return model_name_or_path


def get_base_model_module(model):
    while hasattr(model, "module"):
        model = model.module
    return model.get_base_model() if isinstance(model, PeftModel) else model


def infer_hidden_size(model) -> int:
    base_model = get_base_model_module(model)
    candidate_configs = [
        getattr(base_model, "config", None),
        getattr(getattr(base_model, "config", None), "text_config", None),
    ]
    for config in candidate_configs:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size:
            return int(hidden_size)
    raise ValueError("Could not infer hidden size from model config.")


def resolve_diagnosis_label_count(adapter_path: str | None) -> int:
    if not adapter_path:
        return 0

    adapter_dir = Path(adapter_path)
    candidate_dirs = [adapter_dir, *adapter_dir.parents]
    for directory in candidate_dirs:
        label_map_path = directory / "diagnosis_label_map.json"
        if label_map_path.exists():
            data = json.loads(label_map_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                labels = data.get("labels")
                if isinstance(labels, list):
                    return len(labels)
                label_to_idx = data.get("label_to_idx")
                if isinstance(label_to_idx, dict):
                    return len(label_to_idx)
                return len(data)

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.endswith("diag_head.weight"):
                    shape = handle.get_slice(key).get_shape()
                    if shape:
                        return int(shape[0])
    return 0


def attach_diagnosis_head(model, num_diagnosis_labels: int):
    if num_diagnosis_labels <= 0:
        return model

    base_model = get_base_model_module(model)
    existing_head = getattr(base_model, "diag_head", None)
    hidden_size = infer_hidden_size(model)

    if (
        existing_head is not None
        and getattr(existing_head, "in_features", None) == hidden_size
        and getattr(existing_head, "out_features", None) == num_diagnosis_labels
    ):
        return model

    base_model.diag_head = torch.nn.Linear(hidden_size, num_diagnosis_labels)
    return model


def load_model_and_processor(
    model_name_or_path: str,
    adapter_path: str | None = None,
    trust_remote_code: bool = True,
    dtype: str = "bf16",
):
    torch_dtype = None if dtype == "auto" else resolve_dtype(dtype)
    processor_source = resolve_processor_source(model_name_or_path, adapter_path)
    try:
        processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=trust_remote_code,
        )
    except ImportError as exc:
        if "ReasoningEffort" in str(exc):
            try:
                mistral_common_version = importlib_metadata.version("mistral-common")
            except importlib_metadata.PackageNotFoundError:
                mistral_common_version = "not installed"

            raise ImportError(
                "Failed to load the processor because the current environment has an incompatible "
                f"`mistral-common` version ({mistral_common_version}). "
                f"Your installed transformers is {transformers.__version__}, which expects "
                "`mistral-common[image]>=1.10.0`. "
                "Please run: "
                "`/home/jiazy/miniconda3/envs/qwen_vl/bin/python -m pip install -U \"mistral-common[image]>=1.10.0\"`"
            ) from exc
        if "Torchvision library" in str(exc) and (
            "Qwen3VLVideoProcessor" in str(exc) or "BaseVideoProcessor" in str(exc)
        ):
            from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            from transformers.utils.dummy_torchvision_objects import BaseVideoProcessor

            class ImageOnlyPlaceholderVideoProcessor(BaseVideoProcessor, PushToHubMixin):
                _auto_class = None

                def to_dict(self) -> dict[str, str]:
                    return {"video_processor_type": "BaseVideoProcessor"}

            print(
                "[warn] Torchvision is unavailable; falling back to an image-only Qwen3VL processor. "
                "This is expected for fundus-image evaluation without video inputs."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                processor_source,
                trust_remote_code=trust_remote_code,
            )
            image_processor = Qwen2VLImageProcessorPil.from_pretrained(processor_source)
            video_processor = object.__new__(ImageOnlyPlaceholderVideoProcessor)
            processor = Qwen3VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                chat_template=getattr(tokenizer, "chat_template", None),
            )
        else:
            raise
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
    )
    if adapter_path:
        num_diagnosis_labels = resolve_diagnosis_label_count(adapter_path)
        model = attach_diagnosis_head(model, num_diagnosis_labels)
        model = PeftModel.from_pretrained(model, adapter_path)
    return model, processor


def run_generation_eval(
    model,
    processor,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    output_dir: str | Path,
    split_name: str,
    max_image_side: int | None = None,
    write_predictions: bool = True,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        RetinaSFTDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=QwenVLGenerationCollator(processor, max_image_side=max_image_side),
    )

    device = next(model.parameters()).device
    model.eval()

    predictions_path = output_dir / f"{split_name}_predictions.csv"
    all_scores = []

    handle_context = (
        predictions_path.open("w", encoding="utf-8", newline="")
        if write_predictions
        else nullcontext(None)
    )
    with handle_context as handle:
        writer = None
        if handle is not None:
            writer = csv.DictWriter(
                handle,
                fieldnames=["img_id", "diagnosis", "reference", "prediction", *PREDICTION_ROW_FIELDNAMES],
            )
            writer.writeheader()
        for batch in dataloader:
            metadata = batch.pop("metadata")
            prompt_length = int(batch.pop("prompt_length"))
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                )

            for idx, generated_ids in enumerate(generated):
                new_tokens = generated_ids[prompt_length:]
                prediction = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                reference = metadata[idx]["answer"]
                scores = score_report(prediction, reference)
                all_scores.append(scores)
                if writer is not None:
                    img_id = metadata[idx].get("img_id") or metadata[idx].get("\ufeffimg_id", "")
                    writer.writerow(
                        {
                            "img_id": img_id,
                            "diagnosis": metadata[idx].get("diagnosis", ""),
                            "reference": reference,
                            "prediction": prediction,
                            **prediction_row_metrics(scores),
                        }
                    )

    metrics = aggregate_scores(all_scores)
    save_metrics(metrics, output_dir / f"{split_name}_metrics.json")
    return metrics


def run_cli(
    model_name_or_path: str,
    split_csv: str,
    prompt_path: str,
    output_dir: str,
    adapter_path: str | None = None,
    eval_batch_size: int = 2,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    trust_remote_code: bool = True,
    dtype: str = "bf16",
    max_image_side: int | None = 1024,
) -> dict[str, float]:
    rows = read_split_csv(split_csv, prompt_path=prompt_path)
    rows, invalid_rows = filter_valid_rows(rows, split_name=Path(split_csv).stem)
    if invalid_rows:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "invalid_images.json").write_text(
            json.dumps(invalid_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[warn] Dropped {len(invalid_rows)} unreadable images before evaluation. "
            f"Details written to {Path(output_dir, 'invalid_images.json')}"
        )
    model, processor = load_model_and_processor(
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    model = model.cuda() if torch.cuda.is_available() else model
    metrics = run_generation_eval(
        model=model,
        processor=processor,
        rows=rows,
        batch_size=eval_batch_size,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        output_dir=output_dir,
        split_name=Path(split_csv).stem,
        max_image_side=max_image_side,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
