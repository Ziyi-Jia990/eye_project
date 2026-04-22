from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.distributed.elastic.multiprocessing.errors import record
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from generation_eval import run_generation_eval
from retina_sft_utils import (
    QwenVLChatCollator,
    RetinaSFTDataset,
    build_diagnosis_label_vocab,
    compute_diag_metrics_from_logits,
    diagnosis_to_multihot,
    dump_split_summary,
    filter_valid_rows,
    load_records,
    masked_token_cross_entropy,
    pool_sequence_hidden_states,
    read_split_csv,
    stratified_split,
    write_split_csv,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.hub import PushToHubMixin


VISION_LORA_BLOCK_SUFFIXES = (
    "attn.qkv",
    "attn.proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
)
STRUCTURED_EVAL_METRIC_NAMES = {
    "eval_format_correct_rate",
    "eval_diagnosis_exact_set_acc",
    "eval_diagnosis_micro_f1",
    "eval_diagnosis_macro_f1",
    "eval_diagnosis_family_level_acc",
    "eval_description_exact_match",
    "eval_description_finding_set_f1",
    "eval_description_location_f1",
    "eval_description_count_bucket_acc",
    "eval_description_cdr_mae",
    "eval_description_cdr_tol_acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT a Qwen VL model for retinal fundus report generation.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--data-csv", type=str, default="/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.csv")
    parser.add_argument("--prompt-path", type=str, default="/mnt/hdd/jiazy/eye_project/SFT/prompt.txt")
    parser.add_argument("--images-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="/mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora")
    parser.add_argument("--splits-dir", type=str, default="/mnt/hdd/jiazy/eye_project/SFT/splits")
    parser.add_argument("--train-split", type=str, default="")
    parser.add_argument("--val-split", type=str, default="")
    parser.add_argument("--test-split", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--recursive-images", action="store_true")
    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-image-validation", action="store_true")
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Resize images so their longest side is at most this value before processor tokenization. Set <=0 to disable.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Checkpoint path to resume from, or 'last' to auto-pick the latest checkpoint under output-dir.",
    )
    parser.add_argument(
        "--adapter-init-path",
        type=str,
        default="",
        help="Load an existing LoRA adapter as initialization, but start a fresh optimizer/scheduler state.",
    )

    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--report-to", type=str, default="none")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--lora-vision-merger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to add LoRA to the vision-language merger / connector layers.",
    )
    parser.add_argument(
        "--lora-vision-num-blocks",
        type=int,
        default=4,
        help="Apply LoRA to the last N vision blocks. Set to 0 to disable vision block LoRA.",
    )
    parser.add_argument(
        "--loss-weight-desc-ce",
        type=float,
        default=1.0,
        help="Weight for the description token cross-entropy loss.",
    )
    parser.add_argument(
        "--loss-weight-diag-text-ce",
        type=float,
        default=0.8,
        help="Weight for the diagnosis text token cross-entropy loss.",
    )
    parser.add_argument(
        "--loss-weight-diag-cb-focal",
        type=float,
        default=1.0,
        help="Weight for the diagnosis class-balanced focal BCE loss.",
    )
    parser.add_argument(
        "--loss-weight-diag-bce",
        type=float,
        default=None,
        help="Deprecated alias for --loss-weight-diag-cb-focal. If set, it overrides the new argument.",
    )
    parser.add_argument(
        "--diag-cb-beta",
        type=float,
        default=0.999,
        help="Beta used to compute class-balanced weights from training-set label frequency.",
    )
    parser.add_argument(
        "--diag-focal-gamma",
        type=float,
        default=1.5,
        help="Gamma used in the diagnosis focal BCE loss.",
    )
    parser.add_argument(
        "--diag-pooling",
        type=str,
        choices=("mean", "first", "last"),
        default="mean",
        help="Pooling method used over diagnosis tokens before the lightweight BCE head.",
    )
    parser.add_argument(
        "--metric-for-best-model",
        type=str,
        default="eval_loss",
        choices=(
            "eval_loss",
            "eval_diag_bce_loss",
            "eval_diag_exact_set_acc",
            "eval_diag_micro_f1",
            "eval_diag_macro_f1",
            "eval_format_correct_rate",
            "eval_diagnosis_exact_set_acc",
            "eval_diagnosis_micro_f1",
            "eval_diagnosis_macro_f1",
            "eval_diagnosis_family_level_acc",
            "eval_description_exact_match",
            "eval_description_finding_set_f1",
            "eval_description_location_f1",
            "eval_description_count_bucket_acc",
            "eval_description_cdr_mae",
            "eval_description_cdr_tol_acc",
        ),
        help="Validation metric used for best-checkpoint tracking when load_best_model_at_end is enabled.",
    )
    parser.add_argument(
        "--structured-eval-during-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run generation-based structured validation during training. Defaults to enabled when metric-for-best-model uses a structured metric.",
    )
    parser.add_argument(
        "--structured-eval-max-samples",
        type=int,
        default=None,
        help="Optionally limit the number of validation samples used for generation-based structured evaluation during training.",
    )

    parser.add_argument("--do-eval-after-train", action="store_true")
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-num-beams", type=int, default=1)
    args = parser.parse_args()
    if args.loss_weight_diag_bce is not None:
        args.loss_weight_diag_cb_focal = args.loss_weight_diag_bce
    return args


def prepare_splits(args: argparse.Namespace) -> tuple[str, str, str]:
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_split = args.train_split or str(splits_dir / "train.csv")
    val_split = args.val_split or str(splits_dir / "val.csv")
    test_split = args.test_split or str(splits_dir / "test.csv")

    split_paths = [Path(train_split), Path(val_split), Path(test_split)]
    if all(path.exists() for path in split_paths) and not args.overwrite_splits:
        print("[info] Reusing existing split CSV files.")
        return train_split, val_split, test_split

    records, stats = load_records(
        csv_path=args.data_csv,
        prompt_path=args.prompt_path,
        images_root=args.images_root,
        recursive_images=args.recursive_images,
        max_samples=args.max_samples,
    )

    train_rows, val_rows, test_rows = stratified_split(
        records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    write_split_csv(train_rows, train_split)
    write_split_csv(val_rows, val_split)
    write_split_csv(test_rows, test_split)
    dump_split_summary(
        train_rows,
        val_rows,
        test_rows,
        stats,
        splits_dir / "split_summary.json",
    )
    print(json.dumps({"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}, ensure_ascii=False))
    return train_split, val_split, test_split


def resolve_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def load_processor_with_hint(model_name_or_path: str, trust_remote_code: bool):
    try:
        return AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except ImportError as exc:
        error_text = str(exc)
        if "ReasoningEffort" in error_text:
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
        if "Torchvision library" in error_text and (
            "Qwen3VLVideoProcessor" in error_text or "BaseVideoProcessor" in error_text
        ):
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
            from transformers.utils.dummy_torchvision_objects import BaseVideoProcessor

            class ImageOnlyPlaceholderVideoProcessor(BaseVideoProcessor, PushToHubMixin):
                _auto_class = None

                def to_dict(self) -> dict[str, str]:
                    return {"video_processor_type": "BaseVideoProcessor"}

            print(
                "[warn] Torchvision is unavailable; falling back to an image-only Qwen3VL processor. "
                "This is expected for fundus-image training without video inputs."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            image_processor = Qwen2VLImageProcessorPil.from_pretrained(
                model_name_or_path,
            )
            video_processor = object.__new__(ImageOnlyPlaceholderVideoProcessor)
            return Qwen3VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                chat_template=getattr(tokenizer, "chat_template", None),
            )
        raise


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


def get_diag_head_module(model):
    base_model = get_base_model_module(model)
    diag_head = getattr(base_model, "diag_head", None)
    if diag_head is not None:
        return diag_head

    for name, module in base_model.named_modules():
        if name.endswith("diag_head"):
            return module
    raise AttributeError("Diagnosis head is not attached to the model.")


def get_lm_head_module(model):
    base_model = get_base_model_module(model)
    lm_head = getattr(base_model, "lm_head", None)
    if lm_head is not None:
        return lm_head
    for name, module in base_model.named_modules():
        if name.endswith("lm_head"):
            return module
    raise AttributeError("Could not locate lm_head on the current model.")


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


def resolve_lora_target_modules(model, args: argparse.Namespace) -> list[str]:
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    module_names = {name for name, _ in model.named_modules()}

    if args.lora_vision_merger:
        for module_name in (
            "model.visual.merger.linear_fc1",
            "model.visual.merger.linear_fc2",
        ):
            if module_name in module_names:
                target_modules.append(module_name)

    if args.lora_vision_num_blocks > 0:
        block_indices = sorted(
            {
                int(match.group(1))
                for module_name in module_names
                if (match := re.match(r"model\.visual\.blocks\.(\d+)\.", module_name))
            }
        )
        for block_index in block_indices[-args.lora_vision_num_blocks :]:
            for suffix in VISION_LORA_BLOCK_SUFFIXES:
                module_name = f"model.visual.blocks.{block_index}.{suffix}"
                if module_name in module_names:
                    target_modules.append(module_name)

    deduped_modules: list[str] = []
    seen = set()
    for module_name in target_modules:
        if module_name in seen:
            continue
        seen.add(module_name)
        deduped_modules.append(module_name)
    return deduped_modules


def initialize_lora_from_existing_adapter(model, adapter_path: str | Path) -> None:
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter init path does not exist: {adapter_path}")

    print(f"[info] Initializing LoRA weights from existing adapter: {adapter_path}")
    adapter_state_dict = load_peft_weights(str(adapter_path), device="cpu")
    diag_head = get_diag_head_module(model)
    fallback_state = {
        "base_model.model.diag_head.weight": diag_head.weight.detach().cpu(),
    }
    if getattr(diag_head, "bias", None) is not None:
        fallback_state["base_model.model.diag_head.bias"] = diag_head.bias.detach().cpu()

    missing_diag_keys = [key for key in fallback_state if key not in adapter_state_dict]
    for key in missing_diag_keys:
        adapter_state_dict[key] = fallback_state[key]

    if missing_diag_keys:
        print(
            "[warn] Existing adapter does not contain the new diagnosis head weights. "
            "Keeping the current initialization for: "
            + ", ".join(missing_diag_keys)
        )
    set_peft_model_state_dict(
        model,
        adapter_state_dict,
        adapter_name="default",
        ignore_mismatched_sizes=True,
    )


def save_diagnosis_label_map(output_dir: str | Path, diagnosis_labels: list[str]) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_to_idx = {label: idx for idx, label in enumerate(diagnosis_labels)}
    payload = {
        "labels": diagnosis_labels,
        "label_to_idx": label_to_idx,
    }
    (output_dir / "diagnosis_label_map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def metric_greater_is_better(metric_name: str) -> bool:
    return metric_name not in {"eval_loss", "eval_diag_bce_loss", "eval_description_cdr_mae"}


def uses_structured_eval_metric(metric_name: str) -> bool:
    return metric_name in STRUCTURED_EVAL_METRIC_NAMES


def should_run_structured_eval(args: argparse.Namespace) -> bool:
    if args.structured_eval_during_training is not None:
        return args.structured_eval_during_training
    return uses_structured_eval_metric(args.metric_for_best_model)


def select_structured_eval_rows(
    rows: list[dict[str, Any]] | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if max_samples is None or max_samples <= 0 or max_samples >= len(rows):
        return rows
    return rows[:max_samples]


def extract_scalar_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    scalar_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            scalar_metrics[f"{prefix}_{key}"] = float(value)
        elif isinstance(value, (int, float)):
            scalar_metrics[f"{prefix}_{key}"] = float(value)
    return scalar_metrics


def sync_metrics_via_file(
    metrics: dict[str, float],
    metrics_path: str | Path,
    is_source: bool,
    timeout_seconds: int = 1800,
) -> dict[str, float]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return metrics

    metrics_path = Path(metrics_path)
    if is_source:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        time.sleep(1.0)

    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for structured eval metrics file: {metrics_path}"
    )


def build_model_and_processor(args: argparse.Namespace, num_diagnosis_labels: int):
    torch_dtype = resolve_dtype(args)
    processor = load_processor_with_hint(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dtype=None if args.load_in_4bit else torch_dtype,
        quantization_config=quantization_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model = attach_diagnosis_head(model, num_diagnosis_labels)
    target_modules = resolve_lora_target_modules(model, args)
    print(f"[info] LoRA will target {len(target_modules)} modules/patterns.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=["diag_head"] if num_diagnosis_labels > 0 else None,
    )
    model = get_peft_model(model, lora_config)

    if args.adapter_init_path:
        initialize_lora_from_existing_adapter(model, args.adapter_init_path)
    model.print_trainable_parameters()
    return model, processor


def build_training_arguments(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    load_best_model_at_end = bool(has_eval)

    if load_best_model_at_end and save_steps % eval_steps != 0:
        aligned_save_steps = ((save_steps + eval_steps - 1) // eval_steps) * eval_steps
        print(
            "[warn] load_best_model_at_end=True requires save_steps to be a round multiple of eval_steps. "
            f"Auto-adjusting save_steps from {save_steps} to {aligned_save_steps} to match eval_steps={eval_steps}."
        )
        save_steps = aligned_save_steps

    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        save_strategy="steps",
        report_to=[] if args.report_to == "none" else [args.report_to],
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model if has_eval else None,
        greater_is_better=metric_greater_is_better(args.metric_for_best_model) if has_eval else None,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )

    strategy_value = "steps" if has_eval else "no"
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in training_args_signature.parameters:
        kwargs["eval_strategy"] = strategy_value
    else:
        kwargs["evaluation_strategy"] = strategy_value

    return TrainingArguments(**kwargs)


def disable_plain_dataparallel(training_args: TrainingArguments) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() > 1 and world_size == 1:
        training_args._n_gpu = 1
        print(
            "[warn] Multiple GPUs detected, but the script was launched without distributed training. "
            "Disabling Trainer DataParallel because it is unstable for this Qwen3.5 multimodal setup. "
            "This run will use a single GPU. To use multiple GPUs, launch with torchrun."
        )


def write_invalid_image_report(output_dir: str | Path, invalid_rows: list[dict[str, str]]) -> None:
    if not invalid_rows:
        return

    report_path = Path(output_dir) / "invalid_images.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(invalid_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[warn] Filtered {len(invalid_rows)} unreadable images before training. "
        f"Details written to {report_path}"
    )


def maybe_filter_invalid_rows(
    rows: list[dict[str, str]],
    split_name: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if args.skip_image_validation:
        return rows, []

    valid_rows, invalid_rows = filter_valid_rows(rows, split_name=split_name)
    if invalid_rows:
        print(
            f"[warn] {split_name}: dropped {len(invalid_rows)} unreadable images out of {len(rows)} rows."
        )
    return valid_rows, invalid_rows


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    if not args.resume_from_checkpoint:
        return None
    if args.resume_from_checkpoint == "last":
        checkpoint = get_last_checkpoint(args.output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"`--resume-from-checkpoint last` was set, but no checkpoint was found under {args.output_dir}"
            )
        return checkpoint
    if not Path(args.resume_from_checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.resume_from_checkpoint}")
    return args.resume_from_checkpoint


def build_diag_compute_metrics():
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits = eval_pred.predictions
        targets = eval_pred.label_ids
        logits_tensor = torch.as_tensor(logits) if logits is not None else None
        targets_tensor = torch.as_tensor(targets) if targets is not None else None
        return compute_diag_metrics_from_logits(logits_tensor, targets_tensor)

    return compute_metrics


def count_diagnosis_label_support(
    rows: list[dict[str, Any]],
    label_to_idx: dict[str, int],
) -> torch.Tensor:
    counts = torch.zeros(len(label_to_idx), dtype=torch.float32)
    for row in rows:
        counts += diagnosis_to_multihot(str(row.get("diagnosis", "")), label_to_idx)
    return counts


def build_diag_class_balanced_weights(
    positive_counts: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if positive_counts.numel() == 0:
        return positive_counts.clone()
    if not (0.0 < beta < 1.0):
        raise ValueError(f"`diag_cb_beta` must be in (0, 1), but got {beta}.")

    clipped_counts = positive_counts.clamp_min(1.0)
    beta_tensor = torch.full_like(clipped_counts, beta)
    effective_num = 1.0 - torch.pow(beta_tensor, clipped_counts)
    weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    weights = weights / weights.mean().clamp_min(1e-12)

    zero_positive_mask = positive_counts <= 0
    if zero_positive_mask.any():
        weights = weights.clone()
        weights[zero_positive_mask] = 1.0
    return weights


def class_balanced_focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    if gamma < 0.0:
        raise ValueError(f"`diag_focal_gamma` must be >= 0, but got {gamma}.")

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    focal_factor = torch.pow(1.0 - p_t, gamma)
    loss = bce * focal_factor
    if class_weights is not None and class_weights.numel() > 0:
        loss = loss * class_weights.view(1, -1).to(device=logits.device, dtype=logits.dtype)
    return loss.mean()


class RetinaMultiTaskTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_weight_desc_ce: float = 1.0,
        loss_weight_diag_text_ce: float = 0.8,
        loss_weight_diag_cb_focal: float = 1.0,
        diag_class_weights: torch.Tensor | None = None,
        diag_focal_gamma: float = 1.5,
        diag_pooling: str = "mean",
        processor=None,
        structured_eval_rows: list[dict[str, Any]] | None = None,
        structured_eval_batch_size: int = 1,
        structured_eval_max_new_tokens: int = 256,
        structured_eval_num_beams: int = 1,
        structured_eval_max_image_side: int | None = None,
        structured_eval_split_name: str = "val",
        enable_structured_eval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_weight_desc_ce = loss_weight_desc_ce
        self.loss_weight_diag_text_ce = loss_weight_diag_text_ce
        self.loss_weight_diag_cb_focal = loss_weight_diag_cb_focal
        self.diag_class_weights = (
            diag_class_weights.detach().clone().float()
            if diag_class_weights is not None
            else None
        )
        self.diag_focal_gamma = diag_focal_gamma
        self.diag_pooling = diag_pooling
        self.label_names = ["diagnosis_targets"]
        self.processor = processor
        self.structured_eval_rows = structured_eval_rows or []
        self.structured_eval_batch_size = structured_eval_batch_size
        self.structured_eval_max_new_tokens = structured_eval_max_new_tokens
        self.structured_eval_num_beams = structured_eval_num_beams
        self.structured_eval_max_image_side = structured_eval_max_image_side
        self.structured_eval_split_name = structured_eval_split_name
        self.enable_structured_eval = enable_structured_eval

    def _forward_multitask(
        self,
        model,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        model_inputs = dict(inputs)
        labels_desc = model_inputs.pop("labels_desc")
        labels_diag_text = model_inputs.pop("labels_diag_text")
        diag_token_mask = model_inputs.pop("diag_token_mask")
        diagnosis_targets = model_inputs.pop("diagnosis_targets").float()

        captured_hidden_states: dict[str, torch.Tensor] = {}

        def capture_lm_head_input(_module, module_inputs):
            captured_hidden_states["last_hidden_states"] = module_inputs[0]

        lm_head_handle = None
        if diagnosis_targets.shape[-1] > 0:
            lm_head_handle = get_lm_head_module(model).register_forward_pre_hook(capture_lm_head_input)

        try:
            outputs = model(**model_inputs, return_dict=True)
        finally:
            if lm_head_handle is not None:
                lm_head_handle.remove()

        loss_desc = masked_token_cross_entropy(outputs.logits, labels_desc)
        loss_diag_text = masked_token_cross_entropy(outputs.logits, labels_diag_text)

        diag_logits = None
        loss_diag_cb_focal = outputs.logits.new_zeros(())
        if diagnosis_targets.shape[-1] > 0:
            hidden_states = captured_hidden_states.get("last_hidden_states")
            if hidden_states is None:
                raise RuntimeError("Failed to capture the final hidden states before lm_head.")
            pooled_states = pool_sequence_hidden_states(
                hidden_states,
                diag_token_mask,
                pooling=self.diag_pooling,
            )
            diag_head = get_diag_head_module(model)
            diag_head_param = next(diag_head.parameters(), None)
            if diag_head_param is not None and pooled_states.dtype != diag_head_param.dtype:
                pooled_states = pooled_states.to(diag_head_param.dtype)
            diag_logits = diag_head(pooled_states)
            loss_diag_cb_focal = class_balanced_focal_bce_with_logits(
                diag_logits,
                diagnosis_targets.to(diag_logits.dtype),
                class_weights=self.diag_class_weights,
                gamma=self.diag_focal_gamma,
            )

        total_loss = (
            self.loss_weight_desc_ce * loss_desc
            + self.loss_weight_diag_text_ce * loss_diag_text
            + self.loss_weight_diag_cb_focal * loss_diag_cb_focal
        )
        return total_loss, {
            "diag_logits": diag_logits,
            "diagnosis_targets": diagnosis_targets,
        }

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        del num_items_in_batch
        loss, auxiliary_outputs = self._forward_multitask(model, inputs)
        if return_outputs:
            return loss, auxiliary_outputs
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        del ignore_keys
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = loss.detach()
        if prediction_loss_only:
            return loss, None, None

        diag_logits = outputs["diag_logits"]
        diagnosis_targets = outputs["diagnosis_targets"]
        if diag_logits is None:
            diag_logits = loss.new_zeros((diagnosis_targets.shape[0], 0))
        return loss, diag_logits.detach(), diagnosis_targets.detach()

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if not self.enable_structured_eval or self.processor is None:
            return metrics

        dataset_for_rows = eval_dataset if eval_dataset is not None else self.eval_dataset
        rows = getattr(dataset_for_rows, "rows", None)
        if rows is None:
            rows = self.structured_eval_rows
        if not rows:
            return metrics

        was_training = self.model.training
        rank_is_zero = self.is_world_process_zero()
        structured_output_dir = (
            Path(self.args.output_dir)
            / "structured_eval_during_training"
            / f"step-{self.state.global_step}"
        )
        structured_metrics_path = structured_output_dir / f"{metric_key_prefix}_metrics_rank0.json"
        if rank_is_zero:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            structured_metrics = run_generation_eval(
                model=unwrapped_model,
                processor=self.processor,
                rows=rows,
                batch_size=self.structured_eval_batch_size,
                max_new_tokens=self.structured_eval_max_new_tokens,
                num_beams=self.structured_eval_num_beams,
                output_dir=structured_output_dir,
                split_name=self.structured_eval_split_name,
                max_image_side=self.structured_eval_max_image_side,
                write_predictions=False,
            )
            structured_metrics = extract_scalar_metrics(structured_metrics, metric_key_prefix)
        else:
            structured_metrics = {}

        structured_metrics = sync_metrics_via_file(
            structured_metrics,
            metrics_path=structured_metrics_path,
            is_source=rank_is_zero,
        )
        metrics.update(structured_metrics)
        if structured_metrics:
            self.log(structured_metrics)

        if was_training:
            self.model.train()
        return metrics


@record
def main() -> None:
    args = parse_args()
    if not (0.0 < args.diag_cb_beta < 1.0):
        raise ValueError(f"`--diag-cb-beta` must be in (0, 1), but got {args.diag_cb_beta}")
    if args.diag_focal_gamma < 0.0:
        raise ValueError(f"`--diag-focal-gamma` must be >= 0, but got {args.diag_focal_gamma}")
    if args.resume_from_checkpoint and args.adapter_init_path:
        raise ValueError(
            "`--resume-from-checkpoint` restores optimizer/scheduler state, while `--adapter-init-path` "
            "starts a fresh run from existing adapter weights. Please use only one of them."
        )
    set_seed(args.seed)

    train_split, val_split, test_split = prepare_splits(args)
    if args.prepare_only:
        return

    train_rows = read_split_csv(train_split, prompt_path=args.prompt_path)
    val_rows = read_split_csv(val_split, prompt_path=args.prompt_path)
    test_rows = read_split_csv(test_split, prompt_path=args.prompt_path)

    train_rows, train_invalid = maybe_filter_invalid_rows(train_rows, "train", args)
    val_rows, val_invalid = maybe_filter_invalid_rows(val_rows, "val", args)
    test_rows, test_invalid = maybe_filter_invalid_rows(test_rows, "test", args)
    write_invalid_image_report(
        args.output_dir,
        train_invalid + val_invalid + test_invalid,
    )

    diagnosis_labels = build_diagnosis_label_vocab(train_rows + val_rows + test_rows)
    diagnosis_label_to_idx = {label: idx for idx, label in enumerate(diagnosis_labels)}
    diag_positive_counts = count_diagnosis_label_support(train_rows, diagnosis_label_to_idx)
    diag_class_weights = build_diag_class_balanced_weights(diag_positive_counts, beta=args.diag_cb_beta)
    save_diagnosis_label_map(args.output_dir, diagnosis_labels)
    structured_eval_rows = select_structured_eval_rows(val_rows, args.structured_eval_max_samples)
    zero_positive_labels = [
        diagnosis_labels[idx]
        for idx, count in enumerate(diag_positive_counts.tolist())
        if count <= 0
    ]
    if diagnosis_labels:
        print(
            "[info] Diagnosis CB-Focal config: "
            f"beta={args.diag_cb_beta}, gamma={args.diag_focal_gamma}, "
            f"weight_range=[{diag_class_weights.min().item():.4f}, {diag_class_weights.max().item():.4f}]"
        )
    if zero_positive_labels:
        print(
            "[warn] Some diagnosis labels have zero positive training samples, so their class-balanced weight "
            "falls back to 1.0: "
            + ", ".join(zero_positive_labels)
        )

    model, processor = build_model_and_processor(args, num_diagnosis_labels=len(diagnosis_labels))
    max_image_side = args.max_image_side if args.max_image_side and args.max_image_side > 0 else None
    collator = QwenVLChatCollator(
        processor,
        max_image_side=max_image_side,
        diagnosis_label_to_idx=diagnosis_label_to_idx,
    )

    training_args = build_training_arguments(args, has_eval=bool(val_rows))
    disable_plain_dataparallel(training_args)

    trainer = RetinaMultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=RetinaSFTDataset(train_rows),
        eval_dataset=RetinaSFTDataset(val_rows) if val_rows else None,
        data_collator=collator,
        compute_metrics=build_diag_compute_metrics() if val_rows else None,
        loss_weight_desc_ce=args.loss_weight_desc_ce,
        loss_weight_diag_text_ce=args.loss_weight_diag_text_ce,
        loss_weight_diag_cb_focal=args.loss_weight_diag_cb_focal,
        diag_class_weights=diag_class_weights,
        diag_focal_gamma=args.diag_focal_gamma,
        diag_pooling=args.diag_pooling,
        processor=processor,
        structured_eval_rows=structured_eval_rows,
        structured_eval_batch_size=args.per_device_eval_batch_size,
        structured_eval_max_new_tokens=args.eval_max_new_tokens,
        structured_eval_num_beams=args.eval_num_beams,
        structured_eval_max_image_side=max_image_side,
        structured_eval_split_name="val",
        enable_structured_eval=should_run_structured_eval(args),
    )

    resume_checkpoint = resolve_resume_checkpoint(args)
    if resume_checkpoint:
        print(f"[info] Resuming training from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.do_eval_after_train:
        eval_root = Path(args.output_dir) / "eval"
        model.eval()

        if val_rows:
            val_metrics = run_generation_eval(
                model=model,
                processor=processor,
                rows=val_rows,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=args.eval_max_new_tokens,
                num_beams=args.eval_num_beams,
                output_dir=eval_root,
                split_name="val",
                max_image_side=max_image_side,
            )
            print(json.dumps({"val_metrics": val_metrics}, ensure_ascii=False, indent=2))

        if test_rows:
            test_metrics = run_generation_eval(
                model=model,
                processor=processor,
                rows=test_rows,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=args.eval_max_new_tokens,
                num_beams=args.eval_num_beams,
                output_dir=eval_root,
                split_name="test",
                max_image_side=max_image_side,
            )
            print(json.dumps({"test_metrics": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
