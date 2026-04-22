from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint

from grpo_retina_utils import (
    RewardContributionLoggingCallback,
    build_diagnosis_labels,
    build_grpo_dataset,
    build_reward_configuration,
    default_path_under_project,
    prepare_rows_with_workspace,
    project_root_from_workspace,
    remap_eye_project_path,
    resolve_workspace_root,
    save_invalid_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_ROOT = PROJECT_ROOT / "SFT"

import sys

if str(SFT_ROOT) not in sys.path:
    sys.path.append(str(SFT_ROOT))

from generation_eval import load_model_and_processor, resolve_diagnosis_label_count, resolve_processor_source, run_generation_eval  # noqa: E402
from train_qwen_vl_sft import (  # noqa: E402
    attach_diagnosis_head,
    initialize_lora_from_existing_adapter,
    load_processor_with_hint,
    resolve_dtype,
    resolve_lora_target_modules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a retinal-report GRPO policy on the existing LoRA backbone.")
    parser.add_argument("--workspace-root", type=str, default="")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--adapter-init-path", type=str, default="")
    parser.add_argument("--prompt-path", type=str, default="")
    parser.add_argument("--train-split", type=str, default="")
    parser.add_argument("--val-split", type=str, default="")
    parser.add_argument("--test-split", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--logging-dir", type=str, default="")
    parser.add_argument("--resume-from-checkpoint", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-image-validation", action="store_true")
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Resize images so their longest side is at most this value before processor tokenization. Set <=0 to disable.",
    )

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--report-to", type=str, default="none")

    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-completion-length", type=int, default=160)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--log-completions", action="store_true")

    parser.add_argument(
        "--reward-mode",
        type=str,
        default="",
        help="Comma-separated reward modes: diagnosis, format, description, or all. "
        "If omitted, smoke runs default to diagnosis-only and full runs default to all.",
    )
    parser.add_argument("--reward-weight-format", type=float, default=0.15)
    parser.add_argument("--reward-weight-description", type=float, default=0.25)
    parser.add_argument("--reward-weight-diagnosis", type=float, default=0.6)

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

    parser.add_argument("--do-eval-after-train", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-max-new-tokens", type=int, default=160)
    parser.add_argument("--eval-num-beams", type=int, default=1)
    return parser.parse_args()


def load_grpo_dependencies():
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Failed to import `GRPOConfig` / `GRPOTrainer` from `trl`. "
            "Please make sure the active training environment has a working TRL installation and its "
            "runtime dependencies. In the current qwen_vl environment, TRL additionally required `pandas` "
            "at import time."
        ) from exc
    return GRPOConfig, GRPOTrainer


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    workspace_root = resolve_workspace_root(args.workspace_root or None)
    args.workspace_root = str(workspace_root)

    if not args.prompt_path:
        args.prompt_path = default_path_under_project(workspace_root, "SFT", "prompt.txt")
    else:
        args.prompt_path = remap_eye_project_path(args.prompt_path, workspace_root)

    if not args.train_split:
        args.train_split = default_path_under_project(workspace_root, "SFT", "splits_qc_clean", "train.csv")
    else:
        args.train_split = remap_eye_project_path(args.train_split, workspace_root)

    if not args.val_split:
        args.val_split = default_path_under_project(workspace_root, "SFT", "splits_qc_clean", "val.csv")
    else:
        args.val_split = remap_eye_project_path(args.val_split, workspace_root)

    if not args.test_split:
        args.test_split = default_path_under_project(workspace_root, "SFT", "splits_qc_clean", "test.csv")
    else:
        args.test_split = remap_eye_project_path(args.test_split, workspace_root)

    if not args.adapter_init_path:
        args.adapter_init_path = default_path_under_project(
            workspace_root,
            "SFT",
            "outputs",
            "qwen_retina_lora_stage1_plain_eval_loss",
        )
    else:
        args.adapter_init_path = remap_eye_project_path(args.adapter_init_path, workspace_root)

    if not args.output_dir:
        args.output_dir = default_path_under_project(workspace_root, "RL", "outputs", "qwen_retina_grpo")
    else:
        args.output_dir = str(Path(args.output_dir).expanduser().resolve())

    if not args.logging_dir:
        args.logging_dir = str((Path(args.output_dir) / "logs").resolve())
    else:
        args.logging_dir = str(Path(args.logging_dir).expanduser().resolve())

    if args.bf16 and args.fp16:
        raise ValueError("Please enable only one of `--bf16` or `--fp16`.")

    if not args.reward_mode:
        args.reward_mode = "diagnosis" if args.smoke_run else "all"

    if args.smoke_run:
        if args.num_generations is None:
            args.num_generations = 4
        if args.max_steps < 0:
            args.max_steps = 2
        if args.max_train_samples is None:
            args.max_train_samples = 16
        if args.max_val_samples is None:
            args.max_val_samples = 8
        if args.max_test_samples is None:
            args.max_test_samples = 8
        args.logging_steps = 1
        args.save_steps = max(1, min(args.save_steps, args.max_steps if args.max_steps > 0 else 1))
    elif args.num_generations is None:
        args.num_generations = 6

    if args.num_generations < 2:
        raise ValueError("`--num-generations` must be at least 2 for GRPO.")

    if args.max_image_side is not None and args.max_image_side <= 0:
        args.max_image_side = None

    return args


def build_model_and_processor(args: argparse.Namespace, num_diagnosis_labels: int):
    torch_dtype = resolve_dtype(args)
    processor_source = resolve_processor_source(args.model_name_or_path, args.adapter_init_path or None)
    processor = load_processor_with_hint(
        processor_source,
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


def build_grpo_config(
    args: argparse.Namespace,
    reward_weights: list[float],
):
    GRPOConfig, _ = load_grpo_dependencies()
    report_to = [] if args.report_to == "none" else [args.report_to]
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "logging_dir": args.logging_dir,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "max_grad_norm": args.max_grad_norm,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "report_to": report_to,
        "seed": args.seed,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "generation_batch_size": args.generation_batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "beta": args.beta,
        "reward_weights": reward_weights,
        "generation_kwargs": {"do_sample": True},
        "log_completions": args.log_completions,
    }

    signature = inspect.signature(GRPOConfig.__init__)
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return GRPOConfig(**supported_kwargs)


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: str) -> str | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "last":
        checkpoint = get_last_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                f"`--resume-from-checkpoint last` was set, but no checkpoint was found under {output_dir}"
            )
        return checkpoint

    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    return str(checkpoint_path.resolve())


def build_generation_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        eval_rows.append(
            {
                "img_id": row.get("img_id", ""),
                "image_path": row["image_path"],
                "prompt": row["prompt_text"],
                "answer": row["ground_truth"],
                "diagnosis": row.get("diagnosis", ""),
            }
        )
    return eval_rows


def run_post_train_eval(
    args: argparse.Namespace,
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    eval_root = Path(args.output_dir) / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.output_dir,
        trust_remote_code=args.trust_remote_code,
        dtype="bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
    )
    model = model.cuda() if torch.cuda.is_available() else model

    if val_rows:
        val_metrics = run_generation_eval(
            model=model,
            processor=processor,
            rows=build_generation_eval_rows(val_rows),
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
            num_beams=args.eval_num_beams,
            output_dir=eval_root,
            split_name="val",
            max_image_side=args.max_image_side,
        )
        print(json.dumps({"val_metrics": val_metrics}, ensure_ascii=False, indent=2))

    if test_rows:
        test_metrics = run_generation_eval(
            model=model,
            processor=processor,
            rows=build_generation_eval_rows(test_rows),
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
            num_beams=args.eval_num_beams,
            output_dir=eval_root,
            split_name="test",
            max_image_side=args.max_image_side,
        )
        print(json.dumps({"test_metrics": test_metrics}, ensure_ascii=False, indent=2))


def main() -> None:
    args = normalize_args(parse_args())
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    is_rank_zero = rank == 0

    reward_funcs, reward_weights, reward_name_to_weight = build_reward_configuration(
        reward_mode=args.reward_mode,
        format_weight=args.reward_weight_format,
        description_weight=args.reward_weight_description,
        diagnosis_weight=args.reward_weight_diagnosis,
    )

    print(
        "[info] Reward setup: "
        f"mode={args.reward_mode}, num_generations={args.num_generations}, beta={args.beta}, "
        f"weights={reward_name_to_weight}"
    )
    print(
        "[info] Generation sampling: "
        f"do_sample=True, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
        f"repetition_penalty={args.repetition_penalty}"
    )

    train_rows, train_invalid = prepare_rows_with_workspace(
        split_path=args.train_split,
        prompt_path=args.prompt_path,
        workspace_root=args.workspace_root,
        max_samples=args.max_train_samples,
        validate_images=not args.skip_image_validation,
    )
    val_rows, val_invalid = prepare_rows_with_workspace(
        split_path=args.val_split,
        prompt_path=args.prompt_path,
        workspace_root=args.workspace_root,
        max_samples=args.max_val_samples,
        validate_images=not args.skip_image_validation,
    )
    test_rows, test_invalid = prepare_rows_with_workspace(
        split_path=args.test_split,
        prompt_path=args.prompt_path,
        workspace_root=args.workspace_root,
        max_samples=args.max_test_samples,
        validate_images=not args.skip_image_validation,
    )

    invalid_rows = train_invalid + val_invalid + test_invalid
    if invalid_rows and is_rank_zero:
        invalid_path = Path(args.output_dir) / "invalid_images.json"
        save_invalid_rows(invalid_path, invalid_rows)
        print(
            f"[warn] Filtered {len(invalid_rows)} unreadable images before GRPO training. "
            f"Details written to {invalid_path}"
        )

    diagnosis_labels = build_diagnosis_labels(train_rows + val_rows + test_rows)
    adapter_label_count = resolve_diagnosis_label_count(args.adapter_init_path or None)
    num_diagnosis_labels = max(len(diagnosis_labels), adapter_label_count)

    model, processor = build_model_and_processor(args, num_diagnosis_labels=num_diagnosis_labels)
    train_dataset = build_grpo_dataset(train_rows, max_image_side=args.max_image_side)

    _, GRPOTrainer = load_grpo_dependencies()
    training_args = build_grpo_config(args, reward_weights=reward_weights)

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[RewardContributionLoggingCallback(reward_name_to_weight)],
    )

    config_snapshot = {
        "workspace_root": args.workspace_root,
        "project_root": str(project_root_from_workspace(args.workspace_root)),
        "model_name_or_path": args.model_name_or_path,
        "adapter_init_path": args.adapter_init_path,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "reward_mode": args.reward_mode,
        "reward_name_to_weight": reward_name_to_weight,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "beta": args.beta,
        "do_sample": True,
        "smoke_run": args.smoke_run,
        "max_image_side": args.max_image_side,
    }
    if is_rank_zero:
        (Path(args.output_dir) / "grpo_run_config.json").write_text(
            json.dumps(config_snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_checkpoint:
        print(f"[info] Resuming GRPO training from checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)

    if args.do_eval_after_train and trainer.is_world_process_zero():
        run_post_train_eval(
            args=args,
            val_rows=val_rows,
            test_rows=test_rows,
        )


if __name__ == "__main__":
    main()
