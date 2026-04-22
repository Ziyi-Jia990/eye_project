from __future__ import annotations

import argparse
import json

from generation_eval import run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen VL retinal report generator.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--prompt-path", type=str, default="/mnt/hdd/jiazy/eye_project/SFT/prompt.txt")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Resize images so their longest side is at most this value before processor tokenization. Set <=0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_cli(
        model_name_or_path=args.model_name_or_path,
        split_csv=args.split_csv,
        prompt_path=args.prompt_path,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_image_side=args.max_image_side if args.max_image_side and args.max_image_side > 0 else None,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
