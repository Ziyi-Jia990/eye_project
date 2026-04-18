#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail

python evaluate_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --adapter-path /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/outputs/qwen_retina_lora_no_longtail \
  --split-csv /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/splits/test.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/prompt.txt \
  --output-dir /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/outputs/qwen_retina_lora_no_longtail/test_eval \
  --eval-batch-size 1 \
  --max-new-tokens 160 \
  --dtype bf16 \
  --trust-remote-code
