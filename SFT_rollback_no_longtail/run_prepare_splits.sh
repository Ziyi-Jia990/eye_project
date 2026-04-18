#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail

python train_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --data-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/prompt.txt \
  --splits-dir /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/splits \
  --overwrite-splits \
  --prepare-only
