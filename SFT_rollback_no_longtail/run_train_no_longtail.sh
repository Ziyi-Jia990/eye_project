#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail

torchrun --nproc_per_node=2 train_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --data-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/prompt.txt \
  --output-dir /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/outputs/qwen_retina_lora_no_longtail \
  --splits-dir /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/splits \
  --num-train-epochs 4 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --bf16 \
  --gradient-checkpointing \
  --max-image-side 1024 \
  --trust-remote-code \
  --save-steps 1000 \
  --eval-steps 1000 \
  --metric-for-best-model eval_diag_micro_f1 \
  --loss-weight-desc-ce 1.0 \
  --loss-weight-diag-text-ce 1.0 \
  --loss-weight-diag-bce 0.5 \
  --diag-pooling mean \
  --skip-image-validation \
  --do-eval-after-train \
  --eval-max-new-tokens 160
