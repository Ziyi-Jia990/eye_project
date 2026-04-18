# SFT Rollback: No Long-Tail Improvements

这个目录是一个独立的回滚版本，用来复现实验中“未针对长尾分布做改进”的 SFT/LoRA 方案。

它不会覆盖 `/mnt/hdd/jiazy/eye_project/SFT` 下的当前版本。

## 回滚范围

本版本保留：

- Qwen/Qwen3.5-9B-Base + LoRA
- 描述生成 CE loss
- 诊断文本生成 CE loss
- 诊断辅助 head
- 诊断辅助多标签 BCE loss
- 结构化生成评估
- 训练期 `eval_diag_micro_f1` / `eval_diag_macro_f1` 等 diagnosis-head 指标

本版本移除或不使用：

- TailScore 训练集增强
- tail-augmented train CSV
- 训练集过采样
- Class-Balanced Focal BCE
- `diag_cb_beta`
- `diag_focal_gamma`

## 损失函数

回滚版使用项目汇报说明中的原始多任务损失：

```text
total_loss =
1.0 * description_token_ce
+ 1.0 * diagnosis_text_token_ce
+ 0.5 * diagnosis_multilabel_bce
```

对应参数：

```text
--loss-weight-desc-ce 1.0
--loss-weight-diag-text-ce 1.0
--loss-weight-diag-bce 0.5
```

## 默认路径

`train_qwen_vl_sft.py` 在这个目录中的默认输出路径已经改为：

```text
/mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/outputs/qwen_retina_lora_no_longtail
```

默认 split 路径已经改为：

```text
/mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/splits
```

默认 prompt 路径已经改为：

```text
/mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/prompt.txt
```

## 1. 重新划分数据

如果你想严格基于原始清洗数据重新生成 train/val/test：

```bash
cd /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail

python train_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --data-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/prompt.txt \
  --splits-dir /mnt/hdd/jiazy/eye_project/SFT_rollback_no_longtail/splits \
  --overwrite-splits \
  --prepare-only
```

## 2. 双卡训练

下面命令是不使用长尾增强、不使用 CB-Focal 的回滚训练版本：

```bash
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
```

如果你想按 `eval_diag_macro_f1` 挑 best checkpoint，也可以把上面改成：

```text
--metric-for-best-model eval_diag_macro_f1
```

## 3. 单独评估

```bash
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
```

## 文件说明

- `train_qwen_vl_sft.py`: 回滚版训练脚本，使用普通 BCE 诊断辅助损失。
- `evaluate_qwen_vl_sft.py`: 回滚版离线评估入口，默认 prompt 指向本目录。
- `generation_eval.py`: 生成式结构化评估核心逻辑。
- `retina_sft_utils.py`: 数据读取、split、collator 和 diagnosis-head 指标。
- `metrics.py`: 结构化评估指标。
- `prompt.txt`: 回滚版固定 prompt 副本。
- `项目汇报说明.md`: 回滚设置依据。
