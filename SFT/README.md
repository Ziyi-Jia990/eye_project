# 眼底彩照阅片 SFT

这套脚本用于把 `description.csv` 中的 `img_path -> 阅片文本` 配对起来，做一个基于 Qwen 系列视觉语言模型的 SFT/LoRA 微调。

## 先说明一个关键点

如果目标是“看图生成阅片结果”，底座需要是带视觉编码器的模型，例如 Qwen-VL 系列。`Qwen/Qwen3.5-9B-Base` 这类纯文本底座本身不能直接输入眼底图像，因此不能直接训练成阅片模型。

如果你只是想把已有结构化描述再润色成规范报告，那才可以用纯文本模型。

## 文件

- `train_qwen_vl_sft.py`：训练脚本，包含数据读取、按诊断分层划分、LoRA 微调、可选生成式评估。
- `evaluate_qwen_vl_sft.py`：离线评估脚本。
- `retina_sft_utils.py`：数据集、图像索引、划分和多模态 collator。
- `metrics.py`：评估指标。

## 数据假设

CSV 现在已确认包含四列：

- `img_id`
- `description`
- `institution_name`
- `img_path`

训练脚本会优先直接读取 `img_path`。如果后续有旧版 CSV 没有这列，才会回退到 `--images-root` 下按 `img_id` 扫描匹配。

也就是说，你现在这版数据可以不传 `--images-root`。

## 评价方式

默认实现的是结构化评测指标，分为 3 层：

- 格式：`format_correct_rate`
- 初步诊断：`diagnosis_exact_set_acc`、`diagnosis_micro_f1`、`diagnosis_macro_f1`、`diagnosis_family_level_acc`
- 描述：`description_exact_match`、`description_finding_set_f1`、`description_location_f1`、`description_count_bucket_acc`、`description_cdr_mae`、`description_cdr_tol_acc`

此外还会在 `diagnosis_per_disease` 中汇报每种疾病的 `accuracy / precision / recall / f1 / support`。

这些指标是按当前 `trans.py` 的模板化生成逻辑设计的，所以比单纯的整段 EM 或字符 F1 更贴近任务本身。

如果你希望训练阶段也按这套结构化指标来挑选 best checkpoint，可以在训练时把 `--metric-for-best-model` 设成结构化指标，例如 `eval_diagnosis_micro_f1`。脚本会自动在验证阶段追加一次生成式结构化评测，并用对应的 `eval_*` 标量做 best-model 选择。

## 推荐环境

进入你的 `qwen_vl` 环境后安装依赖：

```bash
cd /mnt/hdd/jiazy/eye_project/SFT
pip install -r requirements.txt
```

如果你要做 4bit QLoRA，请确认 `bitsandbytes` 和 CUDA 可用。

如果你使用 `Qwen/Qwen3.5-9B-Base` 一类较新的 Qwen3 多模态模型，务必保证环境里有：

```bash
python -m pip install -U "mistral-common[image]>=1.10.0"
```

否则可能在 `AutoProcessor.from_pretrained(...)` 时报错：

```text
ImportError: cannot import name 'ReasoningEffort'
```

如果训练中途报错：

```text
OSError: image file is truncated
```

这通常不是模型问题，而是数据里有损坏或未完整写入的图片。当前脚本会在训练开始前先扫描 split 里的图片，把坏图过滤掉，并把详情写到：

- `/mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora/invalid_images.json`

这样就不会在第几百或几千步时因为单张坏图整轮中断。
> 后续可以直接使用 clean

如果你希望后续 split 和训练都直接使用“已经剔除坏图”的 CSV，可以先运行：

```bash
python clean_description_csv.py \
  --input-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.csv \
  --output-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --report-json /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.invalid_images.json
```

清洗后训练建议统一使用：

- `/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv`

如果你之前已经生成过 `/mnt/hdd/jiazy/eye_project/SFT/splits/*.csv`，记得在重新划分时加上 `--overwrite-splits`，这样 split 才会基于清洗后的 CSV 重建。

## 第一步：只做数据划分

```bash
python train_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --data-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT/prompt.txt \
  --splits-dir /mnt/hdd/jiazy/eye_project/SFT/splits \
  --overwrite-splits \
  --prepare-only
```

默认按诊断字段做近似分层划分：

- 训练集：96%
- 验证集：2%
- 测试集：2%

划分结果会写到：

- `/mnt/hdd/jiazy/eye_project/SFT/splits/train.csv`
- `/mnt/hdd/jiazy/eye_project/SFT/splits/val.csv`
- `/mnt/hdd/jiazy/eye_project/SFT/splits/test.csv`

## 第二步：训练

注意：当前脚本在你这种“直接用 `python train_qwen_vl_sft.py` 且机器上有多张 GPU”的场景下，会自动禁用 `Trainer` 的 `DataParallel`，改为单卡训练，因为 `Qwen/Qwen3.5-9B-Base` 在普通 `DataParallel` 路径上不稳定。

这意味着下面这条命令实际上只会使用 1 张卡，所以 `--per-device-train-batch-size` 需要按单卡显存来设置。

### 单卡稳定版

```bash
python train_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --data-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT/prompt.txt \
  --output-dir /mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora \
  --splits-dir /mnt/hdd/jiazy/eye_project/SFT/splits \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --max-image-side 1024 \
  --trust-remote-code \
  --save-steps 1000 \
  --eval-steps 1000 \
  --do-eval-after-train \
  --eval-max-new-tokens 160
```

### 双卡分布式版

如果你希望两张 A800 都参与训练，请不要直接用 `python`，而是用 `torchrun`。此时 `per-device-train-batch-size` 是“每张卡”的 batch size。

下面这组更适合两张 80G 卡，并且全局有效 batch size 仍然是 16：

```bash
torchrun --nproc_per_node=2 train_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --data-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT/prompt.txt \
  --output-dir /mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora \
  --splits-dir /mnt/hdd/jiazy/eye_project/SFT/splits \
  --num-train-epochs 2 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --max-image-side 1024 \
  --trust-remote-code \
  --save-steps 1000 \
  --eval-steps 1000 \
  --loss-weight-desc-ce 1.0 \
  --loss-weight-diag-text-ce 0.8 \
  --loss-weight-diag-cb-focal 1.0 \
  --diag-cb-beta 0.999 \
  --diag-focal-gamma 1.5 \
  --do-eval-after-train \
  --eval-max-new-tokens 160
```

## 断点恢复

训练脚本本来就会按 `--save-steps` 周期保存 checkpoint。比如你现在的输出目录里已经有：

- `/mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora/checkpoint-1000`

如果训练在中途失败，可以直接从最近的 checkpoint 继续：

```bash
torchrun --nproc_per_node=2 train_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --data-csv /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT/prompt.txt \
  --output-dir /mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora \
  --splits-dir /mnt/hdd/jiazy/eye_project/SFT/splits \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --max-image-side 1024 \
  --trust-remote-code \
  --save-steps 1000 \
  --eval-steps 1000 \
  --loss-weight-desc-ce 1.0 \
  --loss-weight-diag-text-ce 0.8 \
  --loss-weight-diag-cb-focal 1.0 \
  --diag-cb-beta 0.999 \
  --diag-focal-gamma 1.5 \
  --do-eval-after-train \
  --eval-max-new-tokens 160 \
  --resume-from-checkpoint last
```

如果你想指定某个 checkpoint，也可以把 `last` 换成具体路径，例如：

```bash
--resume-from-checkpoint /mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora/checkpoint-1000
```

如果双卡训练仍然出现显存不足，优先保留 `--per-device-train-batch-size 1`，并继续降低图像输入尺寸，例如把 `--max-image-side 1024` 改成 `896` 或 `768`。这通常比调整 `gradient-accumulation-steps` 更直接有效，因为显存峰值主要由单步前向/反向里的图像 token 决定。

## 第三步：单独评估

```bash
python evaluate_qwen_vl_sft.py \
  --model-name-or-path Qwen/Qwen3.5-9B-Base \
  --adapter-path /mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora \
  --split-csv /mnt/hdd/jiazy/eye_project/SFT/splits/test.csv \
  --prompt-path /mnt/hdd/jiazy/eye_project/SFT/prompt.txt \
  --output-dir /mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora/test_eval \
  --eval-batch-size 2 \
  --max-new-tokens 256 \
  --dtype bf16 \
  --trust-remote-code
```

评估输出包括：

- `*_predictions.csv`
- `*_metrics.json`
- `summary.json`

## 一些实用建议

- 你的 `description` 已经是很好的监督目标，建议保持两行格式，不要再做额外润色。
- 如果你更关心诊断质量，优先看 `diagnosis_exact_set_acc`、`diagnosis_micro_f1` 和 `diagnosis_family_level_acc`。
- 如果 GPU 显存紧张，优先用 `--load-in-4bit --gradient-checkpointing`。
- 如果你后面还有旧版数据没有 `img_path`，依然可以加上 `--images-root --recursive-images` 走兼容模式。
