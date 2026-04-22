这是一份 **详细执行计划**。目标是：基于你现有工程，构造 DPO 训练数据生成流程，并且按你的要求修改 prompt、train.csv 路径和 evaluator 的格式检查逻辑。

---

# 任务目标

请在当前项目中新增一套**从 `train.csv` 生成 DPO 偏好数据**的流程，核心是：

1. 使用固定 prompt 对 `train.csv` 中的样本做离线推理，得到 stage1 模型输出。
2. 用现有 evaluator 对输出进行单样本打分。
3. 根据打分筛选有价值的样本，构造成 DPO 数据：

   * `prompt`
   * `chosen`
   * `rejected`
4. `chosen` 默认使用 GT 报告，`rejected` 使用 stage1 模型输出。
5. evaluator 中新增“**格式正确性检查**”，并将其纳入综合评分，且其重要性：

   * **低于描述相关**
   * **高于诊断相关**

---

# 已知固定路径

请严格使用下面路径，不要让我再手动改：

* prompt 文件：
  `/mnt/hdd/jiazy/eye_project/SFT/prompt.txt`

* 训练集：
  `/mnt/hdd/jiazy/eye_project/SFT/splits_qc_clean/train.csv`

---

# 总体执行思路

请分成下面 5 个部分实现：

## Part A. 梳理现有代码并复用已有能力

先阅读并确认以下已有模块中哪些可以直接复用：

1. 训练数据读取逻辑
2. prompt 拼接逻辑
3. stage1 模型推理逻辑
4. evaluator 的解析与评分逻辑
5. 当前报告格式定义（尤其是：

   * `描述：...`
   * `初步诊断：...`

目标是尽量复用现有项目里的：

* prompt 读取
* 样本解析
* GT 报告解析
* evaluator 的字段抽取与指标计算

不要自己重新发明一套 incompatible 的格式。

---

## Part B. 新增“离线生成 + 评分”脚本

请新增一个脚本，建议命名类似：

```text
/mnt/hdd/jiazy/eye_project/RL/data/build_dpo_candidates.py
```

这个脚本的职责是：

### 输入

* `train.csv`
* `prompt.txt`
* stage1 最佳模型 / adapter 路径（/mnt/hdd/jiazy/eye_project/SFT/outputs/qwen_retina_lora_stage1_plain_eval_loss）
* 输出路径（/mnt/hdd/jiazy/eye_project/RL/data）

### 处理流程

对 `train.csv` 中每个样本：

1. 读取图像和 GT 报告
2. 用 `prompt.txt` 构造输入 prompt
3. 用 stage1 模型生成一个回答
4. 用 evaluator 对“模型回答 vs GT”做单样本评分
5. 保存一条中间结果记录

### 中间结果建议保存字段

建议输出为 jsonl 或 csv，至少包含：

* `img_id`
* `image_path`
* `prompt`
* `ground_truth`
* `prediction`
* `format_correct`
* `description_exact_match`
* `description_finding_set_f1`
* `description_location_f1`
* `description_count_bucket_acc`
* `description_cdr_mae`
* `description_cdr_tol_acc`
* `diagnosis_exact_set_acc`
* `diagnosis_f1` 或可等价的单样本诊断分数
* `contradiction`（如果已有）
* `final_score`

如果现有 evaluator 没有“单样本综合分”，就在这个脚本里根据 evaluator 的返回字段再计算一个 `final_score`。

---

## Part C. 修改 evaluator：新增“格式正确性检查”

这是你特别要求的部分。

需要增加一个**单样本格式正确性检查**，建议暴露成类似：

```python
format_correct: bool 或 0/1
```

### 格式正确性的定义

至少检查以下内容：

1. 输出是否包含且仅包含两行主结构（或可容忍少量空白行）：

   * 第一部分以 `描述：` 开头
   * 第二部分以 `初步诊断：` 开头

2. 顺序必须正确：

   * `描述：` 在前
   * `初步诊断：` 在后

3. 不允许缺失这两个字段名中的任意一个

4. 允许做适度的空白 / 换行归一化后再判断

### 注意

这个“格式正确性检查”不是替代原有的 description / diagnosis 指标，而是**新增一个额外评分项**。

---

## Part D. 定义综合分数 `final_score`

请在 DPO 样本构造阶段定义一个单样本综合分数，用于筛选和比较 chosen / rejected。

你必须满足这个优先级要求：

> **描述相关 > 格式正确性 > 诊断相关**

也就是：

* 描述最重要
* 格式次之
* 诊断再次之

### 推荐实现方式

请直接实现一个清晰可调的加权公式，例如：

[
\text{final_score}
==================

w_{desc}\cdot S_{desc}
+
w_{format}\cdot S_{format}
+
w_{diag}\cdot S_{diag}
----------------------

w_{penalty}\cdot S_{penalty}
]

并按下面原则设置：

### 1. 描述分数 `S_desc`

可由以下项组成：

* `description_finding_set_f1`
* `description_location_f1`
* `description_count_bucket_acc`
* `description_cdr_tol_acc`
* 或其他已有描述类指标

### 2. 格式分数 `S_format`

直接来自新增的：

* `format_correct`，布尔或 0/1

### 3. 诊断分数 `S_diag`

可由以下项组成：

* `diagnosis_exact_set_acc`

### 4. penalty 项

如果 evaluator 已有 contradiction 或明显错误惩罚项，可以减掉

---

### 权重要求

请把权重写成脚本顶部可配置常量，并满足：

```text
描述总权重 > 格式权重 > 诊断总权重
```

例如可以先设成类似：

* 描述：0.6
* 格式：0.25
* 诊断：0.15

这只是示例，Codex 可以根据现有 evaluator 返回字段做更合理拆分，但必须满足这个优先级。

---

## Part E. 新增 DPO 数据构造脚本

请再新增一个脚本，建议命名：

```text
/mnt/hdd/jiazy/eye_project/RL/data/build_dpo_dataset.py
```

### 输入

读取 Part B 的中间结果文件。

### 处理逻辑

#### 1. 只保留“有学习价值”的样本

请按综合分数和关键错误进行筛选，优先保留：

* 诊断 exact set 错的样本
* 多标签漏检样本
* 描述分数较低的样本
* 格式错误样本

不要把全部 train.csv 都做成 DPO 数据。

#### 2. 构造 chosen / rejected

第一版请这样实现：

* `chosen = ground_truth`
* `rejected = prediction`

#### 3. 过滤掉“差异太小”的样本

请加入一个 margin 过滤机制，例如：

```text
margin = score(chosen) - score(rejected)
```

如果 GT 没有直接算分，也可以把 GT 视为满分模板答案，或单独调用 evaluator 计算 GT 自身分数。

只保留：

* `margin >= threshold`

把 threshold 做成命令行参数或脚本常量。

#### 4. 输出标准 DPO 格式

输出 JSONL，每条格式如下：

```json
{
  "img_id": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "score_chosen": ...,
  "score_rejected": ...,
  "margin": ...
}
```

---

# 关键实现细节要求

## 1. prompt 必须来自固定文件

不要把 prompt 写死在代码里。必须从：

```text
/mnt/hdd/jiazy/eye_project/SFT/prompt.txt
```

读取。

如果项目里已有 prompt loader，优先复用。

---

## 2. train.csv 必须来自固定路径

默认输入应直接指向：

```text
/mnt/hdd/jiazy/eye_project/SFT/splits_qc_clean/train.csv
```

可以允许命令行覆盖，但默认值必须是这个路径。

---

## 3. evaluator 修改要兼容旧逻辑
不要修改 SFT/metrics.py；请参考其逻辑，在 RL/data/metrics.py 中重新实现 RL/DPO 所需的单样本评测接口与格式检查。

新增 `format_correct` 检查时：

* 不要破坏现有 aggregate 逻辑
* 不要影响旧训练/评测脚本除非显式使用新字段
* 尽量把新逻辑做成增量字段，而不是重写旧接口

---

## 4. 尽量支持单样本打分函数

如果现在 evaluator 只有 aggregate 级别接口，请补一个单样本接口，类似：

```python
score_report(pred_text, ref_text) -> dict
```

返回单样本所有明细字段，供 DPO 构造脚本调用。

---

# 期望新增文件

建议至少新增以下文件：

1. `RL/data/build_dpo_candidates.py`

   * 跑 stage1 模型生成预测
   * 调 evaluator
   * 产出中间评分文件

2. `RL/data/build_dpo_dataset.py`

   * 从中间评分文件构造 DPO JSONL

如果需要，也可以新增一个小工具模块，例如：

3. `RL/data/dpo_utils.py`

   * 综合分数计算
   * margin 判断
   * 样本筛选逻辑


---

# 输出与验证要求

完成后，请确保能做到以下验证：

## 验证 1：单样本 evaluator 返回新增字段

对于任意一条预测，能返回：

* `format_correct`
* 描述分数相关字段
* 诊断分数相关字段
* `final_score`

## 验证 2：能生成中间结果文件

运行 `build_dpo_candidates.py` 后，能得到一份包含：

* GT
* prediction
* 各项分数
* final_score

的文件。

## 验证 3：能生成 DPO 数据

运行 `build_dpo_dataset.py` 后，能生成合法 JSONL：

* 每行有 `prompt/chosen/rejected`
* 并且保留 `img_id` 和 score 字段便于追踪

## 验证 4：打印统计信息

请让脚本在结束时打印：

* 总样本数
* 成功生成 prediction 的样本数
* 平均 `final_score`
* 被筛进 DPO 的样本数
* 平均 margin
* 格式错误样本数
* 诊断错误样本数

---

# 给 Codex 的执行顺序建议

请按这个顺序做，不要一上来就大改：

### 第一步

先定位并阅读现有 evaluator 与 prompt / data loading 逻辑

### 第二步

在 evaluator 中新增 `format_correct` 与单样本 `final_score`

### 第三步

实现 `build_dpo_candidates.py`

### 第四步

实现 `build_dpo_dataset.py`

### 第五步

用少量样本 dry-run，确认输出格式正确

### 第六步

再扩展到全量 `train.csv`

---

# 实现偏好

请尽量：

* 少改现有训练主流程
* 新功能以新增脚本为主
* 旧 evaluator 接口保持兼容
* 所有阈值和权重写成可配置常量或命令行参数
* 如果需要，你可以新建一个环境用以运行相关代码
