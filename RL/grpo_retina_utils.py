from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from transformers import TrainerCallback
from torch.utils.data import Dataset


CURRENT_FILE = Path(__file__).resolve()
CURRENT_PROJECT_ROOT = CURRENT_FILE.parents[1]
CURRENT_WORKSPACE_ROOT = CURRENT_PROJECT_ROOT.parent
SFT_ROOT = CURRENT_PROJECT_ROOT / "SFT"
RL_DATA_ROOT = CURRENT_PROJECT_ROOT / "RL" / "data"

if str(SFT_ROOT) not in sys.path:
    sys.path.append(str(SFT_ROOT))
if str(RL_DATA_ROOT) not in sys.path:
    sys.path.append(str(RL_DATA_ROOT))

from dpo_utils import augment_score  # noqa: E402
from retina_sft_utils import (  # noqa: E402
    build_diagnosis_label_vocab,
    build_messages,
    filter_valid_rows,
    load_rgb_image,
    read_split_csv,
    resize_image_longest_side,
)


def _load_sft_metrics_module():
    module_name = "_eye_project_sft_metrics_for_grpo"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = SFT_ROOT / "metrics.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load SFT metrics module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_SFT_METRICS = _load_sft_metrics_module()

REWARD_NAME_TO_FIELD = {
    "format": "format_score",
    "description": "description_score",
    "diagnosis": "diagnosis_score",
}
REWARD_NAME_TO_DEFAULT_WEIGHT = {
    "format": 0.15,
    "description": 0.25,
    "diagnosis": 0.6,
}

_SCORE_CACHE_STEP: int | None = None
_SCORE_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def resolve_workspace_root(workspace_root: str | Path | None) -> Path:
    if workspace_root is None or str(workspace_root).strip() == "":
        return CURRENT_WORKSPACE_ROOT

    candidate = Path(workspace_root).expanduser().resolve()
    if candidate.name == "eye_project" and candidate.exists():
        return candidate.parent
    if (candidate / "eye_project").exists():
        return candidate

    raise FileNotFoundError(
        f"Expected `{candidate}` to either be the workspace root that contains `eye_project/`, "
        "or the `eye_project/` directory itself."
    )


def project_root_from_workspace(workspace_root: str | Path | None) -> Path:
    root = resolve_workspace_root(workspace_root)
    project_root = root / "eye_project"
    if not project_root.exists():
        raise FileNotFoundError(f"`eye_project` was not found under workspace root: {root}")
    return project_root


def remap_eye_project_path(path_value: str | Path, workspace_root: str | Path | None) -> str:
    path = Path(path_value)
    if path.exists():
        return str(path.resolve())

    project_root = project_root_from_workspace(workspace_root)
    if not path.is_absolute():
        candidate = (project_root / path).resolve()
        if candidate.exists():
            return str(candidate)
        return str(candidate)

    marker = f"{Path.sep}eye_project{Path.sep}"
    path_text = str(path)
    if marker in path_text:
        suffix = path_text.split(marker, maxsplit=1)[1]
        remapped = (project_root / suffix).resolve()
        return str(remapped)

    return str(path)


def default_path_under_project(workspace_root: str | Path | None, *parts: str) -> str:
    return str((project_root_from_workspace(workspace_root) / Path(*parts)).resolve())


def prepare_gold_fields(ground_truth: str) -> dict[str, Any]:
    normalized = _SFT_METRICS.normalize_text(ground_truth)
    description = _SFT_METRICS.extract_section(normalized, "描述")
    diagnosis_text = _SFT_METRICS.extract_section(normalized, "初步诊断")
    diagnosis_labels = sorted(_SFT_METRICS.split_diagnosis_labels(diagnosis_text))
    description_structure = _SFT_METRICS.parse_description_structure(description)

    return {
        "ground_truth": ground_truth,
        "ground_truth_normalized": normalized,
        "gold_description": description,
        "gold_diagnosis_text": diagnosis_text,
        "gold_diagnosis_labels": diagnosis_labels,
        "gold_findings": sorted(description_structure["findings"]),
        "gold_locations": sorted(description_structure["locations"]),
        "gold_bucket_map_json": json.dumps(
            description_structure["bucket_map"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "gold_cdr_value": description_structure["cdr_value"],
    }


def build_grpo_prompt(prompt_text: str) -> list[dict[str, Any]]:
    return build_messages(prompt_text)


def prepare_rows_with_workspace(
    split_path: str | Path,
    prompt_path: str | Path,
    workspace_root: str | Path | None,
    max_samples: int | None = None,
    validate_images: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = read_split_csv(split_path, prompt_path=prompt_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["image_path"] = remap_eye_project_path(str(row["image_path"]), workspace_root)
        row.update(prepare_gold_fields(str(row["answer"])))
        row["prompt_text"] = str(row["prompt"])
        row["prompt"] = build_grpo_prompt(row["prompt_text"])
        prepared_rows.append(row)

    if not validate_images:
        return prepared_rows, []

    valid_rows, invalid_rows = filter_valid_rows(prepared_rows, split_name=Path(split_path).stem)
    return valid_rows, invalid_rows


def save_invalid_rows(output_path: str | Path, invalid_rows: list[dict[str, str]]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(invalid_rows, ensure_ascii=False, indent=2), encoding="utf-8")


class RetinaGRPODataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        max_image_side: int | None = None,
    ):
        self.rows = rows
        self.max_image_side = max_image_side

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        image = load_rgb_image(row["image_path"])
        row["image"] = resize_image_longest_side(image, self.max_image_side)
        return row


def build_grpo_dataset(
    rows: list[dict[str, Any]],
    max_image_side: int | None = None,
):
    return RetinaGRPODataset(rows=rows, max_image_side=max_image_side)


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()

    if isinstance(completion, dict):
        return extract_content_text(completion.get("content", ""))

    if isinstance(completion, list):
        parts = [extract_completion_text(item) for item in completion]
        return "\n".join(part for part in parts if part).strip()

    return str(completion).strip()


def extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        return ""

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def decode_bucket_map(bucket_map_json: str | None) -> dict[str, str]:
    if not bucket_map_json:
        return {}
    return json.loads(bucket_map_json)


def _maybe_reset_score_cache(trainer_state: Any) -> None:
    global _SCORE_CACHE_STEP
    global _SCORE_CACHE

    step = getattr(trainer_state, "global_step", None)
    if step != _SCORE_CACHE_STEP:
        _SCORE_CACHE_STEP = step
        _SCORE_CACHE = {}


def score_completion_against_sample(
    completion_text: str,
    sample: Mapping[str, Any],
    trainer_state: Any = None,
) -> dict[str, Any]:
    _maybe_reset_score_cache(trainer_state)

    cache_key = (
        str(sample.get("img_id", "")),
        str(sample.get("ground_truth", "")),
        completion_text,
    )
    cached = _SCORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    normalized_prediction = _SFT_METRICS.normalize_text(completion_text)
    prediction_description = _SFT_METRICS.extract_section(normalized_prediction, "描述")
    prediction_diagnosis = _SFT_METRICS.extract_section(normalized_prediction, "初步诊断")
    prediction_diagnosis_labels = set(_SFT_METRICS.split_diagnosis_labels(prediction_diagnosis))
    prediction_description_structure = _SFT_METRICS.parse_description_structure(prediction_description)

    gold_diagnosis_labels = set(sample.get("gold_diagnosis_labels", []) or [])
    gold_findings = set(sample.get("gold_findings", []) or [])
    gold_locations = set(sample.get("gold_locations", []) or [])
    gold_bucket_map = decode_bucket_map(sample.get("gold_bucket_map_json"))

    cdr_abs_error, cdr_tol_hit = _SFT_METRICS.compare_cdr(
        prediction_description_structure["cdr_value"],
        sample.get("gold_cdr_value"),
    )

    score = {
        "format_correct": float(_SFT_METRICS.is_format_correct(completion_text)),
        "diagnosis_exact_set_acc": float(prediction_diagnosis_labels == gold_diagnosis_labels),
        "description_exact_match": float(prediction_description == str(sample.get("gold_description", ""))),
        "description_finding_set_f1": _SFT_METRICS.set_f1(
            prediction_description_structure["findings"],
            gold_findings,
        ),
        "description_location_f1": _SFT_METRICS.set_f1(
            prediction_description_structure["locations"],
            gold_locations,
        ),
        "description_count_bucket_acc": _SFT_METRICS.score_bucket_accuracy(
            prediction_description_structure["bucket_map"],
            gold_bucket_map,
        ),
        "description_cdr_abs_error": cdr_abs_error,
        "description_cdr_tol_hit": cdr_tol_hit,
        "_pred_diagnosis_labels": sorted(prediction_diagnosis_labels),
        "_ref_diagnosis_labels": sorted(gold_diagnosis_labels),
    }
    augmented = augment_score(score)
    augmented["prediction"] = completion_text
    _SCORE_CACHE[cache_key] = augmented
    return augmented


def _score_batch(
    completions: list[Any],
    trainer_state: Any = None,
    log_extra: Callable[[str, list[Any]], None] | None = None,
    **sample_columns: list[Any],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    predictions: list[str] = []
    final_scores: list[float] = []

    for index, completion in enumerate(completions):
        sample = {key: value[index] for key, value in sample_columns.items()}
        completion_text = extract_completion_text(completion)
        score = score_completion_against_sample(
            completion_text,
            sample=sample,
            trainer_state=trainer_state,
        )
        scores.append(score)
        predictions.append(completion_text)
        final_scores.append(float(score["final_score"]))

    if log_extra is not None:
        log_extra("prediction", predictions)
        log_extra("final_score", final_scores)
    return scores


def format_reward(
    prompts: list[Any],
    completions: list[Any],
    trainer_state: Any = None,
    log_extra: Callable[[str, list[Any]], None] | None = None,
    **sample_columns: list[Any],
) -> list[float]:
    del prompts
    scores = _score_batch(
        completions,
        trainer_state=trainer_state,
        log_extra=log_extra,
        **sample_columns,
    )
    return [float(score["format_score"]) for score in scores]


def description_reward(
    prompts: list[Any],
    completions: list[Any],
    trainer_state: Any = None,
    log_extra: Callable[[str, list[Any]], None] | None = None,
    **sample_columns: list[Any],
) -> list[float]:
    del prompts
    scores = _score_batch(
        completions,
        trainer_state=trainer_state,
        log_extra=log_extra,
        **sample_columns,
    )
    return [float(score["description_score"]) for score in scores]


def diagnosis_reward(
    prompts: list[Any],
    completions: list[Any],
    trainer_state: Any = None,
    log_extra: Callable[[str, list[Any]], None] | None = None,
    **sample_columns: list[Any],
) -> list[float]:
    del prompts
    scores = _score_batch(
        completions,
        trainer_state=trainer_state,
        log_extra=log_extra,
        **sample_columns,
    )
    return [float(score["diagnosis_score"]) for score in scores]


def parse_reward_modes(mode_text: str) -> list[str]:
    if not mode_text.strip():
        raise ValueError("`reward_mode` must not be empty.")

    modes = [item.strip() for item in mode_text.split(",") if item.strip()]
    if "all" in modes:
        return ["format", "description", "diagnosis"]

    unsupported = [mode for mode in modes if mode not in REWARD_NAME_TO_FIELD]
    if unsupported:
        raise ValueError(
            "Unsupported reward modes: "
            + ", ".join(unsupported)
            + ". Expected one or more of: all, format, description, diagnosis."
        )
    if not modes:
        raise ValueError("No valid reward modes were provided.")
    return modes


def build_reward_configuration(
    reward_mode: str,
    format_weight: float,
    description_weight: float,
    diagnosis_weight: float,
) -> tuple[list[Callable[..., list[float]]], list[float], dict[str, float]]:
    reward_map = {
        "format": format_reward,
        "description": description_reward,
        "diagnosis": diagnosis_reward,
    }
    raw_weights = {
        "format": float(format_weight),
        "description": float(description_weight),
        "diagnosis": float(diagnosis_weight),
    }

    reward_names = parse_reward_modes(reward_mode)
    if len(reward_names) == 1:
        reward_name = reward_names[0]
        return [reward_map[reward_name]], [1.0], {reward_name: 1.0}

    reward_funcs = [reward_map[name] for name in reward_names]
    reward_weights = [raw_weights[name] for name in reward_names]
    return reward_funcs, reward_weights, {name: raw_weights[name] for name in reward_names}


class RewardContributionLoggingCallback(TrainerCallback):
    def __init__(self, reward_name_to_weight: Mapping[str, float]):
        self.reward_name_to_weight = dict(reward_name_to_weight)

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args
        del state
        del control
        del kwargs

        if logs is None:
            return

        updates: dict[str, float] = {}
        for reward_name, reward_weight in self.reward_name_to_weight.items():
            mean_key = f"rewards/{reward_name}_reward/mean"
            std_key = f"rewards/{reward_name}_reward/std"
            if mean_key in logs:
                updates[f"rewards/{reward_name}_reward/weighted_contribution"] = (
                    float(logs[mean_key]) * float(reward_weight)
                )
            if std_key in logs:
                updates[f"rewards/{reward_name}_reward/var"] = float(logs[std_key]) ** 2

        if updates:
            logs.update(updates)


def build_diagnosis_labels(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    copied_rows = [dict(row) for row in rows]
    return build_diagnosis_label_vocab(copied_rows)
