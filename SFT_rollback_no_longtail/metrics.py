from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


PUNCT_TRANSLATION = str.maketrans(
    {
        "：": ":",
        "，": ",",
        "。": ".",
        "；": ";",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)

LOCATION_TERMS = (
    "鼻上方",
    "鼻下方",
    "颞上方",
    "颞下方",
    "上方",
    "下方",
    "鼻侧",
    "颞侧",
)
LOCATION_PATTERN = "|".join(LOCATION_TERMS)
QUANT_FINDINGS_PATTERN = "|".join(
    [
        "静脉串珠",
        "视网膜内微血管异常",
        "新生血管",
        "出血",
        "玻璃膜疣",
        "硬性渗出",
        "纤维血管膜",
        "微血管瘤",
        "软性渗出",
    ]
)

RETINA_QUANT_RE = re.compile(
    rf"^(?:(?P<locs>(?:{LOCATION_PATTERN})(?:、(?:{LOCATION_PATTERN}))*)视网膜|视网膜)可见(?P<bucket>散在|大量)?(?P<finding>{QUANT_FINDINGS_PATTERN})$"
)
MACULA_QUANT_RE = re.compile(
    r"^黄斑处可见(?P<finding>出血|玻璃膜疣|硬性渗出|纤维血管膜)$"
)
OCCLUSION_RE = re.compile(
    rf"^静脉迂曲扩张,(?:(?P<locs>(?:{LOCATION_PATTERN})(?:、(?:{LOCATION_PATTERN}))*)视网膜沿血管放射状出血|视网膜沿血管放射状出血)$"
)
CDR_RE = re.compile(r"^杯盘比为\s*(?P<value>\d+(?:\.\d+)?)$")

DIRECT_FINDING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("视盘:出血", re.compile(r"^视盘可见出血$")),
    ("视盘旁:脉络膜萎缩弧", re.compile(r"^视盘旁可见脉络膜萎缩弧$")),
    ("血管:动静脉交叉压迫点", re.compile(r"^可见\d+处动静脉交叉压迫点$")),
    ("视网膜:斑片状萎缩灶", re.compile(r"^可见斑片状萎缩灶$")),
    ("视网膜:漆裂纹", re.compile(r"^视网膜可见漆裂纹$")),
    ("眼底:激光斑", re.compile(r"^眼底可见激光斑$")),
    ("视网膜:神经纤维层缺损", re.compile(r"^可见神经纤维层缺损$")),
    ("玻璃体:浑浊", re.compile(r"^玻璃体可见尘状,条状及块状模糊不清影$")),
    ("屈光介质:混浊", re.compile(r"^屈光介质混浊$")),
    ("视网膜:苍白色混浊", re.compile(r"^视网膜苍白色混浊$")),
    ("血管:动脉变细反光增强", re.compile(r"^视网膜动脉变细,反光增强$")),
    ("血管:动脉迂曲变细反光增强", re.compile(r"^视网膜动脉迂曲变细,反光增强$")),
    ("黄斑:裂孔", re.compile(r"^可见黄斑裂孔$")),
    ("黄斑:前膜", re.compile(r"^黄斑区可见金箔样反光,放射状皱褶,小血管迂曲$")),
    ("黄斑:水肿", re.compile(r"^黄斑水肿$")),
    ("黄斑:中浆", re.compile(r"^黄斑区可见圆形或卵圆形隆起,淡黄或灰黄色边界$")),
    ("视网膜:片状灰白色区", re.compile(r"^视网膜可见片状明亮灰白色区,边缘境界分明$")),
    ("黄斑:黄白色病灶", re.compile(r"^黄斑区可见黄白色病灶$")),
    ("后极部视网膜:皱褶或劈裂", re.compile(r"^后极部视网膜部分区域灰色或青灰色的皱褶或者劈裂$")),
    ("神经纤维:有髓神经纤维", re.compile(r"^沿神经纤维走行的白色不透明有丝样光泽羽毛状区域$")),
    ("视网膜:弥漫性脉络膜萎缩", re.compile(r"^视网膜呈边界模糊,不同程度的黄白色萎缩改变$")),
    ("盘周:脉络膜萎缩弧轻度", re.compile(r"^可见轻度视网膜脉络膜萎缩弧$")),
    ("盘周:脉络膜萎缩弧中度", re.compile(r"^可见中度视网膜脉络膜萎缩弧$")),
    ("盘周:脉络膜萎缩弧重度", re.compile(r"^可见重度视网膜脉络膜萎缩弧$")),
    ("视网膜:豹纹状眼底轻度", re.compile(r"^轻度豹纹状眼底$")),
    ("视网膜:豹纹状眼底中度", re.compile(r"^中度豹纹状眼底$")),
    ("视网膜:豹纹状眼底重度", re.compile(r"^重度豹纹状眼底$")),
    ("视盘:水肿", re.compile(r"^视盘水肿,边界不清$")),
    ("视盘:视神经萎缩", re.compile(r"^视盘淡黄或苍白色,生理凹陷消失$")),
    ("视盘:杯盘比偏大", re.compile(r"^杯盘比偏大$")),
    ("血管:硬化轻度", re.compile(r"^轻度视网膜动脉狭窄,迂曲反光增强$")),
    ("血管:硬化重度", re.compile(r"^动脉银丝样改变$")),
    ("黄斑:大玻璃膜疣", re.compile(r"^黄斑区可见大玻璃膜疣沉积$")),
    ("视网膜:边界清晰灰白色萎缩灶", re.compile(r"^视网膜可见边界清晰的灰白色萎缩灶$")),
)

DIAGNOSIS_FAMILIES = {
    "近视分级": ["轻度近视眼底改变", "中度近视眼底改变", "高度近视眼底改变"],
    "高血压视网膜病变分级": ["高血压视网膜病变轻度", "高血压视网膜病变中度", "高血压视网膜病变重度"],
    "糖尿病视网膜病变分级": [
        "糖尿病视网膜病变轻度非增生期",
        "糖尿病视网膜病变中度非增生期",
        "糖尿病视网膜病变重度非增生期",
        "糖尿病视网膜病变增生期",
    ],
    "黄斑水肿分级": ["黄斑水肿轻度", "黄斑水肿中度", "黄斑水肿重度"],
    "黄斑前膜分期": ["黄斑前膜I期", "黄斑前膜II期", "黄斑前膜III期"],
}

PREDICTION_ROW_FIELDNAMES = [
    "format_correct",
    "diagnosis_exact_set_acc",
    "description_exact_match",
    "description_finding_set_f1",
    "description_location_f1",
    "description_count_bucket_acc",
    "description_cdr_abs_error",
    "description_cdr_tol_hit",
    "pred_diagnosis_labels",
    "ref_diagnosis_labels",
]

CDR_TOLERANCE = 0.1
DEFAULT_BUCKET = "default"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(PUNCT_TRANSLATION)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text


def split_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in normalize_text(text).splitlines() if line.strip()]


def extract_section(text: str, label: str) -> str:
    normalized = normalize_text(text)
    pattern = rf"^{re.escape(label)}\s*:\s*(.+)$"
    match = re.search(pattern, normalized, flags=re.M)
    return match.group(1).strip() if match else ""


def split_diagnosis_labels(diagnosis_text: str) -> list[str]:
    diagnosis_text = normalize_text(diagnosis_text)
    if not diagnosis_text:
        return []
    return [item.strip() for item in diagnosis_text.split("、") if item.strip()]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def set_f1(pred_items: set[str], ref_items: set[str]) -> float:
    if not pred_items and not ref_items:
        return 1.0
    if not pred_items or not ref_items:
        return 0.0

    overlap = len(pred_items & ref_items)
    precision = overlap / len(pred_items)
    recall = overlap / len(ref_items)
    return safe_div(2 * precision * recall, precision + recall)


def is_format_correct(text: str) -> bool:
    lines = split_nonempty_lines(text)
    return (
        len(lines) == 2
        and lines[0].startswith("描述:")
        and lines[1].startswith("初步诊断:")
    )


def split_sentences(description: str) -> list[str]:
    normalized = normalize_text(description)
    sentences: list[str] = []
    current: list[str] = []

    for idx, char in enumerate(normalized):
        if char not in {".", ";"}:
            current.append(char)
            continue

        prev_char = normalized[idx - 1] if idx > 0 else ""
        next_char = normalized[idx + 1] if idx + 1 < len(normalized) else ""
        if char == "." and prev_char.isdigit() and next_char.isdigit():
            current.append(char)
            continue

        sentence = "".join(current).strip(" ,")
        if sentence:
            sentences.append(sentence)
        current = []

    tail = "".join(current).strip(" ,")
    if tail:
        sentences.append(tail)
    return sentences


def parse_locations(location_text: str | None) -> tuple[str, ...]:
    if not location_text:
        return ()
    return tuple(item for item in location_text.split("、") if item)


def make_bucket_key(region: str, finding: str, locations: tuple[str, ...]) -> str:
    location_key = "无位置" if not locations else "、".join(locations)
    return f"{region}|{finding}|{location_key}"


def parse_description_structure(description: str) -> dict[str, Any]:
    findings: set[str] = set()
    locations: set[str] = set()
    bucket_map: dict[str, str] = {}
    cdr_value: float | None = None

    for sentence in split_sentences(description):
        cdr_match = CDR_RE.match(sentence)
        if cdr_match:
            try:
                cdr_value = float(cdr_match.group("value"))
            except ValueError:
                cdr_value = None
            continue

        retina_match = RETINA_QUANT_RE.match(sentence)
        if retina_match:
            finding = retina_match.group("finding")
            location_items = parse_locations(retina_match.group("locs"))
            bucket = retina_match.group("bucket") or DEFAULT_BUCKET
            findings.add(f"视网膜:{finding}")
            locations.update(location_items)
            bucket_map[make_bucket_key("视网膜", finding, location_items)] = bucket
            continue

        macula_match = MACULA_QUANT_RE.match(sentence)
        if macula_match:
            findings.add(f"黄斑处:{macula_match.group('finding')}")
            continue

        occlusion_match = OCCLUSION_RE.match(sentence)
        if occlusion_match:
            location_items = parse_locations(occlusion_match.group("locs"))
            findings.add("视网膜:沿血管放射状出血")
            locations.update(location_items)
            continue

        for canonical_name, pattern in DIRECT_FINDING_PATTERNS:
            if pattern.match(sentence):
                findings.add(canonical_name)
                break

    return {
        "findings": findings,
        "locations": locations,
        "bucket_map": bucket_map,
        "cdr_value": cdr_value,
    }


def score_bucket_accuracy(
    pred_bucket_map: dict[str, str],
    ref_bucket_map: dict[str, str],
) -> float:
    if not ref_bucket_map:
        return 1.0 if not pred_bucket_map else 0.0

    correct = 0
    for key, ref_bucket in ref_bucket_map.items():
        if pred_bucket_map.get(key) == ref_bucket:
            correct += 1
    return correct / len(ref_bucket_map)


def compare_cdr(pred_value: float | None, ref_value: float | None) -> tuple[float, float]:
    if ref_value is None:
        return (0.0, 1.0 if pred_value is None else 0.0)
    if pred_value is None:
        return (1.0, 0.0)

    abs_error = abs(pred_value - ref_value)
    return abs_error, float(abs_error <= CDR_TOLERANCE)


def compute_multilabel_diagnosis_metrics(
    pred_label_sets: list[set[str]],
    ref_label_sets: list[set[str]],
) -> dict[str, Any]:
    all_labels = sorted(set().union(*pred_label_sets, *ref_label_sets))
    if not all_labels:
        return {
            "diagnosis_micro_f1": 0.0,
            "diagnosis_macro_f1": 0.0,
            "diagnosis_per_disease": {},
        }

    total_tp = total_fp = total_fn = 0
    macro_f1_values: list[float] = []
    per_disease: dict[str, Any] = {}
    num_samples = len(ref_label_sets)

    for label in all_labels:
        tp = fp = fn = tn = 0
        for pred_set, ref_set in zip(pred_label_sets, ref_label_sets, strict=True):
            pred_has = label in pred_set
            ref_has = label in ref_set
            if pred_has and ref_has:
                tp += 1
            elif pred_has and not ref_has:
                fp += 1
            elif not pred_has and ref_has:
                fn += 1
            else:
                tn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        macro_f1_values.append(f1)
        per_disease[label] = {
            "accuracy": safe_div(tp + tn, num_samples),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "diagnosis_micro_f1": micro_f1,
        "diagnosis_macro_f1": sum(macro_f1_values) / len(macro_f1_values),
        "diagnosis_per_disease": per_disease,
    }


def compute_family_level_accuracy(
    pred_label_sets: list[set[str]],
    ref_label_sets: list[set[str]],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    family_scores: list[float] = []

    for family_name, family_labels in DIAGNOSIS_FAMILIES.items():
        family_label_set = set(family_labels)
        indices = [
            index
            for index, ref_set in enumerate(ref_label_sets)
            if ref_set & family_label_set
        ]
        if not indices:
            continue

        correct = 0
        for index in indices:
            pred_family = pred_label_sets[index] & family_label_set
            ref_family = ref_label_sets[index] & family_label_set
            if pred_family == ref_family:
                correct += 1

        accuracy = correct / len(indices)
        details[family_name] = {
            "accuracy": accuracy,
            "support": len(indices),
            "labels": family_labels,
        }
        family_scores.append(accuracy)

    return {
        "diagnosis_family_level_acc": sum(family_scores) / len(family_scores) if family_scores else 0.0,
        "diagnosis_family_level_detail": details,
    }


def prediction_row_metrics(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_correct": score["format_correct"],
        "diagnosis_exact_set_acc": score["diagnosis_exact_set_acc"],
        "description_exact_match": score["description_exact_match"],
        "description_finding_set_f1": score["description_finding_set_f1"],
        "description_location_f1": score["description_location_f1"],
        "description_count_bucket_acc": score["description_count_bucket_acc"],
        "description_cdr_abs_error": score["description_cdr_abs_error"],
        "description_cdr_tol_hit": score["description_cdr_tol_hit"],
        "pred_diagnosis_labels": "、".join(score["_pred_diagnosis_labels"]),
        "ref_diagnosis_labels": "、".join(score["_ref_diagnosis_labels"]),
    }


def score_report(prediction: str, reference: str) -> dict[str, Any]:
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    pred_description = extract_section(pred_norm, "描述")
    ref_description = extract_section(ref_norm, "描述")
    pred_diagnosis = extract_section(pred_norm, "初步诊断")
    ref_diagnosis = extract_section(ref_norm, "初步诊断")

    pred_diagnosis_labels = set(split_diagnosis_labels(pred_diagnosis))
    ref_diagnosis_labels = set(split_diagnosis_labels(ref_diagnosis))

    pred_description_structure = parse_description_structure(pred_description)
    ref_description_structure = parse_description_structure(ref_description)

    cdr_abs_error, cdr_tol_hit = compare_cdr(
        pred_description_structure["cdr_value"],
        ref_description_structure["cdr_value"],
    )

    return {
        "format_correct": float(is_format_correct(prediction)),
        "diagnosis_exact_set_acc": float(pred_diagnosis_labels == ref_diagnosis_labels),
        "description_exact_match": float(pred_description == ref_description),
        "description_finding_set_f1": set_f1(
            pred_description_structure["findings"],
            ref_description_structure["findings"],
        ),
        "description_location_f1": set_f1(
            pred_description_structure["locations"],
            ref_description_structure["locations"],
        ),
        "description_count_bucket_acc": score_bucket_accuracy(
            pred_description_structure["bucket_map"],
            ref_description_structure["bucket_map"],
        ),
        "description_cdr_abs_error": cdr_abs_error,
        "description_cdr_tol_hit": cdr_tol_hit,
        "_pred_diagnosis_labels": sorted(pred_diagnosis_labels),
        "_ref_diagnosis_labels": sorted(ref_diagnosis_labels),
    }


def aggregate_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scores = list(scores)
    if not scores:
        return {
            "num_samples": 0,
            "format_correct_rate": 0.0,
            "diagnosis_exact_set_acc": 0.0,
            "diagnosis_micro_f1": 0.0,
            "diagnosis_macro_f1": 0.0,
            "diagnosis_family_level_acc": 0.0,
            "diagnosis_family_level_detail": {},
            "diagnosis_per_disease": {},
            "description_exact_match": 0.0,
            "description_finding_set_f1": 0.0,
            "description_location_f1": 0.0,
            "description_count_bucket_acc": 0.0,
            "description_cdr_mae": 0.0,
            "description_cdr_tol_acc": 0.0,
        }

    num_samples = len(scores)
    aggregated: dict[str, Any] = {
        "num_samples": num_samples,
        "format_correct_rate": sum(score["format_correct"] for score in scores) / num_samples,
        "diagnosis_exact_set_acc": sum(score["diagnosis_exact_set_acc"] for score in scores) / num_samples,
        "description_exact_match": sum(score["description_exact_match"] for score in scores) / num_samples,
        "description_finding_set_f1": sum(score["description_finding_set_f1"] for score in scores) / num_samples,
        "description_location_f1": sum(score["description_location_f1"] for score in scores) / num_samples,
        "description_count_bucket_acc": sum(score["description_count_bucket_acc"] for score in scores) / num_samples,
        "description_cdr_mae": sum(score["description_cdr_abs_error"] for score in scores) / num_samples,
        "description_cdr_tol_acc": sum(score["description_cdr_tol_hit"] for score in scores) / num_samples,
    }

    pred_diagnosis_sets = [set(score["_pred_diagnosis_labels"]) for score in scores]
    ref_diagnosis_sets = [set(score["_ref_diagnosis_labels"]) for score in scores]

    aggregated.update(compute_multilabel_diagnosis_metrics(pred_diagnosis_sets, ref_diagnosis_sets))
    aggregated.update(compute_family_level_accuracy(pred_diagnosis_sets, ref_diagnosis_sets))
    return aggregated


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
