import argparse
import importlib.util
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RL_ANNOTATION_XML = os.path.join(SCRIPT_DIR, "annotations.xml")
TRANS_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "eye_project", "trans_txt", "trans.py")
LOOKUP_EXCEL_FILE = os.path.join(PROJECT_ROOT, "eye_project", "trans_txt", "evisionai.xlsx")
PARAMETER_EXCEL_FILE = os.path.join(PROJECT_ROOT, "eye_project", "trans_txt", "evisionai_updated.xlsx")

OUTPUT_FILE = os.path.join(SCRIPT_DIR, "evisionai_rl_first400_updated.xlsx")
DESCRIPTION_FILE = os.path.join(SCRIPT_DIR, "description_rl_first400.csv")

XML_EYE_SIDE_MAP = {
    "OS": "left",
    "OD": "right",
}

STANDARD_EYE_SIDE_MAP = {
    "os": "left",
    "od": "right",
    "left": "left",
    "right": "right",
    "左眼": "left",
    "右眼": "right",
}

COLUMN_ALIASES = {
    "patient_record_id": ["patient_record_id", "patientrecordid", "record_id", "recordid"],
    "eye_side": ["eye_side", "eyeside", "side"],
    "img_id": ["img_id", "imgid", "image_id", "imageid"],
}


def normalize_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_identifier(value):
    text = normalize_str(value)
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def normalize_eye_side(value):
    text = normalize_str(value)
    if not text:
        return ""
    return STANDARD_EYE_SIDE_MAP.get(text.lower(), STANDARD_EYE_SIDE_MAP.get(text, text.lower()))


def normalize_column_key(value):
    return re.sub(r"[^a-z0-9]+", "", normalize_str(value).lower())


def find_column_by_aliases(columns, aliases, required=True):
    normalized_to_column = {
        normalize_column_key(column): column for column in columns
    }

    for alias in aliases:
        matched = normalized_to_column.get(normalize_column_key(alias))
        if matched:
            return matched

    if required:
        raise KeyError(f"未找到列，候选列名为: {aliases}；实际列名为: {list(columns)}")
    return None


def load_trans_module(trans_script_path):
    spec = importlib.util.spec_from_file_location("trans_original_module", trans_script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_patient_record_from_xml_name(name):
    base_name = os.path.splitext(os.path.basename(normalize_str(name)))[0]
    if not base_name:
        return None

    parts = base_name.split("_")
    if len(parts) < 3:
        return None

    patient_record_id = normalize_identifier(parts[0])
    eye_side = XML_EYE_SIDE_MAP.get(parts[-1].upper(), "")
    if not patient_record_id or not eye_side:
        return None

    return {
        "patient_record_id": patient_record_id,
        "eye_side": eye_side,
        "name": "_".join(parts[1:-1]).strip(),
        "xml_name": base_name,
    }


def parse_rl_xml_subset(xml_path, sample_limit):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    subset_entries = []
    invalid_name_samples = []

    for sample_index, image in enumerate(root.findall(".//image"), start=1):
        if sample_index > sample_limit:
            break

        image_name = image.get("name", "")
        image_info = extract_patient_record_from_xml_name(image_name)
        if not image_info:
            invalid_name_samples.append(image_name)
            continue

        free_text = ""
        evisionai_text = ""

        for tag in image.findall("./tag"):
            label = normalize_str(tag.get("label", ""))
            for attr in tag.findall("./attribute"):
                attr_name = normalize_str(attr.get("name", ""))
                attr_value = normalize_str(attr.text)

                if label == "自由文本" and attr_name == "自由文本":
                    free_text = attr_value
                elif label == "EVisionAI_disease_Tag" and attr_name == "EVisionAI疾病分类":
                    evisionai_text = attr_value

        image_info["free_text"] = free_text
        image_info["evisionai_text"] = evisionai_text
        subset_entries.append(image_info)

    return subset_entries, invalid_name_samples


def build_disease_lookup(df_disease):
    patient_record_col = find_column_by_aliases(
        df_disease.columns,
        COLUMN_ALIASES["patient_record_id"],
    )
    eye_side_col = find_column_by_aliases(
        df_disease.columns,
        COLUMN_ALIASES["eye_side"],
    )
    img_id_col = find_column_by_aliases(
        df_disease.columns,
        COLUMN_ALIASES["img_id"],
    )

    key_to_img_id = {}
    duplicate_keys = {}

    for _, row in df_disease.iterrows():
        patient_record_id = normalize_identifier(row.get(patient_record_col))
        eye_side = normalize_eye_side(row.get(eye_side_col))
        img_id = normalize_identifier(row.get(img_id_col))

        if not patient_record_id or not eye_side or not img_id:
            continue

        key = (patient_record_id, eye_side)
        if key in key_to_img_id and key_to_img_id[key] != img_id:
            duplicate_keys.setdefault(key, set()).update({key_to_img_id[key], img_id})
            continue

        key_to_img_id[key] = img_id

    return {
        "patient_record_col": patient_record_col,
        "eye_side_col": eye_side_col,
        "img_id_col": img_id_col,
        "key_to_img_id": key_to_img_id,
        "duplicate_keys": duplicate_keys,
    }


def build_subset_workbook(
    trans_module,
    lookup_excel_file,
    parameter_excel_file,
    rl_xml_path,
    output_file,
    description_file,
    sample_limit=400,
):
    xml_entries, invalid_name_samples = parse_rl_xml_subset(rl_xml_path, sample_limit)
    lookup_excel_data = pd.read_excel(lookup_excel_file, sheet_name=None)
    parameter_excel_data = pd.read_excel(parameter_excel_file, sheet_name=None)

    if "disease_list" not in lookup_excel_data:
        raise ValueError(f"Excel 中缺少 'disease_list' 表格: {lookup_excel_file}")
    if "parameter" not in parameter_excel_data:
        raise ValueError(f"Excel 中缺少 'parameter' 表格: {parameter_excel_file}")

    df_disease = lookup_excel_data["disease_list"].copy()
    df_parameter = parameter_excel_data["parameter"].copy()

    parameter_img_id_col = find_column_by_aliases(df_parameter.columns, COLUMN_ALIASES["img_id"])

    disease_lookup = build_disease_lookup(df_disease)
    disease_img_id_col = disease_lookup["img_id_col"]

    xml_tag_map = {}
    exclude_img_ids = set()
    selected_img_ids = set()
    unmatched_samples = []

    for entry in xml_entries:
        key = (entry["patient_record_id"], entry["eye_side"])
        img_id = disease_lookup["key_to_img_id"].get(key)
        if not img_id:
            unmatched_samples.append(entry["xml_name"])
            continue

        xml_tag_map[img_id] = trans_module.parse_evisionai_tag_text(entry["evisionai_text"])
        selected_img_ids.add(img_id)

        if entry["free_text"] == "4" or "模糊" in entry["free_text"]:
            exclude_img_ids.add(img_id)

    if not selected_img_ids:
        raise ValueError("前 N 个 XML 样本没有匹配到任何 img_id，请检查 disease_list 的 patient_record_id / eye_side 列。")

    df_parameter[parameter_img_id_col] = df_parameter[parameter_img_id_col].apply(normalize_identifier)
    df_parameter = df_parameter[df_parameter[parameter_img_id_col].isin(selected_img_ids)].copy()

    if "tags" not in df_parameter.columns:
        raise ValueError(f"parameter 表中缺少 'tags' 列: {parameter_excel_file}")

    df_parameter["tags"] = df_parameter[parameter_img_id_col].map(
        lambda img_id: xml_tag_map.get(img_id, [])
    )

    if exclude_img_ids:
        df_parameter = df_parameter[~df_parameter[parameter_img_id_col].isin(exclude_img_ids)].copy()

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in parameter_excel_data.items():
            if sheet_name == "parameter":
                df_parameter.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    trans_module.generate_clinical_description(output_file, description_file)

    print(f"处理完成：{output_file}")
    print(f"描述文件：{description_file}")
    print(f"XML 前 {sample_limit} 个样本中，成功匹配到 {len(selected_img_ids)} 个 img_id。")
    if unmatched_samples:
        print(f"未匹配到 disease_list 的 XML 样本数：{len(unmatched_samples)}")
    if invalid_name_samples:
        print(f"文件名不符合 {{patient_record_id}}_{{name}}_{{eye_side}} 格式的样本数：{len(invalid_name_samples)}")
    if disease_lookup["duplicate_keys"]:
        print(
            f"警告：disease_list 中有 {len(disease_lookup['duplicate_keys'])} 个 patient_record_id + eye_side 对应多个 img_id，脚本默认保留首次出现的 img_id。"
        )
    if exclude_img_ids:
        print(f"因自由文本为“模糊”或“4”剔除 {len(exclude_img_ids)} 个样本。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-file", default=RL_ANNOTATION_XML)
    parser.add_argument("--lookup-excel-file", default=LOOKUP_EXCEL_FILE)
    parser.add_argument("--parameter-excel-file", default=PARAMETER_EXCEL_FILE)
    parser.add_argument("--trans-script", default=TRANS_SCRIPT_PATH)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument("--description-file", default=DESCRIPTION_FILE)
    parser.add_argument("--sample-limit", type=int, default=400)
    args = parser.parse_args()

    trans_module = load_trans_module(args.trans_script)
    build_subset_workbook(
        trans_module=trans_module,
        lookup_excel_file=args.lookup_excel_file,
        parameter_excel_file=args.parameter_excel_file,
        rl_xml_path=args.xml_file,
        output_file=args.output_file,
        description_file=args.description_file,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
