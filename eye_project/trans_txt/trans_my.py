import pandas as pd
import ast
import re
import os
import xml.etree.ElementTree as ET

OUTPUT_FILE = 'evisionai_updated.xlsx'
INPUT_FILE = 'evisionai.xlsx'
ANNOTATION_XML_FILE = 'annotations.xml'
DESCRIPTION_FILE = 'description.csv'

# 图片根目录
IMG_ROOT_DIR = '/mnt/hdd/jiazy/eye_project/image_data/name'

POS_LIST = ['inferiornasal', 'inferiortemporal', 'superiornasal', 'superiortemporal']
BLOODPOS_MAP = {
    "inferiornasal": "鼻下方",
    "inferiortemporal": "颞下方",
    "superiornasal": "鼻上方",
    "superiortemporal": "颞上方",
}

MERGE_RULES = {
    ("鼻下方", "颞下方"): "下方",
    ("鼻上方", "颞上方"): "上方",
    ("鼻上方", "鼻下方"): "鼻侧",
    ("颞上方", "颞下方"): "颞侧"
}

# =========================
# 1. 文本模板
# =========================

DISEASE_PHENOTYPE = {
    # 疾病表型
    "玻璃体浑浊": "玻璃体可见尘状、条状及块状模糊不清影。",
    "屈光介质混浊": "屈光介质混浊。",
    "动脉阻塞": "视网膜苍白色混浊。",
    "高血压视网膜病变轻度": "视网膜动脉变细，反光增强。",
    "高血压视网膜病变中度": "视网膜动脉迂曲变细，反光增强。",
    "高血压视网膜病变重度": "视网膜动脉迂曲变细，反光增强。",
    "黄斑裂孔": "可见黄斑裂孔。",
    "黄斑前膜III期": "黄斑区可见金箔样反光，放射状皱褶，小血管迂曲。",
    "黄斑前膜II期": "黄斑区可见金箔样反光，放射状皱褶，小血管迂曲。",
    "黄斑前膜I期": "黄斑区可见金箔样反光，放射状皱褶，小血管迂曲。",
    "黄斑水肿轻度": "黄斑水肿。",
    "黄斑水肿中度": "黄斑水肿。",
    "黄斑水肿重度": "黄斑水肿。",
    "黄斑中浆": "黄斑区可见圆形或卵圆形隆起，淡黄或灰黄色边界。",
    "脉络膜缺损": "视网膜可见片状明亮灰白色区，边缘境界分明。",
    "年龄相关性黄斑变性进展期": "黄斑区可见黄白色病灶。",
    "视网膜脱离": "后极部视网膜部分区域灰色或青灰色的皱褶或者劈裂。",
    "糖尿病视网膜病变轻度非增生期": "",
    "糖尿病视网膜病变增生期": "",
    "糖尿病视网膜病变中度非增生期": "",
    "糖尿病视网膜病变重度非增生期": "",
    "疑似青光眼": "",
    "有髓神经纤维": "沿神经纤维走行的白色不透明有丝样光泽羽毛状区域。",
    "高度近视眼底改变": "",
    "中度近视眼底改变": "",
    "轻度近视眼底改变": "",
    "中央静脉阻塞": "",
    "分支静脉阻塞": "",
}

LESION_PHENOTYPE = {
    # 病灶表型
    "弥漫性视网膜脉络膜萎缩": "视网膜呈边界模糊、不同程度的黄白色萎缩改变。",
    "盘周视网膜脉络膜萎缩弧轻度": "可见轻度视网膜脉络膜萎缩弧。",
    "盘周视网膜脉络膜萎缩弧中度": "可见中度视网膜脉络膜萎缩弧。",
    "盘周视网膜脉络膜萎缩弧重度": "可见重度视网膜脉络膜萎缩弧。",
    "豹纹状眼底轻度": "轻度豹纹状眼底。",
    "豹纹状眼底中度": "中度豹纹状眼底。",
    "豹纹状眼底重度": "重度豹纹状眼底。",
    "视盘水肿": "视盘水肿，边界不清。",
    "视神经萎缩": "视盘淡黄或苍白色，生理凹陷消失。",
    "杯盘比偏大": "杯盘比偏大。",
    "视盘边界模糊": "",
    "视盘区异常": "",
    "血管硬化轻度": "轻度视网膜动脉狭窄，迂曲反光增强。",
    "血管硬化中": "",
    "血管硬化重度": "动脉银丝样改变。",
    "黄斑区大玻璃膜疣": "黄斑区可见大玻璃膜疣沉积。",
    "黄斑区小玻璃膜疣": "",
    "黄斑区异常": "",
    "黄斑区轻微异常": "",
    "黄斑水肿": "",
    "视网膜激光斑": "眼底可见激光斑。",
    "斑片状视网膜脉络膜萎缩": "视网膜可见边界清晰的灰白色萎缩灶。",
}

# =========================
# 2. tag -> section 映射
# =========================
DISC_TAGS = {
    "盘周视网膜脉络膜萎缩弧轻度",
    "盘周视网膜脉络膜萎缩弧中度",
    "盘周视网膜脉络膜萎缩弧重度",
    "杯盘比偏大",
    "视盘水肿",
    "视神经萎缩",
}

VESSEL_TAGS = {
    "高血压视网膜病变轻度",
    "高血压视网膜病变中度",
    "高血压视网膜病变重度",
    "血管硬化轻度",
    "血管硬化重度",
    "中央静脉阻塞",
    "分支静脉阻塞",
}

MACULA_TAGS = {
    "黄斑裂孔",
    "黄斑前膜I期",
    "黄斑前膜II期",
    "黄斑前膜III期",
    "黄斑水肿轻度",
    "黄斑水肿中度",
    "黄斑水肿重度",
    "黄斑中浆",
    "年龄相关性黄斑变性进展期",
    "黄斑区大玻璃膜疣",
}

RETINA_TAGS = {
    "动脉阻塞",
    "脉络膜缺损",
    "视网膜脱离",
    "有髓神经纤维",
    "弥漫性视网膜脉络膜萎缩",
    "豹纹状眼底轻度",
    "豹纹状眼底中度",
    "豹纹状眼底重度",
    "斑片状视网膜脉络膜萎缩",
    "视网膜激光斑",
}

OTHER_TAGS = {
    "玻璃体浑浊",
    "屈光介质混浊",
}

IGNORE_TAGS = {
    "视盘区异常",
    "黄斑区异常",
    "黄斑区轻微异常",
    "视盘边界模糊",
    "黄斑区小玻璃膜疣",
    "黄斑水肿",
    "血管硬化中",
}

DIAGNOSIS_ONLY_TAGS = {
    "糖尿病视网膜病变轻度非增生期",
    "糖尿病视网膜病变中度非增生期",
    "糖尿病视网膜病变重度非增生期",
    "糖尿病视网膜病变增生期",
    "疑似青光眼",
    "高度近视眼底改变",
    "中度近视眼底改变",
    "轻度近视眼底改变",
}


def safe_parse_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, tuple):
                return list(parsed)
            else:
                return [str(parsed)]
        except Exception:
            return [value]

    return [str(value)]


def normalize_str(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def dedup_keep_order(seq):
    result = []
    seen = set()
    for x in seq:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result


def parse_abnormal_dr_tag(abnormal_list, eye_side):
    result = []

    for item in abnormal_list:
        text = normalize_str(item)
        if not text or text == "未见明显异常":
            continue

        parts = text.split("，", 1)
        if len(parts) != 2:
            continue

        tag_name = parts[0].strip()
        side_text = parts[1].strip()

        if side_text == "右眼" and eye_side == "right":
            result.append(tag_name)
        elif side_text == "左眼" and eye_side == "left":
            result.append(tag_name)
        elif side_text == "双眼":
            result.append(tag_name)
        elif side_text not in ["右眼", "左眼", "双眼"]:
            print(f"位置识别错误，side_text 为 {side_text}")

    return dedup_keep_order(result)


def extract_img_id_from_name(name):
    """
    从 name 中提取最后一个下划线后的 img_id，例如：
    2492961_莫美伍_OD_0807031efd524d6e9bc862a7c6e4a018.jpg
    -> 0807031efd524d6e9bc862a7c6e4a018
    """
    name = normalize_str(name)
    m = re.search(r'_([^_/\\]+)\.[^.]+$', name)
    if m:
        return m.group(1).strip()
    if '_' in name:
        return name.rsplit('_', 1)[-1].strip()
    return name


def parse_evisionai_tag_text(raw_text):
    """
    解析 EVisionAI疾病分类 文本，兼容以下脏数据：
    - None
    - None"杯盘比偏大", ...
    - 多余逗号：..., , ...
    - 尾部多余逗号
    - 重复标签
    """
    text = normalize_str(raw_text)
    if not text or text.lower() == "none":
        return []

    items = re.findall(r'"([^"]+)"', text)
    cleaned = []
    for item in items:
        item = normalize_str(item).strip(',')
        if not item or item.lower() == "none":
            continue
        cleaned.append(item)

    return dedup_keep_order(cleaned)


def parse_annotation_xml(xml_path):
    """
    读取 annotations.xml，返回：
    1. xml_tag_map: img_id -> EVisionAI疾病分类(list)
    2. exclude_img_ids: 自由文本为“模糊”或“4”的 img_id 集合
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_tag_map = {}
    exclude_img_ids = set()

    for image in root.findall('.//image'):
        image_name = image.get('name', '')
        img_id = extract_img_id_from_name(image_name)
        if not img_id:
            continue

        free_text = ""
        evisionai_text = ""

        for tag in image.findall('./tag'):
            label = normalize_str(tag.get('label', ''))
            for attr in tag.findall('./attribute'):
                attr_name = normalize_str(attr.get('name', ''))
                attr_value = normalize_str(attr.text)

                if label == '自由文本' and attr_name == '自由文本':
                    free_text = attr_value
                elif label == 'EVisionAI_disease_Tag' and attr_name == 'EVisionAI疾病分类':
                    evisionai_text = attr_value

        if free_text == '4' or '模糊' in free_text:
            exclude_img_ids.add(img_id)

        xml_tag_map[img_id] = parse_evisionai_tag_text(evisionai_text)

    return xml_tag_map, exclude_img_ids


def sync_tag(file_path, output_file, xml_path=None):
    """
    打开 Excel：
    1. 从 patients_list 中读取 abnormal_list、institution_name、img_id_list
    2. 从 disease_list 中读取 img_id 对应的 eye_side / tag_zh
    3. 更新 parameter sheet 中的 tags 和 institution_name
    4. 若提供 annotations.xml：
       - 从每个 image 的 name 中提取 img_id
       - 用 EVisionAI疾病分类 覆盖对应图片的 tags
       - 自由文本为“模糊”或“4”的图片从最终文件中剔除
    """
    excel_data = pd.read_excel(file_path, sheet_name=None)

    required_sheets = ['patients_list', 'parameter', 'disease_list']
    for sheet in required_sheets:
        if sheet not in excel_data:
            print(f"错误：Excel 中缺少 '{sheet}' 表格")
            return None

    df_patients = excel_data['patients_list']
    df_parameter = excel_data['parameter']
    df_disease = excel_data['disease_list']

    # 1) disease_list 映射：img_id -> eye_side / 所有 tag_zh
    df_disease['img_id'] = df_disease['img_id'].astype(str).str.strip()

    id_to_eye_side_map = {}
    id_to_tag_zh_map = {}

    for _, row in df_disease.iterrows():
        img_id = normalize_str(row.get('img_id'))
        if not img_id:
            continue

        eye_side = normalize_str(row.get('eye_side')).lower()
        tag_zh = row.get('tag_zh', None)

        if img_id not in id_to_eye_side_map or not id_to_eye_side_map[img_id]:
            id_to_eye_side_map[img_id] = eye_side

        if img_id not in id_to_tag_zh_map:
            id_to_tag_zh_map[img_id] = []

        tag_zh_list = safe_parse_list(tag_zh)
        for t in tag_zh_list:
            t = normalize_str(t)
            if t:
                id_to_tag_zh_map[img_id].append(t)

    for img_id in id_to_tag_zh_map:
        id_to_tag_zh_map[img_id] = dedup_keep_order(id_to_tag_zh_map[img_id])

    # 2) patients_list 映射：img_id -> tags(list), institution_name
    id_to_tags_map = {}
    id_to_institution_map = {}

    for _, row in df_patients.iterrows():
        abnormal_value = row.get('abnormal_list', None)
        institution_value = row.get('institution_name', None)
        img_id_list_value = row.get('img_id_list', None)

        if pd.isna(img_id_list_value):
            continue

        abnormal_list = safe_parse_list(abnormal_value)
        img_ids = safe_parse_list(img_id_list_value)

        for img_id in img_ids:
            img_id_str = normalize_str(img_id)
            if not img_id_str:
                continue

            eye_side = id_to_eye_side_map.get(img_id_str, "")

            tags = []
            tags.extend(parse_abnormal_dr_tag(abnormal_list, eye_side))
            tags.extend(id_to_tag_zh_map.get(img_id_str, []))

            id_to_tags_map[img_id_str] = dedup_keep_order(tags)
            id_to_institution_map[img_id_str] = institution_value

    # 3) 读取 XML，覆盖 tags，并记录需要剔除的 img_id
    xml_tag_map = {}
    exclude_img_ids = set()
    if xml_path:
        xml_tag_map, exclude_img_ids = parse_annotation_xml(xml_path)

        for img_id, xml_tags in xml_tag_map.items():
            id_to_tags_map[img_id] = xml_tags

    # 4) 更新 parameter sheet
    df_parameter['img_id'] = df_parameter['img_id'].astype(str).str.strip()
    df_parameter['tags'] = df_parameter['img_id'].map(id_to_tags_map)
    df_parameter['institution_name'] = df_parameter['img_id'].map(id_to_institution_map)
    df_parameter['tags'] = df_parameter['tags'].apply(
        lambda x: x if isinstance(x, list) else []
    )


    # 5) 剔除自由文本为“模糊”或“4”的图片
    if exclude_img_ids:
        df_parameter = df_parameter[~df_parameter['img_id'].isin(exclude_img_ids)].copy()

    def bulit_img_path(row):
        institution_name = normalize_str(row.get('institution_name'))
        img_id = normalize_str(row.get('img_id'))

        if not img_id or not institution_name:
            return None
        
        return os.path.join(IMG_ROOT_DIR, institution_name, f"{img_id}.jpg")
    
    df_parameter['img_path'] = df_parameter.apply(bulit_img_path, axis=1)
    df_parameter['img_exists'] = df_parameter['img_path'].apply(
        lambda x: os.path.exists(x) if isinstance(x,str) and x.strip() else False
    )

    before_count = len(df_parameter)
    df_parameter = df_parameter[df_parameter['img_exists']].copy()
    after_count = len(df_parameter)

    # 删除中间辅助列
    df_parameter.drop(columns=['img_exists'], inplace=True)

    # 6) 保存结果
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in excel_data.items():
            if sheet_name == 'parameter':
                df_parameter.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"处理完成！结果已保存至: {output_file}")
    if exclude_img_ids:
        print(f"已剔除 {len(exclude_img_ids)} 张自由文本为“模糊”或“4”的图片。")
    print(f"因图片文件不存在而额外剔除 {before_count - after_count} 张图片。")
    return output_file


# =========== 表格转文本 ===============
# =========================
# 3. 工具函数
# =========================
def safe_int(v):
    try:
        return int(float(v))
    except:
        return 0


def parse_tags(tags_list_str):
    try:
        if isinstance(tags_list_str, str):
            if tags_list_str.strip() == "" or tags_list_str.lower() == "nan" or tags_list_str == '0':
                return []
            return ast.literal_eval(tags_list_str)
        elif isinstance(tags_list_str, list):
            return tags_list_str
        else:
            return []
    except:
        return []


def get_active_pos(row, prefix):
    active_pos = {BLOODPOS_MAP[p] for p in POS_LIST if safe_int(row.get(prefix + p, 0)) > 0}
    for (a, b), target in MERGE_RULES.items():
        if a in active_pos and b in active_pos:
            active_pos.difference_update({a, b})
            active_pos.add(target)
    return active_pos


def ordered_pos_str(active_pos):
    ordered = ["上方", "下方", "鼻侧", "颞侧", "鼻上方", "鼻下方", "颞上方", "颞下方"]
    return '、'.join([x for x in ordered if x in active_pos])


def quantitative_description_split(row, prefix, label_name, whether_r2=True, whether_num=True):
    """
    返回两个字符串:
    retina_desc: 周边/象限视网膜描述
    macula_desc: 黄斑处描述
    """
    total_num = sum(safe_int(row.get(prefix + pos, 0)) for pos in POS_LIST)
    retina_desc = ""
    macula_desc = ""

    if total_num > 0:
        level = ""
        if whether_num:
            if 0 < total_num < 5:
                level = "散在"
            elif total_num > 50:
                level = "大量"

        active_pos = get_active_pos(row, prefix)
        pos_str = ordered_pos_str(active_pos)
        if pos_str:
            retina_desc = f"{pos_str}视网膜可见{level}{label_name}。"
        else:
            retina_desc = f"视网膜可见{level}{label_name}。"

    if whether_r2 and safe_int(row.get(f"{prefix}r2", 0)) > 0:
        macula_desc = f"黄斑处可见{label_name}。"

    return retina_desc, macula_desc


def append_if(parts, text):
    if text:
        parts.append(text)


def build_occlusion_text(tag, row):
    blood_pos = get_active_pos(row, 'bloodnumberdistribution_')
    blood_str = ordered_pos_str(blood_pos)
    if blood_str:
        return f"静脉迂曲扩张，{blood_str}视网膜沿血管放射状出血。"
    return "静脉迂曲扩张，视网膜沿血管放射状出血。"


# =========================
# 4. 主函数
# =========================
def generate_clinical_description(file_path, output_file):
    df = pd.read_excel(file_path, sheet_name='parameter')
    df = df.fillna(0)

    results = []

    for _, row in df.iterrows():
        img_id = row['img_id']
        institution_name = row.get('institution_name', '')
        img_path = row.get('img_path', '')

        disc_parts = []
        vessel_parts = []
        macula_parts = []
        retina_parts = []
        other_parts = []

        tags_list = [str(x).strip() for x in parse_tags(row.get('tags', 0))]

        append_if(disc_parts, f"杯盘比为{float(row['ratiocdv']):.1f}。")
        if safe_int(row.get('discbloodnumber', 0)) > 0:
            append_if(disc_parts, "视盘可见出血。")
        if safe_int(row.get('ratioatrophymaxwidth', 0)) > 0:
            append_if(disc_parts, "视盘旁可见脉络膜萎缩弧。")

        if safe_int(row.get('arteriovenouscrossingnumber', 0)) > 0:
            append_if(vessel_parts, f"可见{safe_int(row['arteriovenouscrossingnumber'])}处动静脉交叉压迫点。")

        vessel_desc, _ = quantitative_description_split(
            row, 'veinbeadingnumberdistribution_', '静脉串珠', whether_r2=False, whether_num=False
        )
        append_if(vessel_parts, vessel_desc)

        vessel_desc, _ = quantitative_description_split(
            row, 'microvascularabnormalityareadistribution_', '视网膜内微血管异常', whether_r2=False, whether_num=False
        )
        append_if(vessel_parts, vessel_desc)

        vessel_desc, _ = quantitative_description_split(
            row, 'neovascularizationnumberdistribution_', '新生血管', whether_r2=False, whether_num=False
        )
        append_if(vessel_parts, vessel_desc)

        has_vein_occlusion = any(tag in {"中央静脉阻塞", "分支静脉阻塞"} for tag in tags_list)

        if not has_vein_occlusion:
            retina_desc, macula_desc = quantitative_description_split(
                row, 'bloodnumberdistribution_', '出血', whether_r2=True, whether_num=True
            )
            append_if(retina_parts, retina_desc)
            append_if(macula_parts, macula_desc)

        retina_desc, macula_desc = quantitative_description_split(
            row, 'drusennumberdistribution_', '玻璃膜疣', whether_r2=True, whether_num=True
        )
        append_if(retina_parts, retina_desc)
        append_if(macula_parts, macula_desc)

        retina_desc, macula_desc = quantitative_description_split(
            row, 'exudatenumberdistribution_', '硬性渗出', whether_r2=True, whether_num=True
        )
        append_if(retina_parts, retina_desc)
        append_if(macula_parts, macula_desc)

        retina_desc, macula_desc = quantitative_description_split(
            row, 'fibrovascularmembranenumberdistribution_', '纤维血管膜', whether_r2=True, whether_num=True
        )
        append_if(retina_parts, retina_desc)
        append_if(macula_parts, macula_desc)

        retina_desc, _ = quantitative_description_split(
            row, 'micropointnumberdistribution_', '微血管瘤', whether_r2=False, whether_num=True
        )
        append_if(retina_parts, retina_desc)

        retina_desc, _ = quantitative_description_split(
            row, 'softexudatenumberdistribution_', '软性渗出', whether_r2=False, whether_num=True
        )
        append_if(retina_parts, retina_desc)

        if safe_int(row.get('atrophicfocinumber', 0)) > 0:
            append_if(retina_parts, "可见斑片状萎缩灶。")

        if safe_int(row.get('lacquercracknumber', 0)) > 0:
            append_if(retina_parts, "视网膜可见漆裂纹。")

        laserspot_sum = sum(safe_int(row.get(f"laserspotnumberdistribution_{p}", 0)) for p in POS_LIST)
        if laserspot_sum > 0:
            append_if(retina_parts, "眼底可见激光斑。")

        if safe_int(row.get('rnfldarea', 0)) > 0:
            append_if(retina_parts, "可见神经纤维层缺损。")

        disease_list = []

        for tag in tags_list:
            if tag in IGNORE_TAGS:
                continue

            if tag in DISEASE_PHENOTYPE:
                disease_list.append(tag)

            if tag in {"中央静脉阻塞", "分支静脉阻塞"}:
                append_if(vessel_parts, build_occlusion_text(tag, row))
                continue

            if tag in DIAGNOSIS_ONLY_TAGS:
                continue

            text = ""
            if tag in LESION_PHENOTYPE:
                text = LESION_PHENOTYPE[tag]
            elif tag in DISEASE_PHENOTYPE:
                text = DISEASE_PHENOTYPE[tag]

            if not text:
                continue

            if tag in DISC_TAGS:
                append_if(disc_parts, text)
            elif tag in VESSEL_TAGS:
                append_if(vessel_parts, text)
            elif tag in MACULA_TAGS:
                append_if(macula_parts, text)
            elif tag in RETINA_TAGS:
                append_if(retina_parts, text)
            elif tag in OTHER_TAGS:
                append_if(other_parts, text)
            else:
                append_if(other_parts, text)

        disc_parts = dedup_keep_order(disc_parts)
        vessel_parts = dedup_keep_order(vessel_parts)
        macula_parts = dedup_keep_order(macula_parts)
        retina_parts = dedup_keep_order(retina_parts)
        other_parts = dedup_keep_order(other_parts)

        desc = "描述：" + "".join(
            disc_parts + vessel_parts + macula_parts + retina_parts + other_parts
        )

        diagnosis_final = dedup_keep_order(disease_list)
        diagnosis = "、".join(diagnosis_final) if diagnosis_final else "未见明显异常表征"

        full_text = desc + "\n" + f"初步诊断：{diagnosis}"

        results.append({
            'img_id': img_id,
            'description': full_text,
            'institution_name': institution_name,
            'img_path': img_path,
        })

    df_description = pd.DataFrame(results)
    df_description.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"文件转换成功！结果已保存至: {output_file}")


if __name__ == "__main__":
    sync_tag(INPUT_FILE, OUTPUT_FILE, ANNOTATION_XML_FILE)
    generate_clinical_description(OUTPUT_FILE, DESCRIPTION_FILE)