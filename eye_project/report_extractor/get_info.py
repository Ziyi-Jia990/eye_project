import re
import json
import pandas as pd
from html import unescape
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup


# =========================
# 1. 你给定的 schema
# =========================

# 出血渗出 - 第一种表格
EXUDATION_QUADRANT_COLUMNS = [
    "内容",
    "右眼/总面积(mm²)",
    "右眼/最大面积(mm²)",
    "右眼/个数/TS",
    "右眼/个数/TI",
    "右眼/个数/NS",
    "右眼/个数/NI",
    "右眼/总个数",
    "左眼/总面积(mm²)",
    "左眼/最大面积(mm²)",
    "左眼/个数/TS",
    "左眼/个数/TI",
    "左眼/个数/NS",
    "左眼/个数/NI",
    "左眼/总个数",
]

EXUDATION_QUADRANT_ROWS = [
    "出血",
    "渗出",
    "棉絮斑/硬性渗出",
    "玻璃膜疣",
]

EXUDATION_QUADRANT_FIRST_COL = [
    '内容', 
    '内容', 
    '内容',
    "出血",
    "渗出",
    "棉絮斑/硬性渗出",
    "玻璃膜疣",
]

# 出血渗出 - 第二种表格
EXUDATION_SUMMARY_COLUMNS = [
    "内容",
    "右眼/总面积(mm²)",
    "右眼/最大面积(mm²)",
    "右眼/个数",
    "左眼/总面积(mm²)",
    "左眼/最大面积(mm²)",
    "左眼/个数",
]

EXUDATION_SUMMARY_ROWS = [
    "出血",
    "渗出",
]

EXUDATION_SUMMARY_FIRST_COL = [
    "内容",
    "内容",
    "出血",
    "渗出",
]

# 血管参数
VESSEL_PARAMETER_COLUMNS = [
    "血管参数",
    "右眼",
    "左眼",
]

VESSEL_PARAMETER_ROWS = [
    "B区视网膜中央动脉当量(CRAE)(PD)",
    "B区视网膜中央静脉当量(CRVE)(PD)",
    "B区视网膜动静脉比(AVR)",
]

VESSEL_PARAMETER_FIRST_COL = [
    "血管参数",
    "B区视网膜中央动脉当量(CRAE)(PD)",
    "B区视网膜中央静脉当量(CRVE)(PD)",
    "B区视网膜动静脉比(AVR)",
]

# 视神经参数
OPTIC_DISC_PARAMETER_COLUMNS = [
    "内容",
    "右眼/水平",
    "右眼/垂直",
    "左眼/水平",
    "左眼/垂直",
]

OPTIC_DISC_PARAMETER_ROWS = [
    "视杯直径(μm)",
    "视盘直径(μm)",
    "杯盘比值",
]

OPTIC_DISC_PARAMETER_FIRST_COL = [
    "内容",
    "内容",
    "视杯直径(μm)",
    "视盘直径(μm)",
    "杯盘比值",
]

# 近视相关参数
MYOPIA_PARAMETER_COLUMNS = [
    "内容",
    "右眼",
    "左眼",
]

MYOPIA_PARAMETER_ROWS = [
    "弧形斑和视盘面积比",
    "弧形斑直径(PD)",
    "豹纹斑平均密度",
]

MYOPIA_PARAMETER_FIRST_COL = [
    "内容",
    "弧形斑和视盘面积比",
    "弧形斑直径(PD)",
    "豹纹斑平均密度",
]

# 视神经纤维层缺损分析
RETINAL_NERVE_DEFECT_COLUMNS = [
    "内容",
    "是否发现疑似神经纤维层缺损",
    "疑似缺损位置",
    "疑似缺损面积(mm²)",
]

RETINAL_NERVE_DEFECT_ROWS = [
    "右眼",
    "左眼",
]

RETINAL_NERVE_DEFECT_FIRST_COL = [
    "内容",
    "右眼",
    "左眼",
]

# 杯盘比分析
CUP_DISC_RATIO_COLUMNS = [
    "内容",
    "右眼/水平",
    "右眼/垂直",
    "右眼/平均",
    "左眼/水平",
    "左眼/垂直",
    "左眼/平均",
]

CUP_DISC_RATIO_ROWS = [
    "视杯直径(μm)",
    "视盘直径(μm)",
    "杯盘比值",
]

CUP_DISC_RATIO_FIRST_COL = [
    "内容",
    "内容",
    "视杯直径(μm)",
    "视盘直径(μm)",
    "杯盘比值",
]

# 盘沿分析
NEURORETINAL_RIM_COLUMNS = [
    "内容",
    "右眼盘沿宽度(μm)",
    "左眼盘沿宽度(μm)",
]

NEURORETINAL_RIM_ROWS = [
    "I区",
    "S区",
    "N区",
    "T区",
]

NEURORETINAL_RIM_FIRST_COL = [
    "内容",
    "I区",
    "S区",
    "N区",
    "T区",
]

# 视盘面积分析
OPTIC_DISC_AREA_COLUMNS = [
    "内容",
    "右眼",
    "左眼",
]

OPTIC_DISC_AREA_ROWS = [
    "视盘面积(mm²)",
    "视杯面积(mm²)",
    "盘沿面积(mm²)",
    "视杯/视盘面积比",
]

OPTIC_DISC_AREA_FIRST_COL = [
    '内容',
    "视盘面积(mm²)",
    "视杯面积(mm²)",
    "盘沿面积(mm²)",
    "视杯/视盘面积比",
]

TABLE_SCHEMAS = {
    "出血渗出": {
        "columns": EXUDATION_QUADRANT_COLUMNS,
        "rows": EXUDATION_QUADRANT_ROWS,
        "alt_columns": EXUDATION_SUMMARY_COLUMNS,
        "alt_rows": EXUDATION_SUMMARY_ROWS,
    },
    "血管参数": {
        "columns": VESSEL_PARAMETER_COLUMNS,
        "rows": VESSEL_PARAMETER_ROWS,
    },
    "视神经参数": {
        "columns": OPTIC_DISC_PARAMETER_COLUMNS,
        "rows": OPTIC_DISC_PARAMETER_ROWS,
    },
    "近视相关参数": {
        "columns": MYOPIA_PARAMETER_COLUMNS,
        "rows": MYOPIA_PARAMETER_ROWS,
    },
    "视神经纤维层缺损分析": {
        "columns": RETINAL_NERVE_DEFECT_COLUMNS,
        "rows": RETINAL_NERVE_DEFECT_ROWS,
    },
    "杯盘比分析": {
        "columns": CUP_DISC_RATIO_COLUMNS,
        "rows": CUP_DISC_RATIO_ROWS,
    },
    "盘沿分析": {
        "columns": NEURORETINAL_RIM_COLUMNS,
        "rows": NEURORETINAL_RIM_ROWS,
    },
    "视盘面积分析": {
        "columns": OPTIC_DISC_AREA_COLUMNS,
        "rows": OPTIC_DISC_AREA_ROWS,
    },
}

EXCEL_PATH = "data/label.xlsx"

# =========================
# 2. 通用工具
# =========================

import re
from html import unescape

def normalize_text(text: str) -> str:
    """统一空白、特殊字符、中文括号、单位写法。"""
    if text is None:
        return ""
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u3000", " ")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("＜", "<").replace("＞", ">")
    text = text.replace("μ", "μ")

    # 1. 处理带有 <sup> 标签的原生 HTML 文本
    text = re.sub(r"mm<sup>\s*2\s*</sup>", "mm²", text, flags=re.I)
    text = re.sub(r"m<sup>\s*2\s*</sup>", "m²", text, flags=re.I)

    # 2. 处理标签被剥离后的情况，并在正则末尾加上 \s* 把后面的空格一起吞掉
    text = re.sub(r"mm\s*2\b\s*", "mm²", text, flags=re.I)
    text = re.sub(r"(?<!m)m\s*2\b\s*", "m²", text, flags=re.I) 

    # 3. 规范化所有空白字符
    text = re.sub(r"\s+", " ", text).strip()

    # 4. 新增：清理括号内侧的冗余空格，把 "( mm² )" 彻底变成 "(mm²)"
    text = text.replace("( ", "(").replace(" )", ")")

    # 常见 OCR/HTML 误差修正
    text = text.replace("m²)", "mm²)") if "最大面积(" in text and "mm²" not in text else text
    text = text.replace("m²", "mm²") if "最大面积(" in text and "mm²" not in text else text
    text = text.replace("虎纹斑", "豹纹斑")
    
    return text


def fuzzy_line_eq(a: str, b: str) -> bool:
    """轻量级相等判断，不依赖第三方模糊匹配库。"""
    a = normalize_text(a)
    b = normalize_text(b)
    if a == b:
        return True
    if a in b or b in a:
        return True
    return False


def clean_md(md: str) -> str:
    """去掉图片行，但保留章节和普通文本。"""
    lines = md.splitlines()
    kept = []
    for line in lines:
        if re.match(r"^\s*!\[\]\(.*?\)\s*$", line):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept)

def first_col_values(grid: List[List[str]]) -> List[str]:
    """
    取表格第一列的非空值。
    """
    vals = []
    for row in normalize_grid(grid):
        if row and normalize_text(row[0]):
            vals.append(normalize_text(row[0]))
    return vals


def row_signature_score(first_col: List[str], wanted_rows: List[str]) -> int:
    """
    用第一列内容给某个 schema 打分。
    命中越多，越可能是该表。
    """
    score = 0
    for target in wanted_rows:
        for cell in first_col:
            if fuzzy_line_eq(cell, target):
                score += 1
                break
    return score


def find_best_matching_row(grid: List[List[str]], target: str) -> Optional[List[str]]:
    """
    在整个 grid 中找最匹配 target 的那一行。
    """
    best_row = None
    best_score = -1

    for row in normalize_grid(grid):
        if not row:
            continue
        first = normalize_text(row[0])
        if not first:
            continue

        score = 0
        if first == normalize_text(target):
            score = 100
        elif target in first or first in target:
            score = 80
        elif fuzzy_line_eq(first, target):
            score = 60

        if score > best_score:
            best_score = score
            best_row = row

    if best_score <= 0:
        return None
    return best_row


def is_header_like_row(row: List[str]) -> bool:
    """
    粗略判断是不是表头行。
    """
    row_text = " ".join(normalize_text(x) for x in row)
    header_keywords = ["内容", "右眼", "左眼", "水平", "垂直", "平均", "总面积", "最大面积", "总个数", "个数"]
    hit = sum(1 for k in header_keywords if k in row_text)
    return hit >= 2

# =========================
# 3. 基本信息提取
# =========================

def extract_basic_info(md: str) -> Dict[str, Optional[str]]:
    text = clean_md(md)

    def pick(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.M)
        return m.group(1).strip() if m else None # 取第一个捕获组，.strip()去除前后空格

    return {
        "性别": pick(r"性别[:：]\s*([^\n]+)"),  
        "年龄": pick(r"年龄[:：]\s*([^\n]+)"),  
        "检查号": pick(r"检查号[:：]\s*([^\n]+)"), 
        "检查日期": pick(r"检查日期[:：]\s*([^\n]+)"), 
    }
    # [:：]字符集，匹配两个中的一个即可
    # \s* 匹配0个/多个空白字符
    # [^\n] 除换行符之外的任意字符，() 表示要捕获的内容
    # + 匹配一次或多次


# =========================
# 4. 章节提取
# =========================

def split_non_table_and_tables(md: str):
    """把 markdown 切成 非表格文本段 + 表格段。"""
    table_pattern = re.compile(r"<table>.*?</table>", flags=re.S | re.I)
    parts = []
    last = 0
    for m in table_pattern.finditer(md):
        if m.start() > last:
            parts.append(("text", md[last:m.start()]))
        parts.append(("table", m.group(0)))
        last = m.end()
    if last < len(md):
        parts.append(("text", md[last:]))
    return parts


def extract_dr_analysis(md: str) -> Optional[str]:
    """提取 糖尿病性视网膜病变分析 章节正文。"""
    text = clean_md(md)
    lines = [x.strip() for x in text.splitlines()]

    for i, line in enumerate(lines):
        if "糖尿病性视网膜病变分析" in line:
            # 取后续第一条非空正文，直到碰到医师/日期/标题
            collected = []
            for j in range(i + 1, len(lines)):
                cur = lines[j].strip()
                if not cur:
                    continue
                if cur.startswith("医师"):
                    break
                if cur.startswith("日期"):
                    break
                if cur.startswith("#"):
                    break
                collected.append(cur)
            result = " ".join(collected).strip()
            return result or None
    return None



def extract_assessment_result(md: str) -> Dict[str, Optional[str]]:
    """提取 评估结果 中的 OD / OS 段落。"""

    # --- 新增：从 Excel 加载疾病字典的辅助函数 ---
    def load_disease_list() -> List[str]:
        """读取 Excel 第一列的所有诊断名称，返回列表"""
        df = pd.read_excel(EXCEL_PATH)
        # 假设"诊断名称"在第 1 列 (索引为 0)
        diseases = df.iloc[:, 0].dropna().astype(str).tolist()
        
        # 去除首尾空格，并过滤掉空字符串
        cleaned_diseases = [d.strip() for d in diseases if d.strip()]
        
        # 【可选优化】按字符串长度降序排序。
        # 这样在后续拓展中，如果遇到类似“豹纹样改变”和“豹纹样改变（明显）”，可以防止短词被优先匹配而漏掉长词。
        cleaned_diseases.sort(key=len, reverse=True)
        
        return cleaned_diseases
    
    text = clean_md(md)
    lines = [x.strip() for x in text.splitlines()]

    in_section = False
    section_lines = []
    for line in lines:
        if re.match(r"^#*\s*评估结果\s*$", line):
            in_section = True
            continue
        if in_section:
            if re.match(r"^#+\s+", line):
                break
            section_lines.append(line)

    section_text = "\n".join(section_lines)

    # 提取逻辑保持不变
    od_match = re.search(
        r"(OD[（(]右眼[）)][:：].*?)(?=OS[（(]左眼[）)][:：]|医师[:：]|日期[:：]|$)",
        section_text,
        flags=re.S
    )
    os_match = re.search(
        r"(OS[（(]左眼[）)][:：].*?)(?=医师[:：]|日期[:：]|$)",
        section_text,
        flags=re.S
    )

    od_text = normalize_text(od_match.group(1)) if od_match else None
    os_text = normalize_text(os_match.group(1)) if os_match else None

    return {
        "OD": od_text,
        "OS": os_text,
    }


# =========================
# 5. HTML 表格解析
# =========================

def parse_html_table_to_grid(table_html: str) -> List[List[str]]:
    """
    把 HTML table 展开成二维网格，处理 rowspan / colspan。
    """
    soup = BeautifulSoup(table_html, "html.parser")
    trs = soup.find_all("tr")
    grid: List[List[str]] = []
    span_map = {}  # col_idx -> [remaining_rows, value]

    for tr in trs:
        row_cells = []
        col_idx = 0

        # 先填补上一行遗留的 rowspan
        while col_idx in span_map:
            remain, value = span_map[col_idx]
            row_cells.append(value)
            if remain - 1 == 0:
                del span_map[col_idx]
            else:
                span_map[col_idx] = [remain - 1, value]
            col_idx += 1

        cells = tr.find_all(["td", "th"])
        if not cells and not row_cells:
            continue

        for cell in cells:
            # 如果中间某些列被 rowspan 占据，先补进去
            while col_idx in span_map:
                remain, value = span_map[col_idx]
                row_cells.append(value)
                if remain - 1 == 0:
                    del span_map[col_idx]
                else:
                    span_map[col_idx] = [remain - 1, value]
                col_idx += 1

            text = normalize_text(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            for _ in range(colspan):
                row_cells.append(text)
                if rowspan > 1:
                    span_map[col_idx] = [rowspan - 1, text]
                col_idx += 1

        # 行末尾若还有 rowspan 残留，也要补齐
        while col_idx in span_map:
            remain, value = span_map[col_idx]
            row_cells.append(value)
            if remain - 1 == 0:
                del span_map[col_idx]
            else:
                span_map[col_idx] = [remain - 1, value]
            col_idx += 1

        if any(x != "" for x in row_cells):
            grid.append(row_cells)

    # 补齐所有行长度
    max_len = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (max_len - len(r)) for r in grid]
    return grid


def find_tables_with_context(md: str) -> List[Dict[str, Any]]:
    """
    提取每个表格及其前面的近邻文本，用于判断表名。
    """
    parts = split_non_table_and_tables(md)
    results = []

    text_so_far = ""
    for part_type, content in parts:
        if part_type == "text":
            text_so_far += content
        else:
            context_lines = [x.strip() for x in clean_md(text_so_far).splitlines() if x.strip()]
            prev_lines = context_lines[-8:]  # 最近几行上下文
            results.append({
                "html": content,
                "grid": parse_html_table_to_grid(content),
                "prev_lines": prev_lines,
            })

    return results


# =========================
# 6. 表格类型识别
# =========================

def infer_table_type(prev_lines: List[str], grid: List[List[str]]) -> Optional[str]:
    """
    优先根据第一列内容判断表格类型，不依赖上下文标题。
    只有在确实无法判断时，才用少量辅助信息。
    """
    grid = normalize_grid(grid)
    if not grid:
        return None

    first_col = first_col_values(grid)

    # 1) 出血渗出
    # score_exud = row_signature_score(
    #     first_col,
    #     EXUDATION_QUADRANT_ROWS + EXUDATION_SUMMARY_ROWS
    # )
    # if score_exud >= 2:
    #     return "出血渗出"
    if first_col == EXUDATION_QUADRANT_FIRST_COL:
        return "出血渗出"

    # 2) 血管参数
    # score_vessel = row_signature_score(first_col, VESSEL_PARAMETER_ROWS)
    # if score_vessel >= 2:
    #     return "血管参数"
    if first_col == VESSEL_PARAMETER_FIRST_COL:
        return "血管参数"

    # 3) 近视相关参数
    # score_myopia = row_signature_score(first_col, MYOPIA_PARAMETER_ROWS)
    # if score_myopia >= 2:
    #     return "近视相关参数"
    if first_col == MYOPIA_PARAMETER_FIRST_COL:
        return "近视相关参数"

    # 4) 盘沿分析
    # score_rim = row_signature_score(first_col, NEURORETINAL_RIM_ROWS)
    # if score_rim >= 3:
    #     return "盘沿分析"
    if first_col == NEURORETINAL_RIM_FIRST_COL:
        return "盘沿分析"

    # 5) 视盘面积分析
    # score_disc_area = row_signature_score(first_col, OPTIC_DISC_AREA_ROWS)
    # if score_disc_area >= 2:
    #     return "视盘面积分析"
    if first_col == OPTIC_DISC_AREA_FIRST_COL:
        return "视盘面积分析"

    # 6) 视神经纤维层缺损分析
    # score_nerve = row_signature_score(first_col, RETINAL_NERVE_DEFECT_ROWS)
    # grid_text = " ".join(" ".join(r) for r in grid)
    # if score_nerve >= 2 and "是否发现疑似神经纤维层缺损" in grid_text:
    #     return "视神经纤维层缺损分析"
    if first_col == RETINAL_NERVE_DEFECT_FIRST_COL:
        return "视神经纤维层缺损分析"

    # 7) 视神经参数 / 杯盘比分析
    # score_disc = row_signature_score(first_col, OPTIC_DISC_PARAMETER_ROWS)
    # if score_disc >= 2:
    #     width = max(len(r) for r in grid)
    #     if width >= 7:
    #         return "杯盘比分析"
    #     return "视神经参数"
    if first_col == OPTIC_DISC_PARAMETER_FIRST_COL:
        width = max(len(r) for r in grid)
        if width >= 7:
            return "杯盘比分析"
        return "视神经参数"

    # 最后才弱依赖上下文
    ctx = " ".join(prev_lines)
    if "出血渗出" in ctx:
        return "出血渗出"
    if "血管参数" in ctx:
        return "血管参数"
    if "近视相关参数" in ctx:
        return "近视相关参数"
    if "视神经纤维层缺损分析" in ctx:
        return "视神经纤维层缺损分析"
    if "杯盘比分析" in ctx:
        return "杯盘比分析"
    if "盘沿分析" in ctx:
        return "盘沿分析"
    if "视盘面积分析" in ctx:
        return "视盘面积分析"

    return None


# =========================
# 7. 表格模板重建 / 抽取
# =========================

def row_is_empty(row: List[str]) -> bool:
    return all(not normalize_text(x) for x in row)


def normalize_grid(grid: List[List[str]]) -> List[List[str]]:
    grid = [[normalize_text(x) for x in row] for row in grid]
    grid = [row for row in grid if not row_is_empty(row)]
    if not grid:
        return []
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    return grid


def find_rows_by_schema(grid: List[List[str]], wanted_rows: List[str]) -> List[List[str]]:
    """
    从表格中按第一列匹配需要的行。
    """
    found = []
    for target in wanted_rows:
        matched = None
        for row in grid:
            if row and fuzzy_line_eq(row[0], target):
                matched = row
                break
        if matched:
            found.append(matched)
        else:
            found.append([target])  # 占位，后续补空
    return found


def build_row_dict(columns: List[str], row: List[str]) -> Dict[str, Optional[str]]:
    row = row + [""] * (len(columns) - len(row))
    return {columns[i]: (row[i] if i < len(row) and row[i] != "" else None) for i in range(len(columns))}


def parse_exudation_table(grid: List[List[str]]) -> Dict[str, Any]:
    """
    出血渗出：
    - 不依赖原始复杂表头
    - 只依赖第一列行名和数据列顺序
    - 优先重建为四象限表
    """
    grid = normalize_grid(grid)
    result = {
        "schema_type": None,
        "columns": None,
        "rows": {},
        "warnings": [],
    }

    # 找真实数据行
    matched_rows = {}
    for row_name in EXUDATION_QUADRANT_ROWS:
        matched = find_best_matching_row(grid, row_name)
        if matched is not None:
            matched_rows[row_name] = matched

    if not matched_rows:
        result["warnings"].append("未找到出血渗出数据行")
        return result

    # 根据实际行宽判断是哪种 schema
    max_width = max(len(r) for r in matched_rows.values())

    # 四象限表优先
    if max_width >= 15 or len(matched_rows) >= 3:
        result["schema_type"] = "EXUDATION_QUADRANT"
        result["columns"] = EXUDATION_QUADRANT_COLUMNS

        for row_name in EXUDATION_QUADRANT_ROWS:
            matched = matched_rows.get(row_name)
            if matched is None:
                result["warnings"].append(f"未找到行: {row_name}")
                padded = [row_name] + [""] * (15 - 1)
            else:
                padded = matched + [""] * (15 - len(matched))
                padded = padded[:15]

            result["rows"][row_name] = build_row_dict(EXUDATION_QUADRANT_COLUMNS, padded)

        return result

    # 否则按 summary 表
    result["schema_type"] = "EXUDATION_SUMMARY"
    result["columns"] = EXUDATION_SUMMARY_COLUMNS

    for row_name in EXUDATION_SUMMARY_ROWS:
        matched = find_best_matching_row(grid, row_name)
        if matched is None:
            result["warnings"].append(f"未找到行: {row_name}")
            padded = [row_name] + [""] * (7 - 1)
        else:
            padded = matched + [""] * (7 - len(matched))
            padded = padded[:7]

        result["rows"][row_name] = build_row_dict(EXUDATION_SUMMARY_COLUMNS, padded)

    return result


def parse_fixed_schema_table(
    grid: List[List[str]],
    columns: List[str],
    wanted_rows: List[str],
) -> Dict[str, Any]:
    """
    按固定 schema 抽取：
    - 用第一列匹配真实数据行
    - 不依赖上下文
    - 不依赖原始表头
    """
    grid = normalize_grid(grid)
    result = {
        "columns": columns,
        "rows": {},
        "warnings": [],
    }

    target_width = len(columns)

    for row_name in wanted_rows:
        matched = find_best_matching_row(grid, row_name)

        if matched is None:
            result["warnings"].append(f"未找到行: {row_name}")
            padded = [row_name] + [""] * (target_width - 1)
        else:
            # 有些表会在前面混入表头；如果匹配到的是表头要跳过
            if is_header_like_row(matched) and not fuzzy_line_eq(matched[0], row_name):
                result["warnings"].append(f"匹配到疑似表头而非数据行: {row_name}")
                padded = [row_name] + [""] * (target_width - 1)
            else:
                padded = matched + [""] * (target_width - len(matched))
                padded = padded[:target_width]

        result["rows"][row_name] = build_row_dict(columns, padded)

    return result


def extract_tables(md: str) -> Dict[str, Any]:
    tables = find_tables_with_context(md)
    results = {}
    unknown_tables = []

    for i, item in enumerate(tables):
        grid = item["grid"]
        prev_lines = item["prev_lines"]
        table_type = infer_table_type(prev_lines, grid)
        print(f"[DEBUG] table #{i} => {table_type}, first_col={first_col_values(grid)}")

        if table_type == "出血渗出":
            results["出血渗出"] = parse_exudation_table(grid)

        elif table_type == "血管参数":
            results["血管参数"] = parse_fixed_schema_table(
                grid, VESSEL_PARAMETER_COLUMNS, VESSEL_PARAMETER_ROWS
            )

        elif table_type == "视神经参数":
            results["视神经参数"] = parse_fixed_schema_table(
                grid, OPTIC_DISC_PARAMETER_COLUMNS, OPTIC_DISC_PARAMETER_ROWS
            )

        elif table_type == "近视相关参数":
            results["近视相关参数"] = parse_fixed_schema_table(
                grid, MYOPIA_PARAMETER_COLUMNS, MYOPIA_PARAMETER_ROWS
            )

        elif table_type == "视神经纤维层缺损分析":
            results["视神经纤维层缺损分析"] = parse_fixed_schema_table(
                grid, RETINAL_NERVE_DEFECT_COLUMNS, RETINAL_NERVE_DEFECT_ROWS
            )

        elif table_type == "杯盘比分析":
            results["杯盘比分析"] = parse_fixed_schema_table(
                grid, CUP_DISC_RATIO_COLUMNS, CUP_DISC_RATIO_ROWS
            )

        elif table_type == "盘沿分析":
            results["盘沿分析"] = parse_fixed_schema_table(
                grid, NEURORETINAL_RIM_COLUMNS, NEURORETINAL_RIM_ROWS
            )

        elif table_type == "视盘面积分析":
            results["视盘面积分析"] = parse_fixed_schema_table(
                grid, OPTIC_DISC_AREA_COLUMNS, OPTIC_DISC_AREA_ROWS
            )

        else:
            unknown_tables.append({
                "index": i,
                "prev_lines": prev_lines,
                "grid": grid,
            })

    if unknown_tables:
        results["_unknown_tables"] = unknown_tables

    return results


# =========================
# 8. 总入口
# =========================

def extract_report_from_md(md: str) -> Dict[str, Any]:
    basic_info = extract_basic_info(md)
    dr_analysis = extract_dr_analysis(md)
    assessment = extract_assessment_result(md)
    tables = extract_tables(md)

    return {
        "基本信息": basic_info,
        "糖尿病性视网膜病变分析": dr_analysis,
        "评估结果": assessment,
        "表格": tables,
    }


def extract_report_from_md_file(md_path: str, output_json_path: Optional[str] = None) -> Dict[str, Any]:
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    result = extract_report_from_md(md)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# =========================
# 9. 示例 main
# =========================

if __name__ == "__main__":
    # 改成你的 markdown 文件路径
    md_path = "/data1/jiazy/eye_project/report_extractor/data/ir_output/example4/example4.md"
    output_path = "/data1/jiazy/eye_project/report_extractor/data/ir_output/example4/report_extracted.json"

    result = extract_report_from_md_file(md_path, output_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))