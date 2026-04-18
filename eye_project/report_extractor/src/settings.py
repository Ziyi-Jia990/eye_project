from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_PDF_DIR = DATA_DIR / "input_pdfs"
MONKEY_RAW_DIR = DATA_DIR / "monkey_raw"
IR_OUTPUT_DIR = DATA_DIR / "ir_output"

# 你当前关注的四个表
TARGET_TABLE_TITLES = [
    "出血渗出",
    "血管参数",
    "视神经参数",
    "近视相关参数",
]

# 后面章节抽取会用到，这里先留着
TARGET_SECTION_TITLES = [
    "评估结果",
    "糖尿病性视网膜病变分析",
]

# 出血渗出
# 第一种表格
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

# 第二种表格
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