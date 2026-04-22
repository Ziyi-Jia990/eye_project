import pandas as pd

src = "/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.csv"
cleaned = "/mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.csv"

df_src = pd.read_csv(src)
df_clean = pd.read_csv(cleaned)

# 统一成字符串，避免类型不一致
src_ids = df_src["img_id"].astype(str)
clean_ids = set(df_clean["img_id"].astype(str))

# 保留那些 img_id 也出现在 description.cleaned.csv 里的样本
df_result = df_src[src_ids.isin(clean_ids)].copy()

# 覆盖保存
df_result.to_csv(cleaned, index=False)

print(f"原始 description.csv 行数: {len(df_src)}")
print(f"原始 description.cleaned.csv 行数: {len(df_clean)}")
print(f"删除后保存的行数: {len(df_result)}")
print(f"已覆盖保存到: {cleaned}")