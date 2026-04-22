import numpy as np
import pandas as pd


df = pd.read_csv("/mnt/hdd/jiazy/eye_project/SFT/splits_qc_clean/train.csv")
print(df.__len__)

df_aug = pd.read_csv("/mnt/hdd/jiazy/eye_project/SFT/splits_qc_clean/train.tail_aug.csv")
print(df_aug.__len__)