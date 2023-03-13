import pandas as pd
import math


def infomation_val(df, col):
    df_gb = df.groupby(by=col)[col].count()
    total_cnt = sum(df_gb)
    df_rt = round(df_gb / total_cnt, 5)
    df_rslt = pd.DataFrame()
    df_rslt["info_val"] = df_rt.map(lambda x :- math.log2(x))
    df_rslt["prob"] = df_rt
    return df_rslt


def Information_entropy(df):
    df["val"] = df["info_val"] * df["prob"]
    return df["val"].sum()


def create_iformation_entropy(df, columns):
    entropy_vals = []
    for col in columns:
        entropy_val = Information_entropy(infomation_val(df[[col]], col))
        entropy_vals.append(entropy_val)
    df = pd.DataFrame()
    df["cols"] = columns
    df["entropy_val"] = entropy_vals
    return df

