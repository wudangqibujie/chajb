import pandas as pd
from Information.utils import Information_entropy, infomation_val, create_iformation_entropy
from jay_plot.utils import plot_two


df = pd.read_csv("data/train.csv")
# df_rslt = infomation_val(df, "area_id")
# plot_two(df_rslt, "area_id", "info_val", "prob")
df_entrpy = create_iformation_entropy(df, df.columns)
print(df_entrpy.sort_values(by="entropy_val"))

