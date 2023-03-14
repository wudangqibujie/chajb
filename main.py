import pandas as pd
from Information.utils import Information_entropy, infomation_val, create_iformation_entropy, conditional_entropy, union_entrpy, mutual_information, relative_entropy
from jay_plot.utils import plot_two


df = pd.read_csv("data/train.csv")

# df_rslt = infomation_val(df, "area_id")
# plot_two(df_rslt, "area_id", "info_val", "prob")
# df_entrpy = create_iformation_entropy(df, df.columns)
# print(df_entrpy.sort_values(by="entropy_val"))


for col in ["main_account_active_loan_no", "passport_flag", "Driving_flag"]:
    val = conditional_entropy(df, x_col=col,  y_col="loan_default")
    union_val = union_entrpy(df, x_col=col,  y_col="loan_default")
    mutual_val = mutual_information(df, x_col="loan_default",  y_col=col)
    KL = relative_entropy(df, x_col=col,  y_col="loan_default")
    print(KL, col)


