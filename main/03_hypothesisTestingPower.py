from scipy import ttest_ind
import pandas as pd

df = pd.read_csv("data/synthetic/customers.csv")


treated = df.loc[df['treatment'] == 1, 'revenue90d']

control = df.loc[df['treatment'] == 0, 'revenue90d']

result = ttest_ind(treated, control, equal_var=False)

print("T-statistic: ", result.statistic)
print("P-value: ", result.pvalue)