from scipy.stats import ttest_ind
from statsmodels.stats.power import TTestIndPower
import numpy as np
import pandas as pd

df = pd.read_csv("data/synthetic/customers.csv")

treated = df.loc[df['treatment'] == 1, 'revenue90d']

control = df.loc[df['treatment'] == 0, 'revenue90d']

result = ttest_ind(treated, control, equal_var=False)

print("T-statistic: ", result.statistic)
print("P-value: ", result.pvalue)

alpha = 0.05
power = 0.80
mde = 50

historicalStd = df['revenue90d'].std()
effectSize = mde / historicalStd

analysis = TTestIndPower()

requiredN = analysis.solve_power(
    effect_size = effectSize,
    alpha = alpha,
    power = power,
    ratio = 1,
    alternative= 'two-sided'
)

print("Sample per group:", np.ceil(requiredN))
print("Total Sample", 2 * np.ceil(requiredN))