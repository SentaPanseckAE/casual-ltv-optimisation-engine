import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from IPython.core.pylabtools import figsize
from sympy.polys.rootisolation import dup_count_real_roots
from lifelines import CoxPHFitter
from src.survivalModels import calculateRMST
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/synthetic/customers.csv")

survialFeatures = ['treatment', 'historicalSpend', 'engagement', 'tenureMonths', 'purchaseFrequency']
survival_df = df[survialFeatures + ['durationMonths', 'churnEvent']].copy()

cph = CoxPHFitter(penalizer=0.01)
cph.fit(survival_df, duration_col='durationMonths', event_col='churnEvent')
cph.print_summary()
print(cph.hazard_ratios_)

customerProfile = pd.DataFrame({
    'treatment': [0, 1],

    'historicalSpend': [df['historicalSpend'].median()] * 2,

    'engagement': [df['engagement'].median()] * 2,

    'tenureMonths': [df['tenureMonths'].median()] * 2,

    'purchaseFrequency': [df['purchaseFrequency'].median()] * 2
})

survivalCurves = (cph.predict_survival_function(customerProfile))
survivalCurves.columns = ["No Discount", "Discount"]

# survivalCurves.plot(figsize=(9, 5))
# plt.title("Predicted Survival Probability")
# plt.xlabel("Months")
# plt.ylabel("Probability of Remaining Active")
# plt.show()

# cph.check_assumptions(
#     survival_df,
#     p_value_threshold=0.05,
#     show_plots=True
# )

rmstControl = calculateRMST(survivalCurves['No Discount'], horizon=36)
rmstTreatment = calculateRMST(survivalCurves['Discount'], horizon=36)

print("Control RMST: ", rmstControl)
print("Treatment RMST: ", rmstTreatment)
print("RMST Difference: ", rmstTreatment -rmstControl)
