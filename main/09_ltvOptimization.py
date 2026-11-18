from IPython.core.pylabtools import figsize
from sympy.polys.rootisolation import dup_count_real_roots
from lifelines import CoxPHFitter
from econml.dml import LinearDML
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("data/synthetic/customers.csv")

X = df[['historicalSpend', 'engagement', 'tenureMonths', 'purchaseFrequency']]
T = df['treatment']
Y = df['revenue90d']

dml = LinearDML(
    model_y=RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    ),

    model_t=RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    ),

    discrete_treatment=True,
    cv=3,
    random_state=42

)

dml.fit(Y, T, X=X)

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

monthlyMargin = 300

futureGrid = np.arange(3, 36.01, 0.25)

controlSurvival = np.interp(futureGrid, survivalCurves.index, survivalCurves["No Discount"])
treatmentSurvival = np.interp(futureGrid, survivalCurves.index, survivalCurves["Discount"])

incrementalSurvival = (treatmentSurvival - controlSurvival)

incrementalMonths = np.trapezoid(incrementalSurvival, futureGrid)
print("Additionel expected active months: ", incrementalMonths)

futureSurvivalValue = (monthlyMargin * incrementalMonths)
print("Estimated future contribution value:", futureSurvivalValue)

contributionMarginRate = 0.40

discountCost = 200

estimated90dIncrementalRevenue = 150

incremental90dMargin = (estimated90dIncrementalRevenue * contributionMarginRate)

incrementalProfit = (incremental90dMargin + futureSurvivalValue - discountCost)

print("Expected incremental profit: ", incrementalProfit)

df["estimatedCate"] = dml.effect(X)
df['interventionCost'] = 200
df['expectedIncrementalProfit'] = (df['estimatedCate'] * contributionMarginRate - df['interventionCost'])

priorityCustomers = df.sort_values("expectedIncrementalProfit", ascending=False)

print(priorityCustomers[["customerId", "estimatedCate", "expectedIncrementalProfit"]].head(20))

budget = 100000
costPerCustomer = 200
maxCustomers = int(budget // costPerCustomer)

eligible = df[df["expectedIncrementalProfit"] > 0].copy()

selected = eligible.nlargest(maxCustomers, "expected_incremental_profit")

print("Customers selected: ", len(selected))
print("Intervention cost: ", len(selected) * costPerCustomer)

print("Estimated incremental profit: ", selected["expected_incremental_profit"].sum())