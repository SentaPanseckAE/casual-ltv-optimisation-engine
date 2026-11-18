from econml.dml import LinearDML
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

ate = dml.ate(X=X)
print("Estimated ATE:", ate)

ateInterval = dml.ate_interval(X=X, alpha=0.05)

print(ateInterval)
print("True ATE: ", (df['revenue1'] - df['revenue0'].mean()))
print("Estimated DML ATE: ", ate)

cate = dml.effect(X)
df['estimatedCate'] = cate

sns.histplot(df['estimatedCate'], kde=True)

plt.title("Distribution of Estimated CATE")
plt.xlabel("Estimated Incremental Revenue")
plt.show()

df['engagementGroup'] = pd.qcut(df['engagement'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

print(df.groupby('engagementGroup')['estimatedCate'].mean())