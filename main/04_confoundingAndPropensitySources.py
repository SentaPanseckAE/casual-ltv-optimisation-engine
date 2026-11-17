from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/synthetic/customers.csv")

covariates = [
    'historicalSpend',
    'engagement',
    'tenureMonths',
    'purchaseFrequency'
]

X = df[covariates]
T = df['treatment']

propensityModel = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

propensityModel.fit(X, T)

df['propensityScore'] = (propensityModel.predict_proba(X)[:, 1])

print(df[['customerId', 'treatment', "propensityScore"]].head(10))

sns.histplot(
    data = df,
    x = 'propensityScore',
    hue = 'treatment',
    bins = 30,
    stat = 'density',
    common_norm = False
)
plt.title("Propensity Score Overlap")
plt.show()