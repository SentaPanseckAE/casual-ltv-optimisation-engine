import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/synthetic/customers.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.describe().T)

missing = (df.isnull().sum().sort_values(ascending=False))
print(missing)

print(df['customerId']).duplicated().sum()

print(df['treatment'].value_counts(normalize=True))

plt.figure()
sns.histplot(
    data=df, 
    x='historicalSpend', 
    bins=40
)
plt.title("Customer Historical Spending")
plt.show()

plt.figure()
sns.histplot(
    data=df,
    x='engagement',
    bins=30
)
plt.title("Customer Engagement Distribution")
plt.show()

plt.figure()
sns.histplot(
    data=df,
    x='tenureMonths',
    bins=30
)
plt.title("Customer Tenure Distribution")
plt.show()

features = [
    "historicalSpend",
    "engagement",
    "tenureMonths",
    "purchaseFrequency",
    "revenue90d"
]

comparison = (df.groupby)