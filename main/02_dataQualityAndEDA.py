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

print(df['customerId'].duplicated().sum())

print(df['treatment'].value_counts(normalize=True))

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
    data=df, 
    x='historicalSpend', 
    bins=40,
    ax=ax
)
plt.title("Customer Historical Spending")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
    data=df,
    x='engagement',
    bins=30,
    ax=ax
)
plt.title("Customer Engagement Distribution")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(
    data=df,
    x='tenureMonths',
    bins=30,
    ax=ax
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

comparison = (df.groupby('treatment')[features].mean().T)
print(comparison)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df,
    x='treatment', 
    y='revenue90d', 
    ax=ax
)
plt.title("Observed Revenue: Treated vs Control")
plt.show()

rawDifference = (
    df.loc[df.treatment == 1, "revenue90d"].mean()
    - df.loc[df.treatment == 0, "revenue90d"].mean()
)

print("Raw revenue difference: ", rawDifference)

