from econml.dml import LinearDML
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
import pandas as pd

df = pd.read_csv("data/synthetic/customers.csv")

X = df[['historicalSpend', 'engagement', 'tenureMonths', 'purhcaseFrequency']]