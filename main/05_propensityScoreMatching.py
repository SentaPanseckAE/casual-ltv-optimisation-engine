import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import t
from src.preprocessing import calculateSMD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
smdBefore = calculateSMD(df, 'treatment', covariates)

eps = 1e-6

df['logitPs'] = np.log(df['propensityScore'].clip(eps, 1 - eps) / (1 - df['propensityScore'].clip(eps, 1 - eps)))

treated = df[df['treatment'] == 1].sort_values('logitPs')
control = df[df['treatment'] == 0].copy()

caliper = (0.2 * df['logitPs'].std())

availableControls = set(control.index)

pairs = []

for treatedIdx, treatedRow in treated.iterrows():
    if not availableControls:
        break

    candidateIds = list(availableControls)

    distances = (control.loc[candidateIds, 'logitPs'] - treatedRow['logitPs']).abs()

    nearestControlIdx = distances.idxmin()

    distance = distances.loc[nearestControlIdx]

    if distance <= caliper:
        pairs.append({
            'treatedIdx': treatedIdx,
            'controlIdx': nearestControlIdx,
            'distance': distance
        })

        availableControls.remove(nearestControlIdx)
    
pairs_df = pd.DataFrame(pairs)
print("Matched pairs: ", len(pairs_df))

matchedTreated = df.loc[pairs_df['treatedIdx']].reset_index(drop=True)
matchedControl = df.loc[pairs_df['controlIdx']].reset_index(drop=True)

pairEffects = (matchedTreated['revenue90d'].to_numpy() - matchedControl['revenue90d'].to_numpy())

att = pairEffects.mean()

print("Estimated ATT: ", att)

nPairs = len(pairEffects)

standardError = (pairEffects.std(ddof=1) / np.sqrt(nPairs))
criticalValue = t.ppf(0.975, df = nPairs - 1)

lower = att - criticalValue * standardError
upper = att + criticalValue * standardError

print("ATT: ", att)
print("95% CI:", lower, upper)

matched_df = pd.concat([matchedTreated, matchedControl])
smdAfter = calculateSMD(matched_df, 'treatment', covariates)

balance = pd.DataFrame({'before': smdBefore, 'after': smdAfter})
print(balance)


# balance.abs().plot(kind='barh', figsize=(9,5))
# plt.axvline(0.10, linestyle='--', color='red')

# plt.title("Balance Before and After PSM")
# plt.show()
