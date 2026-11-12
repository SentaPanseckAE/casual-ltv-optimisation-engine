import numpy as np
import pandas as pd
from scipy.special import expit


def generateCustomerData(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    tenure = rng.gamma(3, 8, n)
    spend = rng.lognormal(7, 0.6, n)
    engagement = rng.beta(2, 2, n)
    frequency = rng.poisson(4, n)

    spendZ = (spend - spend.mean()) / (spend.std())
    tenureZ = (tenure - tenure.mean()) / (tenure.std())
    frequencyZ = (frequency - frequency.mean()) / (frequency.std() + 1e-8)
    truePropensity = expit(-1.0 + 0.55*spendZ + 0.75*engagement + 0.25*tenureZ + 0.35*frequencyZ)

    treatment = rng.binomial(1, truePropensity)

    trueCate = (100 + 120*(1 -engagement) + 20*spendZ)
    baselineRevenue = (350 + 75*spendZ + 55*engagement + 15*tenureZ + 30*frequencyZ + rng.normal(0, 100, n))

    revenue0 = baselineRevenue
    revenue1 = baselineRevenue + trueCate

    revenue90d = np.where(treatment == 1, revenue1, revenue0)

    logHazard = (-3.5 - 0.95*engagement - 0.15*spendZ - 0.25*treatment)
    individualHazard = np.exp(logHazard)

    eventTime = rng.exponential(scale=1 / individualHazard)

    censorTime = rng.uniform(12, 36, n)

    duration = np.minimum(eventTime, censorTime)

    event = (eventTime <= censorTime).astype(int)

    df = pd.DataFrame({
        "customerId": np.arange(n),
        "tenureMonths": tenure,
        "historicalSpend": spend,
        "engagement": engagement,
        "purchaseFrequency": frequency,
        "treatment": treatment,
        "revenue90d": revenue90d,
        "churnEvent": event,
        "durationMonths": duration,
        "trueCate": trueCate,
        "truePropensity": truePropensity,
        "revenue0": revenue0,
        "revenue1": revenue1

    })

    return df