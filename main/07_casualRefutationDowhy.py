from joblib.parallel import method
from dowhy import CausalModel
import pandas as pd

df = pd.read_csv("data/synthetic/customers.csv")

covariates = [
    'historicalSpend',
    'engagement',
    'tenureMonths',
    'purchaseFrequency'
]

casual_df = df[covariates + ['treatment', 'revenue90d']].copy()

model = CausalModel(
    data=casual_df,
    treatment="treatment",
    outcome='revenue90d',
    common_causes=covariates
)

estimand = model.identify_effect()
print(estimand)

estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression", target_units="ate")
print("Dowhy ATE: ", estimate.value)

randomRefutation = model.refute_estimate(estimand, estimate, method_name="random_common_cause")
print(randomRefutation)

placeboRefutation = model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute")
print(placeboRefutation)

subsetRefutation = model.refute_estimate(estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.9)
print(subsetRefutation)