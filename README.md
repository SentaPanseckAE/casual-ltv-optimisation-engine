# Causal LTV Optimization Engine

A Python project for estimating the causal effect of a marketing intervention on customer revenue and retention, then using those estimates to prioritize customers for a budget-limited campaign.

The project combines synthetic data generation, statistical testing, causal inference, survival analysis, and incremental-profit calculations. It is designed for learning and experimentation, not as a production decision-making service.

## Project Goal

A conventional model can identify customers who are likely to generate high revenue or churn. That is useful for prediction, but it does not necessarily identify customers who will benefit from an intervention.

For example, engaged or high-spending customers may be more likely to receive a discount and may also generate more revenue without it. A comparison of all treated and untreated customers can therefore mix the intervention effect with differences that existed before treatment.

This project asks four related questions:

1. Did the intervention cause higher 90-day revenue?
2. How much did the effect vary between customer types?
3. Did the intervention reduce customer churn?
4. Which customers have positive expected incremental profit within a fixed budget?

The synthetic data include known treatment effects and propensity scores. These values allow estimated results to be checked against the data-generating process, while the observed columns remain representative of a normal observational analysis.

## What It Does

- Generates 10,000 synthetic customer records with a fixed random seed.
- Examines data quality and the distribution of features and outcomes.
- Compares treated and control revenue using a Welch two-sample t-test.
- Estimates the sample size needed to detect a specified effect.
- Estimates propensity scores and examines overlap between groups.
- Matches similar customers and checks covariate balance.
- Uses Double Machine Learning to estimate average and customer-level effects.
- Applies DoWhy refutation tests as robustness checks.
- Fits a Cox proportional hazards model to estimate retention effects.
- Estimates restricted mean survival time for treated and control profiles.
- Converts modeled effects into incremental profit and ranks customers for targeting.

## Analysis Workflow

### 1. Synthetic Data Generation

`main/01_syntheticDataGeneration.py` calls `generateCustomerData()` in `src/dataGeneration.py`. It creates a CSV containing customer features, treatment assignment, revenue, churn outcomes, survival duration, and ground-truth variables.

The default generator uses 10,000 rows and seed 42. Tenure is sampled from a Gamma distribution, historical spend from a log-normal distribution, engagement from a Beta distribution, and purchase frequency from a Poisson distribution.

Treatment is not assigned randomly. The probability of treatment depends on standardized historical spend, engagement, tenure, and purchase frequency through a logistic function. This creates the confounding pattern that the later methods attempt to address.

The revenue outcome includes baseline revenue plus a customer-specific treatment effect. Synthetic churn time follows an exponential hazard. Censoring time is sampled from a uniform distribution, and observed duration is the earlier of the event or censoring time.

### 2. Data Quality and Exploratory Analysis

`main/02_dataQualityAndEDA.py` reads the generated CSV and reports its shape, schema, summary statistics, missing values, duplicate identifiers, and treatment prevalence.

It also creates basic visualizations for customer spending and feature distributions. This stage establishes whether the generated data are structurally valid before running causal or survival methods.

### 3. Hypothesis Testing and Power Analysis

`main/03_hypothesisTestingPower.py` compares 90-day revenue between treated and control customers with `scipy.stats.ttest_ind(..., equal_var=False)`, corresponding to Welch's t-test.

The script reports the test statistic and p-value. It also defines a 50-unit minimum detectable effect and calculates a required sample size for 80 percent power at a 5 percent significance level.

The test estimates a raw association, not a causal effect. It does not adjust for the observed differences that affect treatment assignment.

### 4. Confounding and Propensity Scores

`main/04_confoundingAndPropensitySources.py` uses a standardized logistic regression model to estimate each customer's propensity for treatment from:

- `historicalSpend`
- `engagement`
- `tenureMonths`
- `purchaseFrequency`

A propensity score is:

```text
P(treatment = 1 | observed customer features)
```

The script stores the estimated score and plots treated and control score distributions. Overlapping distributions indicate that comparable treated and control observations exist. Limited overlap indicates extrapolation risk or a weak basis for causal comparison.

This step demonstrates the selection process but does not by itself prove that a treatment effect is unbiased.

### 5. Propensity Score Matching

`main/05_propensityScoreMatching.py` matches each treated customer to the nearest unused control customer using logit-transformed propensity scores.

It applies a caliper of 0.2 times the standard deviation of the logit score. A pair is retained only when its control is close enough to the treated customer.

For every retained pair, the script calculates control revenue subtracted from treated revenue. The average paired difference estimates the average treatment effect on the treated, or ATT.

It also calculates standardized mean differences for the four covariates before and after matching. Lower differences after matching indicate improved observed balance. A common informal threshold is 0.10, although this threshold is not a guarantee of valid causal inference.

### 6. Double Machine Learning

`main/06_doubleMachineLearning.py` uses EconML's `LinearDML` implementation.

The model uses:

- A random forest regressor to model revenue.
- A random forest classifier to model treatment.
- Three-fold cross-fitting.
- A linear treatment-effect specification.
- Random seed 42 for reproducibility.

DML separates nuisance prediction from treatment-effect estimation. It reports:

- The estimated average treatment effect.
- A 95 percent confidence interval for the ATE.
- An individual conditional average treatment effect for every customer.
- A distribution of estimated effects.
- CATE averages across engagement quartiles.
- CATE root mean squared error against the synthetic ground truth.

The CATE is a conditional average rather than an exact observed effect for an individual customer. It may vary across customers because the synthetic outcome function differs by engagement and spend. Individual estimates should be interpreted with uncertainty.

### 7. Causal Refutation

`main/07_casualRefutationDowhy.py` uses DoWhy to define the treatment, outcome, and observed common causes. It identifies an effect and estimates it with backdoor linear regression.

The script performs three perturbation tests:

- **Random common cause:** adds an irrelevant random cause and checks whether the estimate changes substantially.
- **Placebo treatment:** permutes treatment assignment and checks whether the estimated effect disappears.
- **Data subset:** re-estimates the effect on a randomly selected 90 percent subset.

A placebo estimate near zero is reassuring. Stable estimates across subset samples support robustness but do not prove that unmeasured confounding is absent.

### 8. Survival Analysis

`main/08_survivalAnalysisCox.py` fits a Cox proportional hazards model to treatment, historical spend, engagement, tenure, and purchase frequency.

The duration column is used as follow-up time. The churn-event column is used as the event indicator. The model reports coefficients and hazard ratios.

A hazard ratio below one indicates a lower estimated churn hazard relative to the reference category or variable level. A ratio above one indicates a higher estimated hazard.

The script predicts survival curves for a median customer profile under treatment and control. The curves estimate the probability of remaining active over time.

It also calculates restricted mean survival time, or RMST, to 36 months. RMST is the expected active time represented by the area under a survival curve up to a fixed horizon. The difference is an interpretable estimate of additional active months for the modeled profile.

The Cox model assumes proportional hazards and an appropriate covariate specification. These assumptions are not automatically satisfied by the fitting process.

### 9. LTV and Customer Targeting

`main/09_ltvOptimization.py` demonstrates how the modeled outputs can inform a customer-selection decision.

The script:

1. Fits a DML model and estimates customer-level treatment effects.
2. Fits a Cox survival model using treatment and customer features.
3. Predicts treated and control survival curves for a median profile.
4. Integrates the survival difference across a 36-month future grid.
5. Converts additional active months into future contribution value.
6. Adds a 90-day incremental margin estimate.
7. Subtracts intervention cost.
8. Ranks customers by expected incremental profit.
9. Selects customers under a fixed budget.

The customer-level CATE is multiplied by a contribution-margin assumption. Future value is calculated from additional modeled active months and a monthly margin assumption. The customer is a candidate for targeting only when expected incremental profit is positive.

These LTV values are approximations. They combine uncertainty from the causal model, survival model, margin assumptions, and intervention cost. They should not be read as guaranteed future profit.

## Dataset

The generated dataset has the following main columns:

| Column | Meaning |
|---|---|
| `customerId` | Unique synthetic customer identifier |
| `tenureMonths` | Customer relationship length |
| `historicalSpend` | Historical customer spending |
| `engagement` | Synthetic engagement score |
| `purchaseFrequency` | Synthetic purchase count |
| `treatment` | Whether the customer received the intervention |
| `revenue90d` | Observed revenue over 90 days |
| `churnEvent` | Whether churn was observed |
| `durationMonths` | Observed active or censored time |
| `trueCate` | Ground-truth conditional treatment effect |
| `truePropensity` | Ground-truth treatment probability |
| `revenue0` | Potential outcome without treatment |
| `revenue1` | Potential outcome with treatment |

## Project Structure

```text
data/
  synthetic/customers.csv
main/
  01_syntheticDataGeneration.py
  02_dataQualityAndEDA.py
  03_hypothesisTestingPower.py
  04_confoundingAndPropensitySources.py
  05_propensityScoreMatching.py
  06_doubleMachineLearning.py
  07_casualRefutationDowhy.py
  08_survivalAnalysisCox.py
  09_ltvOptimization.py
src/
  dataGeneration.py
  preprocessing.py
  survivalModels.py
tests/
requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Run the Project

Run the scripts from the repository root in numbered order:

```bash
python main/01_syntheticDataGeneration.py
python main/02_dataQualityAndEDA.py
python main/03_hypothesisTestingPower.py
python main/04_confoundingAndPropensitySources.py
python main/05_propensityScoreMatching.py
python main/06_doubleMachineLearning.py
python main/07_casualRefutationDowhy.py
python main/08_survivalAnalysisCox.py
python main/09_ltvOptimization.py
```

Some scripts open Matplotlib windows. Close each chart before continuing if the process is waiting for the window to close.

## Technologies

- NumPy and Pandas for data and numerical operations
- SciPy and Statsmodels for hypothesis tests and power analysis
- Matplotlib and Seaborn for visualization
- scikit-learn for preprocessing and propensity models
- EconML for Double Machine Learning
- DoWhy for causal identification and refutation
- Lifelines for Cox survival modeling
- Pytest for tests

## Interpretation and Limitations

- Observational causal estimates depend on the assumption of no unmeasured confounding.
- Propensity score overlap and matching balance do not prove that all confounding is removed.
- Synthetic ground truth will not appear in real-world datasets.
- Customer-level treatment effects are model estimates and may be uncertain or unstable.
- Cross-method differences can reveal sensitivity and should be investigated rather than averaged automatically.
- The Cox model assumes proportional hazards and can require additional diagnostics.
- Censoring assumptions and survival follow-up must be valid for the RMST estimate to be meaningful.
- LTV calculations depend on fixed margin, cost, and horizon assumptions.
- Refutation tests evaluate robustness; they do not establish a causal effect by themselves.
- The scripts are research examples and do not provide a production API or continuous monitoring pipeline.