# Causal LTV Optimization Engine

A Python project that estimates the causal effect of a marketing intervention on
customer revenue and retention, then uses those estimates to rank customers for
budget-constrained targeting.

The project combines synthetic data, causal inference, survival analysis, and
incremental profit calculations. It is intended for learning and experimentation,
not as a production decision-making system.

## What It Does

- Generates 10,000 synthetic customer records.
- Tests and explores treatment effects and confounding.
- Uses propensity score matching and Double Machine Learning (DML).
- Runs causal refutation checks with DoWhy.
- Estimates retention with a Cox proportional hazards model.
- Estimates incremental customer profit and prioritizes customers within a fixed budget.

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

## Data

The synthetic dataset contains customer demographics and activity, treatment
assignment, 90-day revenue, churn status, and duration. It also includes
ground-truth variables for experimentation:

- `historicalSpend`
- `engagement`
- `tenureMonths`
- `purchaseFrequency`
- `treatment`
- `revenue90d`
- `churnEvent`
- `durationMonths`
- `trueCate`
- `truePropensity`

Run the first script to generate `data/synthetic/customers.csv`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
.venv\Scripts\activate
```

## Run

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

Some analysis scripts open Matplotlib charts. Close each chart to continue.

## Technologies

NumPy, Pandas, SciPy, Matplotlib, Seaborn, scikit-learn, Statsmodels,
EconML, DoWhy, Lifelines, and Pytest.

## Limitations

- Causal estimates depend on the assumption of no unmeasured confounding.
- Synthetic results may not represent real customer behavior.
- Customer-level treatment estimates are uncertain.
- The LTV calculation combines modeled short-term and retention effects with
  fixed business assumptions.
- Cox and other models require assumptions that should be checked before use.
- The current scripts are research examples and do not include a production API.

## Tests

```bash
pytest
```
