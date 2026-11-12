import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataGeneration import generateCustomerData

df = generateCustomerData(n=10000)

Path("data/synthetic").mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    "data/synthetic/customers.csv",
    index=False
)

print(df.shape)
print(df.head())