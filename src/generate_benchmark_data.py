
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"

CUSTOMER_COUNT = 10_000

PURCHASE_SIZES = [
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000
]

RANDOM_SEED = 42


def generate_customers():
    return pd.DataFrame({
        "customer_id": np.arange(
            1, CUSTOMER_COUNT + 1, dtype=np.int32
        ),
        "customer_name": [
            f"Customer_{i}"
            for i in range(1, CUSTOMER_COUNT + 1)
        ]
    })


def generate_purchases(size, rng):
    return pd.DataFrame({
        "purchase_id": np.arange(
            1, size + 1, dtype=np.int32
        ),
        "customer_id": rng.integers(
            1,
            CUSTOMER_COUNT + 1,
            size=size,
            dtype=np.int32
        ),
        "amount": np.round(
            rng.uniform(100, 10_000, size=size),
            2
        )
    })


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    customers = generate_customers()

    print("=" * 55)
    print("SCALABLE HASH JOIN DATASET GENERATOR")
    print("=" * 55)

    print(f"Customers per dataset: {CUSTOMER_COUNT:,}")

    for size in PURCHASE_SIZES:
        dataset_dir = DATA_DIR / str(size)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        purchases = generate_purchases(size, rng)

        customers.to_csv(
            dataset_dir / "customers.csv",
            index=False
        )

        purchases.to_csv(
            dataset_dir / "purchases.csv",
            index=False
        )

        print(f"\nDataset: {size:,} purchases")
        print(f"Customers: {len(customers):,}")
        print(f"Purchases: {len(purchases):,}")
        print(f"Saved to: {dataset_dir}")

    print("\nAll benchmark datasets generated successfully.")


if __name__ == "__main__":
    main()
