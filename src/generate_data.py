
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time


def generate_data(customers_count, purchases_count):

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating datasets...")
    start = time.perf_counter()

    rng = np.random.default_rng(42)

    # Generate customers
    customers = pd.DataFrame({
        "customer_id": np.arange(1, customers_count + 1),
        "customer_name": [
            f"Customer_{i}" for i in range(1, customers_count + 1)
        ]
    })

    # Generate purchases
    purchases = pd.DataFrame({
        "purchase_id": np.arange(1, purchases_count + 1),
        "customer_id": rng.integers(
            1, customers_count + 1, size=purchases_count
        ),
        "amount": np.round(
            rng.uniform(100, 10000, size=purchases_count), 2
        )
    })

    customers.to_csv(output_dir / "customers.csv", index=False)
    purchases.to_csv(output_dir / "purchases.csv", index=False)

    elapsed = time.perf_counter() - start

    print("\nDataset generation completed!")
    print(f"Customers: {len(customers):,}")
    print(f"Purchases: {len(purchases):,}")
    print(f"Generation time: {elapsed:.2f} seconds")

    print("\nFiles saved:")
    print("data/customers.csv")
    print("data/purchases.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--customers", type=int, default=10000)
    parser.add_argument("--purchases", type=int, default=50000)

    args = parser.parse_args()

    generate_data(args.customers, args.purchases)
