
import pandas as pd
import time
import argparse
from pathlib import Path


def sequential_hash_join(customers, purchases):

    # PHASE 1: BUILD
    build_start = time.perf_counter()

    hash_table = {}

    for row in customers.itertuples(index=False):
        hash_table[row.customer_id] = row.customer_name

    build_time = time.perf_counter() - build_start

    # PHASE 2: PROBE
    probe_start = time.perf_counter()

    results = []

    for row in purchases.itertuples(index=False):
        customer_name = hash_table.get(row.customer_id)

        if customer_name is not None:
            results.append({
                "purchase_id": row.purchase_id,
                "customer_id": row.customer_id,
                "customer_name": customer_name,
                "amount": row.amount
            })

    probe_time = time.perf_counter() - probe_start

    return results, build_time, probe_time


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing customers.csv and purchases.csv"
    )
    parser.add_argument(
        "--output",
        default="results/sequential_output.csv",
        help="Path for the joined output CSV"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    print("\n========== SEQUENTIAL HASH JOIN ==========")

    customers = pd.read_csv(data_dir / "customers.csv")
    purchases = pd.read_csv(data_dir / "purchases.csv")

    print(f"\nCustomers loaded: {len(customers):,}")
    print(f"Purchases loaded: {len(purchases):,}")

    results, build_time, probe_time = sequential_hash_join(
        customers, purchases
    )

    total_time = build_time + probe_time

    print("\n---------- PERFORMANCE RESULTS ----------")
    print(f"Build Time:  {build_time:.6f} seconds")
    print(f"Probe Time:  {probe_time:.6f} seconds")
    print(f"Total Time:  {total_time:.6f} seconds")

    print(f"\nMatching Records: {len(results):,}")

    throughput = len(purchases) / total_time if total_time > 0 else 0

    print(f"Throughput: {throughput:,.2f} records/second")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame(results)
    output.to_csv(output_path, index=False)

    print(f"\nOutput saved: {output_path}")

    print("\n---------- SAMPLE JOINED RECORDS ----------")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
