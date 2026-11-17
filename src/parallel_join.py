
import os
import time
import math
import argparse
import pandas as pd

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

CUSTOMER_HASH = {}


def initialize_worker(customer_hash):
    global CUSTOMER_HASH
    CUSTOMER_HASH = customer_hash


def probe_chunk(purchases_chunk):

    purchases_chunk = purchases_chunk.copy()

    purchases_chunk["customer_name"] = (
        purchases_chunk["customer_id"].map(CUSTOMER_HASH)
    )

    result = purchases_chunk.dropna(
        subset=["customer_name"]
    )

    return result[
        ["purchase_id", "customer_id", "customer_name", "amount"]
    ]


def parallel_hash_join(data_dir="data", output_path="results/parallel_output.csv"):

    print("\n========== PARALLEL HASH JOIN ==========\n")

    data_dir = Path(data_dir)
    output_path = Path(output_path)

    customers = pd.read_csv(data_dir / "customers.csv")
    purchases = pd.read_csv(data_dir / "purchases.csv")

    print(f"Customers loaded: {len(customers):,}")
    print(f"Purchases loaded: {len(purchases):,}")

    # BUILD PHASE
    build_start = time.perf_counter()

    customer_hash = dict(
        zip(customers["customer_id"], customers["customer_name"])
    )

    build_time = time.perf_counter() - build_start

    # CHUNK DIVISION
    workers = min(os.cpu_count() or 1, 4)

    chunk_size = math.ceil(len(purchases) / workers)

    chunks = [
        purchases.iloc[i:i + chunk_size].copy()
        for i in range(0, len(purchases), chunk_size)
    ]

    print(f"CPU logical processors: {os.cpu_count()}")
    print(f"Workers used: {workers}")
    print(f"Chunks created: {len(chunks)}")

    # PARALLEL PROBE PHASE
    probe_start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(customer_hash,)
    ) as executor:

        results = list(
            executor.map(probe_chunk, chunks)
        )

    probe_time = time.perf_counter() - probe_start

    # COMBINE RESULTS
    final_result = pd.concat(
        results,
        ignore_index=True
    )

    total_time = build_time + probe_time
    matching_records = len(final_result)

    throughput = (
        len(purchases) / total_time
        if total_time > 0 else 0
    )

    # VALIDATION
    print("\n---------- VALIDATION ----------")

    print(
        "All purchases matched:",
        matching_records == len(purchases)
    )

    print(
        "Unique purchase IDs:",
        final_result["purchase_id"].nunique()
    )

    # SAVE OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_result.to_csv(output_path, index=False)

    # PERFORMANCE RESULTS
    print("\n---------- PERFORMANCE RESULTS ----------")

    print(f"Build Time:  {build_time:.6f} seconds")
    print(f"Probe Time:  {probe_time:.6f} seconds")
    print(f"Total Time:  {total_time:.6f} seconds")

    print(f"\nMatching Records: {matching_records:,}")
    print(f"Throughput: {throughput:,.2f} records/second")

    print(f"\nOutput saved: {output_path}")

    print("\n---------- SAMPLE JOINED RECORDS ----------")
    print(final_result.head(10).to_string(index=False))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing customers.csv and purchases.csv"
    )

    parser.add_argument(
        "--output",
        default="results/parallel_output.csv",
        help="Path for the joined output CSV"
    )

    args = parser.parse_args()

    parallel_hash_join(
        data_dir=args.data_dir,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
