
import re
import sys
import subprocess
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "scalability"

DATASET_SIZES = [
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000
]

SEQUENTIAL_SCRIPT = PROJECT_ROOT / "src" / "sequential_join.py"
PARALLEL_SCRIPT = PROJECT_ROOT / "src" / "parallel_join.py"
GPU_SCRIPT = PROJECT_ROOT / "src" / "gpu_join.py"


def run_script(script, name, arguments):

    print("\n" + "=" * 65)
    print(f"RUNNING {name}")
    print("=" * 65)

    command = [
        sys.executable,
        str(script)
    ] + [str(arg) for arg in arguments]

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    print(process.stdout)

    if process.returncode != 0:
        print(process.stderr)
        raise RuntimeError(f"{name} failed.")

    return process.stdout


def extract_metric(output, pattern):

    match = re.search(pattern, output)

    if not match:
        raise ValueError(
            f"Could not extract metric: {pattern}"
        )

    return float(
        match.group(1).replace(",", "")
    )


def extract_records(output):

    return int(
        extract_metric(
            output,
            r"Matching Records:\s*([\d,]+)"
        )
    )


def validate_outputs(sequential_path, parallel_path, gpu_path):

    print("\n---------- OUTPUT VALIDATION ----------")

    sequential = pd.read_csv(sequential_path)
    parallel = pd.read_csv(parallel_path)
    gpu = pd.read_csv(gpu_path)

    sequential = (
        sequential.sort_values("purchase_id")
        .reset_index(drop=True)
    )

    parallel = (
        parallel.sort_values("purchase_id")
        .reset_index(drop=True)
    )

    gpu = (
        gpu.sort_values("purchase_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        sequential,
        parallel,
        check_dtype=False,
        check_exact=True
    )

    print("Sequential vs Parallel: PASSED")

    pd.testing.assert_frame_equal(
        sequential,
        gpu,
        check_dtype=False,
        check_exact=True
    )

    print("Sequential vs GPU: PASSED")

    print(
        f"All implementations match: "
        f"{len(sequential):,} records"
    )

    del sequential, parallel, gpu


def benchmark_dataset(size):

    print("\n\n")
    print("#" * 70)
    print(f"BENCHMARKING DATASET: {size:,} PURCHASES")
    print("#" * 70)

    data_dir = (
        PROJECT_ROOT / "data" / "benchmarks" / str(size)
    )

    output_dir = RESULTS_DIR / str(size)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequential_path = output_dir / "sequential.csv"
    parallel_path = output_dir / "parallel.csv"
    gpu_path = output_dir / "gpu.csv"

    # ------------------------------------------------
    # SEQUENTIAL CPU
    # ------------------------------------------------

    sequential_output = run_script(
        SEQUENTIAL_SCRIPT,
        f"SEQUENTIAL CPU - {size:,}",
        [
            "--data-dir", data_dir,
            "--output", sequential_path
        ]
    )

    sequential_time = extract_metric(
        sequential_output,
        r"Total Time:\s*([\d.]+)"
    )

    sequential_build = extract_metric(
        sequential_output,
        r"Build Time:\s*([\d.]+)"
    )

    sequential_probe = extract_metric(
        sequential_output,
        r"Probe Time:\s*([\d.]+)"
    )

    sequential_records = extract_records(sequential_output)

    sequential_throughput = extract_metric(
        sequential_output,
        r"Throughput:\s*([\d,.]+)"
    )

    # ------------------------------------------------
    # PARALLEL CPU
    # ------------------------------------------------

    parallel_output = run_script(
        PARALLEL_SCRIPT,
        f"PARALLEL CPU - {size:,}",
        [
            "--data-dir", data_dir,
            "--output", parallel_path
        ]
    )

    parallel_time = extract_metric(
        parallel_output,
        r"Total Time:\s*([\d.]+)"
    )

    parallel_build = extract_metric(
        parallel_output,
        r"Build Time:\s*([\d.]+)"
    )

    parallel_probe = extract_metric(
        parallel_output,
        r"Probe Time:\s*([\d.]+)"
    )

    parallel_records = extract_records(parallel_output)

    parallel_throughput = extract_metric(
        parallel_output,
        r"Throughput:\s*([\d,.]+)"
    )

    workers_match = re.search(
        r"Workers used:\s*(\d+)",
        parallel_output
    )

    workers = (
        int(workers_match.group(1))
        if workers_match else 1
    )

    # ------------------------------------------------
    # INTEL GPU OPENCL
    # ------------------------------------------------

    gpu_output = run_script(
        GPU_SCRIPT,
        f"INTEL GPU OPENCL - {size:,}",
        [
            "--data-dir", data_dir,
            "--output", gpu_path,
            "--expected", sequential_path
        ]
    )

    gpu_cold_time = extract_metric(
        gpu_output,
        r"Total execution time:\s*([\d.]+)"
    )

    gpu_initialization = extract_metric(
        gpu_output,
        r"Initialization time:\s*([\d.]+)"
    )

    gpu_cold_build = extract_metric(
        gpu_output,
        r"GPU hash build time:\s*([\d.]+)"
    )

    gpu_cold_probe = extract_metric(
        gpu_output,
        r"GPU probe time:\s*([\d.]+)"
    )

    gpu_cold_copy = extract_metric(
        gpu_output,
        r"GPU result copy time:\s*([\d.]+)"
    )

    gpu_warm_time = extract_metric(
        gpu_output,
        r"Average warm total time:\s*([\d.]+)"
    )

    gpu_warm_build = extract_metric(
        gpu_output,
        r"Average build time:\s*([\d.]+)"
    )

    gpu_warm_probe = extract_metric(
        gpu_output,
        r"Average probe time:\s*([\d.]+)"
    )

    gpu_warm_copy = extract_metric(
        gpu_output,
        r"Average result copy time:\s*([\d.]+)"
    )

    gpu_warm_throughput = extract_metric(
        gpu_output,
        r"Warm Throughput:\s*([\d,.]+)"
    )

    gpu_records = extract_records(gpu_output)

    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------

    if not (
        sequential_records
        == parallel_records
        == gpu_records
        == size
    ):
        raise AssertionError(
            f"Record count mismatch for dataset {size}"
        )

    validate_outputs(
        sequential_path,
        parallel_path,
        gpu_path
    )

    # ------------------------------------------------
    # COLLECT RESULTS
    # ------------------------------------------------

    rows = [
        {
            "Dataset_Size": size,
            "Implementation": "Sequential CPU",
            "Mode": "Single execution",
            "Build_Time": sequential_build,
            "Probe_Time": sequential_probe,
            "Total_Time": sequential_time,
            "Throughput": sequential_throughput,
            "Initialization_Time": 0.0,
            "Result_Copy_Time": 0.0,
            "Workers": 1
        },
        {
            "Dataset_Size": size,
            "Implementation": "Parallel CPU",
            "Mode": "Single execution",
            "Build_Time": parallel_build,
            "Probe_Time": parallel_probe,
            "Total_Time": parallel_time,
            "Throughput": parallel_throughput,
            "Initialization_Time": 0.0,
            "Result_Copy_Time": 0.0,
            "Workers": workers
        },
        {
            "Dataset_Size": size,
            "Implementation": "Intel GPU OpenCL",
            "Mode": "Cold execution",
            "Build_Time": gpu_cold_build,
            "Probe_Time": gpu_cold_probe,
            "Total_Time": gpu_cold_time,
            "Throughput": size / gpu_cold_time,
            "Initialization_Time": gpu_initialization,
            "Result_Copy_Time": gpu_cold_copy,
            "Workers": None
        },
        {
            "Dataset_Size": size,
            "Implementation": "Intel GPU OpenCL",
            "Mode": "Warm average (5 runs)",
            "Build_Time": gpu_warm_build,
            "Probe_Time": gpu_warm_probe,
            "Total_Time": gpu_warm_time,
            "Throughput": gpu_warm_throughput,
            "Initialization_Time": 0.0,
            "Result_Copy_Time": gpu_warm_copy,
            "Workers": None
        }
    ]

    for row in rows:

        row["Speedup_vs_Sequential"] = (
            sequential_time / row["Total_Time"]
            if row["Total_Time"] > 0 else 0
        )

        if row["Implementation"] == "Parallel CPU":
            row["Efficiency_Percent"] = (
                row["Speedup_vs_Sequential"] / workers * 100
            )
        else:
            row["Efficiency_Percent"] = None

    print("\n---------- DATASET SUMMARY ----------")
    print(f"Dataset size: {size:,}")
    print(f"Sequential CPU: {sequential_time:.6f} seconds")
    print(f"Parallel CPU:   {parallel_time:.6f} seconds")
    print(f"GPU cold:       {gpu_cold_time:.6f} seconds")
    print(f"GPU warm:       {gpu_warm_time:.6f} seconds")

    return rows


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PARALLEL HASH JOIN - SCALABILITY BENCHMARK")
    print("=" * 70)

    all_results = []

    for size in DATASET_SIZES:

        data_dir = (
            PROJECT_ROOT / "data" / "benchmarks" / str(size)
        )

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_dir}"
            )

        rows = benchmark_dataset(size)
        all_results.extend(rows)

        # Save progress after each dataset.
        progress_df = pd.DataFrame(all_results)

        progress_path = (
            PROJECT_ROOT / "results" /
            "scalability_results.csv"
        )

        progress_df.to_csv(
            progress_path,
            index=False
        )

        print(f"\nProgress saved: {progress_path}")

    # ------------------------------------------------
    # FINAL RESULTS
    # ------------------------------------------------

    df = pd.DataFrame(all_results)

    results_path = (
        PROJECT_ROOT / "results" /
        "scalability_results.csv"
    )

    df.to_csv(results_path, index=False)

    print("\n\n")
    print("=" * 70)
    print("FINAL SCALABILITY BENCHMARK RESULTS")
    print("=" * 70)

    print(
        df[
            [
                "Dataset_Size",
                "Implementation",
                "Mode",
                "Total_Time",
                "Throughput",
                "Speedup_vs_Sequential"
            ]
        ].to_string(
            index=False,
            formatters={
                "Total_Time": lambda x: f"{x:.6f} s",
                "Throughput": lambda x: f"{x:,.2f}",
                "Speedup_vs_Sequential": lambda x: f"{x:.2f}x"
            }
        )
    )

    print("\nResults saved to:")
    print(results_path)

    print("\nAll dataset benchmarks completed successfully.")


if __name__ == "__main__":
    main()
