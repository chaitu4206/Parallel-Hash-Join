
import re
import sys
import subprocess
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SEQUENTIAL_SCRIPT = PROJECT_ROOT / "src" / "sequential_join.py"
PARALLEL_SCRIPT = PROJECT_ROOT / "src" / "parallel_join.py"
GPU_SCRIPT = PROJECT_ROOT / "src" / "gpu_join.py"


def run_script(script_path, name):
    print(f"\n{'=' * 60}")
    print(f"RUNNING {name.upper()}")
    print("=" * 60)

    process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    print(process.stdout)

    if process.returncode != 0:
        print(process.stderr)
        raise RuntimeError(f"{name} execution failed.")

    return process.stdout


def extract_metric(output, pattern):
    match = re.search(pattern, output)

    if not match:
        raise ValueError(f"Could not find metric: {pattern}")

    return float(match.group(1).replace(",", ""))


def extract_records(output):
    return int(
        extract_metric(output, r"Matching Records:\s*([\d,]+)")
    )


def benchmark_sequential():
    output = run_script(SEQUENTIAL_SCRIPT, "Sequential CPU")

    return {
        "Implementation": "Sequential CPU",
        "Mode": "Single execution",
        "Build_Time": extract_metric(
            output, r"Build Time:\s*([\d.]+)"
        ),
        "Probe_Time": extract_metric(
            output, r"Probe Time:\s*([\d.]+)"
        ),
        "Total_Time": extract_metric(
            output, r"Total Time:\s*([\d.]+)"
        ),
        "Matching_Records": extract_records(output),
        "Throughput": extract_metric(
            output, r"Throughput:\s*([\d,.]+)"
        ),
        "Workers": 1,
        "Initialization_Time": 0.0
    }


def benchmark_parallel():
    output = run_script(PARALLEL_SCRIPT, "Parallel CPU")

    workers_match = re.search(r"Workers used:\s*(\d+)", output)
    workers = int(workers_match.group(1)) if workers_match else 1

    return {
        "Implementation": "Parallel CPU",
        "Mode": "Single execution",
        "Build_Time": extract_metric(
            output, r"Build Time:\s*([\d.]+)"
        ),
        "Probe_Time": extract_metric(
            output, r"Probe Time:\s*([\d.]+)"
        ),
        "Total_Time": extract_metric(
            output, r"Total Time:\s*([\d.]+)"
        ),
        "Matching_Records": extract_records(output),
        "Throughput": extract_metric(
            output, r"Throughput:\s*([\d,.]+)"
        ),
        "Workers": workers,
        "Initialization_Time": 0.0
    }


def benchmark_gpu():
    output = run_script(GPU_SCRIPT, "Intel GPU OpenCL")

    records = extract_records(output)

    # Cold execution: includes GPU initialization and first execution.
    cold_total = extract_metric(
        output, r"Total execution time:\s*([\d.]+)"
    )

    initialization = extract_metric(
        output, r"Initialization time:\s*([\d.]+)"
    )

    cold_build = extract_metric(
        output, r"GPU hash build time:\s*([\d.]+)"
    )

    cold_probe = extract_metric(
        output, r"GPU probe time:\s*([\d.]+)"
    )

    cold_copy = extract_metric(
        output, r"GPU result copy time:\s*([\d.]+)"
    )

    # Warm execution: reuses the initialized GPU context and buffers.
    warm_total = extract_metric(
        output, r"Average warm total time:\s*([\d.]+)"
    )

    warm_build = extract_metric(
        output, r"Average build time:\s*([\d.]+)"
    )

    warm_probe = extract_metric(
        output, r"Average probe time:\s*([\d.]+)"
    )

    warm_copy = extract_metric(
        output, r"Average result copy time:\s*([\d.]+)"
    )

    warm_throughput = extract_metric(
        output, r"Warm Throughput:\s*([\d,.]+)"
    )

    cold_throughput = records / cold_total if cold_total > 0 else 0

    cold_result = {
        "Implementation": "Intel GPU OpenCL",
        "Mode": "Cold execution",
        "Build_Time": cold_build,
        "Probe_Time": cold_probe,
        "Total_Time": cold_total,
        "Matching_Records": records,
        "Throughput": cold_throughput,
        "Workers": None,
        "Initialization_Time": initialization,
        "Result_Copy_Time": cold_copy
    }

    warm_result = {
        "Implementation": "Intel GPU OpenCL",
        "Mode": "Warm average (5 runs)",
        "Build_Time": warm_build,
        "Probe_Time": warm_probe,
        "Total_Time": warm_total,
        "Matching_Records": records,
        "Throughput": warm_throughput,
        "Workers": None,
        "Initialization_Time": 0.0,
        "Result_Copy_Time": warm_copy
    }

    return cold_result, warm_result


def validate_outputs():
    print("\n" + "=" * 60)
    print("VALIDATING OUTPUT FILES")
    print("=" * 60)

    files = {
        "Sequential": RESULTS_DIR / "sequential_output.csv",
        "Parallel": RESULTS_DIR / "parallel_output.csv",
        "GPU": RESULTS_DIR / "gpu_output.csv"
    }

    dataframes = {}

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} output not found: {path}")

        df = pd.read_csv(path)
        df = df.sort_values("purchase_id").reset_index(drop=True)
        dataframes[name] = df

        print(f"{name}: {len(df):,} records loaded")

    reference = dataframes["Sequential"]

    for name in ["Parallel", "GPU"]:
        pd.testing.assert_frame_equal(
            reference,
            dataframes[name],
            check_dtype=False
        )
        print(f"{name} output matches Sequential: PASSED")

    print("\nAll three implementations produced identical results.")


def calculate_metrics(results):
    sequential_time = next(
        row["Total_Time"]
        for row in results
        if row["Implementation"] == "Sequential CPU"
    )

    for row in results:
        row["Speedup_vs_Sequential"] = (
            sequential_time / row["Total_Time"]
            if row["Total_Time"] > 0 else 0
        )

        if row["Implementation"] == "Parallel CPU":
            workers = row["Workers"]
            row["Efficiency_Percent"] = (
                row["Speedup_vs_Sequential"] / workers * 100
                if workers else None
            )
        else:
            row["Efficiency_Percent"] = None

    return results


def benchmark():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("PARALLEL HASH JOIN - PERFORMANCE BENCHMARK")
    print("=" * 60)

    sequential = benchmark_sequential()
    parallel = benchmark_parallel()
    gpu_cold, gpu_warm = benchmark_gpu()

    validate_outputs()

    results = calculate_metrics([
        sequential,
        parallel,
        gpu_cold,
        gpu_warm
    ])

    df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE COMPARISON")
    print("=" * 60)

    display_columns = [
        "Implementation",
        "Mode",
        "Total_Time",
        "Throughput",
        "Speedup_vs_Sequential"
    ]

    print(
        df[display_columns].to_string(
            index=False,
            formatters={
                "Total_Time": lambda x: f"{x:.6f} s",
                "Throughput": lambda x: f"{x:,.2f}",
                "Speedup_vs_Sequential": lambda x: f"{x:.2f}x"
            }
        )
    )

    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    fastest = min(results, key=lambda row: row["Total_Time"])

    print(
        f"Fastest measured execution: "
        f"{fastest['Implementation']} ({fastest['Mode']})"
    )

    print(
        f"Sequential CPU baseline: {sequential['Total_Time']:.6f} seconds"
    )

    print(
        f"GPU cold execution: {gpu_cold['Total_Time']:.6f} seconds"
    )

    print(
        f"GPU warm average: {gpu_warm['Total_Time']:.6f} seconds"
    )

    print(
        "\nNote: GPU warm execution reuses the OpenCL context, "
        "compiled kernels, and input buffers. It represents repeated "
        "joins with reusable GPU resources, not a cold-start comparison."
    )

    print(
        "\nParallel CPU efficiency: "
        f"{parallel['Efficiency_Percent']:.2f}%"
    )

    print(f"\nBenchmark results saved to: {output_path}")


if __name__ == "__main__":
    benchmark()
