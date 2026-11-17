
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "scalability_results.csv"
OUTPUT_DIR = ROOT / "results" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["Dataset_Size"] = pd.to_numeric(df["Dataset_Size"])

# Consistent order for all graphs.
order = [
    "Sequential CPU",
    "Parallel CPU",
    "Intel GPU OpenCL"
]

# --------------------------------------------------
# 1. EXECUTION TIME VS DATASET SIZE
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for implementation in order:

    subset = df[df["Implementation"] == implementation]

    if implementation == "Intel GPU OpenCL":

        for mode in ["Cold execution", "Warm average (5 runs)"]:

            mode_data = subset[subset["Mode"] == mode]

            plt.plot(
                mode_data["Dataset_Size"],
                mode_data["Total_Time"],
                marker="o",
                linewidth=2,
                label=f"{implementation} - {mode}"
            )

    else:

        plt.plot(
            subset["Dataset_Size"],
            subset["Total_Time"],
            marker="o",
            linewidth=2,
            label=implementation
        )

plt.title("Hash Join Execution Time vs Dataset Size")
plt.xlabel("Number of Purchase Records")
plt.ylabel("Execution Time (seconds)")
plt.yscale("log")
plt.xticks(
    sorted(df["Dataset_Size"].unique()),
    ["50K", "100K", "250K", "500K", "1M"]
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "execution_time_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 2. THROUGHPUT VS DATASET SIZE
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for implementation in order:

    subset = df[df["Implementation"] == implementation]

    if implementation == "Intel GPU OpenCL":

        for mode in ["Cold execution", "Warm average (5 runs)"]:

            mode_data = subset[subset["Mode"] == mode]

            plt.plot(
                mode_data["Dataset_Size"],
                mode_data["Throughput"] / 1_000_000,
                marker="o",
                linewidth=2,
                label=f"GPU - {mode}"
            )

    else:

        plt.plot(
            subset["Dataset_Size"],
            subset["Throughput"] / 1_000_000,
            marker="o",
            linewidth=2,
            label=implementation
        )

plt.title("Hash Join Throughput vs Dataset Size")
plt.xlabel("Number of Purchase Records")
plt.ylabel("Throughput (Million Records/Second)")
plt.xticks(
    sorted(df["Dataset_Size"].unique()),
    ["50K", "100K", "250K", "500K", "1M"]
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "throughput_comparison.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 3. SPEEDUP VS SEQUENTIAL CPU
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for implementation, mode in [
    ("Parallel CPU", "Single execution"),
    ("Intel GPU OpenCL", "Cold execution"),
    ("Intel GPU OpenCL", "Warm average (5 runs)")
]:

    subset = df[
        (df["Implementation"] == implementation)
        & (df["Mode"] == mode)
    ]

    label = (
        implementation
        if implementation != "Intel GPU OpenCL"
        else f"GPU - {mode}"
    )

    plt.plot(
        subset["Dataset_Size"],
        subset["Speedup_vs_Sequential"],
        marker="o",
        linewidth=2,
        label=label
    )

plt.axhline(
    y=1,
    linestyle="--",
    label="Sequential baseline (1x)"
)

plt.title("Speedup Compared with Sequential CPU")
plt.xlabel("Number of Purchase Records")
plt.ylabel("Speedup (x)")
plt.xticks(
    sorted(df["Dataset_Size"].unique()),
    ["50K", "100K", "250K", "500K", "1M"]
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "speedup_comparison.png",
    dpi=300
)

plt.show()


print("\nAll graphs generated successfully!")

print(f"\nSaved in: {OUTPUT_DIR}")

for file in OUTPUT_DIR.glob("*.png"):
    print(file.name)
