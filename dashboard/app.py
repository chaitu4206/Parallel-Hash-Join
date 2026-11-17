
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "results" / "scalability_results.csv"
GRAPH_DIR = ROOT / "results" / "graphs"


st.set_page_config(
    page_title="Parallel Hash Join Dashboard",
    page_icon="⚡",
    layout="wide"
)


@st.cache_data
def load_results():
    return pd.read_csv(DATA_FILE)


df = load_results()


st.title("⚡ Parallel Hash Join Performance Dashboard")

st.markdown(
    """
    Compare **Sequential CPU**, **Parallel CPU**, and
    **Intel GPU OpenCL** hash join implementations
    across multiple dataset sizes.
    """
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dashboard Controls")

available_sizes = sorted(df["Dataset_Size"].unique())

selected_sizes = st.sidebar.multiselect(
    "Select dataset sizes",
    available_sizes,
    default=available_sizes
)

if not selected_sizes:
    st.warning("Select at least one dataset size.")
    st.stop()


filtered = df[
    df["Dataset_Size"].isin(selected_sizes)
].copy()

# --------------------------------------------------
# VALIDATION STATUS
# --------------------------------------------------

st.subheader("Implementation Validation")

validation_results = []

for size in selected_sizes:

    folder = ROOT / "results" / "scalability" / str(int(size))

    sequential_path = folder / "sequential.csv"
    parallel_path = folder / "parallel.csv"
    gpu_path = folder / "gpu.csv"

    try:
        sequential = pd.read_csv(sequential_path)
        parallel = pd.read_csv(parallel_path)
        gpu = pd.read_csv(gpu_path)

        sequential = sequential.sort_values(
            "purchase_id"
        ).reset_index(drop=True)

        parallel = parallel.sort_values(
            "purchase_id"
        ).reset_index(drop=True)

        gpu = gpu.sort_values(
            "purchase_id"
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            sequential, parallel,
            check_dtype=False, check_exact=True
        )

        pd.testing.assert_frame_equal(
            sequential, gpu,
            check_dtype=False, check_exact=True
        )

        validation_results.append({
            "Dataset": f"{int(size):,}",
            "Records": len(sequential),
            "Status": "PASSED"
        })

    except Exception as error:

        validation_results.append({
            "Dataset": f"{int(size):,}",
            "Records": 0,
            "Status": f"FAILED: {error}"
        })

validation_df = pd.DataFrame(validation_results)

st.dataframe(
    validation_df,
    width="stretch",
    hide_index=True
)
st.download_button(
    label="Download Complete Benchmark Results",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="scalability_results.csv",
    mime="text/csv"
)
# --------------------------------------------------
# METRICS
# --------------------------------------------------

largest_size = max(selected_sizes)

largest_data = filtered[
    filtered["Dataset_Size"] == largest_size
]

sequential = largest_data[
    largest_data["Implementation"] == "Sequential CPU"
].iloc[0]

gpu_warm = largest_data[
    largest_data["Mode"] == "Warm average (5 runs)"
].iloc[0]

parallel = largest_data[
    largest_data["Implementation"] == "Parallel CPU"
].iloc[0]

st.subheader("Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Largest Dataset",
    f"{largest_size:,}"
)

col2.metric(
    "Sequential CPU",
    f"{sequential['Total_Time']:.4f} s"
)

col3.metric(
    "GPU Warm Average",
    f"{gpu_warm['Total_Time']:.6f} s"
)

col4.metric(
    "GPU Warm Speedup",
    f"{gpu_warm['Speedup_vs_Sequential']:.2f}x"
)


# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------

st.divider()

st.subheader("Benchmark Results")

display_columns = [
    "Dataset_Size",
    "Implementation",
    "Mode",
    "Build_Time",
    "Probe_Time",
    "Result_Copy_Time",
    "Total_Time",
    "Throughput",
    "Speedup_vs_Sequential"
]

st.dataframe(
    filtered[display_columns].sort_values(
        ["Dataset_Size", "Implementation", "Mode"]
    ),
    width="stretch",
    hide_index=True
)


# --------------------------------------------------
# EXECUTION TIME CHART
# --------------------------------------------------

st.divider()

st.subheader("Execution Time Comparison")

fig, ax = plt.subplots(figsize=(11, 5))

for implementation in [
    "Sequential CPU",
    "Parallel CPU",
    "Intel GPU OpenCL"
]:

    subset = filtered[
        filtered["Implementation"] == implementation
    ]

    if implementation == "Intel GPU OpenCL":

        for mode in [
            "Cold execution",
            "Warm average (5 runs)"
        ]:

            mode_data = subset[
                subset["Mode"] == mode
            ]

            ax.plot(
                mode_data["Dataset_Size"],
                mode_data["Total_Time"],
                marker="o",
                label=f"GPU - {mode}"
            )

    else:

        ax.plot(
            subset["Dataset_Size"],
            subset["Total_Time"],
            marker="o",
            label=implementation
        )

ax.set_xlabel("Dataset Size")
ax.set_ylabel("Execution Time (seconds)")
ax.set_yscale("log")
ax.set_title("Execution Time vs Dataset Size")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# THROUGHPUT CHART
# --------------------------------------------------

st.divider()

st.subheader("Throughput Comparison")

fig, ax = plt.subplots(figsize=(11, 5))

for implementation in [
    "Sequential CPU",
    "Parallel CPU",
    "Intel GPU OpenCL"
]:

    subset = filtered[
        filtered["Implementation"] == implementation
    ]

    if implementation == "Intel GPU OpenCL":

        for mode in [
            "Cold execution",
            "Warm average (5 runs)"
        ]:

            mode_data = subset[
                subset["Mode"] == mode
            ]

            ax.plot(
                mode_data["Dataset_Size"],
                mode_data["Throughput"] / 1_000_000,
                marker="o",
                label=f"GPU - {mode}"
            )

    else:

        ax.plot(
            subset["Dataset_Size"],
            subset["Throughput"] / 1_000_000,
            marker="o",
            label=implementation
        )

ax.set_xlabel("Dataset Size")
ax.set_ylabel("Million Records / Second")
ax.set_title("Throughput vs Dataset Size")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# SPEEDUP CHART
# --------------------------------------------------

st.divider()

st.subheader("Speedup Compared with Sequential CPU")

fig, ax = plt.subplots(figsize=(11, 5))

for implementation, mode, label in [
    ("Parallel CPU", "Single execution", "Parallel CPU"),
    ("Intel GPU OpenCL", "Cold execution", "GPU Cold"),
    ("Intel GPU OpenCL", "Warm average (5 runs)", "GPU Warm")
]:

    subset = filtered[
        (filtered["Implementation"] == implementation)
        & (filtered["Mode"] == mode)
    ]

    ax.plot(
        subset["Dataset_Size"],
        subset["Speedup_vs_Sequential"],
        marker="o",
        label=label
    )

ax.axhline(
    1,
    linestyle="--",
    label="Sequential baseline"
)

ax.set_xlabel("Dataset Size")
ax.set_ylabel("Speedup (x)")
ax.set_title("Relative Performance")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# GENERATED GRAPH IMAGES
# --------------------------------------------------

st.divider()

st.subheader("Exported Performance Graphs")

for graph_name, title in [
    ("execution_time_comparison.png", "Execution Time"),
    ("throughput_comparison.png", "Throughput"),
    ("speedup_comparison.png", "Speedup")
]:

    graph_path = GRAPH_DIR / graph_name

    if graph_path.exists():

        st.markdown(f"**{title} Graph**")
        st.image(str(graph_path), width="stretch")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Parallel Hash Join | Python | Pandas | "
    "Multiprocessing | Intel OpenCL | Streamlit"
)

st.caption(
    "Note: GPU warm measurements reuse initialized GPU resources "
    "and should not be interpreted as cold-start execution time."
)
