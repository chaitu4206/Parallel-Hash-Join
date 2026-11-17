
import time
import argparse
import numpy as np
import pandas as pd
import pyopencl as cl

from pathlib import Path


OPENCL_CODE = """
__kernel void build_hash_table(
    __global const int *customer_ids,
    __global int *hash_table,
    const int customer_count
) {
    int i = get_global_id(0);

    if (i < customer_count) {
        int key = customer_ids[i];
        hash_table[key] = key;
    }
}

__kernel void probe_hash_table(
    __global const int *purchase_customer_ids,
    __global const int *hash_table,
    __global int *matched_customer_ids,
    const int purchase_count,
    const int max_customer_id
) {
    int i = get_global_id(0);

    if (i < purchase_count) {
        int key = purchase_customer_ids[i];

        if (key > 0 && key <= max_customer_id &&
            hash_table[key] == key) {
            matched_customer_ids[i] = key;
        } else {
            matched_customer_ids[i] = -1;
        }
    }
}
"""


class GPUHashJoin:

    def __init__(self, customers, purchases):

        self.customer_ids = np.ascontiguousarray(
            customers["customer_id"].to_numpy(dtype=np.int32)
        )

        self.purchase_customer_ids = np.ascontiguousarray(
            purchases["customer_id"].to_numpy(dtype=np.int32)
        )

        self.customer_count = len(self.customer_ids)
        self.purchase_count = len(self.purchase_customer_ids)

        self.max_customer_id = int(self.customer_ids.max())

        # Find OpenCL GPU.
        self.device = None

        for platform in cl.get_platforms():
            for device in platform.get_devices(
                device_type=cl.device_type.GPU
            ):
                self.device = device
                break

            if self.device is not None:
                break

        if self.device is None:
            raise RuntimeError("No OpenCL GPU found.")

        # Initialize OpenCL resources once.
        self.context = cl.Context([self.device])

        self.queue = cl.CommandQueue(
            self.context,
            self.device,
            properties=cl.command_queue_properties.PROFILING_ENABLE
        )

        self.program = cl.Program(
            self.context,
            OPENCL_CODE
        ).build()

        # Cache kernel objects.
        self.build_kernel = cl.Kernel(
            self.program, "build_hash_table"
        )

        self.probe_kernel = cl.Kernel(
            self.program, "probe_hash_table"
        )

        # Allocate buffers once.
        self.table_size = self.max_customer_id + 1

        self.customer_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.customer_ids
        )

        self.purchase_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.purchase_customer_ids
        )

        self.hash_table_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE,
            size=self.table_size * np.dtype(np.int32).itemsize
        )

        self.matched_buffer = cl.Buffer(
            self.context,
            cl.mem_flags.WRITE_ONLY,
            size=self.purchase_count * np.dtype(np.int32).itemsize
        )

        self.output = np.empty(
            self.purchase_count,
            dtype=np.int32
        )

        self.customer_lookup = (
            customers.set_index("customer_id")["customer_name"]
        )

        self.purchases = purchases

    def execute(self):

        start = time.perf_counter()

        # Reset hash table.
        cl.enqueue_fill_buffer(
            self.queue,
            self.hash_table_buffer,
            np.int32(0),
            0,
            self.table_size * np.dtype(np.int32).itemsize
        )

        # Build phase.
        build_event = self.build_kernel(
            self.queue,
            (self.customer_count,),
            None,
            self.customer_buffer,
            self.hash_table_buffer,
            np.int32(self.customer_count)
        )

        # Probe phase.
        probe_event = self.probe_kernel(
            self.queue,
            (self.purchase_count,),
            None,
            self.purchase_buffer,
            self.hash_table_buffer,
            self.matched_buffer,
            np.int32(self.purchase_count),
            np.int32(self.max_customer_id)
        )

        # Copy results to CPU.
        copy_event = cl.enqueue_copy(
            self.queue,
            self.output,
            self.matched_buffer
        )

        copy_event.wait()

        elapsed = time.perf_counter() - start

        build_time = (
            build_event.profile.end - build_event.profile.start
        ) / 1_000_000_000

        probe_time = (
            probe_event.profile.end - probe_event.profile.start
        ) / 1_000_000_000

        copy_time = (
            copy_event.profile.end - copy_event.profile.start
        ) / 1_000_000_000

        # Construct joined DataFrame.
        valid_mask = self.output != -1

        result = self.purchases.loc[valid_mask].copy()

        result["customer_id"] = self.output[valid_mask]

        result["customer_name"] = result["customer_id"].map(
            self.customer_lookup
        )

        result = result[
            ["purchase_id", "customer_id", "customer_name", "amount"]
        ]

        return result, build_time, probe_time, copy_time, elapsed


def validate_output(gpu_result, expected_path):

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Expected sequential output not found: {expected_path}"
        )

    expected = pd.read_csv(expected_path)

    actual = (
        gpu_result.sort_values("purchase_id")
        .reset_index(drop=True)
    )

    expected = (
        expected.sort_values("purchase_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=True
    )

    print("Validation: PASSED")
    print("GPU and sequential outputs are identical.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing customers.csv and purchases.csv"
    )

    parser.add_argument(
        "--output",
        default="results/gpu_output.csv",
        help="Path for GPU joined output"
    )

    parser.add_argument(
        "--expected",
        default="results/sequential_output.csv",
        help="Sequential output CSV used for validation"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    expected_path = Path(args.expected)

    print("=" * 60)
    print("INTEL GPU HASH JOIN - PERSISTENT OPENCL")
    print("=" * 60)

    customers = pd.read_csv(data_dir / "customers.csv")
    purchases = pd.read_csv(data_dir / "purchases.csv")

    print("\nCustomers loaded:", len(customers))
    print("Purchases loaded:", len(purchases))

    # COLD EXECUTION
    cold_start = time.perf_counter()

    gpu = GPUHashJoin(customers, purchases)

    initialization_time = time.perf_counter() - cold_start

    (
        cold_result,
        cold_build,
        cold_probe,
        cold_copy,
        cold_execution
    ) = gpu.execute()

    cold_total = time.perf_counter() - cold_start

    print("\nGPU Device:", gpu.device.name)
    print("OpenCL Version:", gpu.device.version)
    print("Compute Units:", gpu.device.max_compute_units)

    print("\n--- COLD EXECUTION ---")
    print(f"Initialization time:      {initialization_time:.6f} seconds")
    print(f"GPU hash build time:       {cold_build:.6f} seconds")
    print(f"GPU probe time:            {cold_probe:.6f} seconds")
    print(f"GPU result copy time:      {cold_copy:.6f} seconds")
    print(f"Total execution time:      {cold_total:.6f} seconds")

    # WARM EXECUTIONS
    warm_runs = 5

    warm_times = []
    warm_build_times = []
    warm_probe_times = []
    warm_copy_times = []

    print(f"\n--- WARM EXECUTIONS ({warm_runs} RUNS) ---")

    for run in range(1, warm_runs + 1):

        (
            warm_result,
            build_time,
            probe_time,
            copy_time,
            elapsed
        ) = gpu.execute()

        warm_times.append(elapsed)
        warm_build_times.append(build_time)
        warm_probe_times.append(probe_time)
        warm_copy_times.append(copy_time)

        print(
            f"Run {run}: {elapsed:.6f} seconds "
            f"(Build: {build_time:.6f}, "
            f"Probe: {probe_time:.6f})"
        )

    # AVERAGES
    average_warm_time = float(np.mean(warm_times))
    average_build = float(np.mean(warm_build_times))
    average_probe = float(np.mean(warm_probe_times))
    average_copy = float(np.mean(warm_copy_times))

    print("\n--- WARM EXECUTION AVERAGES ---")
    print(f"Average build time:       {average_build:.6f} seconds")
    print(f"Average probe time:       {average_probe:.6f} seconds")
    print(f"Average result copy time: {average_copy:.6f} seconds")
    print(f"Average warm total time:  {average_warm_time:.6f} seconds")

    # VALIDATION
    validate_output(warm_result, expected_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    warm_result.to_csv(output_path, index=False)

    matching_records = len(warm_result)

    throughput = (
        len(purchases) / average_warm_time
        if average_warm_time > 0 else 0
    )

    print("\nMatching Records:", matching_records)
    print(f"Warm Throughput: {throughput:,.2f} records/second")

    print(f"\nOutput saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
