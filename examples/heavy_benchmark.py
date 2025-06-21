import multiprocessing as mp
import pandas as pd
import numpy as np
import pyarrow.compute as pc
import time
import arrowshelf
import sys
import os

# --- A More Realistic, Heavy Worker Function ---
def heavy_computation(data):
    """
    Simulates a more demanding data science task involving multiple steps.
    Accepts either a Pandas DataFrame (for pickle) or an Arrow Table.
    """
    if isinstance(data, pd.DataFrame):
        # This is the Pickle path
        df = data
        # Simulate multiple operations
        v1 = df['A'] * np.sin(df['B'])
        v2 = df['C'] * np.cos(df['D'])
        v3 = np.log(np.abs(df['E']) + 1)
        # Final result calculation
        return (v1 - v2 + v3).mean()
    else:
        # This is the ArrowShelf Native path
        table = data
        # Perform the same operations using PyArrow's C++ functions
        v1 = pc.multiply(table.column('A'), pc.sin(table.column('B')))
        v2 = pc.multiply(table.column('C'), pc.cos(table.column('D')))
        v3 = pc.ln(pc.add(pc.abs(table.column('E')), 1))
        
        final_array = pc.subtract(pc.add(v3, v1), v2)
        return pc.mean(final_array).as_py()


def worker_pickle(df_chunk):
    return heavy_computation(df_chunk)

def worker_arrowshelf_native(key):
    table = arrowshelf.get_arrow(key)
    return heavy_computation(table)


def run_heavy_benchmark(num_rows=10_000_000, num_processes=8):
    print("\n--- HEAVY WORKLOAD BENCHMARK ---")
    print(f"Simulating a demanding task on {num_rows:,} rows with {num_processes} cores.")

    # Generate a larger, more realistic DataFrame
    df = pd.DataFrame(np.random.randn(num_rows, 5), columns=list('ABCDE'))
    df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"DataFrame size: {df_size_mb:.2f} MB\n")

    # --- Pickle Workflow ---
    print("Running Pickle workflow...")
    pickle_start = time.time()
    # Split the data for the workers
    df_chunks = np.array_split(df, num_processes)
    with mp.Pool(processes=num_processes) as pool:
        pool.map(worker_pickle, df_chunks)
    pickle_duration = time.time() - pickle_start
    print(f"-> Pickle Total Time: {pickle_duration:.4f} seconds")

    # --- ArrowShelf Workflow ---
    print("\nRunning ArrowShelf workflow...")
    arrowshelf_start = time.time()
    # Pay the one-time cost to put data on the shelf
    key = arrowshelf.put(df)
    put_time = time.time() - arrowshelf_start
    print(f"  (Initial 'put' cost: {put_time:.4f} seconds)")

    compute_start = time.time()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(worker_arrowshelf_native, [key] * num_processes)
    compute_time = time.time() - compute_start
    arrowshelf_duration = time.time() - arrowshelf_start
    print(f"  (Parallel computation time: {compute_time:.4f} seconds)")
    print(f"-> ArrowShelf Total Time: {arrowshelf_duration:.4f} seconds")

    speedup = pickle_duration / arrowshelf_duration
    print(f"\n>>>>> HEAVY WORKLOAD SPEEDUP: {speedup:.2f}x <<<<<")
    
    arrowshelf.delete(key)

if __name__ == "__main__":
    print("--- ArrowShelf Heavy Workload Benchmark ---")
    try:
        print("\nINFO: Pinging ArrowShelf server...")
        arrowshelf.list_keys()
        print("INFO: Connected to ArrowShelf server successfully.")
        
        # Determine a high number of cores to use, up to the machine's max
        max_cores = os.cpu_count() or 1
        cores_to_test = min(16, max_cores) # Use up to 16 cores if available
        
        run_heavy_benchmark(num_processes=cores_to_test)

    except arrowshelf.ConnectionError as e:
        print(f"\nBenchmark failed due to a connection error: {e}")
    finally:
        arrowshelf.close()
        print("\nBenchmark finished.")