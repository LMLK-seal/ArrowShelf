import multiprocessing as mp
import pandas as pd
import numpy as np
import pyarrow.compute as pc
import time
import arrowshelf
import sys
import os

# --- Worker Functions (for a more realistic workload) ---
def complex_computation_pickle(df_chunk):
    """A worker that receives a pickled DataFrame chunk."""
    res = (df_chunk['E'] * np.sin(df_chunk['A'])).sum()
    return res

def complex_computation_native_arrow(key):
    """A worker that uses ArrowShelf's native Arrow access."""
    table = arrowshelf.get_arrow(key)
    res = pc.sum(pc.multiply(table.column('E'), pc.sin(table.column('A')))).as_py()
    return res

# --- SCENARIO 1: Massively Parallel Computation ---
def run_scaling_benchmark(num_rows=5_000_000):
    print("\n\n--- SCENARIO 1: Massively Parallel Core Scaling ---")
    print(f"Testing with a {num_rows:,} row DataFrame across different core counts.")
    
    df = pd.DataFrame(np.random.randn(num_rows, 5), columns=list('ABCDE'))
    key = arrowshelf.put(df)
    
    print("\n| Num Cores | Pickle Time (s) | ArrowShelf Time (s) | Speedup Factor |")
    print("|-----------|-----------------|---------------------|----------------|")
    
    core_counts = [2, 4, 8] # A shorter list for quicker tests
    max_cores = os.cpu_count() or 1
    if max_cores > 8 and max_cores not in core_counts:
        core_counts.append(max_cores)

    for cores in core_counts:
        df_chunks = np.array_split(df, cores)
        start = time.time()
        with mp.Pool(processes=cores) as pool:
            pool.map(complex_computation_pickle, df_chunks)
        pickle_duration = time.time() - start

        start = time.time()
        with mp.Pool(processes=cores) as pool:
            pool.map(complex_computation_native_arrow, [key] * cores)
        arrowshelf_duration = time.time() - start

        speedup = pickle_duration / arrowshelf_duration
        print(f"| {cores:<9} | {pickle_duration:<15.4f} | {arrowshelf_duration:<19.4f} | {speedup:<14.2f}x |")
        
    arrowshelf.delete(key)

# --- SCENARIO 2: Iterative Analysis ---
def run_iterative_benchmark(num_rows=5_000_000, num_processes=8):
    print("\n\n--- SCENARIO 2: Iterative & Interactive Analysis ---")
    print(f"Simulating 5 sequential parallel tasks on the same {num_rows:,} row DataFrame.")
    
    df = pd.DataFrame(np.random.randn(num_rows, 5), columns=list('ABCDE'))
    
    print("\nRunning Pickle workflow (pays data transfer cost every time)...")
    pickle_start = time.time()
    df_chunks = np.array_split(df, num_processes)
    for _ in range(5): # Loop for 5 tasks
        with mp.Pool(processes=num_processes) as pool: pool.map(complex_computation_pickle, df_chunks)
    pickle_total_time = time.time() - pickle_start
    print(f"-> Pickle total time for 5 tasks: {pickle_total_time:.4f} seconds")

    print("\nRunning ArrowShelf workflow (pays data setup cost only once)...")
    arrowshelf_start = time.time()
    key = arrowshelf.put(df)
    put_time = time.time() - arrowshelf_start
    print(f"  (Initial 'put' cost: {put_time:.4f} seconds)")

    tasks_start = time.time()
    for _ in range(5): # Loop for 5 tasks
        with mp.Pool(processes=num_processes) as pool: pool.map(complex_computation_native_arrow, [key] * num_processes)
    arrowshelf_total_time = time.time() - arrowshelf_start
    tasks_only_time = time.time() - tasks_start
    
    print(f"  (Time for 5 parallel tasks: {tasks_only_time:.4f} seconds)")
    print(f"-> ArrowShelf total time for 5 tasks: {arrowshelf_total_time:.4f} seconds")
    
    speedup = pickle_total_time / arrowshelf_total_time
    print(f"\n>> Iterative Workflow Speedup: {speedup:.2f}x <<")
    
    arrowshelf.delete(key)


if __name__ == "__main__":
    print("--- ArrowShelf Advanced Benchmark (V2.1) ---")
    print("This benchmark demonstrates ArrowShelf's power in two key scenarios:")
    
    try:
        print("\nINFO: Pinging ArrowShelf server...")
        arrowshelf.list_keys()
        print("INFO: Connected to ArrowShelf server successfully.")
        
        # Run the actual benchmarks
        run_scaling_benchmark()
        run_iterative_benchmark()

    except arrowshelf.ConnectionError as e:
        print(f"\nBenchmark failed due to a connection error: {e}")
    finally:
        arrowshelf.close()
        print("\nAdvanced benchmark finished.")