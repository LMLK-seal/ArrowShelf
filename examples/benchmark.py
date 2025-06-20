import multiprocessing as mp
import pandas as pd
import numpy as np
import time
import arrowshelf
import sys

print("--- ArrowShelf Benchmark (V1 / TCP) ---")
print("INFO: This script assumes the ArrowShelf server is running in a separate terminal.")
print("      (Run 'python -m arrowshelf.server' to launch it)")

try:
    if not arrowshelf.client._connection.is_connected():
        print("\nERROR: Could not connect to ArrowShelf server. Please start it first.")
        sys.exit(1)
except arrowshelf.ConnectionError as e:
    print(f"\nERROR: Connection failed: {e}")
    sys.exit(1)

print("INFO: Connected to ArrowShelf server successfully.")

def worker_pickle(df):
    return df.shape

def worker_arrowshelf(key):
    df = arrowshelf.get(key)
    return df.shape

def run_benchmark(num_rows):
    print(f"\n--- Benchmarking with {num_rows:,} rows ---")
    df = pd.DataFrame(np.random.randn(num_rows, 5), columns=list('ABCDE'))
    df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"DataFrame size: {df_size_mb:.2f} MB")

    num_processes = 4
    tasks = range(num_processes)

    start_time_pickle = time.time()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(worker_pickle, [df for _ in tasks])
    pickle_duration = time.time() - start_time_pickle
    print(f"Standard (Pickle): {pickle_duration:.4f} seconds")

    key = None
    try:
        start_time_memry = time.time()
        key = arrowshelf.put(df)
        with mp.Pool(processes=num_processes) as pool:
            pool.map(worker_arrowshelf, [key for _ in tasks])
        end_time_memry = time.time() - start_time_memry
        print(f"ArrowShelf (TCP):   {end_time_memry:.4f} seconds")

        speedup = pickle_duration / end_time_memry
        print(f"Speedup: {speedup:.2f}x")
        return df_size_mb, pickle_duration, end_time_memry
    finally:
        if key:
            arrowshelf.delete(key)

if __name__ == "__main__":
    row_counts = [100_000, 1_000_000, 5_000_000, 10_000_000]
    results = []
    
    try:
        for rows in row_counts:
            results.append(run_benchmark(rows))
    except arrowshelf.ConnectionError as e:
        print(f"\nBenchmark failed due to a connection error: {e}")
        print("Please ensure the ArrowShelf server is still running.")
    finally:
        arrowshelf.close()
        print("\nBenchmark finished. You can now stop the server with Ctrl+C in its terminal.")
    
    print("\n--- Summary ---")
    print("| DataFrame Size (MB) | Pickle (s) | ArrowShelf (s) | Speedup |")
    print("|---------------------|------------|----------------|---------|")
    for size, p_dur, m_dur in results:
        speedup = p_dur / m_dur if m_dur > 0 else float('inf')
        print(f"| {size:<19.2f} | {p_dur:<10.4f} | {m_dur:<14.4f} | {speedup:<7.2f}x |")