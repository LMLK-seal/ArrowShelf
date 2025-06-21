"""
A real-world example script demonstrating how to use ArrowShelf to accelerate
the processing of a large CSV file using multiprocessing.

This script showcases the high-performance "native Arrow" pattern.

-----------------------------------------------------------------------------
HOW TO USE THIS SCRIPT:
-----------------------------------------------------------------------------

1.  Make sure the ArrowShelf server is running in a separate terminal:
    >> python -m arrowshelf.server

2.  If you don't have a large CSV, create one by running this script
    with the --create-sample flag:
    >> python process_data.py --create-sample large_dataset.csv

3.  Run the processing on your CSV file:
    >> python process_data.py large_dataset.csv --cores 8

    (Replace 8 with the number of CPU cores you want to use)
-----------------------------------------------------------------------------
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc

import arrowshelf


def high_performance_worker(data_key: str) -> float:
    """
    This is the function that runs on each parallel process.
    It demonstrates the optimal way to use ArrowShelf for maximum speed.
    """
    try:
        # 1. Get a zero-copy reference to the Arrow Table from shared memory.
        #    This is an extremely fast operation.
        arrow_table = arrowshelf.get_arrow(data_key)

        # 2. Perform a complex, multi-step calculation directly on the
        #    Arrow data using PyArrow's fast C++ compute functions.
        #    This avoids the slow step of converting the data back to Pandas.
        col_a = arrow_table.column('A')
        col_b = arrow_table.column('B')
        col_c = arrow_table.column('C')
        col_d = arrow_table.column('D')
        
        # Simulate a realistic workload
        v1 = pc.multiply(col_a, pc.sin(col_b))
        v2 = pc.multiply(col_c, pc.cos(col_d))
        final_array = pc.subtract(v1, v2)
        
        # 3. Return the final result.
        return pc.mean(final_array).as_py()

    except Exception as e:
        print(f"[Worker Error] An error occurred: {e}")
        return 0.0


def create_sample_csv(filepath: str, num_rows: int = 10_000_000):
    """Generates a large sample CSV file for demonstration purposes."""
    print(f"Creating a sample dataset with {num_rows:,} rows at '{filepath}'...")
    df = pd.DataFrame(np.random.randn(num_rows, 5), columns=list('ABCDE'))
    df.to_csv(filepath, index=False)
    print("Sample dataset created successfully.")


def run_processing(filepath: str, num_cores: int):
    """
    The main function that loads and processes the data using ArrowShelf.
    """
    # --- 1. Load Data ---
    print(f"\nLoading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Successfully loaded DataFrame with {len(df):,} rows ({df_size_mb:.2f} MB).")
    except FileNotFoundError:
        print(f"ERROR: File not found at '{filepath}'.")
        print("You can create a sample file with: python process_data.py --create-sample large_dataset.csv")
        sys.exit(1)

    # --- 2. Connect to ArrowShelf Server ---
    try:
        print("Connecting to ArrowShelf server...")
        arrowshelf.list_keys() # Ping the server
        print("Connection successful.")
    except arrowshelf.ConnectionError as e:
        print(f"ERROR: Could not connect to ArrowShelf server. Please ensure it is running. ({e})")
        sys.exit(1)

    key = None
    try:
        # --- 3. Process Data with ArrowShelf ---
        print(f"\nStarting parallel processing on {num_cores} cores...")
        total_start_time = time.perf_counter()

        # Place the entire DataFrame onto the shared memory shelf ONCE.
        put_start_time = time.perf_counter()
        key = arrowshelf.put(df)
        put_duration = time.perf_counter() - put_start_time
        print(f" -> Data placed on shelf in {put_duration:.4f} seconds.")

        # Create a list of keys to send to the workers.
        # Each worker will access the same shared data.
        keys_for_workers = [key] * num_cores

        # Start the parallel processing pool.
        compute_start_time = time.perf_counter()
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(high_performance_worker, keys_for_workers)
        compute_duration = time.perf_counter() - compute_start_time
        
        total_duration = time.perf_counter() - total_start_time

        print(f" -> Parallel computation finished in {compute_duration:.4f} seconds.")
        print("-" * 40)
        print(f"✅ Total processing time: {total_duration:.4f} seconds")
        print(f"   Results from workers: {results}")
        print("-" * 40)

    finally:
        # --- 4. Clean Up ---
        print("\nCleaning up shared memory object from shelf...")
        if key:
            arrowshelf.delete(key)
        arrowshelf.close()
        print("Cleanup complete.")


if __name__ == "__main__":
    # Use argparse for professional command-line argument handling
    parser = argparse.ArgumentParser(
        description="Accelerate CSV processing with ArrowShelf.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        "input_file",
        nargs='?', # Makes the argument optional
        default=None,
        help="Path to the large CSV file to process."
    )
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=os.cpu_count() or 4, # Default to all available cores, or 4 if undetectable
        help="Number of CPU cores to use for parallel processing."
    )
    parser.add_argument(
        "--create-sample",
        metavar="FILENAME",
        help="Create a large sample CSV file and exit."
    )

    args = parser.parse_args()

    # Handle the --create-sample flag first
    if args.create_sample:
        create_sample_csv(args.create_sample)
        sys.exit(0)

    # If no input file is provided, print help and exit
    if not args.input_file:
        parser.print_help()
        sys.exit(1)

    # Run the main processing function
    run_processing(args.input_file, args.cores)