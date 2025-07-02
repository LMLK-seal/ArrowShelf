import multiprocessing as mp
import threading
import pandas as pd
import numpy as np
import time
import arrowshelf
import sys
import os

try:
    import faiss
except ImportError:
    print("ERROR: FAISS is not installed. Please run 'pip install faiss-cpu'.")
    sys.exit(1)

from multiprocessing.pool import ThreadPool
threading.Pool = ThreadPool

# --- FAISS IVF Worker ---
def worker_faiss_ivf(task_data):
    key, start_index, end_index = task_data

    table = arrowshelf.get_arrow(key).combine_chunks()
    x = table.column("x").chunk(0).to_numpy(zero_copy_only=True)
    y = table.column("y").chunk(0).to_numpy(zero_copy_only=True)
    z = table.column("z").chunk(0).to_numpy(zero_copy_only=True)

    all_points = np.stack([x, y, z], axis=1).astype(np.float32)
    query_chunk = all_points[start_index:end_index]

    d = 3  # 3D points
    nlist = 100  # Number of Voronoi cells (adjust for speed vs. accuracy)
    quantizer = faiss.IndexFlatL2(d)  # coarse quantizer
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    assert not index.is_trained
    index.train(all_points)  # required before add()
    index.add(all_points)

    index.nprobe = 10  # Number of cells to search (lower = faster, higher = better accuracy)

    _, dists = index.search(query_chunk, 11)
    avg_dist = np.mean(np.sqrt(dists[:, 1:]))  # exclude self
    return avg_dist

# --- MAIN BENCHMARK ---

def run_approximate_faiss_benchmark(num_points=100_000, num_cores=6):
    print("\n--- SCENARIO 2: Approximate k-NN with FAISS IVF ---")
    print(f"Testing on {num_points:,} 3D points with {num_cores} cores (IndexIVFFlat)\n")

    df = pd.DataFrame(np.random.rand(num_points, 3), columns=['x', 'y', 'z'])
    df_size_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"DataFrame size: {df_size_mb:.2f} MB")

    key = arrowshelf.put(df)
    chunk_size = num_points // num_cores
    faiss_tasks = [
        (key, i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)
    ]

    print("Running ArrowShelf + FAISS IVF (approximate search)...")
    start_time = time.perf_counter()
    with ThreadPool(processes=num_cores) as pool:
        results = pool.map(worker_faiss_ivf, faiss_tasks)

    duration = time.perf_counter() - start_time
    avg_dist = np.mean(results)

    print(f"\n✅ Approximate 10-NN avg distance: {avg_dist:.6f}")
    print(f"⚡ Total FAISS-IVF time: {duration:.4f} seconds")
    print("==================================================")

    arrowshelf.delete(key)

# --- ENTRY POINT ---

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    try:
        print("--- ArrowShelf Approximate FAISS Benchmark ---")
        print("INFO: Checking ArrowShelf server...")
        arrowshelf.list_keys()
        print("INFO: Connection OK.")

        PHYSICAL_CORES = 6  # Adjust to your hardware
        run_approximate_faiss_benchmark(num_points=100_000, num_cores=PHYSICAL_CORES)

    except arrowshelf.ConnectionError as e:
        print(f"\nERROR: ArrowShelf server unavailable. ({e})")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        arrowshelf.close()
        print("\nBenchmark finished.")
