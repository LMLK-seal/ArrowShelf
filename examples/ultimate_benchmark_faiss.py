import numpy as np
import pandas as pd
import faiss
import time

def faiss_knn_average_distance(points_np: np.ndarray, k: int = 10) -> float:
    points32 = points_np.astype(np.float32)
    index = faiss.IndexFlatL2(3)
    index.add(points32)
    _, distances = index.search(points32, k + 1)

    print("\nFirst 5 FAISS raw distances (squared):")
    print(distances[:5, 1:])

    print("\nFirst 5 FAISS real distances (sqrt):")
    print(np.sqrt(distances[:5, 1:]))

    avg_dist = np.mean(np.sqrt(distances[:, 1:]))  # 💥 THIS IS THE FIX
    return avg_dist

def run_optimized_faiss_benchmark(num_points=40_000):
    print("\n=== OPTIMIZED FAISS k-NN BENCHMARK (CPU-only) ===")
    print(f"Testing on {num_points:,} 3D points...\n")

    df = pd.DataFrame(np.random.rand(num_points, 3), columns=['x', 'y', 'z'])
    df_size_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"DataFrame size: {df_size_mb:.2f} MB")

    points_np = df[['x', 'y', 'z']].to_numpy()
    print("Point value range (x, y, z):")
    print("Min:", points_np.min(axis=0))
    print("Max:", points_np.max(axis=0))

    print("\nRunning FAISS k-NN (exact search, CPU)...")
    start_time = time.perf_counter()
    avg_dist = faiss_knn_average_distance(points_np, k=10)
    duration = time.perf_counter() - start_time

    print(f"\n✅ Average distance to 10 nearest neighbors: {avg_dist:.6f}")
    print(f"Total FAISS time: {duration:.4f} seconds")
    print("===============================================")

if __name__ == "__main__":
    run_optimized_faiss_benchmark(num_points=40_000)
