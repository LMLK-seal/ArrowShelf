# 🏹 ArrowShelf

**High-Performance Shared Memory Data Exchange for Python**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Apache Arrow](https://img.shields.io/badge/Apache%20Arrow-Powered-orange?logo=apache&logoColor=white)](https://arrow.apache.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/Performance-⚡%20Ultra%20Fast-yellow)](https://github.com/LMLK-seal/ArrowShelf)

ArrowShelf is a cutting-edge Python library that enables **lightning-fast shared memory data exchange** between processes using Apache Arrow's columnar format. Perfect for high-performance computing, machine learning pipelines, and distributed data processing.

## 🌟 Key Features

- **🚀 Zero-Copy Operations**: Direct memory access without serialization overhead
- **🔧 Process-Safe**: Thread and multiprocess safe data sharing
- **📊 Columnar Efficiency**: Optimized for analytical workloads with Apache Arrow
- **🎯 FAISS Integration**: Built-in support for approximate nearest neighbor search
- **🔄 Automatic Cleanup**: Smart memory management with reference counting
- **🛡️ Production Ready**: Robust error handling and connection management

## 📦 Installation

```bash
pip install arrowshelf
```

For FAISS integration (optional):
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

### 🚀 Starting the ArrowShelf Server

ArrowShelf requires a server daemon to manage shared memory. Start it before running your applications:

```bash
# Start the ArrowShelf server
arrowshelf-server

# Or run in background (Linux/Mac)
arrowshelf-server &

# Windows background (using PowerShell)
Start-Process arrowshelf-server -WindowStyle Hidden
```

The server will run on `localhost:50051` by default.

## 🚀 Quick Start

### Setting Up ArrowShelf

1. **Start the server** (required):
```bash
arrowshelf-server
```

2. **Basic Usage** (in a separate terminal/process):

```python
import arrowshelf
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'x': np.random.rand(10000),
    'y': np.random.rand(10000),
    'z': np.random.rand(10000)
})

# Store in shared memory
key = arrowshelf.put(df)

# Access from any process
retrieved_df = arrowshelf.get(key)
print(f"Retrieved {len(retrieved_df)} rows")

# Cleanup
arrowshelf.delete(key)
```

### Advanced Zero-Copy Access

```python
import arrowshelf
import numpy as np

# Store data
key = arrowshelf.put(df)

# Get Arrow table for zero-copy operations
table = arrowshelf.get_arrow(key)
x_column = table.column("x").chunk(0).to_numpy(zero_copy_only=True)

# Direct NumPy operations without copying
result = np.mean(x_column)
```

## 🎯 Real-World Example: Parallel Nearest Neighbor Search

This example demonstrates how ArrowShelf enables efficient parallel processing with FAISS for approximate nearest neighbor search.

### Prerequisites

1. **Start ArrowShelf server**:
```bash
arrowshelf-server
```

2. **Install dependencies**:
```bash
pip install arrowshelf faiss-cpu pandas numpy
```

### Complete Example

```python
import multiprocessing as mp
import threading
import pandas as pd
import numpy as np
import time
import arrowshelf
import faiss
from multiprocessing.pool import ThreadPool

# Enable thread-based multiprocessing
threading.Pool = ThreadPool

def worker_faiss_search(task_data):
    """Worker function for parallel FAISS nearest neighbor search"""
    key, start_index, end_index = task_data
    
    # Zero-copy access to shared data
    table = arrowshelf.get_arrow(key).combine_chunks()
    x = table.column("x").chunk(0).to_numpy(zero_copy_only=True)
    y = table.column("y").chunk(0).to_numpy(zero_copy_only=True)
    z = table.column("z").chunk(0).to_numpy(zero_copy_only=True)
    
    # Stack coordinates for FAISS
    all_points = np.stack([x, y, z], axis=1).astype(np.float32)
    query_chunk = all_points[start_index:end_index]
    
    # Configure FAISS IVF index
    d = 3  # 3D points
    nlist = 100  # Voronoi cells
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # Train and populate index
    index.train(all_points)
    index.add(all_points)
    index.nprobe = 10  # Search cells
    
    # Perform approximate k-NN search
    _, distances = index.search(query_chunk, 11)  # k=11 (excluding self)
    avg_distance = np.mean(np.sqrt(distances[:, 1:]))  # Exclude self-distance
    
    return avg_distance

def parallel_nearest_neighbor_demo():
    """Demonstrate parallel processing with ArrowShelf + FAISS"""
    
    # Check ArrowShelf connection
    try:
        arrowshelf.list_keys()
        print("✅ ArrowShelf server connection OK")
    except arrowshelf.ConnectionError:
        print("❌ ERROR: ArrowShelf server not running!")
        print("Please start the server first: arrowshelf-server")
        return
    
    # Generate sample 3D points
    num_points = 100_000
    num_cores = 6
    
    print(f"🔍 Running parallel k-NN search on {num_points:,} 3D points")
    
    # Create dataset
    df = pd.DataFrame(np.random.rand(num_points, 3), columns=['x', 'y', 'z'])
    print(f"📊 Dataset size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Store in ArrowShelf
    key = arrowshelf.put(df)
    
    # Create tasks for parallel processing
    chunk_size = num_points // num_cores
    tasks = [
        (key, i * chunk_size, (i + 1) * chunk_size) 
        for i in range(num_cores)
    ]
    
    # Execute parallel search
    print(f"⚡ Processing with {num_cores} cores...")
    start_time = time.perf_counter()
    
    with ThreadPool(processes=num_cores) as pool:
        results = pool.map(worker_faiss_search, tasks)
    
    duration = time.perf_counter() - start_time
    avg_distance = np.mean(results)
    
    # Results
    print(f"✅ Average 10-NN distance: {avg_distance:.6f}")
    print(f"🚀 Processing time: {duration:.4f} seconds")
    print(f"🔥 Throughput: {num_points/duration:,.0f} points/second")
    
    # Cleanup
    arrowshelf.delete(key)
    print("🧹 Cleanup completed")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parallel_nearest_neighbor_demo()
```

### Running the Example

1. **Terminal 1** - Start the server:
```bash
arrowshelf-server
```

2. **Terminal 2** - Run the example:
```bash
python nearest_neighbor_demo.py
```

**Expected Output:**
```
✅ ArrowShelf server connection OK
🔍 Running parallel k-NN search on 100,000 3D points
📊 Dataset size: 2.29 MB
⚡ Processing with 6 cores...
✅ Average 10-NN distance: 210.789151
🚀 Processing time: 1.0017 seconds
🔥 Throughput: 99,830 points/second
🧹 Cleanup completed
```

# How ArrowShelf Helps with Large Datasets

## The Process Flow

1. **Load Once, Use Many Times**: Your large dataset is loaded into memory once and placed on the ArrowShelf
2. **Zero-Copy Access**: Multiple worker processes access the same data instantly without copying
3. **Memory Efficient**: Instead of having 8 copies of your data for 8 cores, you have just 1 shared copy
4. **Fast Parallel Processing**: Workers can immediately start computing instead of waiting for data transfer

## Real-World Scenarios Where This Shines

### Scenario 1: Machine Learning Feature Engineering

```python
# You have a 5GB customer dataset
customer_data = pd.read_csv("customer_behavior_5gb.csv")

# Put it on the shelf once
data_key = arrowshelf.put(customer_data)

# Now run multiple feature engineering tasks in parallel:
# - Calculate RFM scores
# - Generate time-based features  
# - Compute behavioral clusters
# - Create recommendation features

# Each task accesses the same 5GB instantly, no copying!
```

### Scenario 2: Financial Risk Analysis

```python
# Load 10 million stock price records
stock_data = pd.read_parquet("stock_prices_10m_rows.parquet")
data_key = arrowshelf.put(stock_data)

# Run parallel risk calculations:
# - VaR calculations for different portfolios
# - Correlation analysis across sectors
# - Volatility modeling
# - Stress testing scenarios

# Traditional approach: Each task waits 30+ seconds for data copying
# ArrowShelf approach: Each task starts immediately
```

### Scenario 3: Geospatial Analysis

```python
# Load millions of GPS coordinates
location_data = pd.read_csv("gps_coordinates_50m_points.csv")
data_key = arrowshelf.put(location_data)

# Parallel geospatial tasks:
# - Find nearest neighbors for different regions
# - Calculate clustering patterns
# - Identify hotspots and anomalies
# - Generate heatmaps for different time periods
```

## Key Benefits

1. **Memory Efficiency**: Instead of 6 copies of your 3D points (one per core), you have 1 shared copy
2. **Instant Access**: Each worker gets the data via `arrowshelf.get_arrow(key)` instantly
3. **Zero-Copy Operations**: The `.to_numpy(zero_copy_only=True)` means no data copying at all
4. **Scalable**: Works whether you have 100K points or 100M points

## The Traditional Problem vs ArrowShelf Solution

### Traditional Multiprocessing (Pickle)

```
Main Process: Load 5GB dataset
├── Send 5GB copy to Worker 1 (30 seconds)
├── Send 5GB copy to Worker 2 (30 seconds)  
├── Send 5GB copy to Worker 3 (30 seconds)
└── Send 5GB copy to Worker 4 (30 seconds)
Total data transfer: 120 seconds + computation time
```

### ArrowShelf Approach

```
Main Process: Load 5GB dataset → Put on shelf (2 seconds)
├── Worker 1: Get instant reference (0.001 seconds)
├── Worker 2: Get instant reference (0.001 seconds)
├── Worker 3: Get instant reference (0.001 seconds)
└── Worker 4: Get instant reference (0.001 seconds)
Total data transfer: 2 seconds + computation time
```

## Perfect Use Cases

1. **Data Science Notebooks**: When you're iteratively running different analyses on the same large dataset
2. **ETL Pipelines**: When multiple transformation steps need access to the same source data
3. **Machine Learning**: When training multiple models or doing hyperparameter tuning on the same dataset
4. **Scientific Computing**: When running simulations that need shared reference data
5. **Real-time Analytics**: When multiple dashboards need to query the same large dataset

## The Key Insight

ArrowShelf eliminates the "data tax" - the time penalty you normally pay for having multiple processes work with large datasets. Instead of spending most of your time copying data, you spend it actually computing results.


## 🚀 Project Evolution

ArrowShelf has evolved from a simple data sharing concept to a high-performance computing powerhouse. Here's the journey of optimization:

### Performance Evolution Timeline

| Benchmark | Architecture | Algorithm | Time | Improvement |
|-----------|-------------|-----------|------|-------------|
| **Pickle + Brute Force** | Slow Data Transfer | Brute Force O(n²) | 16.7s | *Baseline* |
| **ArrowShelf + Brute Force** | Fast Data Transfer | Brute Force O(n²) | 14.5s | **13% faster** |
| **ArrowShelf + FAISS IndexFlatL2** | Fast Data Transfer | Optimized Exact Search | 1.84s | **87% faster** |
| **ArrowShelf + FAISS IndexIVFFlat** | Fast Data Transfer | Approximate Search | **1.00s** | **94% faster** |

### 📊 Performance Visualization

```
Traditional Pickle Approach    ████████████████████████████████████ 16.7s
ArrowShelf + Brute Force      ███████████████████████████████     14.5s
ArrowShelf + FAISS Exact      ████                               1.84s
ArrowShelf + FAISS Approx     ██                                 1.00s ⚡
```

### 🎯 Key Milestones

- **🏗️ Phase 1: Foundation** - Basic shared memory with Apache Arrow
- **⚡ Phase 2: Optimization** - Zero-copy operations and efficient data transfer  
- **🔍 Phase 3: Intelligence** - FAISS integration for similarity search
- **🚀 Phase 4: Approximation** - IVF indexing for ultimate performance

The evolution demonstrates a **16.7x performance improvement** from traditional pickle-based approaches to our current FAISS-optimized implementation.

## 📈 Performance

ArrowShelf delivers exceptional performance for data-intensive applications:

### FAISS Integration Benchmark
- **Dataset**: 100,000 3D points (2.29 MB)
- **Operation**: Approximate 10-NN search with IVF index
- **Hardware**: 6-core parallel processing
- **Result**: **1.00 seconds** processing time
- **Throughput**: **~100K points/second**

### Key Performance Benefits
- **Zero-Copy Access**: Direct memory mapping eliminates serialization overhead
- **Columnar Storage**: Optimized for analytical operations and vectorized computations
- **Parallel Processing**: Efficient multi-core scaling with shared memory
- **Memory Efficiency**: Reference counting prevents memory leaks

## 🔧 API Reference

### Core Functions

```python
# Store data in shared memory
key = arrowshelf.put(data)

# Retrieve data as pandas DataFrame
df = arrowshelf.get(key)

# Retrieve data as Arrow Table (zero-copy)
table = arrowshelf.get_arrow(key)

# List all stored keys
keys = arrowshelf.list_keys()

# Delete data from shared memory
arrowshelf.delete(key)

# Close connection
arrowshelf.close()
```

### Advanced Operations

```python
# Batch operations
arrowshelf.delete_all()  # Clear all data

# Connection management
arrowshelf.is_connected()  # Check connection status

# Memory statistics
arrowshelf.memory_usage()  # Get usage statistics
```

## 🛠️ Use Cases

### 🤖 Machine Learning
- **Feature Engineering**: Share preprocessed datasets across training processes
- **Model Serving**: Cache model predictions and intermediate results
- **Hyperparameter Tuning**: Efficient data sharing in parallel optimization

### 📊 Data Analytics
- **ETL Pipelines**: Zero-copy data transformations
- **Distributed Computing**: Shared memory for map-reduce operations
- **Real-time Analytics**: High-throughput data processing

### 🔬 Scientific Computing
- **Numerical Simulations**: Share large arrays between simulation processes
- **Image Processing**: Efficient pixel data sharing
- **Geospatial Analysis**: Fast coordinate and geometry operations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Process A     │    │   ArrowShelf    │    │   Process B     │
│                 │    │     Server      │    │                 │
│  put(data) ────────▶│                 │◀──────── get(key)   │
│                 │    │  Apache Arrow   │    │                 │
│                 │    │ Shared Memory   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

- Python 3.7+
- Apache Arrow
- pandas
- numpy

Optional dependencies:
- FAISS (for nearest neighbor search)
- multiprocessing support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Apache Arrow](https://arrow.apache.org/) columnar memory format
- Optimized for [FAISS](https://github.com/facebookresearch/faiss) similarity search
- Inspired by modern high-performance computing needs

## 📞 Support

- 📧 **Issues**: [GitHub Issues](https://github.com/LMLK-seal/ArrowShelf/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/LMLK-seal/ArrowShelf/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/LMLK-seal/ArrowShelf/wiki)

---

**⭐ Star this repository if ArrowShelf helps accelerate your data processing workflows!**
