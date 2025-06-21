# ArrowShelf

### 🛑 Stop Pickling. 🚀 Start Sharing.

[![PyPI version](https://img.shields.io/pypi/v/arrowshelf.svg)](https://pypi.org/project/arrowshelf/)
![Python Version](https://img.shields.io/pypi/pyversions/arrowshelf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ArrowShelf** is a high-performance, zero-copy, cross-process data store for Python. It uses Apache Arrow and shared memory to eliminate the crippling overhead of `pickle` in multiprocessing workflows, allowing you to unlock the full power of your multi-core CPU for data science and analysis.

---

## The Problem: Python's Multiprocessing Bottleneck

When using Python's `multiprocessing` library, sharing large DataFrames between processes is incredibly slow. Python must `pickle` the data, send the bytes over a pipe, and `unpickle` it in each child process. For gigabytes of data, this overhead can make your parallel code even slower than single-threaded code, wasting your time and your expensive hardware.

## The ArrowShelf Solution: The Shared Memory Bookshelf

ArrowShelf runs a tiny, high-performance daemon (written in Rust) that coordinates access to data stored in shared memory. Instead of slowly sending a massive copy of your data to each process, you place it on the "shelf" **once**. Your worker processes can then read this data instantly with zero copy overhead.

**The Analogy:** Instead of photocopying a 1,000-page book for every colleague (the `pickle` way), you place the book on a magic, shared bookshelf and just tell them its location (`ArrowShelf`). Access is instantaneous.

---

## 🚀 Quick Start

**1. Installation**
```bash
pip install arrowshelf
```

**2. Start the Server**
In your first terminal, start the ArrowShelf server. It will run in the foreground.

```bash
python -m arrowshelf.server
```

**3. Run Your High-Performance Code**
In a second terminal, run your processing script. To get maximum performance, use arrowshelf.get_arrow() and compute directly with PyArrow's C++-backed functions.

```python
import multiprocessing as mp
import pandas as pd
import numpy as np
import pyarrow.compute as pc # Import PyArrow's compute functions
import arrowshelf

def high_performance_worker(data_key):
    # 1. Get a zero-copy reference to the Arrow Table. This is instant.
    arrow_table = arrowshelf.get_arrow(data_key)
    
    # 2. Perform calculations directly on the Arrow data.
    #    This avoids the slow .to_pandas() step.
    result = pc.sum(arrow_table.column('value')).as_py()
    return result

if __name__ == "__main__":
    large_df = pd.DataFrame(np.random.rand(10_000_000, 1), columns=['value'])

    # 1. Put the data onto the shelf ONCE.
    data_key = arrowshelf.put(large_df)

    # 2. Pass only the tiny key string to the workers.
    with mp.Pool(processes=4) as pool:
        results = pool.map(high_performance_worker, [data_key] * 4)

    # 3. Clean up the data from the shelf.
    arrowshelf.delete(data_key)
    print("ArrowShelf processing complete!")
```

---

## ⚡ Performance: Understanding the Trade-Offs

ArrowShelf is designed to attack the data transfer bottleneck (`pickle`). Benchmarks show that for workloads that are truly limited by data serialization, ArrowShelf provides a significant advantage.

However, for heavily CPU-bound parallel tasks on a single machine, performance is ultimately constrained by Python's Global Interpreter Lock (GIL).

### Benchmark: Heavy CPU Workload

*Test: A complex mathematical simulation on a 10,000,000 row DataFrame using 12 cores.*

| Workflow   | Total Time | Breakdown                                       |
|------------|------------|-------------------------------------------------|
| Pickle     | **1.72 s** | (Each worker gets a small, independent data chunk) |
| ArrowShelf | 4.02 s     | (1.3s one-time `put` + 2.7s parallel computation) |
| **Speedup**| **0.43x**  |                                                 |

**The Verdict & Analysis:**

In this CPU-bound scenario, the overhead of coordinating 12 processes accessing a single large shared memory object, combined with GIL contention, makes the `pickle` strategy of "divide and conquer" more effective. This is a classic example of Amdahl's Law: we successfully eliminated the data transfer bottleneck, only to reveal that the next bottleneck is the GIL itself.

**Where ArrowShelf truly shines is in I/O-bound or interactive workflows where the one-time `put` cost is amortized over many operations.** For example, in a Jupyter notebook where a data scientist loads a large dataset once and then runs dozens of different parallel analyses on it, ArrowShelf's "pay-once" model provides a massive productivity boost that `pickle` cannot match.

---

## 📖 API Reference

| Function                    | Description                                                           |
|-----------------------------|-----------------------------------------------------------------------|
| `arrowshelf.put(df)`        | 📥 Stores a Pandas DataFrame on the shelf, returns a key.            |
| `arrowshelf.get(key)`       | 📤 Retrieves a copy as a Pandas DataFrame (for convenience).         |
| `arrowshelf.get_arrow(key)` | 🚀 Retrieves a zero-copy reference as a PyArrow Table (for high-performance). |
| `arrowshelf.delete(key)`    | 🗑️ Removes an object from the shelf.                                |
| `arrowshelf.list_keys()`    | 📋 Returns a list of all keys on the shelf.                         |

---

## 🔮 Future Roadmap

- **In-Server Querying (V3.0):** Run SQL queries directly on the in-memory data via DataFusion.
- **Enhanced Data Types:** Native support for NumPy arrays, Polars DataFrames, and more.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request on our GitHub repository.

---

## 📄 License

This project is licensed under the MIT License.