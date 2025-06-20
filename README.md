# ArrowShelf

### 🛑 Stop Pickling. 🚀 Start Sharing.

[![PyPI version](https://badge.fury.io/py/arrowshelf.svg)](https://badge.fury.io/py/arrowshelf)
![Python Version](https://img.shields.io/pypi/pyversions/arrowshelf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ArrowShelf** is a high-performance, cross-process data store for Python, designed to eliminate the crippling overhead of serialization (`pickle`) in multiprocessing workflows. It allows multiple Python processes to access large Pandas DataFrames with minimal overhead, unlocking the full power of your multi-core CPU.

---

## The Problem: Python's Multiprocessing Bottleneck

When using Python's `multiprocessing` library, sharing large objects like Pandas DataFrames between processes is incredibly slow. Python must `pickle` the data, send the bytes over a pipe, and `unpickle` it in each child process. For gigabytes of data, this overhead can make your parallel code even slower than your single-threaded code, wasting your time and your computer's expensive hardware.

## The ArrowShelf Solution: The Shared Library Shelf

ArrowShelf runs a tiny, high-performance daemon (written in Rust) that manages a central data store. Instead of slowly sending a massive copy of your data to each process, you place it on the "shelf" **once**, and then pass a tiny string key to your child processes.

**The Analogy:** Instead of photocopying a 1,000-page book for every colleague (the `pickle` way), you place the book on a shared library shelf and just tell them its location (`ArrowShelf`). Access is instantaneous.

## Key Features

*   **⚡ Blazingly Fast:** V2.0 with shared memory is designed to be 10-100x faster than pickling for large datasets.
*   **🦀 Rust-Powered Core:** The background server is written in Rust for memory safety, concurrency, and raw speed.
*   **🐍 Simple & Pythonic API:** A clean, intuitive API (`put`, `get`, `delete`) that feels natural to any Python developer.
*   **🖥️ Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.

---

## Who Is This For? Real-World Scenarios

### 👩‍🔬 The Data Scientist: Supercharging Data Preparation

**The Pain:** Priya has a 10 GB dataset. Her data cleaning and feature engineering script takes **90 minutes** to run on a single core. Trying to parallelize it with `multiprocessing` is even slower because of the pickling overhead.

**The ArrowShelf Solution:** Priya puts her DataFrame on the shelf once with `arrowshelf.put(df)`. She then gives the key to 16 worker processes. All 16 of her CPU cores light up, and the job finishes in **under 7 minutes**.

**The Benefit:** Priya can now run over 10 experiments in a single morning, instead of just one. Her "idea-to-result" cycle shrinks from hours to minutes, making her massively more productive.

### 👨‍💼 The Financial Analyst: Accelerating Backtests

**The Pain:** David needs to backtest a trading strategy against a 50 GB dataset of stock data. Each simulation takes hours, and running thousands of variations is impossible.

**The ArrowShelf Solution:** David loads the entire 50 GB dataset into ArrowShelf once. He launches hundreds of simulation processes, each instantly accessing the shared data with zero copy overhead.

**The Benefit:** His overnight backtesting jobs are now completed in his lunch break. He can test more complex strategies and find profitable signals faster than his competitors.

---

# 🚀 Quick Start

Get up and running with ArrowShelf in two minutes.

## 📦 1. Installation

First, install ArrowShelf from PyPI using pip.

```bash
pip install arrowshelf
```

## 🖥️ 2. Start the Server

ArrowShelf requires a small, dedicated server process to be running in the background.

Open your first terminal and run the following command. The server will launch and take over this terminal, printing log messages as it runs.

```bash
python -m arrowshelf.server
```

You will see a confirmation message like this, and the terminal will wait for connections:

```
--- Launching ArrowShelf Server from: C:\...\arrowshelf\bin\arrowshelfd.exe ---
[INFO] ArrowShelf daemon listening on tcp://127.0.0.1:56789
[INFO] Press Ctrl+C to shut down.
```

⚠️ **Important**: Leave this terminal running.

## 💻 3. Run Your Code

Now, open a second terminal. This is where you will run your actual data processing script. Notice how minimal the code change is to get the benefit of ArrowShelf.

### ❌ Before ArrowShelf (The Pain)

This is a typical multiprocessing script. It's slow because the `large_df` is copied over and over again.

```python
import multiprocessing as mp
import pandas as pd
import numpy as np

def process_chunk(df):
    # This is SLOW because the large_df is pickled and sent for each task.
    return df['value'].sum()

if __name__ == "__main__":
    large_df = pd.DataFrame(np.random.rand(10_000_000, 1), columns=['value'])
    with mp.Pool(processes=4) as pool:
        results = pool.map(process_chunk, [large_df] * 4)
    print("Standard multiprocessing is done.")
```

### ✅ After ArrowShelf (The Power)

With ArrowShelf, you simply put the data on the shelf first and pass the key.

```python
import multiprocessing as mp
import pandas as pd
import numpy as np
import arrowshelf  # Import the library

def process_chunk_from_shelf(data_key):
    # This is FAST because there is no data copy.
    df = arrowshelf.get(data_key)
    return df['value'].sum()

if __name__ == "__main__":
    large_df = pd.DataFrame(np.random.rand(10_000_000, 1), columns=['value'])

    # 1. Put the data onto the shelf ONCE.
    data_key = arrowshelf.put(large_df)
    print(f"DataFrame placed on shelf with key: {data_key}")

    # 2. Pass only the tiny key string to the workers.
    with mp.Pool(processes=4) as pool:
        results = pool.map(process_chunk_from_shelf, [data_key] * 4)

    # 3. Clean up the data from the shelf.
    arrowshelf.delete(data_key)
    print("ArrowShelf processing is done.")
```

---

## 📖 API Reference

The ArrowShelf API is designed to be simple and intuitive.

| Function | Description |
|----------|-------------|
| `arrowshelf.put(df: pd.DataFrame) -> str` | 📥 Stores a Pandas DataFrame on the shelf and returns a unique string key |
| `arrowshelf.get(key: str) -> pd.DataFrame` | 📤 Retrieves a copy of the DataFrame from the shelf using its key |
| `arrowshelf.delete(key: str)` | 🗑️ Removes a DataFrame from the shelf to free up memory |
| `arrowshelf.list_keys() -> list` | 📋 Returns a list of all keys currently on the shelf |
| `arrowshelf.close()` | 🔌 Closes the current Python script's connection to the server |
| `arrowshelf.shutdown_server()` | 🛑 Remotely tells the ArrowShelf server to shut down |

---

## ⚡ Performance & Roadmap

### 🏷️ Current Version (V1.0.0)
This version uses a robust TCP-based transport layer. It successfully proves the architecture works but does not yet show a significant speedup over pickling due to the overhead of converting data to and from the Arrow format.

### 🚀 Coming Soon (V2.0)
<<<<<<< HEAD
The real magic comes in V2.0, which will replace the TCP layer with **true zero-copy shared memory**. This will eliminate the data transfer bottleneck entirely and is projected to deliver a **10-100x speedup** over standard pickling for large datasets. Stay tuned! 
=======
The real magic comes in V2.0, which will replace the TCP layer with **true zero-copy shared memory**. This will eliminate the data transfer bottleneck entirely and is projected to deliver a larger speedup over standard pickling for large datasets. Stay tuned! 
>>>>>>> 85733d764e5e4e041a31dd45ba8524e8e972b790

### 🔮 Future Versions
- **📊 In-Server Querying (V3.0)**: Run SQL queries directly on the in-memory data without ever moving it to Python
- **🔧 Enhanced Data Types**: Support for NumPy arrays, Polars DataFrames, and more

---

## 🤝 Contributing

Contributions are what make the open-source community amazing! If you have ideas for features, bug reports, or improvements, please:

- 🐛 Open an issue for bug reports
- 💡 Submit feature requests
- 🔧 Create pull requests for improvements

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
