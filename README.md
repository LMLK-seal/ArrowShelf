# ArrowShelf

[![PyPI version](https://badge.fury.io/py/arrowshelf.svg)](https://badge.fury.io/py/arrowshelf) <!-- You'll activate this when you publish -->
![Python Version](https://img.shields.io/pypi/pyversions/arrowshelf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### *Stop Pickling. Start Sharing.*

**ArrowShelf** is a high-performance, cross-process data store for Python, designed to eliminate the crippling overhead of serialization (`pickle`) in multiprocessing workflows. It allows multiple Python processes to access large Pandas DataFrames with minimal overhead, unlocking the full power of your multi-core CPU for data analysis.

## The Problem

When using Python's `multiprocessing` library, sharing large objects like Pandas DataFrames between processes is incredibly slow. Python must `pickle` the data, send the bytes over a pipe, and `unpickle` it in each child process. For gigabytes of data, this overhead can make your parallel code even slower than your single-threaded code.

## The ArrowShelf Solution

ArrowShelf runs a tiny, high-performance daemon (written in Rust) that manages a central data store. Instead of sending your huge DataFrame to each process, you place it on the "shelf" **once**, and then pass a tiny string key to your child processes.

**The Analogys:**

**The Data Scientist:**
The Pain: A data scientist, let's call her Priya, has a 10 GB CSV file of user activity. She needs to clean it, calculate new features, and prepare it for a machine learning model. Her Python script, running on a powerful 16-core machine, takes 90 minutes to run. Why? Because Pandas runs on a single core. She tries to use multiprocessing, but finds it's even slower because Python spends all its time pickling and unpickling 10 GB of data for each of the 16 processes.
How Memry Helps (The Real-World Benefit):
Priya runs key = memry.put(my_dataframe) once. This takes a few seconds.
She starts her 16 worker processes, but instead of sending them a massive DataFrame, she sends them the tiny string key.
Each of the 16 processes instantly gets access to the full, shared DataFrame with zero copy overhead (using the V2.0 shared memory feature).
All 16 CPU cores light up to 100% utilization, each working on a separate chunk of the data.
Her 90-minute job now finishes in under 7 minutes.
What this gives her: She can now run 10 different feature engineering experiments in a single morning, instead of one or two per day. Her "idea-to-result" cycle shrinks from hours to minutes, making her massively more productive and creative.

**The Financial Analyst:**
The Pain: A quantitative analyst, David, needs to backtest a new trading strategy against 20 years of high-frequency stock data (a 50 GB dataset). His simulation needs to be run thousands of times with different parameters. Each run takes hours, and the dataset is too large to split up easily.
How Memry Helps (The Real-World Benefit):
David loads the entire 50 GB dataset into Memry once.
He launches hundreds of simulation processes. Each one gets the key to the master dataset.
The simulations run in a massively parallel way, each reading the historical data it needs from the zero-copy shared memory.
His overnight backtesting jobs can now be completed in his lunch break.
What this gives him: He can test more complex strategies and find profitable signals faster than his competitors. The firm gets a better return on its research.

## Quick Start

**1. Installation**

```bash
pip install arrowshelf
=======
A lightning-fast, zero-copy, cross-process data store for Python using Apache Arrow.
>>>>>>> 7eb98e272c91d464bae2a9eca681bfa8a2560984
