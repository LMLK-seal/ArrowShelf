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

**The Analogy:** Instead of photocopying a 1,000-page book for every colleague (pickling), you place the book on a shared library shelf and just tell them its location (`ArrowShelf`).

## Quick Start

**1. Installation**

```bash
pip install arrowshelf