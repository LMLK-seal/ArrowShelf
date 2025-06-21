"""
ArrowShelf: A lightning-fast, zero-copy, cross-process data store for Python.
Stop Pickling. Start Sharing.
"""
__version__ = "2.1.0"

from .client import put, get, get_arrow, delete, list_keys, close, shutdown_server
from .exceptions import ArrowShelfError, ConnectionError, ServerError