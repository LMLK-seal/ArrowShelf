"""
ArrowShelf: A lightning-fast, zero-copy, cross-process data store for Python.
Stop Pickling. Start Sharing.
"""
__version__ = "2.2.0"

from .client import (
    put, 
    get, 
    get_arrow, 
    get_numpy_view, 
    delete, 
    list_keys, 
    close, 
    shutdown_server
)
from .exceptions import ArrowShelfError, ConnectionError, ServerError