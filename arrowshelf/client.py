import json
import socket
import io
import mmap
import os
import threading
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc
from .exceptions import ConnectionError, ServerError

SERVER_ADDRESS = ("127.0.0.1", 56789)

# --- THREAD-SAFE CONNECTION MANAGEMENT ---
# We use a thread-local object to ensure that each thread gets its own
# independent connection to the server. This prevents race conditions.
thread_local_storage = threading.local()

def get_connection():
    """
    Returns a connection object that is unique to the current thread.
    Creates a new connection if one doesn't exist for this thread.
    """
    if not hasattr(thread_local_storage, 'connection'):
        # This thread does not have a connection yet, so create one.
        thread_local_storage.connection = ArrowShelfConnection()
        thread_local_storage.connection.connect()
    return thread_local_storage.connection

class ArrowShelfConnection:
    def __init__(self): self.sock,self.reader = None,None
    def connect(self):
        if self.sock: return
        try:
            self.sock = socket.create_connection(SERVER_ADDRESS, timeout=5) # Increased timeout
            self.reader = self.sock.makefile('rb')
        except (socket.error, ConnectionRefusedError) as e:
            raise ConnectionError(f"Could not connect to ArrowShelf daemon") from e
    def close(self):
        if self.sock: self.sock.close()
        self.sock, self.reader = None, None
    def _send_command(self, cmd):
        if not self.sock: self.connect()
        try: self.sock.sendall((json.dumps(cmd) + '\n').encode('utf-8'))
        except (socket.error, BrokenPipeError): self.connect(); self.sock.sendall((json.dumps(cmd) + '\n').encode('utf-8'))
    def _receive_response(self):
        if not self.sock: self.connect()
        line = self.reader.readline()
        if not line: raise ConnectionError("Daemon closed connection")
        resp = json.loads(line)
        if resp.get('status') == 'Error': raise ServerError(resp.get('message'))
        return resp

# --- API Functions now use the thread-safe get_connection() ---

def put(df: pd.DataFrame) -> str:
    conn = get_connection()
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    data_bytes = sink.getvalue()
    conn._send_command({"action": "RequestPath"})
    response = conn._receive_response()
    key = response['message']
    path = response['path']
    with open(path, "wb") as f: f.truncate(len(data_bytes))
    with open(path, "r+b") as f:
        with mmap.mmap(f.fileno(), 0) as mm: mm.write(data_bytes)
    return key

def get(key: str) -> pd.DataFrame:
    table = get_arrow(key)
    return table.to_pandas()

def get_arrow(key: str) -> pa.Table:
    conn = get_connection()
    conn._send_command({"action": "GetPath", "key": key})
    response = conn._receive_response()
    path = response['path']
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with pa.ipc.open_stream(mm) as reader:
            return reader.read_all()

def get_numpy_view(key: str, column_name: str) -> np.ndarray:
    table = get_arrow(key) # This now uses a thread-safe connection
    column = table.column(column_name)
    if column.num_chunks != 1:
        # For simplicity, let's combine chunks if needed.
        table = table.combine_chunks()
        column = table.column(column_name)
    data_buffer = column.chunk(0).buffers()[1]
    numpy_dtype = column.type.to_pandas_dtype()
    return np.frombuffer(data_buffer, dtype=numpy_dtype)

def delete(key: str):
    conn = get_connection()
    conn._send_command({"action": "Delete", "key": key})
    conn._receive_response()

def list_keys() -> list:
    conn = get_connection()
    conn._send_command({"action": "ListKeys"})
    response = conn._receive_response()
    return response.get('keys', [])

def close():
    """Closes the connection FOR THE CURRENT THREAD."""
    if hasattr(thread_local_storage, 'connection'):
        thread_local_storage.connection.close()
        del thread_local_storage.connection

def shutdown_server():
    conn = get_connection()
    print("Sending shutdown command to ArrowShelf daemon...")
    try:
        conn._send_command({"action": "Shutdown"})
    except (ConnectionError, BrokenPipeError):
        print("Could not connect to server, it might already be stopped.")
    finally:
        close()