import json
import socket
import io
import mmap
import os
import pandas as pd
import pyarrow as pa
import pyarrow.ipc
from .exceptions import ConnectionError, ServerError

SERVER_ADDRESS = ("127.0.0.1", 56789)

class ArrowShelfConnection:
    def __init__(self): self.sock,self.reader = None,None
    def connect(self):
        if self.sock: return
        try:
            self.sock = socket.create_connection(SERVER_ADDRESS, timeout=2)
            self.reader = self.sock.makefile('rb')
        except (socket.error, ConnectionRefusedError) as e:
            raise ConnectionError(f"Could not connect to ArrowShelf daemon at {SERVER_ADDRESS}") from e
    def is_connected(self):
        try:
            with socket.create_connection(SERVER_ADDRESS, timeout=0.5) as s: return True
        except (socket.error, ConnectionRefusedError): return False
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

_connection = ArrowShelfConnection()

def put(df: pd.DataFrame) -> str:
    """Stores a DataFrame in a shared memory-mapped file."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    data_bytes = sink.getvalue()
    data_len = len(data_bytes)

    _connection._send_command({"action": "RequestPath"})
    response = _connection._receive_response()
    key = response['message']
    path = response['path']

    with open(path, "wb") as f:
        f.truncate(data_len)
    with open(path, "r+b") as f:
        with mmap.mmap(f.fileno(), 0) as mm:
            mm.write(data_bytes)
    return key

def get(key: str) -> pd.DataFrame:
    """
    Retrieves a DataFrame from shared memory. This involves a final copy
    from Arrow's memory layout to Pandas' memory layout. For highest
    performance, use get_arrow() instead.
    """
    table = get_arrow(key)
    return table.to_pandas()

# --- THE NEW, HIGH-PERFORMANCE FUNCTION ---
def get_arrow(key: str) -> pa.Table:
    """
    Retrieves a zero-copy reference to the Arrow Table from shared memory.
    This is the highest-performance way to access data, avoiding the final
    conversion to a Pandas DataFrame.
    """
    _connection._send_command({"action": "GetPath", "key": key})
    response = _connection._receive_response()
    path = response['path']
    # The memory-map provides a zero-copy view into the file's contents.
    # PyArrow can read directly from this memory view.
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with pa.ipc.open_stream(mm) as reader:
            return reader.read_all()

def delete(key: str):
    _connection._send_command({"action": "Delete", "key": key})
    _connection._receive_response()

def list_keys() -> list:
    _connection._send_command({"action": "ListKeys"})
    response = _connection._receive_response()
    return response.get('keys', [])

def close():
    _connection.close()

def shutdown_server():
    print("Sending shutdown command to ArrowShelf daemon...")
    try:
        _connection._send_command({"action": "Shutdown"})
    except (ConnectionError, BrokenPipeError):
        print("Could not connect to server, it might already be stopped.")
    finally:
        close()