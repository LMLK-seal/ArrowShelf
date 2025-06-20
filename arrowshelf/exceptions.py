class ArrowShelfError(Exception):
    """Base exception for the ArrowShelf client."""
    pass

class ConnectionError(ArrowShelfError):
    """Raised when the client cannot connect to the ArrowShelf daemon."""
    pass

class ServerError(ArrowShelfError):
    """Raised when the server returns an error message."""
    pass