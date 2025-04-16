import contextlib
import sys
import io

@contextlib.contextmanager
def suppress_tqdm_cleanup():
    """Context manager to suppress tqdm cleanup messages."""
    original_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = original_stderr