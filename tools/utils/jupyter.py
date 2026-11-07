"""Jupyter notebook utility functions"""

from IPython import get_ipython


def in_jupyter_notebook() -> bool:
    """Check if running in a jupyter notebook.

    Returns:
        bool: True if running in Jupyter notebook, False otherwise
    """
    try:
        ipython = get_ipython()
        if ipython is None or "IPKernelApp" not in ipython.config:
            raise ImportError("console")
        return True
    except ImportError, AttributeError, NameError:
        return False


def stop_notebook(msg: str = "stop here"):
    """Stop notebook execution at the cell where this function is called.

    Args:
        msg: Message to display when stopping

    Raises:
        SystemExit: Always raised to halt execution
    """
    raise SystemExit(msg)
