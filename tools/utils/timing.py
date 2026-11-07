"""Time measurement and formatting utilities."""

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any


def print_time(prefix: str, start: float) -> None:
    """Print elapsed time in seconds since a start timestamp.

    Args:
        prefix: Label to print before the time value
        start: Start time from time.time()
    """
    print(f"{prefix}: {time.time() - start:.3f} seconds")


def print_time_elapsed(
    since: int | float, newline: bool = False, prefix: str = "time elapsed"
) -> None:
    """Print elapsed time in HH:MM:SS.s clock format.

    Args:
        since: Start time from time.time()
        newline: Whether to add a trailing newline
        prefix: Label to print before the time value
    """
    msg = f"{prefix}: {seconds_as_clock(time.time() - since)}"
    print(f"{msg}\n" if newline else msg)


def get_printable_timedelta(td: timedelta, precision: int = 1) -> str:
    """Format a timedelta as a human-readable clock string.

    Args:
        td: Timedelta object to format
        precision: Number of decimal places for milliseconds

    Returns:
        Formatted time string in HH:MM:SS.s format
    """
    seconds = td.total_seconds()
    timestamp = seconds_as_clock(seconds, precision=precision)
    return timestamp


def seconds_as_clock(seconds: float, precision: int = 1) -> str:
    """Convert seconds to HH:MM:SS.s clock format string.

    Args:
        seconds: Number of seconds to convert
        precision: Number of decimal places for sub-second display

    Returns:
        Formatted clock string
    """
    rounded = round(seconds, precision)
    timestamp = time.strftime("%H:%M:%S", time.gmtime(rounded))
    if precision == 0:
        return timestamp
    fraction = f"{rounded:.{precision}f}".split(".")[-1]
    return f"{timestamp}.{fraction}"


def stopwatch(
    prefix: str = "elapsed time",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to time function execution and print elapsed time.

    Args:
        prefix: Label to print before the elapsed time

    Returns:
        Decorator function that wraps the target function with timing
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = datetime.now()
            result = func(*args, **kwargs)
            elapsed = get_printable_timedelta(datetime.now() - now)
            print(f"{prefix}: {elapsed}")
            return result

        return wrapper

    return decorator
