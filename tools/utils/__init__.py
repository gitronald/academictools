"""General utilities: printing, timing, text, jupyter."""

from .jupyter import in_jupyter_notebook, stop_notebook
from .printing import (
    get_script_header,
    get_script_path,
    print_green,
    print_line,
    print_red,
    wraprint,
)
from .qr import make_qr
from .text import clean_text, remove_mentions, remove_urls
from .timing import (
    get_printable_timedelta,
    print_time,
    print_time_elapsed,
    seconds_as_clock,
    stopwatch,
)

__all__ = [
    # Printing
    "get_script_path",
    "get_script_header",
    "wraprint",
    "print_line",
    "print_green",
    "print_red",
    # Timing
    "print_time",
    "print_time_elapsed",
    "get_printable_timedelta",
    "seconds_as_clock",
    "stopwatch",
    # Text
    "clean_text",
    "remove_mentions",
    "remove_urls",
    # Jupyter
    "in_jupyter_notebook",
    "stop_notebook",
    # QR
    "make_qr",
]
