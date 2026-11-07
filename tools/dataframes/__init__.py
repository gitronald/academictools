"""DataFrame operations: frequency tables, inspection, reshaping, display."""

from .dates import decompose_dates, process_timestamps
from .describe import describe
from .dft import (
    binned_col,
    dft,
    get_binned_dft,
    neat_n,
    neat_p,
    presentation_table,
)
from .display import (
    display_tables,
    display_tables_html,
    display_tables_text,
    pd_options,
    set_css,
)
from .inspection import (
    dfshape,
    get_closest_row,
    get_dataframe_details,
    get_nonzero_min,
    print_row,
    unlist,
)
from .reshaping import (
    crosstab,
    crosstab_np,
    groupby_value_counts,
    join_df_list,
)
from .sessions import add_sessions

__all__ = [
    # DFT
    "dft",
    "neat_n",
    "neat_p",
    "binned_col",
    "get_binned_dft",
    "presentation_table",
    # Describe
    "describe",
    # Inspection
    "dfshape",
    "unlist",
    "get_dataframe_details",
    "print_row",
    "get_closest_row",
    "get_nonzero_min",
    # Reshaping
    "groupby_value_counts",
    "crosstab",
    "crosstab_np",
    "join_df_list",
    # Dates
    "process_timestamps",
    "decompose_dates",
    # Sessions
    "add_sessions",
    # Display
    "display_tables",
    "display_tables_text",
    "display_tables_html",
    "set_css",
    "pd_options",
]
