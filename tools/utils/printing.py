"""Print and display formatting utilities."""

import inspect
import textwrap


def get_script_path() -> str:
    """Get the file path of the top-level calling script.

    Returns:
        Absolute path of the outermost script in the call stack
    """
    return inspect.stack()[-1].filename


def get_script_header(
    arrow_style: str = "bold dark_orange",
    script_style: str = "bold sky_blue3",
    line_style: str = "light_goldenrod3",
    line_char: str = "=",
    lines: bool = False,
    arrow_style_end: bool = False,
) -> str:
    """Build a styled script header with arrow and path for rich console output.

    Args:
        arrow_style: Rich style for the arrow character
        script_style: Rich style for the script path
        line_style: Rich style for the separator lines
        line_char: Character used for separator lines
        lines: Whether to include separator lines above and below
        arrow_style_end: Whether to add a trailing arrow character

    Returns:
        Formatted rich markup string for console display
    """
    script_path = get_script_path()
    script_line = f"[{script_style}]{script_path}[{script_style}]"
    arrow_char = "\U0001e89c\u27dd\u27f6"  # 𞢜⟝⟶
    arrow_char_styled = f"[{arrow_style}]{arrow_char}[/{arrow_style}]"
    script_line = f"{arrow_char_styled} {script_line}"

    if arrow_style_end:
        arrow_char_end = "\u27dd\u2b0e"  # ⟝⬎
        arrow_char_end_styled = f"[{arrow_style}]{arrow_char_end}[/{arrow_style}]"
        script_line = f"{script_line} {arrow_char_end_styled}"

    format_line = f"[{line_style}]{line_char * len(script_path)}[{line_style}]"
    return f"{format_line}\n{script_line}\n{format_line}" if lines else script_line


def wraprint(text: str, width: int = 100, return_str: bool = False) -> str | None:
    """Wrap text to a specified width and print or return it.

    Args:
        text: Text string to wrap
        width: Maximum line width in characters
        return_str: Whether to return the wrapped string instead of printing
    """
    wrapped = textwrap.fill(text, width)
    if return_str:
        return wrapped
    else:
        print(wrapped, "\n")


def print_line(word: str = "", length: int = 80, newline: bool = False) -> None:
    """Print a word followed by a line to a specified length.

    Args:
        word: Text to display before the line
        length: Total length of the line (default: 80)
        newline: Whether to add a newline after the line (default: False)
    """
    line = "".join(["-"] * length)
    word_line = f"{word} {line[len(word) + 1 :]}" + ("\n" if newline else "")
    print(word_line if word else line)


def print_green(text: str = "green text") -> None:
    """Print text in green using ANSI escape codes.

    Args:
        text: Text string to print in green
    """
    print(f"\033[32m{text}\033[0m")


def print_red(text: str = "red text") -> None:
    """Print text in red using ANSI escape codes.

    Args:
        text: Text string to print in red
    """
    print(f"\033[91m{text}\033[0m")
