"""Text cleaning utilities."""

import re

import polars as pl

# Shared patterns so clean_text and the scalar helpers stay in sync
MENTION_PATTERN = r"@(\w+)\s?"
COLON_PATTERN = r":?"
URL_PATTERN = r"http\S+|www\S+"


def clean_text(text: pl.Series) -> pl.Series:
    """Lowercase text, remove mentions and URLs, and strip whitespace.

    Args:
        text: Polars Series of text strings to clean

    Returns:
        Cleaned polars Series
    """
    return (
        text.str.to_lowercase()
        .str.replace_all(MENTION_PATTERN + COLON_PATTERN, "")
        .str.replace_all(URL_PATTERN, "")
        .str.strip_chars()
    )


def remove_mentions(text: str, remove_colons: bool = True) -> str:
    """Remove all @mention instances from text using regex.

    Args:
        text: Input text string
        remove_colons: Whether to also remove colons following mentions

    Returns:
        Text with mentions removed
    """
    pattern = MENTION_PATTERN + COLON_PATTERN if remove_colons else MENTION_PATTERN
    return re.sub(pattern, "", text)


def remove_urls(text: str) -> str:
    """Remove all URLs from text using regex.

    Args:
        text: Input text string

    Returns:
        Text with URLs removed
    """
    return re.sub(URL_PATTERN, "", text)
