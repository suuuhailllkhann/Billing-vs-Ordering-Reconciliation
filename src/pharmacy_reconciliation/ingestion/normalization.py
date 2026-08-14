"""Deterministic header normalization for source-column comparison."""

import re
import unicodedata


def normalize_header(value: object) -> str:
    """Return a stable comparison key without changing source data values.

    Unicode text is normalized, surrounding whitespace/punctuation is removed,
    letters are lowercased, and runs of non-alphanumeric characters become one
    underscore. Original header text remains available in mapping metadata.
    """
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

