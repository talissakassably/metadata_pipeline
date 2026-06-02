# -*- coding: utf-8 -*-

"""
General utility functions for metadata extraction.
"""

from datetime import datetime, date


def make_json_safe(value):
    """
    Convert non-JSON-serializable objects into JSON-safe objects.

    Handles:
        - basic Python types
        - dict/list/tuple/set
        - pathlib paths
        - datetime/date
        - NumPy types
        - pandas NaN
        - quantities.Quantity
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return str(value)

    try:
        from pathlib import Path

        if isinstance(value, Path):
            return str(value)
    except Exception:
        pass

    try:
        import quantities as pq

        if isinstance(value, pq.Quantity):
            try:
                if value.size == 1:
                    return {
                        "value": float(value.magnitude),
                        "unit": str(value.units),
                    }
                return {
                    "value": value.magnitude.tolist(),
                    "unit": str(value.units),
                }
            except Exception:
                return str(value)
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

    except Exception:
        pass

    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    try:
        return str(value)
    except Exception:
        return None


def unique_values(values):
    """
    Return sorted unique values while removing None and empty values.
    """

    clean_values = []

    for value in values:
        if value is not None and value != "":
            clean_values.append(str(value))

    return sorted(list(set(clean_values)))


def summarize_numeric_values(values):
    """
    Summarize a list of numeric values.
    """

    clean_values = []

    for value in values:
        try:
            clean_values.append(float(value))
        except Exception:
            pass

    if len(clean_values) == 0:
        return None

    return {
        "min": min(clean_values),
        "max": max(clean_values),
        "mean": sum(clean_values) / len(clean_values),
        "n": len(clean_values),
    }


def dataframe_column_summary(df, column_name):
    """
    Summarize one numeric pandas dataframe column.
    """

    if df is None:
        return None

    if column_name not in df.columns:
        return None

    try:
        values = df[column_name]
        values = values.dropna()
    except Exception:
        return None

    numeric_values = []

    for value in values:
        try:
            numeric_values.append(float(value))
        except Exception:
            pass

    if len(numeric_values) == 0:
        return None

    return {
        "min": min(numeric_values),
        "max": max(numeric_values),
        "mean": sum(numeric_values) / len(numeric_values),
        "n": len(numeric_values),
    }


def dataframe_unique_values(df, column_name, max_values=50):
    """
    Return unique values from a dataframe column.
    """

    if df is None:
        return []

    if column_name not in df.columns:
        return []

    try:
        values = df[column_name].dropna().astype(str).unique().tolist()
        values = sorted(values)
        return values[:max_values]
    except Exception:
        return []


def safe_len(value):
    """
    Return len(value), or 0 if unavailable.
    """

    try:
        return len(value)
    except Exception:
        return 0