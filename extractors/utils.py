# -*- coding: utf-8 -*-
"""
Shared utility functions for extractor scripts.

This file is intentionally self-contained because the extractor scripts are
executed as standalone scripts from the `extractors/` folder.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math


def make_json_safe(value):
    """
    Convert Python / NumPy / pandas / Neo / quantities objects into JSON-safe
    values without expanding large arrays.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        try:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
        except Exception:
            pass
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return str(value)

    # NumPy objects
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, np.ndarray):
            if value.size > 50:
                return {
                    "array_summary": True,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            return [make_json_safe(x) for x in value.tolist()]
    except Exception:
        pass

    # Pandas values
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass

    # Quantities / Neo time values
    try:
        if hasattr(value, "magnitude") and hasattr(value, "units"):
            magnitude = value.magnitude
            try:
                import numpy as np
                arr = np.asarray(magnitude)
                if arr.size == 1:
                    return {
                        "value": make_json_safe(arr.reshape(-1)[0]),
                        "unit": str(value.units),
                    }
                return {
                    "quantity_summary": True,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "unit": str(value.units),
                }
            except Exception:
                return str(value)
    except Exception:
        pass

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[str(k)] = make_json_safe(v)
        return out

    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if len(values) > 100:
            return {
                "list_summary": True,
                "length": len(values),
                "preview": [make_json_safe(x) for x in values[:10]],
            }
        return [make_json_safe(x) for x in values]

    try:
        return str(value)
    except Exception:
        return None


def unique_values(values, max_values=50):
    """
    Return sorted unique JSON-safe values from an iterable.
    """
    try:
        values = list(values)
    except Exception:
        return []

    cleaned = []
    for value in values:
        safe = make_json_safe(value)
        if safe is None:
            continue
        if isinstance(safe, (dict, list)):
            safe = str(safe)
        cleaned.append(str(safe))

    unique = sorted(set(cleaned))
    return unique[:max_values]


def dataframe_unique_values(df, column_name, max_values=50):
    """
    Return unique values from a dataframe column.
    """
    if df is None or column_name not in df.columns:
        return []

    try:
        return unique_values(df[column_name].dropna().tolist(), max_values=max_values)
    except Exception:
        return []


def dataframe_column_summary(df, column_name):
    """
    Summarize a numeric dataframe column.
    """
    if df is None or column_name not in df.columns:
        return None

    try:
        import pandas as pd

        values = pd.to_numeric(df[column_name], errors="coerce").dropna()

        if len(values) == 0:
            return None

        return make_json_safe({
            "n": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
        })
    except Exception:
        return None


def safe_file_size_mb(path):
    try:
        path = Path(path)
        return round(path.stat().st_size / 1024 / 1024, 3)
    except Exception:
        return None


def safe_relpath(path, root):
    try:
        return str(Path(path).relative_to(Path(root)))
    except Exception:
        try:
            return str(Path(path))
        except Exception:
            return ""
