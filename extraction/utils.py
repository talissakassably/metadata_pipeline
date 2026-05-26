# -*- coding: utf-8 -*-

"""
General utility functions for metadata extraction.
"""


def make_json_safe(value):
    """
    Convert non-JSON-serializable objects into JSON-safe objects.
    """

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    return str(value)


def unique_values(values):
    """
    Return sorted unique values while removing None and empty values.
    """

    clean_values = []

    for value in values:
        if value is not None and value != "":
            clean_values.append(value)

    return sorted(list(set(clean_values)))