# -*- coding: utf-8 -*-

"""
Extract metadata from pickle files.

This is a fallback for legacy .pkl files.
It does not replace Neo extraction.
"""

import pickle

from utils import make_json_safe


def summarize_python_object(obj):
    """
    Summarize a Python object loaded from pickle.
    """

    summary = {
        "object_type": type(obj).__name__,
        "object_module": type(obj).__module__,
    }

    if isinstance(obj, dict):
        summary["n_keys"] = len(obj)
        summary["keys"] = list(obj.keys())

    elif isinstance(obj, list):
        summary["list_length"] = len(obj)

        if len(obj) > 0:
            summary["first_item_type"] = type(obj[0]).__name__
            summary["first_item_module"] = type(obj[0]).__module__

    if hasattr(obj, "segments"):

        summary["looks_like_neo_block"] = True
        summary["n_segments"] = len(obj.segments)

        n_analogsignals = 0
        n_spiketrains = 0
        n_events = 0
        n_epochs = 0

        for segment in obj.segments:
            n_analogsignals += len(getattr(segment, "analogsignals", []))
            n_spiketrains += len(getattr(segment, "spiketrains", []))
            n_events += len(getattr(segment, "events", []))
            n_epochs += len(getattr(segment, "epochs", []))

        summary["n_analogsignals"] = n_analogsignals
        summary["n_spiketrains"] = n_spiketrains
        summary["n_events"] = n_events
        summary["n_epochs"] = n_epochs

    else:
        summary["looks_like_neo_block"] = False

    return summary


def extract_pickle_metadata(file_path):
    """
    input:
        file_path: str

    output:
        pickle metadata dictionary
    """

    metadata = {
        "attempted": True,
        "success": False,
        "error": None,
        "object_summary": None,
    }

    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        metadata["success"] = True
        metadata["object_summary"] = summarize_python_object(obj)

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)