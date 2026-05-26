# -*- coding: utf-8 -*-

"""
Extract electrophysiology metadata using Neo.

Neo is the main extractor for electrophysiology formats.
"""

from utils import make_json_safe


def read_with_neo(file_path):
    """
    Read one file with Neo.

    Some formats support lazy loading and some do not.
    Therefore, the function first tries lazy=True and then lazy=False.
    """

    from neo import get_io

    io = get_io(file_path)

    try:
        data = io.read(lazy=True)[0]
        loading_mode = "lazy"
    except Exception:
        data = io.read(lazy=False)[0]
        loading_mode = "non_lazy"

    return io, data, loading_mode


def get_signal_annotations(signal):
    """
    Extract signal annotations if present.
    """

    annotations = {}

    try:
        annotations["annotations"] = dict(signal.annotations)
    except Exception:
        annotations["annotations"] = {}

    try:
        annotations["array_annotations"] = {
            key: list(value)
            for key, value in signal.array_annotations.items()
        }
    except Exception:
        annotations["array_annotations"] = {}

    return annotations


def extract_neo_metadata(file_path):
    """
    input:
        file_path: str

    output:
        Neo extraction dictionary
    """

    metadata = {
        "attempted": True,
        "success": False,
        "error": None,
        "neo_io_class": None,
        "loading_mode": None,
    }

    try:
        io, block, loading_mode = read_with_neo(file_path)

        metadata["success"] = True
        metadata["neo_io_class"] = io.__class__.__name__
        metadata["loading_mode"] = loading_mode

        metadata["block_name"] = getattr(block, "name", None)
        metadata["block_description"] = getattr(block, "description", None)

        try:
            metadata["block_annotations"] = dict(block.annotations)
        except Exception:
            metadata["block_annotations"] = {}

        metadata["n_segments"] = len(block.segments)

        n_analogsignals = 0
        n_spiketrains = 0
        n_events = 0
        n_epochs = 0

        n_channels_per_segment = []
        sampling_rates = []
        units = []
        durations = []
        signal_names = []
        signal_shapes = []

        segment_summaries = []

        for segment_index, segment in enumerate(block.segments):

            segment_summary = {
                "segment_index": segment_index,
                "segment_name": getattr(segment, "name", None),
                "n_analogsignals": len(getattr(segment, "analogsignals", [])),
                "n_spiketrains": len(getattr(segment, "spiketrains", [])),
                "n_events": len(getattr(segment, "events", [])),
                "n_epochs": len(getattr(segment, "epochs", [])),
                "analogsignals": [],
            }

            try:
                segment_summary["segment_annotations"] = dict(segment.annotations)
            except Exception:
                segment_summary["segment_annotations"] = {}

            n_analogsignals += segment_summary["n_analogsignals"]
            n_spiketrains += segment_summary["n_spiketrains"]
            n_events += segment_summary["n_events"]
            n_epochs += segment_summary["n_epochs"]

            segment_channels = 0

            for signal_index, signal in enumerate(getattr(segment, "analogsignals", [])):

                signal_summary = {
                    "signal_index": signal_index,
                    "name": getattr(signal, "name", None),
                    "description": getattr(signal, "description", None),
                    "shape": list(signal.shape),
                    "sampling_rate_hz": None,
                    "units": None,
                    "duration_s": None,
                    "n_channels": None,
                }

                try:
                    if len(signal.shape) == 1:
                        n_channels = 1
                    else:
                        n_channels = signal.shape[1]

                    signal_summary["n_channels"] = n_channels
                    segment_channels += n_channels
                except Exception:
                    pass

                try:
                    sampling_rate = signal.sampling_rate.rescale("Hz").item()
                    signal_summary["sampling_rate_hz"] = sampling_rate
                    sampling_rates.append(sampling_rate)
                except Exception:
                    pass

                try:
                    unit = signal.units.dimensionality.string
                    signal_summary["units"] = unit
                    units.append(unit)
                except Exception:
                    pass

                try:
                    duration = (signal.t_stop - signal.t_start).rescale("s").item()
                    signal_summary["duration_s"] = duration
                    durations.append(duration)
                except Exception:
                    pass

                signal_summary.update(get_signal_annotations(signal))

                signal_names.append(signal_summary["name"])
                signal_shapes.append(signal_summary["shape"])

                segment_summary["analogsignals"].append(signal_summary)

            n_channels_per_segment.append(segment_channels)
            segment_summaries.append(segment_summary)

        metadata["n_analogsignals"] = n_analogsignals
        metadata["n_spiketrains"] = n_spiketrains
        metadata["n_events"] = n_events
        metadata["n_epochs"] = n_epochs

        metadata["n_channels_per_segment"] = list(set(n_channels_per_segment))
        metadata["sampling_rates_hz"] = list(set(sampling_rates))
        metadata["units"] = list(set(units))
        metadata["durations_s"] = list(set(durations))
        metadata["signal_names"] = list(set([name for name in signal_names if name is not None]))
        metadata["signal_shapes"] = signal_shapes

        metadata["has_analogsignals"] = n_analogsignals > 0
        metadata["has_spiketrains"] = n_spiketrains > 0
        metadata["has_events"] = n_events > 0
        metadata["has_epochs"] = n_epochs > 0

        metadata["segments"] = segment_summaries

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)