# -*- coding: utf-8 -*-

"""
Extract electrophysiology metadata using Neo.

Neo is the main extractor for many electrophysiology formats.
For NWB files, PyNWB is usually preferred, but Neo can still be used optionally.
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


def get_object_annotations(obj):
    """
    Extract annotations and array_annotations if present.
    """

    output = {
        "annotations": {},
        "array_annotations": {},
    }

    try:
        output["annotations"] = dict(obj.annotations)
    except Exception:
        pass

    try:
        output["array_annotations"] = {
            key: make_json_safe(value)
            for key, value in obj.array_annotations.items()
        }
    except Exception:
        pass

    return output


def summarize_analogsignal(signal, signal_index):
    """
    Summarize one Neo AnalogSignal.
    """

    signal_summary = {
        "signal_index": signal_index,
        "name": getattr(signal, "name", None),
        "description": getattr(signal, "description", None),
        "shape": None,
        "n_channels": None,
        "n_samples": None,
        "sampling_rate_hz": None,
        "units": None,
        "duration_s": None,
        "t_start_s": None,
    }

    try:
        signal_summary["shape"] = list(signal.shape)
        signal_summary["n_samples"] = int(signal.shape[0])

        if len(signal.shape) == 1:
            signal_summary["n_channels"] = 1
        else:
            signal_summary["n_channels"] = int(signal.shape[1])
    except Exception:
        pass

    try:
        signal_summary["sampling_rate_hz"] = float(signal.sampling_rate.rescale("Hz").item())
    except Exception:
        pass

    try:
        signal_summary["units"] = str(signal.units)
    except Exception:
        pass

    try:
        signal_summary["duration_s"] = float((signal.t_stop - signal.t_start).rescale("s").item())
    except Exception:
        pass

    try:
        signal_summary["t_start_s"] = float(signal.t_start.rescale("s").item())
    except Exception:
        pass

    signal_summary.update(get_object_annotations(signal))

    return make_json_safe(signal_summary)


def summarize_spiketrain(spiketrain, spiketrain_index):
    """
    Summarize one Neo SpikeTrain.
    """

    spiketrain_summary = {
        "spiketrain_index": spiketrain_index,
        "name": getattr(spiketrain, "name", None),
        "description": getattr(spiketrain, "description", None),
        "n_spikes": None,
        "units": None,
        "t_start_s": None,
        "t_stop_s": None,
        "duration_s": None,
        "firing_rate_hz": None,
    }

    try:
        spiketrain_summary["n_spikes"] = int(len(spiketrain))
    except Exception:
        pass

    try:
        spiketrain_summary["units"] = str(spiketrain.units)
    except Exception:
        pass

    try:
        t_start = spiketrain.t_start.rescale("s").item()
        t_stop = spiketrain.t_stop.rescale("s").item()
        duration = t_stop - t_start

        spiketrain_summary["t_start_s"] = float(t_start)
        spiketrain_summary["t_stop_s"] = float(t_stop)
        spiketrain_summary["duration_s"] = float(duration)

        if duration > 0 and spiketrain_summary["n_spikes"] is not None:
            spiketrain_summary["firing_rate_hz"] = float(spiketrain_summary["n_spikes"] / duration)

    except Exception:
        pass

    spiketrain_summary.update(get_object_annotations(spiketrain))

    return make_json_safe(spiketrain_summary)


def summarize_event(event, event_index):
    """
    Summarize one Neo Event.
    """

    event_summary = {
        "event_index": event_index,
        "name": getattr(event, "name", None),
        "description": getattr(event, "description", None),
        "n_events": None,
        "units": None,
        "labels_preview": [],
    }

    try:
        event_summary["n_events"] = int(len(event))
    except Exception:
        pass

    try:
        event_summary["units"] = str(event.units)
    except Exception:
        pass

    try:
        event_summary["labels_preview"] = make_json_safe(list(event.labels[:20]))
    except Exception:
        pass

    event_summary.update(get_object_annotations(event))

    return make_json_safe(event_summary)


def summarize_epoch(epoch, epoch_index):
    """
    Summarize one Neo Epoch.
    """

    epoch_summary = {
        "epoch_index": epoch_index,
        "name": getattr(epoch, "name", None),
        "description": getattr(epoch, "description", None),
        "n_epochs": None,
        "units": None,
        "labels_preview": [],
    }

    try:
        epoch_summary["n_epochs"] = int(len(epoch))
    except Exception:
        pass

    try:
        epoch_summary["units"] = str(epoch.units)
    except Exception:
        pass

    try:
        epoch_summary["labels_preview"] = make_json_safe(list(epoch.labels[:20]))
    except Exception:
        pass

    epoch_summary.update(get_object_annotations(epoch))

    return make_json_safe(epoch_summary)


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

        n_spikes_total = 0
        spike_counts = []

        n_channels_per_segment = []
        sampling_rates = []
        units = []
        durations = []
        signal_names = []
        signal_shapes = []

        spiketrain_names = []
        event_names = []
        epoch_names = []

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
                "spiketrains": [],
                "events": [],
                "epochs": [],
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

            # -----------------------------------------------------
            # Analog signals
            # -----------------------------------------------------
            for signal_index, signal in enumerate(getattr(segment, "analogsignals", [])):

                signal_summary = summarize_analogsignal(signal, signal_index)

                if signal_summary.get("n_channels") is not None:
                    segment_channels += signal_summary["n_channels"]

                if signal_summary.get("sampling_rate_hz") is not None:
                    sampling_rates.append(signal_summary["sampling_rate_hz"])

                if signal_summary.get("units") is not None:
                    units.append(signal_summary["units"])

                if signal_summary.get("duration_s") is not None:
                    durations.append(signal_summary["duration_s"])

                signal_names.append(signal_summary.get("name"))
                signal_shapes.append(signal_summary.get("shape"))

                segment_summary["analogsignals"].append(signal_summary)

            # -----------------------------------------------------
            # Spike trains
            # -----------------------------------------------------
            for spiketrain_index, spiketrain in enumerate(getattr(segment, "spiketrains", [])):

                spiketrain_summary = summarize_spiketrain(spiketrain, spiketrain_index)

                if spiketrain_summary.get("n_spikes") is not None:
                    spike_counts.append(spiketrain_summary["n_spikes"])
                    n_spikes_total += spiketrain_summary["n_spikes"]

                spiketrain_names.append(spiketrain_summary.get("name"))

                segment_summary["spiketrains"].append(spiketrain_summary)

            # -----------------------------------------------------
            # Events
            # -----------------------------------------------------
            for event_index, event in enumerate(getattr(segment, "events", [])):

                event_summary = summarize_event(event, event_index)
                event_names.append(event_summary.get("name"))
                segment_summary["events"].append(event_summary)

            # -----------------------------------------------------
            # Epochs
            # -----------------------------------------------------
            for epoch_index, epoch in enumerate(getattr(segment, "epochs", [])):

                epoch_summary = summarize_epoch(epoch, epoch_index)
                epoch_names.append(epoch_summary.get("name"))
                segment_summary["epochs"].append(epoch_summary)

            n_channels_per_segment.append(segment_channels)
            segment_summaries.append(segment_summary)

        metadata["n_analogsignals"] = n_analogsignals
        metadata["n_spiketrains"] = n_spiketrains
        metadata["n_events"] = n_events
        metadata["n_epochs"] = n_epochs

        metadata["n_spikes_total"] = int(n_spikes_total) if len(spike_counts) > 0 else None
        metadata["min_spikes_per_spiketrain"] = min(spike_counts) if len(spike_counts) > 0 else None
        metadata["max_spikes_per_spiketrain"] = max(spike_counts) if len(spike_counts) > 0 else None
        metadata["mean_spikes_per_spiketrain"] = (
            sum(spike_counts) / len(spike_counts)
            if len(spike_counts) > 0
            else None
        )

        metadata["n_channels_per_segment"] = list(set(n_channels_per_segment))
        metadata["sampling_rates_hz"] = list(set(sampling_rates))
        metadata["units"] = list(set(units))
        metadata["durations_s"] = list(set(durations))
        metadata["signal_names"] = list(set([name for name in signal_names if name is not None]))
        metadata["signal_shapes"] = signal_shapes

        metadata["spiketrain_names"] = list(set([name for name in spiketrain_names if name is not None]))
        metadata["event_names"] = list(set([name for name in event_names if name is not None]))
        metadata["epoch_names"] = list(set([name for name in epoch_names if name is not None]))

        metadata["has_analogsignals"] = n_analogsignals > 0
        metadata["has_spiketrains"] = n_spiketrains > 0
        metadata["has_spike_metadata"] = len(spike_counts) > 0
        metadata["has_events"] = n_events > 0
        metadata["has_epochs"] = n_epochs > 0

        metadata["segments"] = segment_summaries

    except Exception as error:
        metadata["error"] = str(error)

    return make_json_safe(metadata)