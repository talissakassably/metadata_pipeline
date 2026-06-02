# -*- coding: utf-8 -*-

"""
Dataset-specific metadata extractor for open-field CA1 dataset.

Dataset:
    d-40faae41-7e72-4c3c-9abf-91dea149158d

This dataset contains:
    - Neuralynx sessions: .ncs, .ntt, .nvt, .nev, .t64, .clusters, .fd
    - Axona sessions: .set, .pos, .1-.8, .eeg/.eeg2/.eeg3/.eeg4
    - MClust cluster-cut timestamp files: .t/.t32/.t64
    - OptiTrack CSV files

This extractor works at SESSION-FOLDER level:
    data/<subject_id>/<session_date>/
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np


# Compatibility patch for dataset scripts with NumPy 2.x
if not hasattr(np, "string_"):
    np.string_ = np.bytes_


CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))


def make_json_safe(value):
    """
    Convert values to JSON-safe objects.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime,)):
        return str(value)

    try:
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

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    try:
        return str(value)
    except Exception:
        return None


def summarize_numeric_list(values):
    """
    Summarize a numeric list.
    """

    clean = []

    for value in values:
        try:
            clean.append(float(value))
        except Exception:
            pass

    if len(clean) == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "sum": None,
            "n": 0,
        }

    return {
        "min": float(min(clean)),
        "max": float(max(clean)),
        "mean": float(sum(clean) / len(clean)),
        "sum": float(sum(clean)),
        "n": int(len(clean)),
    }


def count_files(session_folder):
    """
    Count known file types in one session folder.
    """

    file_counts = {
        "n_files_total": 0,
        "n_ncs_files": 0,
        "n_ntt_files": 0,
        "n_nvt_files": 0,
        "n_nev_files": 0,
        "n_set_files": 0,
        "n_pos_files": 0,
        "n_axona_spike_files": 0,
        "n_eeg_files": 0,
        "n_egf_files": 0,
        "n_t_files": 0,
        "n_t32_files": 0,
        "n_t64_files": 0,
        "n_fd_files": 0,
        "n_clusters_files": 0,
        "n_csv_files": 0,
    }

    extensions = {}

    for file_path in session_folder.iterdir():
        if not file_path.is_file():
            continue

        file_counts["n_files_total"] += 1

        suffix = file_path.suffix.lower()
        extensions[suffix] = extensions.get(suffix, 0) + 1

        if suffix == ".ncs":
            file_counts["n_ncs_files"] += 1
        elif suffix == ".ntt":
            file_counts["n_ntt_files"] += 1
        elif suffix == ".nvt":
            file_counts["n_nvt_files"] += 1
        elif suffix == ".nev":
            file_counts["n_nev_files"] += 1
        elif suffix == ".set":
            file_counts["n_set_files"] += 1
        elif suffix == ".pos":
            file_counts["n_pos_files"] += 1
        elif suffix in [".eeg", ".eeg1", ".eeg2", ".eeg3", ".eeg4"]:
            file_counts["n_eeg_files"] += 1
        elif suffix in [".egf", ".egf1", ".egf2", ".egf3", ".egf4"]:
            file_counts["n_egf_files"] += 1
        elif suffix == ".t":
            file_counts["n_t_files"] += 1
        elif suffix == ".t32":
            file_counts["n_t32_files"] += 1
        elif suffix == ".t64":
            file_counts["n_t64_files"] += 1
        elif suffix == ".fd":
            file_counts["n_fd_files"] += 1
        elif suffix == ".clusters":
            file_counts["n_clusters_files"] += 1
        elif suffix == ".csv":
            file_counts["n_csv_files"] += 1
        elif suffix.startswith(".") and suffix[1:].isdigit():
            file_counts["n_axona_spike_files"] += 1

    file_counts["extensions"] = extensions

    return file_counts


def infer_recording_system(file_counts):
    """
    Infer recording system from files.
    """

    if file_counts["n_ntt_files"] > 0 or file_counts["n_ncs_files"] > 0:
        return "Neuralynx"

    if file_counts["n_set_files"] > 0:
        return "Axona"

    if file_counts["n_csv_files"] > 0:
        return "OptiTrack_only_or_tracking"

    return "unknown"


def find_session_folders(dataset_path):
    """
    Find folders under data/<subject_id>/<session_date>.
    """

    dataset_path = Path(dataset_path)
    data_folder = dataset_path / "data"

    session_folders = []

    if not data_folder.exists():
        return session_folders

    for subject_folder in sorted(data_folder.iterdir()):
        if not subject_folder.is_dir():
            continue

        for session_folder in sorted(subject_folder.iterdir()):
            if not session_folder.is_dir():
                continue

            file_counts = count_files(session_folder)

            if file_counts["n_files_total"] > 0:
                session_folders.append(session_folder)

    return session_folders


def get_subject_and_date(session_folder):
    """
    Parse subject_id and session_date from data/<subject_id>/<session_date>.
    """

    subject_id = session_folder.parent.name
    session_date = session_folder.name
    session_id = subject_id + "_" + session_date

    return subject_id, session_date, session_id


def import_dataset_scripts(dataset_path):
    """
    Import dataset-provided scripts.
    """

    dataset_path = Path(dataset_path)

    if str(dataset_path) not in sys.path:
        sys.path.insert(0, str(dataset_path))

    from scripts import axona
    from scripts import neuralynx
    from scripts import mclust
    from scripts import opticsv

    return axona, neuralynx, mclust, opticsv


def extract_neuralynx_metadata(session_folder, neuralynx):
    """
    Extract metadata from a Neuralynx session.
    """

    output = {
        "attempted": True,
        "success": False,
        "error": None,

        "recording_name": None,
        "recording_time": None,
        "recording_duration_s": None,
        "tetrode_list": [],

        "n_spike_tetrodes": None,
        "n_spike_events_total": None,
        "spike_events_per_tetrode": {},

        "n_lfp_channels": None,
        "lfp_records_per_channel": {},
        "lfp_sampling_rates_hz": [],

        "position_attempted": False,
        "position_success": False,
        "position_error": None,
        "n_position_records": None,

        "events_attempted": False,
        "events_success": False,
        "events_error": None,
        "n_event_records": None,
    }

    try:
        recording = neuralynx.NeuralynxSession(session_folder)

        output["recording_name"] = recording.recording_name
        output["recording_time"] = make_json_safe(recording.recording_time)
        output["recording_duration_s"] = make_json_safe(recording.recording_duration)
        output["tetrode_list"] = make_json_safe(recording.tetrode_list)

        # Spikes
        try:
            spikes = recording.spikes

            if spikes:
                output["n_spike_tetrodes"] = len(spikes)

                spike_counts = {}

                for tetrode_id, tetrode_data in spikes.items():
                    spike_counts[str(tetrode_id)] = len(tetrode_data.get("data", []))

                output["spike_events_per_tetrode"] = spike_counts
                output["n_spike_events_total"] = int(sum(spike_counts.values()))

        except Exception as error:
            output["spikes_error"] = repr(error)

        # LFP / EEG from .ncs
        try:
            eeg = recording.eeg

            if eeg:
                output["n_lfp_channels"] = len(eeg)

                lfp_counts = {}
                sampling_rates = []

                for channel_id, channel_data in eeg.items():
                    records = channel_data.get("data", [])
                    header = channel_data.get("header", {})

                    lfp_counts[str(channel_id)] = len(records)

                    # Sample frequency may be available in data records or header.
                    try:
                        if len(records) > 0:
                            first_record = records[0]
                            sample_freq = first_record["dwSampleFreq"][0]
                            sampling_rates.append(float(sample_freq))
                    except Exception:
                        pass

                    try:
                        if "SamplingFrequency" in header:
                            sampling_rates.append(float(np.ravel(header["SamplingFrequency"])[0]))
                    except Exception:
                        pass

                output["lfp_records_per_channel"] = lfp_counts
                output["lfp_sampling_rates_hz"] = sorted(list(set(sampling_rates)))

        except Exception as error:
            output["lfp_error"] = repr(error)

        # Position can fail because of old dtype/overflow issues. Keep error, do not crash.
        output["position_attempted"] = True
        try:
            position = recording.position
            output["position_success"] = True
            output["n_position_records"] = len(position[1])
        except Exception as error:
            output["position_success"] = False
            output["position_error"] = repr(error)

        # Events may not exist in every folder.
        output["events_attempted"] = True
        try:
            events = recording.events
            output["events_success"] = True
            output["n_event_records"] = len(events[1])
        except Exception as error:
            output["events_success"] = False
            output["events_error"] = repr(error)

        output["success"] = True

    except Exception as error:
        output["success"] = False
        output["error"] = repr(error)
        output["traceback"] = traceback.format_exc()

    return make_json_safe(output)


def extract_axona_metadata(session_folder, axona):
    """
    Extract metadata from an Axona session.
    """

    output = {
        "attempted": True,
        "success": False,
        "error": None,

        "recording_name": None,
        "recording_time": None,
        "recording_duration_s": None,
        "tetrode_list": [],

        "n_spike_channels": None,
        "n_spikes_total": None,
        "spikes_per_channel": {},
        "n_channels_per_spike_file": {},

        "position_success": False,
        "position_error": None,
        "position_sample_rate_hz": None,
        "n_position_samples": None,
        "position_coordinates_shape": None,

        "n_eeg_channels": None,
        "eeg_sample_rates_hz": [],
        "eeg_signal_shapes": {},
    }

    try:
        set_files = sorted(session_folder.glob("*.set"))

        if len(set_files) == 0:
            output["error"] = "No .set file found."
            return output

        basepath = set_files[0].with_suffix("")
        recording = axona.AxonaRecording(basepath)

        output["recording_name"] = basepath.name
        output["recording_time"] = make_json_safe(recording.recording_time)
        output["recording_duration_s"] = make_json_safe(recording.recording_duration)
        output["tetrode_list"] = make_json_safe(recording.tetrode_list)

        # Spikes
        try:
            spikes = recording.spikes

            output["n_spike_channels"] = len(spikes)

            spikes_per_channel = {}
            n_channels_per_spike_file = {}

            for channel_id, spike_data in spikes.items():
                spikes_per_channel[str(channel_id)] = int(spike_data.n_spikes)
                n_channels_per_spike_file[str(channel_id)] = int(spike_data.n_channels)

            output["spikes_per_channel"] = spikes_per_channel
            output["n_channels_per_spike_file"] = n_channels_per_spike_file
            output["n_spikes_total"] = int(sum(spikes_per_channel.values()))

        except Exception as error:
            output["spikes_error"] = repr(error)

        # Position
        try:
            position = recording.position

            if position is not None:
                output["position_success"] = True
                output["position_sample_rate_hz"] = float(position.sample_rate)
                output["n_position_samples"] = int(len(position.timestamps))
                output["position_coordinates_shape"] = list(position.coordinates.shape)

        except Exception as error:
            output["position_success"] = False
            output["position_error"] = repr(error)

        # EEG
        try:
            eeg = recording.eeg

            output["n_eeg_channels"] = len(eeg)

            sample_rates = []
            signal_shapes = {}

            for channel_id, eeg_data in eeg.items():
                sample_rates.append(float(eeg_data.sample_rate))
                signal_shapes[str(channel_id)] = list(eeg_data.signal.shape)

            output["eeg_sample_rates_hz"] = sorted(list(set(sample_rates)))
            output["eeg_signal_shapes"] = signal_shapes

        except Exception as error:
            output["eeg_error"] = repr(error)

        output["success"] = True

    except Exception as error:
        output["success"] = False
        output["error"] = repr(error)
        output["traceback"] = traceback.format_exc()

    return make_json_safe(output)


def extract_mclust_metadata(session_folder, mclust):
    """
    Extract metadata from MClust .t/.t32/.t64 files.
    """

    output = {
        "attempted": True,
        "success": False,
        "error": None,

        "n_tetrodes_with_units": None,
        "n_units": None,
        "n_spikes_total": None,
        "units_per_tetrode": {},
        "spikes_per_unit": {},
        "unit_header_keys": [],
    }

    try:
        cuts = mclust.load_all_cuts(str(session_folder))

        if not cuts:
            output["success"] = True
            output["n_tetrodes_with_units"] = 0
            output["n_units"] = 0
            output["n_spikes_total"] = 0
            return output

        output["n_tetrodes_with_units"] = len(cuts)

        units_per_tetrode = {}
        spikes_per_unit = {}
        total_units = 0
        total_spikes = 0
        header_keys = set()

        for tetrode_id, units in cuts.items():
            units_per_tetrode[str(tetrode_id)] = len(units)

            for unit_id, unit_data in units.items():
                header, timestamps = unit_data

                unit_key = str(tetrode_id) + "_" + str(unit_id)

                n_spikes = len(timestamps)
                spikes_per_unit[unit_key] = int(n_spikes)

                total_units += 1
                total_spikes += int(n_spikes)

                for key in header.keys():
                    header_keys.add(str(key))

        output["units_per_tetrode"] = units_per_tetrode
        output["spikes_per_unit"] = spikes_per_unit
        output["n_units"] = int(total_units)
        output["n_spikes_total"] = int(total_spikes)
        output["unit_header_keys"] = sorted(list(header_keys))

        output["success"] = True

    except Exception as error:
        output["success"] = False
        output["error"] = repr(error)
        output["traceback"] = traceback.format_exc()

    return make_json_safe(output)


def extract_optitrack_metadata(session_folder, opticsv):
    """
    Extract OptiTrack CSV metadata if CSV exists.
    """

    output = {
        "attempted": True,
        "success": False,
        "error": None,

        "n_csv_files": 0,
        "csv_files": [],

        "tracking_name": None,
        "recording_time": None,
        "sample_rate_hz": None,
        "meta_keys": [],
        "rigid_body_names": [],
        "n_timestamps": None,
    }

    csv_files = sorted(session_folder.glob("*.csv"))
    output["n_csv_files"] = len(csv_files)
    output["csv_files"] = [file.name for file in csv_files]

    if len(csv_files) == 0:
        output["success"] = True
        return output

    try:
        opti = opticsv.OptiCSV(csv_files[0])

        output["tracking_name"] = opti.tracking_name
        output["recording_time"] = make_json_safe(opti.recording_time)
        output["sample_rate_hz"] = float(opti.sample_rate)
        output["meta_keys"] = list(opti.meta.keys())

        try:
            output["rigid_body_names"] = list(opti.data["rigid_bodies"].keys())
            output["n_timestamps"] = int(len(opti.data["t"]))
        except Exception as error:
            output["data_error"] = repr(error)

        output["success"] = True

    except Exception as error:
        output["success"] = False
        output["error"] = repr(error)
        output["traceback"] = traceback.format_exc()

    return make_json_safe(output)


def extract_one_session(session_folder, dataset_path, scripts):
    """
    Extract one session folder.
    """

    axona, neuralynx, mclust, opticsv = scripts

    subject_id, session_date, session_id = get_subject_and_date(session_folder)
    file_counts = count_files(session_folder)
    recording_system = infer_recording_system(file_counts)

    session_metadata = {
        "session_id": session_id,
        "subject_id": subject_id,
        "session_date": session_date,
        "session_folder": str(session_folder),
        "relative_session_folder": str(session_folder.relative_to(dataset_path)),
        "recording_system": recording_system,
        "file_counts": file_counts,

        "neuralynx_metadata": None,
        "axona_metadata": None,
        "mclust_metadata": None,
        "optitrack_metadata": None,

        "summary": None,
    }

    if recording_system == "Neuralynx":
        print("Extracting Neuralynx session:", session_id)
        session_metadata["neuralynx_metadata"] = extract_neuralynx_metadata(
            session_folder,
            neuralynx,
        )

    if recording_system == "Axona":
        print("Extracting Axona session:", session_id)
        session_metadata["axona_metadata"] = extract_axona_metadata(
            session_folder,
            axona,
        )

    # MClust can exist for Neuralynx sessions.
    if (
        file_counts["n_t_files"] > 0
        or file_counts["n_t32_files"] > 0
        or file_counts["n_t64_files"] > 0
    ):
        session_metadata["mclust_metadata"] = extract_mclust_metadata(
            session_folder,
            mclust,
        )

    # OptiTrack CSV can exist independently.
    if file_counts["n_csv_files"] > 0:
        session_metadata["optitrack_metadata"] = extract_optitrack_metadata(
            session_folder,
            opticsv,
        )

    neuralynx_metadata = session_metadata.get("neuralynx_metadata") or {}
    axona_metadata = session_metadata.get("axona_metadata") or {}
    mclust_metadata = session_metadata.get("mclust_metadata") or {}
    optitrack_metadata = session_metadata.get("optitrack_metadata") or {}

    n_spikes_total_candidates = [
        neuralynx_metadata.get("n_spike_events_total"),
        axona_metadata.get("n_spikes_total"),
        mclust_metadata.get("n_spikes_total"),
    ]

    n_spikes_total = None

    for value in n_spikes_total_candidates:
        if isinstance(value, int):
            n_spikes_total = value
            break

    n_units = mclust_metadata.get("n_units")

    if n_units is None:
        n_units = axona_metadata.get("n_spike_channels")

    n_lfp_channels = neuralynx_metadata.get("n_lfp_channels")

    if n_lfp_channels is None:
        n_lfp_channels = axona_metadata.get("n_eeg_channels")

    recording_time = (
        neuralynx_metadata.get("recording_time")
        or axona_metadata.get("recording_time")
        or optitrack_metadata.get("recording_time")
    )

    recording_duration_s = (
        neuralynx_metadata.get("recording_duration_s")
        or axona_metadata.get("recording_duration_s")
    )

    session_metadata["summary"] = {
        "recording_time": recording_time,
        "recording_duration_s": recording_duration_s,

        "n_units": n_units,
        "n_spikes_total": n_spikes_total,
        "n_lfp_channels": n_lfp_channels,

        "has_spike_metadata": n_spikes_total is not None,
        "has_unit_metadata": n_units is not None,
        "has_lfp_metadata": n_lfp_channels is not None,
        "has_position_metadata": (
            neuralynx_metadata.get("position_success") is True
            or axona_metadata.get("position_success") is True
            or optitrack_metadata.get("success") is True and optitrack_metadata.get("n_timestamps") is not None
        ),
        "has_optitrack_metadata": optitrack_metadata.get("success") is True and optitrack_metadata.get("n_csv_files", 0) > 0,
        "has_mclust_metadata": mclust_metadata.get("success") is True and mclust_metadata.get("n_units", 0) > 0,
    }

    return make_json_safe(session_metadata)


def build_dataset_summary(dataset_metadata):
    """
    Build dataset-level summary.
    """

    sessions = dataset_metadata["sessions"]

    subject_ids = sorted(list(set([session["subject_id"] for session in sessions])))
    recording_systems = sorted(list(set([session["recording_system"] for session in sessions])))

    n_spikes_values = []
    n_units_values = []
    n_lfp_values = []

    for session in sessions:
        summary = session.get("summary") or {}

        if isinstance(summary.get("n_spikes_total"), int):
            n_spikes_values.append(summary.get("n_spikes_total"))

        if isinstance(summary.get("n_units"), int):
            n_units_values.append(summary.get("n_units"))

        if isinstance(summary.get("n_lfp_channels"), int):
            n_lfp_values.append(summary.get("n_lfp_channels"))

    return make_json_safe({
        "n_sessions": len(sessions),
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "recording_systems": recording_systems,

        "n_neuralynx_sessions": sum(1 for s in sessions if s["recording_system"] == "Neuralynx"),
        "n_axona_sessions": sum(1 for s in sessions if s["recording_system"] == "Axona"),
        "n_unknown_sessions": sum(1 for s in sessions if s["recording_system"] == "unknown"),

        "total_spikes_across_sessions": int(sum(n_spikes_values)) if n_spikes_values else None,
        "min_spikes_per_session": int(min(n_spikes_values)) if n_spikes_values else None,
        "max_spikes_per_session": int(max(n_spikes_values)) if n_spikes_values else None,

        "total_units_across_sessions": int(sum(n_units_values)) if n_units_values else None,
        "min_units_per_session": int(min(n_units_values)) if n_units_values else None,
        "max_units_per_session": int(max(n_units_values)) if n_units_values else None,

        "min_lfp_channels_per_session": int(min(n_lfp_values)) if n_lfp_values else None,
        "max_lfp_channels_per_session": int(max(n_lfp_values)) if n_lfp_values else None,
    })


def extract_openfield_dataset(dataset_path, output_folder=None):
    """
    Extract all openfield sessions.
    """

    dataset_path = Path(dataset_path)

    if output_folder is None:
        output_folder = Path(os.getcwd()) / "outputs" / "extracted_metadata"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    scripts = import_dataset_scripts(dataset_path)

    session_folders = find_session_folders(dataset_path)

    dataset_metadata = {
        "dataset_name": dataset_path.name,
        "dataset_folder": str(dataset_path),
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extractor": "openfield_dataset_specific_extractor",
        "n_session_folders_found": len(session_folders),
        "sessions": [],
        "dataset_summary": None,
    }

    print("Session folders found:", len(session_folders))

    for session_folder in session_folders:
        session_metadata = extract_one_session(
            session_folder=session_folder,
            dataset_path=dataset_path,
            scripts=scripts,
        )

        dataset_metadata["sessions"].append(session_metadata)

    dataset_metadata["dataset_summary"] = build_dataset_summary(dataset_metadata)

    output_json = output_folder / (dataset_path.name + "_openfield_metadata.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(dataset_metadata), f, indent=4)

    print("Openfield extraction finished")
    print("Output file:", output_json)

    return dataset_metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract metadata from open-field CA1 dataset"
    )

    parser.add_argument(
        "dataset_path",
        help="Path to d-40faae41-7e72-4c3c-9abf-91dea149158d"
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        help="Output folder"
    )

    args = parser.parse_args()

    extract_openfield_dataset(
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
    )