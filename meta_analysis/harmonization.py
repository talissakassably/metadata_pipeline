# -*- coding: utf-8 -*-
"""Dataset-specific harmonization functions.

Each extractor produces a different JSON structure. This module converts each
JSON into a common session/file-level row format used by the ML pipeline.
"""

from pathlib import Path

from .utils import (
    as_bool,
    has_any_text,
    infer_dataset_short_name,
    infer_source_type,
    list_to_text,
    normalize_text,
    sum_dict_values,
    to_number,
)

def harmonize_openfield(dataset):
    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for session in dataset.get("sessions", []):
        summary = session.get("summary") or {}
        file_counts = session.get("file_counts") or {}
        neuralynx = session.get("neuralynx_metadata") or {}
        axona = session.get("axona_metadata") or {}
        mclust = session.get("mclust_metadata") or {}
        optitrack = session.get("optitrack_metadata") or {}

        recording_system = session.get("recording_system") or ""

        lfp_sampling_rates = []
        if neuralynx.get("lfp_sampling_rates_hz"):
            lfp_sampling_rates.extend(neuralynx.get("lfp_sampling_rates_hz") or [])
        if axona.get("eeg_sample_rates_hz"):
            lfp_sampling_rates.extend(axona.get("eeg_sample_rates_hz") or [])
        if axona.get("eeg_sampling_rates_hz"):
            lfp_sampling_rates.extend(axona.get("eeg_sampling_rates_hz") or [])

        n_position_samples = (
            axona.get("n_position_samples")
            or neuralynx.get("n_position_records")
            or optitrack.get("n_timestamps")
            or optitrack.get("n_position_samples")
        )

        # The openfield extractor has existed in two naming variants.
        # Support both to avoid silently zeroing useful metadata.
        n_units_best = (
            summary.get("n_units_best_available")
            or summary.get("n_units")
            or mclust.get("sorted_units_mclust")
            or mclust.get("n_units")
        )
        n_spikes_best = (
            summary.get("sorted_unit_spikes_total_mclust")
            or mclust.get("sorted_unit_spikes_total_mclust")
            or mclust.get("n_spikes_total")
            or summary.get("n_spikes_total")
        )
        raw_spike_events = (
            summary.get("raw_spike_events_total")
            or neuralynx.get("raw_spike_events_total")
            or neuralynx.get("n_spike_events_total")
            or axona.get("n_spikes_total")
            or summary.get("n_spikes_total")
        )
        sorted_units = (
            summary.get("sorted_units_mclust")
            or mclust.get("sorted_units_mclust")
            or mclust.get("n_units")
        )
        sorted_unit_spikes = (
            summary.get("sorted_unit_spikes_total_mclust")
            or mclust.get("sorted_unit_spikes_total_mclust")
            or mclust.get("n_spikes_total")
        )

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "openfield",
            "source_format": "Axona/Neuralynx/MClust/OptiTrack",
            "session_id": session.get("session_id"),
            "subject_id": session.get("subject_id"),
            "session_date": session.get("session_date"),
            "recording_system": recording_system,
            "behavioral_context": "open field / navigation",
            "extraction_success": True,

            "n_files": file_counts.get("n_files_total"),
            "file_extensions": list_to_text(file_counts.get("extensions")),

            "recording_duration_s": summary.get("recording_duration_s"),
            "n_units": n_units_best,
            "n_spiketrains": n_units_best,
            "n_spikes_total": n_spikes_best,
            "raw_spike_events_total": raw_spike_events,
            "sorted_units_mclust": sorted_units,
            "sorted_unit_spikes_total_mclust": sorted_unit_spikes,
            "n_lfp_channels": summary.get("n_lfp_channels"),
            "n_electrodes": None,
            "n_trials": 0,
            "n_event_objects": None,
            "n_event_times_total": neuralynx.get("n_event_records"),
            "n_position_samples": n_position_samples,

            "sampling_rates_hz": list_to_text(lfp_sampling_rates),
            "brain_regions": "",
            "unit_source": summary.get("n_units_source"),

            "has_subject_metadata": has_any_text(session.get("subject_id")),
            "has_session_metadata": has_any_text(session.get("session_id")),
            "has_session_date": has_any_text(session.get("session_date")),
            "has_recording_duration": summary.get("recording_duration_s") is not None,
            "has_spike_metadata": as_bool(summary.get("has_raw_spike_metadata")) or as_bool(summary.get("has_sorted_unit_metadata")) or as_bool(summary.get("has_spike_metadata")) or to_number(raw_spike_events, 0) > 0,
            "has_unit_metadata": as_bool(summary.get("has_sorted_unit_metadata")) or as_bool(summary.get("has_unit_metadata")) or to_number(n_units_best, 0) > 0,
            "has_sorted_unit_metadata": as_bool(summary.get("has_sorted_unit_metadata")) or as_bool(summary.get("has_unit_metadata")) or to_number(n_units_best, 0) > 0,
            "has_lfp_metadata": as_bool(summary.get("has_lfp_metadata")),
            "has_trial_metadata": False,
            "has_event_metadata": neuralynx.get("events_success") is True,
            "has_position_metadata": as_bool(summary.get("has_position_metadata")),
            "has_sampling_rate_metadata": len(lfp_sampling_rates) > 0,
            "has_brain_region_metadata": False,
            "has_standardized_format": False,
            "has_openminds_candidate_metadata": True,

            "error": "",
        }

        rows.append(row)

    return rows


def harmonize_nwb(dataset):
    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for item in dataset.get("files", []):
        file_meta = item.get("file_metadata") or {}
        nwb = item.get("nwb_extraction") or {}

        subject = nwb.get("subject") or {}

        success = nwb.get("success") is True

        electrode_locations = nwb.get("electrode_locations") or []
        processing_modules = nwb.get("processing_modules") or []
        intervals = nwb.get("intervals") or []
        unit_columns = nwb.get("unit_columns") or []
        electrode_columns = nwb.get("electrode_columns") or []

        session_description = nwb.get("session_description") or ""
        behavioral_context = session_description
        if not behavioral_context:
            behavioral_context = "NWB electrophysiology"

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "nwb",
            "source_format": "NWB",
            "session_id": nwb.get("identifier") or file_meta.get("session_id") or Path(file_meta.get("file_name", "")).stem,
            "subject_id": subject.get("subject_id") or file_meta.get("subject_id") or file_meta.get("animal_id"),
            "session_date": str(nwb.get("session_start_time") or "")[:10],
            "recording_system": list_to_text(nwb.get("devices")),
            "behavioral_context": behavioral_context,
            "extraction_success": success,

            "n_files": 1,
            "file_extensions": file_meta.get("file_extension"),

            "recording_duration_s": None,
            "n_units": nwb.get("n_units"),
            "n_spiketrains": nwb.get("n_units"),
            "n_spikes_total": None,
            "raw_spike_events_total": None,
            "sorted_units_mclust": None,
            "sorted_unit_spikes_total_mclust": None,
            "n_lfp_channels": None,
            "n_electrodes": nwb.get("n_electrodes"),
            "n_trials": nwb.get("n_trials"),
            "n_event_objects": len(intervals),
            "n_event_times_total": None,
            "n_position_samples": None,

            "sampling_rates_hz": "unit sampling_rate column" if "sampling_rate" in unit_columns else "",
            "brain_regions": list_to_text(electrode_locations),
            "unit_source": "NWB units table",

            "has_subject_metadata": has_any_text(subject.get("subject_id")),
            "has_session_metadata": has_any_text(nwb.get("identifier")),
            "has_session_date": has_any_text(nwb.get("session_start_time")),
            "has_recording_duration": False,
            "has_spike_metadata": to_number(nwb.get("n_units"), 0) > 0,
            "has_unit_metadata": to_number(nwb.get("n_units"), 0) > 0,
            "has_sorted_unit_metadata": to_number(nwb.get("n_units"), 0) > 0,
            "has_lfp_metadata": False,
            "has_trial_metadata": to_number(nwb.get("n_trials"), 0) > 0,
            "has_event_metadata": len(intervals) > 0,
            "has_position_metadata": "behavior" in processing_modules,
            "has_sampling_rate_metadata": "sampling_rate" in unit_columns,
            "has_brain_region_metadata": len(electrode_locations) > 0,
            "has_standardized_format": True,
            "has_openminds_candidate_metadata": True,

            "species": subject.get("species"),
            "sex": subject.get("sex"),
            "institution": nwb.get("institution"),
            "experimenter": list_to_text(nwb.get("experimenter")),
            "electrode_columns": list_to_text(electrode_columns),
            "unit_columns": list_to_text(unit_columns),
            "error": "" if success else normalize_text(nwb.get("error")),
        }

        rows.append(row)

    return rows


def harmonize_legacy_touchscreen(dataset):
    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for session in dataset.get("sessions", []):
        summary = session.get("summary") or {}

        event_counts = summary.get("event_counts") or {}
        event_total = sum_dict_values(event_counts)

        unit_categories = summary.get("unit_categories") or {}
        lfp_categories = summary.get("lfp_categories") or {}
        trial_categories = summary.get("trial_categories") or {}

        regions = []
        for key in ["area", "recording_group", "tetrode_area"]:
            if key in unit_categories:
                regions.extend(unit_categories[key].get("values", []))
        if "recording_group" in lfp_categories:
            regions.extend(lfp_categories["recording_group"].get("values", []))
        regions = sorted(set(str(x) for x in regions if str(x).strip() and str(x) != "no_data"))

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "legacy_touchscreen",
            "source_format": "legacy Neo/NIX",
            "session_id": session.get("session_id"),
            "subject_id": session.get("animal_id"),
            "session_date": session.get("session_date"),
            "recording_system": "legacy Neo/NIX",
            "behavioral_context": "touchscreen / tactile-visual task",
            "extraction_success": session.get("success") is True,

            "n_files": None,
            "file_extensions": ".pkl; .nio",

            "recording_duration_s": None,
            "n_units": summary.get("n_units_from_dataframe"),
            "n_spiketrains": summary.get("n_spiketrains"),
            "n_spikes_total": summary.get("n_spikes_total"),
            "raw_spike_events_total": None,
            "sorted_units_mclust": None,
            "sorted_unit_spikes_total_mclust": None,
            "n_lfp_channels": summary.get("n_lfp_channels_from_dataframe"),
            "n_electrodes": None,
            "n_trials": summary.get("n_trials_from_dataframe"),
            "n_event_objects": summary.get("n_events"),
            "n_event_times_total": event_total,
            "n_position_samples": None,

            "sampling_rates_hz": "",
            "brain_regions": list_to_text(regions),
            "unit_source": "Neo spiketrains / unit dataframe",

            "has_subject_metadata": has_any_text(session.get("animal_id")),
            "has_session_metadata": has_any_text(session.get("session_id")),
            "has_session_date": has_any_text(session.get("session_date")),
            "has_recording_duration": False,
            "has_spike_metadata": as_bool(summary.get("has_spike_metadata")),
            "has_unit_metadata": to_number(summary.get("n_units_from_dataframe"), 0) > 0,
            "has_sorted_unit_metadata": to_number(summary.get("n_units_from_dataframe"), 0) > 0,
            "has_lfp_metadata": as_bool(summary.get("has_lfp_metadata")),
            "has_trial_metadata": as_bool(summary.get("has_trial_metadata")),
            "has_event_metadata": as_bool(summary.get("has_event_metadata")),
            "has_position_metadata": (
                "has_nosetracking" in trial_categories or "has_whiskertracking" in trial_categories
            ),
            "has_sampling_rate_metadata": False,
            "has_brain_region_metadata": len(regions) > 0,
            "has_standardized_format": False,
            "has_openminds_candidate_metadata": True,

            "error": "" if session.get("success") is True else normalize_text(session.get("error")),
        }

        rows.append(row)

    return rows


def harmonize_regular_pickle_or_touchandsee(dataset):
    """
    Handles the regular extraction JSON for TouchAndSee.

    Important:
        In the currently provided file, pickle/Neo extraction failed for all .pkl
        files because of old Neo pickle compatibility. We still include these
        sessions in the harmonized table as metadata-limited rows, using file
        metadata only. Once the pickle compatibility is fixed and this JSON is
        regenerated, this function can be extended to read object_summary.
    """

    rows = []
    dataset_name = dataset.get("dataset_name")
    dataset_short_name = infer_dataset_short_name(dataset_name)

    for item in dataset.get("files", []):
        file_meta = item.get("file_metadata") or {}
        pickle_info = item.get("pickle_extraction") or {}
        neo_info = item.get("neo_extraction") or {}
        touchandsee_info = item.get("touchandsee_extraction") or {}

        ext = file_meta.get("file_extension")

        # Keep session-like data files as rows. Ignore code/descriptor files.
        if ext not in [".pkl", ".nwb"]:
            continue

        pickle_success = pickle_info.get("success") is True
        neo_success = neo_info.get("success") is True
        touchandsee_success = touchandsee_info.get("success") is True
        success = pickle_success or neo_success or touchandsee_success

        object_summary = (
            touchandsee_info.get("object_summary")
            or pickle_info.get("object_summary")
            or neo_info.get("object_summary")
            or {}
        )

        row = {
            "dataset_short_name": dataset_short_name,
            "dataset_name": dataset_name,
            "source_type": "touchandsee_pickle",
            "source_format": "Neo pickle",
            "session_id": file_meta.get("session_id") or Path(file_meta.get("file_name", "")).stem,
            "subject_id": file_meta.get("subject_id") or file_meta.get("animal_id"),
            "session_date": file_meta.get("session_date"),
            "recording_system": "Neo pickle",
            "behavioral_context": file_meta.get("task_label") or "TouchAndSee",
            "extraction_success": success,

            "n_files": 1,
            "file_extensions": ext,

            # If object_summary exists in future, these can be populated.
            "recording_duration_s": object_summary.get("duration_s"),
            "n_units": object_summary.get("n_units"),
            "n_spiketrains": object_summary.get("n_spiketrains"),
            "n_spikes_total": object_summary.get("n_spikes_total"),
            "raw_spike_events_total": None,
            "sorted_units_mclust": None,
            "sorted_unit_spikes_total_mclust": None,
            "n_lfp_channels": object_summary.get("n_lfp_channels") or object_summary.get("n_analogsignals"),
            "n_electrodes": object_summary.get("n_electrodes"),
            "n_trials": object_summary.get("n_trials") or object_summary.get("n_segments"),
            "n_event_objects": object_summary.get("n_events"),
            "n_event_times_total": object_summary.get("n_event_times_total"),
            "n_position_samples": object_summary.get("n_position_samples"),

            "sampling_rates_hz": list_to_text(object_summary.get("sampling_rates_hz")),
            "brain_regions": list_to_text(object_summary.get("brain_regions")),
            "unit_source": "Neo pickle" if success else "not loaded",

            "has_subject_metadata": has_any_text(file_meta.get("subject_id") or file_meta.get("animal_id")),
            "has_session_metadata": has_any_text(file_meta.get("session_id")),
            "has_session_date": has_any_text(file_meta.get("session_date")),
            "has_recording_duration": object_summary.get("duration_s") is not None,
            "has_spike_metadata": to_number(object_summary.get("n_spikes_total"), 0) > 0,
            "has_unit_metadata": to_number(object_summary.get("n_units"), 0) > 0 or to_number(object_summary.get("n_spiketrains"), 0) > 0,
            "has_sorted_unit_metadata": to_number(object_summary.get("n_units"), 0) > 0,
            "has_lfp_metadata": to_number(object_summary.get("n_lfp_channels") or object_summary.get("n_analogsignals"), 0) > 0,
            "has_trial_metadata": to_number(object_summary.get("n_trials") or object_summary.get("n_segments"), 0) > 0,
            "has_event_metadata": to_number(object_summary.get("n_events"), 0) > 0,
            "has_position_metadata": to_number(object_summary.get("n_position_samples"), 0) > 0,
            "has_sampling_rate_metadata": has_any_text(object_summary.get("sampling_rates_hz")),
            "has_brain_region_metadata": has_any_text(object_summary.get("brain_regions")),
            "has_standardized_format": False,
            "has_openminds_candidate_metadata": True,

            "file_size_mb": file_meta.get("file_size_mb"),
            "error": "" if success else normalize_text(pickle_info.get("error") or neo_info.get("error")),
        }

        rows.append(row)

    return rows


def harmonize_dataset(dataset):
    source_type = infer_source_type(dataset)

    if source_type == "openfield":
        return harmonize_openfield(dataset)

    if source_type == "nwb":
        return harmonize_nwb(dataset)

    if source_type == "legacy_touchscreen":
        return harmonize_legacy_touchscreen(dataset)

    if source_type == "regular_pickle_or_touchandsee":
        return harmonize_regular_pickle_or_touchandsee(dataset)

    raise ValueError("Could not infer dataset source type for: {}".format(dataset.get("dataset_name")))
