# -*- coding: utf-8 -*-
"""
Inspect whether the input files are suitable for the perfect case study.

Run:

python case_studies\inspect_perfect_case_study_inputs.py ^
  --nwb_json outputs\extracted_metadata\d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json ^
  --touchandsee_json outputs\extracted_metadata\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json ^
  --legacy_json outputs\extracted_metadata\legacy_touchscreen_metadata.json ^
  --openfield_json outputs\extracted_metadata\d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json
"""

from pathlib import Path
import argparse
import json


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_nwb(path):
    d = load(path)
    files = d.get("files", [])
    ok = [x for x in files if (x.get("nwb_extraction", {}) or {}).get("success")]
    regions = set()
    contexts = set()
    for x in ok:
        nwb = x.get("nwb_extraction", {}) or {}
        for r in nwb.get("electrode_locations", []) or []:
            regions.add(str(r))
        contexts.add(str(nwb.get("session_description")))
    print("\nNWB")
    print(" files:", len(files), "success:", len(ok))
    print(" region examples:", list(sorted(regions))[:10])
    print(" context examples:", list(sorted(contexts))[:8])


def inspect_touch(path):
    d = load(path)
    files = d.get("files", [])
    n_with_segments = 0
    n_preview = 0
    for x in files:
        obj = ((x.get("touchandsee_extraction", {}) or {}).get("object_summary", {}) or {})
        segs = obj.get("segments", [])
        if isinstance(segs, list) and len(segs) > 0:
            n_with_segments += 1
        if isinstance(segs, dict) and len(segs.get("preview", []) or []) > 0:
            n_with_segments += 1
            n_preview += 1
    print("\nTouchAndSee")
    print(" files:", len(files), "with segment info:", n_with_segments, "preview only:", n_preview)


def inspect_legacy(path):
    d = load(path)
    sessions = d.get("sessions", [])
    candidate_keys = [
        "trial_records", "trial_dataframe_records", "trial_table_records",
        "trials_records", "trial_preview_records",
    ]
    counts = {k: 0 for k in candidate_keys}
    for s in sessions:
        for k in candidate_keys:
            if isinstance(s.get(k), list) and len(s.get(k)) > 0:
                counts[k] += 1
    print("\nLegacy touchscreen")
    print(" sessions:", len(sessions))
    print(" trial record keys:", counts)


def inspect_open(path):
    d = load(path)
    sessions = d.get("sessions", [])
    sorted_ok = 0
    durations = 0
    for s in sessions:
        summary = s.get("summary", {}) or {}
        if summary.get("has_sorted_unit_metadata"):
            sorted_ok += 1
        if summary.get("recording_duration_s"):
            durations += 1
    print("\nOpenfield")
    print(" sessions:", len(sessions), "with sorted units:", sorted_ok, "with duration:", durations)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nwb_json", required=True)
    p.add_argument("--touchandsee_json", required=True)
    p.add_argument("--legacy_json", required=True)
    p.add_argument("--openfield_json", required=True)
    args = p.parse_args()

    inspect_nwb(args.nwb_json)
    inspect_touch(args.touchandsee_json)
    inspect_legacy(args.legacy_json)
    inspect_open(args.openfield_json)


if __name__ == "__main__":
    main()
