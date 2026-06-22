# -*- coding: utf-8 -*-
r"""
Environment-aware pipeline runner.

Some datasets can be extracted with the normal/base Python environment.
Other legacy datasets require old Python / old Neo / old NIX dependencies.

Recommended workflow:

1) BASE:
   conda activate base
   py run_pipeline_env_aware.py --config configs\config_template.json --steps extract_base

2) NEO_LEGACY:
   conda activate neo_legacy
   py extractors\extract_legacy_touchscreen_metadata.py ^
     "C:\Users\tkassably\Downloads\d-d406a98c-ae5c-4fb3-9f0c-4cf4de9b1094-hbp-data-002061-data" ^
     --read_lfp ^
     --output_folder outputs\extracted_metadata

3) BASE:
   conda activate base
   py run_pipeline_env_aware.py --config configs\config_template.json --steps biological_tables case_study
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd, title, allow_fail=False):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(" ".join(str(x) for x in cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        if allow_fail:
            print(f"\nWARNING: step failed but allow_fail=True. Return code: {result.returncode}")
            return
        raise subprocess.CalledProcessError(result.returncode, cmd)


def run_extract_base(config):
    """
    Run extraction steps expected to work in the normal/base environment.
    Does NOT run the legacy touchscreen extractor.
    """
    paths = config["paths"]
    outputs = config.get("outputs", {})
    extracted = outputs.get("extracted_metadata", "outputs/extracted_metadata")
    Path(extracted).mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            "extractors/extract_openfield_metadata.py",
            paths["openfield_root"],
            "--output_folder",
            extracted,
        ],
        "Extracting openfield CA1 dataset in base environment",
    )

    run_cmd(
        [
            sys.executable,
            "extractors/extract_metadata_pipeline.py",
            paths["nwb_root"],
            "--extensions",
            ".nwb",
            "--output_folder",
            extracted,
        ],
        "Extracting NWB CA1/LEC dataset in base environment",
    )

    run_cmd(
        [
            sys.executable,
            "extractors/extract_touchandsee_metadata.py",
            paths["touchandsee_root"],
            "--no_object_details",
            "--output_folder",
            extracted,
        ],
        "Extracting TouchAndSee dataset in base environment",
        allow_fail=True,
    )

    print("\nBase extraction finished.")
    print("IMPORTANT: legacy touchscreen extraction is intentionally NOT run here.")
    print("Run it separately inside the neo_legacy environment.")


def print_legacy_command(config):
    paths = config["paths"]
    outputs = config.get("outputs", {})
    extracted = outputs.get("extracted_metadata", "outputs/extracted_metadata")

    print("\nRun this in neo_legacy:")
    print("conda activate neo_legacy")
    print(
        "py extractors\\extract_legacy_touchscreen_metadata.py "
        f"\"{paths['legacy_touchscreen_root']}\" "
        "--read_lfp "
        f"--output_folder {extracted}"
    )


def infer_json_paths(config):
    extracted = Path(config.get("outputs", {}).get("extracted_metadata", "outputs/extracted_metadata"))
    return {
        "openfield_json": extracted / "d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json",
        "nwb_json": extracted / "d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json",
        "legacy_json": extracted / "legacy_touchscreen_metadata.json",
        "touchandsee_json": extracted / "p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json",
    }


def check_required_jsons(config):
    jsons = infer_json_paths(config)
    missing = [str(path) for path in jsons.values() if not Path(path).exists()]

    if missing:
        print("\nMissing JSON files:")
        for item in missing:
            print(" -", item)
        print("\nYou probably still need to run one extraction step.")
        print_legacy_command(config)
        raise FileNotFoundError("Missing extracted metadata JSON files.")

    print("\nAll required JSON files found.")


def run_biological_tables(config):
    check_required_jsons(config)
    outputs = config.get("outputs", {})
    table_out = outputs.get("biological_tables", "outputs/biological_tables")
    jsons = infer_json_paths(config)

    run_cmd(
        [
            sys.executable,
            "biological_tables/export_biological_tables.py",
            "--openfield_json",
            str(jsons["openfield_json"]),
            "--nwb_json",
            str(jsons["nwb_json"]),
            "--legacy_json",
            str(jsons["legacy_json"]),
            "--touchandsee_json",
            str(jsons["touchandsee_json"]),
            "--output_dir",
            table_out,
        ],
        "Exporting harmonized biological tables",
    )


def run_case_study(config):
    outputs = config.get("outputs", {})
    table_out = outputs.get("biological_tables", "outputs/biological_tables")
    case_out = outputs.get("case_study", "outputs/case_studies/biological_cross_dataset")

    run_cmd(
        [
            sys.executable,
            "case_studies/run_biological_case_study_from_tables.py",
            "--input_dir",
            table_out,
            "--output_dir",
            case_out,
        ],
        "Running biological cross-dataset case study",
    )


def main():
    parser = argparse.ArgumentParser(description="Run environment-aware CA1 metadata pipeline.")
    parser.add_argument("--config", default="configs/config_template.json")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["extract_base", "biological_tables", "case_study"],
        choices=["extract_base", "print_legacy_command", "biological_tables", "case_study"],
    )
    args = parser.parse_args()

    config = load_config(args.config)

    for step in args.steps:
        if step == "extract_base":
            run_extract_base(config)
        elif step == "print_legacy_command":
            print_legacy_command(config)
        elif step == "biological_tables":
            run_biological_tables(config)
        elif step == "case_study":
            run_case_study(config)

    print("\nEnvironment-aware pipeline finished.")


if __name__ == "__main__":
    main()
