# -*- coding: utf-8 -*-
"""
Main organized pipeline runner.

This script gives you one clean entry point for the project.

Recommended usage:

1. Edit configs/config_template.json with your local paths.
2. Run all steps:

py run_pipeline.py --config configs/config_template.json --steps extract biological_tables case_study

You can also run only one step:

py run_pipeline.py --config configs/config_template.json --steps biological_tables
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


def run_cmd(cmd, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def run_extract(config):
    paths = config["paths"]
    outputs = config.get("outputs", {})
    extracted = outputs.get("extracted_metadata", "outputs/extracted_metadata")
    Path(extracted).mkdir(parents=True, exist_ok=True)

    # 1. Openfield dataset-specific extractor
    run_cmd(
        [
            sys.executable,
            "extractors/extract_openfield_metadata.py",
            paths["openfield_root"],
            "--output_folder",
            extracted,
        ],
        "Extracting openfield CA1 dataset",
    )

    # 2. NWB generic extractor
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
        "Extracting NWB CA1/LEC dataset",
    )

    # 3. Legacy touchscreen dataset-specific extractor
    # This can require the neo_legacy conda environment.
    run_cmd(
        [
            sys.executable,
            "extractors/extract_legacy_touchscreen_metadata.py",
            paths["legacy_touchscreen_root"],
            "--read_lfp",
            "--output_folder",
            extracted,
        ],
        "Extracting legacy touchscreen dataset",
    )

    # 4. TouchAndSee dataset-specific extractor
    run_cmd(
        [
            sys.executable,
            "extractors/extract_touchandsee_metadata.py",
            paths["touchandsee_root"],
            "--no_object_details",
            "--output_folder",
            extracted,
        ],
        "Extracting TouchAndSee dataset",
    )


def infer_json_paths(config):
    outputs = config.get("outputs", {})
    extracted = Path(outputs.get("extracted_metadata", "outputs/extracted_metadata"))

    return {
        "openfield_json": extracted / "d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json",
        "nwb_json": extracted / "d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json",
        "legacy_json": extracted / "legacy_touchscreen_metadata.json",
        "touchandsee_json": extracted / "p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json",
    }


def run_biological_tables(config):
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


def run_meta_analysis(config):
    outputs = config.get("outputs", {})
    extracted = outputs.get("extracted_metadata", "outputs/extracted_metadata")
    meta_out = outputs.get("meta_analysis", "outputs/meta_analysis")

    run_cmd(
        [
            sys.executable,
            "meta_analysis/run_ca1_metadata_meta_analysis.py",
            "--input_dir",
            extracted,
            "--output_dir",
            meta_out,
        ],
        "Running metadata meta-analysis",
    )


def main():
    parser = argparse.ArgumentParser(description="Run the organized CA1 metadata pipeline.")
    parser.add_argument("--config", default="configs/config_template.json")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["extract", "biological_tables", "case_study"],
        choices=["extract", "biological_tables", "case_study", "meta_analysis"],
    )
    args = parser.parse_args()

    config = load_config(args.config)

    for step in args.steps:
        if step == "extract":
            run_extract(config)
        elif step == "biological_tables":
            run_biological_tables(config)
        elif step == "case_study":
            run_case_study(config)
        elif step == "meta_analysis":
            run_meta_analysis(config)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
