# -*- coding: utf-8 -*-
"""Run the clean CA1 metadata-enriched meta-analysis pipeline.

Example:
py meta_analysis\run_ca1_metadata_meta_analysis.py `
  outputs\extracted_metadata\d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json `
  outputs\extracted_metadata\d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json `
  outputs\extracted_metadata\legacy_touchscreen_metadata.json `
  outputs\extracted_metadata\p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json `
  --output_dir meta_analysis\outputs
"""

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from meta_analysis.io_utils import load_json
from meta_analysis.harmonization import harmonize_dataset, infer_source_type
from meta_analysis.features import prepare_dataframe
from meta_analysis.ml_analysis import run_quantitative_pca_clustering, run_categorical_acm
from meta_analysis.reporting import (
    make_dataset_summary,
    make_missing_metadata_report,
    make_reuse_recommendations,
    write_narrative_summary,
)
from meta_analysis.plotting import (
    plot_completeness_by_dataset,
    plot_missingness_heatmap,
    plot_quantitative_pca,
    plot_pca_loadings,
    plot_reuse_score_by_dataset,
    plot_acm,
    plot_acm_variable_importance,
)


def _make_output_dirs(output_dir):
    output_dir = Path(output_dir)
    dirs = {
        "root": output_dir,
        "main": output_dir / "main",
        "quality": output_dir / "quality_reports",
        "pca": output_dir / "pca_outputs",
        "categorical": output_dir / "categorical_outputs",
        "figures": output_dir / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_pipeline(input_paths, output_dir="meta_analysis/outputs", n_clusters=4):
    dirs = _make_output_dirs(output_dir)
    all_rows = []

    for path in input_paths:
        path = Path(path)
        print("Loading:", path)
        dataset = load_json(path)
        rows = harmonize_dataset(dataset)
        print("  rows:", len(rows), "source:", infer_source_type(dataset), "dataset:", dataset.get("dataset_name"))
        all_rows.extend(rows)

    df = prepare_dataframe(all_rows)
    df_ml, pca, features_used, q_sil, pca_loadings, pca_explained, pca_quality = run_quantitative_pca_clustering(df, n_clusters=n_clusters)
    acm_coords, acm_loadings, acm_variable_importance, acm_explained, acm_quality, acm_sil = run_categorical_acm(df_ml, n_clusters=n_clusters)

    dataset_summary = make_dataset_summary(df_ml)
    missing_report = make_missing_metadata_report(df_ml)
    reuse_recommendations = make_reuse_recommendations(df_ml)

    # Main outputs: only the tables that matter for interpretation.
    df_ml.to_csv(dirs["main"] / "ca1_metadata_enriched_features.csv", index=False)
    dataset_summary.to_csv(dirs["main"] / "ca1_dataset_summary.csv", index=False)
    reuse_recommendations.to_csv(dirs["main"] / "ca1_reuse_recommendations.csv", index=False)
    write_narrative_summary(dirs["root"], df_ml, dataset_summary, pca_loadings, pca_explained, acm_variable_importance, q_sil, acm_sil)

    # Quality/debug outputs, separated so the root is not messy.
    missing_report.to_csv(dirs["quality"] / "ca1_missing_metadata_report.csv", index=False)
    if pca_quality is not None:
        pca_quality.to_csv(dirs["quality"] / "ca1_pca_quality_report.csv", index=False)
    if acm_quality is not None:
        acm_quality.to_csv(dirs["quality"] / "ca1_acm_quality_report.csv", index=False)

    # PCA outputs.
    if "pca_1" in df_ml.columns:
        pca_cols = [c for c in [
            "dataset_short_name", "session_id", "subject_id", "metadata_profile_label",
            "recommended_analysis_type", "recording_profile_group", "cluster", "pca_1", "pca_2",
        ] if c in df_ml.columns]
        df_ml[pca_cols].to_csv(dirs["pca"] / "ca1_quantitative_pca_coordinates.csv", index=False)
    if pca_loadings is not None:
        pca_loadings.to_csv(dirs["pca"] / "ca1_quantitative_pca_loadings.csv", index=False)
    if pca_explained is not None:
        pca_explained.to_csv(dirs["pca"] / "ca1_quantitative_pca_explained_variance.csv", index=False)

    # Categorical/ACM outputs.
    if acm_coords is not None:
        acm_coords.to_csv(dirs["categorical"] / "ca1_categorical_acm_coordinates.csv", index=False)
    if acm_loadings is not None:
        acm_loadings.to_csv(dirs["categorical"] / "ca1_categorical_acm_category_loadings.csv", index=False)
    if acm_variable_importance is not None:
        acm_variable_importance.to_csv(dirs["categorical"] / "ca1_categorical_acm_variable_importance.csv", index=False)
    if acm_explained is not None:
        acm_explained.to_csv(dirs["categorical"] / "ca1_categorical_acm_explained_variance.csv", index=False)

    # Figures.
    plot_completeness_by_dataset(dataset_summary, dirs["figures"])
    plot_missingness_heatmap(df_ml, dirs["figures"])
    plot_quantitative_pca(df_ml, dirs["figures"])
    plot_pca_loadings(pca_loadings, dirs["figures"])
    plot_reuse_score_by_dataset(dataset_summary, dirs["figures"])
    plot_acm(acm_coords, dirs["figures"])
    plot_acm_variable_importance(acm_variable_importance, dirs["figures"])

    print("\nDone.")
    print("Output directory:", dirs["root"])
    print("Main outputs:", dirs["main"])
    print("PCA outputs:", dirs["pca"])
    print("Categorical outputs:", dirs["categorical"])
    print("Quality reports:", dirs["quality"])
    print("Figures:", dirs["figures"])
    print("Narrative summary:", dirs["main"] / "ca1_meta_analysis_summary.md")
    return df_ml


def main():
    parser = argparse.ArgumentParser(description="Run clean CA1 metadata-enriched meta-analysis.")
    parser.add_argument("json_files", nargs="+", help="Extracted metadata JSON files to combine.")
    parser.add_argument("--output_dir", default="meta_analysis/outputs", help="Output directory.")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of KMeans clusters.")
    args = parser.parse_args()
    run_pipeline(args.json_files, output_dir=args.output_dir, n_clusters=args.n_clusters)


if __name__ == "__main__":
    main()
