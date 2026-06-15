# -*- coding: utf-8 -*-
"""
Run the updated CA1 metadata-enriched meta-analysis pipeline.

Example from the repository root:

py meta_analysis/run_ca1_metadata_meta_analysis.py ^
  outputs/extracted_metadata/d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json ^
  outputs/extracted_metadata/d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json ^
  outputs/extracted_metadata/d-d406a98c-ae5c-4fb3-9f0c-4cf4de9b1094-hbp-data-002061-data_legacy_touchscreen_metadata.json ^
  outputs/extracted_metadata/p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json ^
  --output_dir meta_analysis/outputs
"""

import argparse
from pathlib import Path

# Allows running both as module and as script.
if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from meta_analysis.io_utils import load_json
from meta_analysis.harmonization import harmonize_dataset, infer_source_type
from meta_analysis.features import prepare_dataframe
from meta_analysis.ml_analysis import run_pca_and_clustering, run_categorical_afc_mca
from meta_analysis.reporting import (
    make_dataset_summary,
    make_availability_by_dataset,
    make_missing_metadata_report,
    make_cluster_profiles,
    write_interpretation_markdown,
)
from meta_analysis.plotting import (
    plot_completeness_by_dataset,
    plot_missingness_heatmap,
    plot_pca,
    plot_clusters,
    plot_reuse_score_by_dataset,
    plot_categorical_mca,
)


def run_pipeline(input_paths, output_dir, n_clusters=4):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for path in input_paths:
        path = Path(path)
        print("Loading:", path)
        dataset = load_json(path)
        rows = harmonize_dataset(dataset)
        print("  rows:", len(rows), "source:", infer_source_type(dataset), "dataset:", dataset.get("dataset_name"))
        all_rows.extend(rows)

    # This now builds both the classical metadata scores and the enriched quantitative features.
    df = prepare_dataframe(all_rows)

    # Quantitative PCA / KMeans.
    df_ml, pca, feature_columns, silhouette, pca_loadings, pca_explained, pca_quality = run_pca_and_clustering(
        df,
        n_clusters=n_clusters,
    )

    # AFC/ACM-like categorical analysis.
    mca_coordinates, mca_loadings, mca_explained, mca_quality, mca_silhouette = run_categorical_afc_mca(
        df_ml,
        n_clusters=n_clusters,
    )

    dataset_summary = make_dataset_summary(df_ml)
    availability = make_availability_by_dataset(df_ml)
    missing_report = make_missing_metadata_report(df_ml)
    cluster_profiles = make_cluster_profiles(df_ml)

    # Main tables.
    df_ml.to_csv(output_dir / "ca1_harmonized_sessions.csv", index=False)
    # Alias with a clearer name for the updated version.
    df_ml.to_csv(output_dir / "ca1_metadata_enriched_features.csv", index=False)
    dataset_summary.to_csv(output_dir / "ca1_dataset_summary.csv", index=False)
    availability.to_csv(output_dir / "ca1_metadata_availability_by_dataset.csv", index=False)
    missing_report.to_csv(output_dir / "ca1_missing_metadata_report.csv", index=False)

    # Quantitative PCA outputs.
    if "pca_1" in df_ml.columns:
        pca_cols = [
            "dataset_short_name", "session_id", "subject_id", "metadata_profile_label",
            "recommended_analysis_type", "recording_profile_group", "cluster", "pca_1", "pca_2",
        ]
        pca_cols = [col for col in pca_cols if col in df_ml.columns]
        df_ml[pca_cols].to_csv(output_dir / "ca1_pca_coordinates.csv", index=False)
        df_ml[pca_cols].to_csv(output_dir / "ca1_quantitative_pca_coordinates.csv", index=False)

    if pca_loadings is not None:
        pca_loadings.to_csv(output_dir / "ca1_quantitative_pca_loadings.csv", index=False)
    if pca_explained is not None:
        pca_explained.to_csv(output_dir / "ca1_quantitative_pca_explained_variance.csv", index=False)
    if pca_quality is not None:
        pca_quality.to_csv(output_dir / "ca1_pca_quality_report.csv", index=False)

    # Cluster profiles.
    if not cluster_profiles.empty:
        cluster_profiles.to_csv(output_dir / "ca1_cluster_profiles.csv", index=False)

    # Reuse recommendations table.
    reuse_cols = [
        "dataset_short_name", "session_id", "subject_id", "session_date", "behavioral_context",
        "recording_system", "brain_regions", "best_unit_count", "best_spike_count",
        "n_lfp_channels", "n_trials", "n_event_times_total", "recording_duration_s",
        "spikes_per_unit", "spikes_per_minute", "can_do_spike_analysis",
        "can_do_lfp_analysis", "can_do_behavior_analysis", "can_do_position_analysis",
        "can_do_spike_behavior_analysis", "can_do_spike_position_analysis",
        "can_do_lfp_behavior_analysis", "can_do_cross_dataset_comparison",
        "cross_dataset_reuse_score", "data_analysis_potential_score",
        "recommended_analysis_type", "missing_requirements",
    ]
    reuse_cols = [col for col in reuse_cols if col in df_ml.columns]
    df_ml[reuse_cols].to_csv(output_dir / "ca1_reuse_recommendations.csv", index=False)

    # AFC/ACM outputs.
    if mca_coordinates is not None:
        mca_coordinates.to_csv(output_dir / "ca1_categorical_mca_coordinates.csv", index=False)
    if mca_loadings is not None:
        mca_loadings.to_csv(output_dir / "ca1_categorical_mca_loadings.csv", index=False)
    if mca_explained is not None:
        mca_explained.to_csv(output_dir / "ca1_categorical_mca_explained_variance.csv", index=False)
    if mca_quality is not None:
        mca_quality.to_csv(output_dir / "ca1_categorical_mca_quality_report.csv", index=False)

    # Figures.
    plot_completeness_by_dataset(dataset_summary, figures_dir)
    plot_missingness_heatmap(df_ml, figures_dir)
    plot_pca(df_ml, figures_dir)
    plot_clusters(df_ml, figures_dir)
    plot_reuse_score_by_dataset(dataset_summary, figures_dir)
    plot_categorical_mca(mca_coordinates, figures_dir)

    write_interpretation_markdown(
        output_dir,
        df_ml,
        dataset_summary,
        missing_report,
        silhouette,
        pca_loadings=pca_loadings,
        mca_silhouette=mca_silhouette,
    )

    print("\nDone.")
    print("Output directory:", output_dir)
    print("Main enriched table:", output_dir / "ca1_metadata_enriched_features.csv")
    print("PCA loadings:", output_dir / "ca1_quantitative_pca_loadings.csv")
    print("PCA quality report:", output_dir / "ca1_pca_quality_report.csv")
    print("AFC/ACM loadings:", output_dir / "ca1_categorical_mca_loadings.csv")
    print("Figures:", figures_dir)

    return df_ml


def main():
    parser = argparse.ArgumentParser(
        description="Run updated CA1 metadata-enriched ML meta-analysis from extracted metadata JSON files."
    )
    parser.add_argument("json_files", nargs="+", help="Extracted metadata JSON files to combine.")
    parser.add_argument("--output_dir", default="meta_analysis/outputs", help="Output directory. Default: meta_analysis/outputs")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of KMeans clusters. Default: 4")
    args = parser.parse_args()
    run_pipeline(input_paths=args.json_files, output_dir=args.output_dir, n_clusters=args.n_clusters)


if __name__ == "__main__":
    main()
