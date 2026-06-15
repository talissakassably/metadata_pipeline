# Organized CA1 metadata and biological reuse pipeline

This is the updated clean organization of the pipeline.

It separates the project into four layers:

```text
metadata_pipeline_organized_update/
├── extractors/              # original + dataset-specific metadata extractors
├── biological_tables/       # corrected biological extraction output layer
├── meta_analysis/           # metadata completeness / PCA / ACM analysis
├── case_studies/            # biological cross-dataset case study
├── data_preparation/        # file discovery helpers
├── configs/                 # local path configuration
├── outputs/                 # generated outputs
└── run_pipeline.py          # single entry point
```

## Why this organization is better

The old pipeline produced rich but heterogeneous JSON files. The key biological variables were buried inside different structures.

The updated pipeline makes the workflow explicit:

```text
raw datasets
→ dataset-specific extraction JSONs
→ harmonized biological tables
→ biological cross-dataset case study
→ metadata meta-analysis
```

## Step 1 — configure paths

Edit:

```text
configs/config_template.json
```

with your local dataset paths.

## Step 2 — run the full recommended pipeline

```powershell
py run_pipeline.py --config configs\config_template.json --steps extract biological_tables case_study
```

## Step 3 — run only the corrected biological table export

If you already have the JSON files in `outputs/extracted_metadata`, run:

```powershell
py run_pipeline.py --config configs\config_template.json --steps biological_tables case_study
```

## Main outputs

### Metadata JSON outputs

```text
outputs/extracted_metadata/
├── d-40faae41-7e72-4c3c-9abf-91dea149158d_openfield_metadata.json
├── d-885b4936-9345-43bd-880e-eebc19898ded_extracted_metadata.json
├── legacy_touchscreen_metadata.json
└── p25b4e-Pennartz_SGA1_T3.3.3-hbp-01681_touchandsee_metadata.json
```

### Corrected biological tables

```text
outputs/biological_tables/
├── harmonized_biological_sessions.csv
├── harmonized_biological_units.csv
├── harmonized_biological_trials.csv
├── dataset_biological_summary.csv
├── extraction_coverage_report.csv
└── biological_extraction_report.md
```

### Biological case study

```text
outputs/case_studies/biological_cross_dataset/
├── biological_case_dataset_summary.csv
├── biological_case_context_summary.csv
├── biological_case_kruskal_by_dataset.csv
├── biological_case_kruskal_by_context.csv
├── biological_case_pca_coordinates.csv
├── biological_case_pca_loadings.csv
├── biological_case_pca_explained_variance.csv
├── biological_case_study_report.md
└── figures/
```

## Important note about the legacy touchscreen extractor

The legacy touchscreen extractor may require your `neo_legacy` conda environment because it depends on older Neo/NIX objects.

If the full `extract` step fails at the legacy extractor, run that extractor separately inside `neo_legacy`, then run:

```powershell
py run_pipeline.py --config configs\config_template.json --steps biological_tables case_study
```

## What to say in the meeting

> I reorganized the pipeline into clear extraction, harmonization, meta-analysis and case-study layers. The extraction step still saves the full metadata JSONs, but the corrected layer now also exports harmonized biological tables: sessions, units and trials. This makes the downstream cross-dataset case study cleaner and biologically interpretable, because it compares unit yield, spike yield, firing-rate summaries, trial structure, events, LFP availability and recording duration rather than only metadata completeness.
