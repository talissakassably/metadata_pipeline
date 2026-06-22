# Metadata Pipeline for In-depth Neuroscience Metadata Extraction and Reuse

This repository contains the metadata extraction, harmonization, meta-analysis, and biological reuse pipeline developed during the internship project.

The objective of the pipeline is to transform heterogeneous neuroscience datasets into structured, comparable, and reusable metadata outputs. The workflow combines dataset-specific metadata extraction, harmonized JSON/CSV outputs, metadata quality assessment, exploratory analysis, and a biological case study based on metadata-guided dataset reuse.

## Repository structure

```text
metadata_pipeline/
├── extractors/              # Dataset-specific metadata extraction scripts
├── data_preparation/        # File discovery and preprocessing helpers
├── configs/                 # Local path configuration files
├── meta_analysis/           # Metadata scoring, feature engineering, PCA, reuse analyses
├── case_studies/            # Biological reuse case studies
├── outputs/                 # Generated metadata, tables, figures and reports
└── run_pipeline.py          # Main entry point for running the workflow
```

## Workflow overview

The pipeline is organized into successive processing layers:

```text
Raw datasets
→ Dataset-specific metadata extraction
→ Structured JSON metadata outputs
→ Metadata harmonization
→ Metadata quality and reuse scoring
→ Exploratory metadata analysis
→ Metadata-guided biological case study
```

This organization separates the extraction of metadata from its downstream use. The extraction modules generate structured metadata records, while the analysis modules use these records to assess metadata completeness, compare datasets, identify reusable resources, and perform biological analyses.

## Datasets supported

The pipeline was developed for heterogeneous neuroscience datasets including:

* OpenField CA1 electrophysiology dataset
* TouchAndSee object-memory electrophysiology dataset
* Entorhinal–Hippocampal NWB dataset
* Legacy touchscreen electrophysiology dataset
* Preliminary BIDS-organized neuroimaging dataset

The electrophysiology datasets constitute the main validated part of the workflow. The BIDS component is exploratory and remains under development.

## Main pipeline components

### 1. Metadata extraction

Dataset-specific extraction scripts recover metadata from raw files, standardized formats, and associated publications.

Extracted metadata include:

* subject information
* anatomical targets
* recording sessions
* neuronal units
* spike and LFP information
* behavioral events
* trials and task structure
* recording duration
* sampling rates
* file-format information

The extraction step generates structured JSON outputs in:

```text
outputs/extracted_metadata/
```

### 2. Metadata harmonization

The heterogeneous JSON outputs are mapped into a common representation so that datasets originating from different formats and laboratories can be compared.

The harmonized metadata are used to generate dataset-level and session-level summaries.

### 3. Metadata quality and reuse scoring

The pipeline computes metadata-derived indicators including:

* metadata completeness
* electrophysiological richness
* behavioral richness
* openMINDS readiness
* cross-dataset reuse potential
* overall analysis potential

These scores are calculated from the availability and completeness of relevant metadata fields and exported as CSV tables for downstream visualization.

### 4. Exploratory metadata analysis

The harmonized metadata are converted into quantitative feature vectors. These features are used for exploratory analyses such as PCA and dataset similarity visualization.

These analyses are based on metadata-derived features only, not directly on neuronal recordings.

### 5. Metadata-guided biological case study

The biological case study evaluates whether metadata can guide the selection of datasets suitable for downstream analysis.

The main case study focuses on temporal population drift in lateral entorhinal cortex recordings from the NWB dataset. Metadata are used to select relevant sessions and group recordings by behavioral context. Neuronal recordings are then analyzed using population vectors, cosine-distance metrics, and temporal organization indices.

## Configuration

Before running the pipeline, edit the configuration file:

```text
configs/config_template.json
```

Update it with the local paths to the datasets on your machine.

## Running the pipeline

Run the complete recommended workflow:

```powershell
py run_pipeline.py --config configs\config_template.json --steps extract meta_analysis case_study
```

Run only the analysis steps if metadata JSON outputs already exist:

```powershell
py run_pipeline.py --config configs\config_template.json --steps meta_analysis case_study
```

Run only extraction:

```powershell
py run_pipeline.py --config configs\config_template.json --steps extract
```

## Main outputs

### Extracted metadata JSON files

```text
outputs/extracted_metadata/
├── openfield_metadata.json
├── touchandsee_metadata.json
├── nwb_metadata.json
├── legacy_touchscreen_metadata.json
└── bids_metadata.json
```

### Metadata analysis outputs

```text
outputs/meta_analysis/
├── metadata_quality_scores.csv
├── metadata_feature_matrix.csv
├── metadata_pca_coordinates.csv
├── metadata_pca_loadings.csv
├── metadata_pca_explained_variance.csv
├── biological_analysis_affordance_scores.csv
└── figures/
```

### Biological case study outputs

```text
outputs/case_studies/
├── lec_sessions_by_context.csv
├── lec_distance_lag_curves.csv
├── lec_temporal_organization_indices.csv
├── lec_context_summary.csv
└── figures/
```

## Legacy Neo environment

The legacy touchscreen dataset depends on historical Neo-compatible objects. This part of the workflow may require a separate environment using:

```text
Python 3.7
Neo 0.8.0
```

The rest of the pipeline was developed in a modern Python environment using:

```text
Python 3.12
Neo 0.14.4
```

If the full extraction step fails because of the legacy dataset, run the legacy extractor separately in the legacy environment, then run the remaining harmonization and analysis steps in the modern environment.

## Relationship to openMINDS and EBRAINS

The metadata outputs are inspired by openMINDS concepts and designed to support future standardization efforts. The current pipeline focuses on metadata extraction, harmonization, and reuse. Full conversion into openMINDS objects and direct publication to the EBRAINS Knowledge Graph are outside the current scope, but the generated outputs are structured to facilitate future integration.

## Notes

This repository contains an internship research pipeline. Some components are fully functional, while others are exploratory or dataset-specific. In particular:

* electrophysiology extraction and metadata analysis are the main validated components;
* the BIDS neuroimaging workflow is preliminary;
* the legacy touchscreen dataset may require a separate environment;
* metadata scores are intended as practical comparative indicators, not absolute measures of dataset quality.

## Summary

This pipeline demonstrates how in-depth metadata can be extracted from heterogeneous neuroscience datasets, harmonized into common representations, and reused for dataset comparison and biological analysis. It supports the broader objective of transforming poorly comparable public datasets into structured and reusable scientific resources.
