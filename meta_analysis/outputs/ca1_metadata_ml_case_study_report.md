# CA1 metadata-enriched meta-analysis report

## Research question

Can automatically extracted metadata, including quantitative recording-level variables, be used to compare heterogeneous CA1-related electrophysiology sessions and evaluate their reuse potential?

## Input overview

- Total rows in harmonized table: 238
- Datasets included: legacy_touchscreen, nwb_ca1, openfield_ca1, touchandsee

## Dataset-level summary

- **legacy_touchscreen**: 25 rows, 25 successful extractions, mean completeness=0.91, mean data-analysis potential=1.00
- **nwb_ca1**: 53 rows, 53 successful extractions, mean completeness=0.82, mean data-analysis potential=0.67
- **openfield_ca1**: 114 rows, 114 successful extractions, mean completeness=0.71, mean data-analysis potential=0.88
- **touchandsee**: 46 rows, 46 successful extractions, mean completeness=0.64, mean data-analysis potential=0.51

## Quantitative PCA / ML outputs

- PCA was computed using quantitative metadata and recording-level features, including unit counts, spike counts, LFP channel counts, trial/event counts, recording duration and derived rates.
- PCA coordinates, loadings and a PCA quality report were saved.
- KMeans clusters were computed on the standardized quantitative feature matrix.
- Quantitative PCA/KMeans silhouette score: 0.778
- Main PC1 contributors: log_spikes_per_minute, log_recording_duration_s, log_best_unit_count, log_spikes_per_unit, log_n_position_samples
- Main PC2 contributors: log_n_lfp_channels, openminds_readiness_score, log_n_electrodes, log_n_trials, log_n_event_times_total

## AFC/ACM-like categorical analysis

- Categorical metadata were analysed separately with an ACM-like approach based on one-hot encoded metadata categories.
- Categorical clustering silhouette score: 0.632

## Interpretation angle

The analysis uses metadata as an integration layer. The updated version no longer relies only on binary metadata presence/absence, but explicitly includes quantitative metadata and recording-level descriptors to evaluate cross-dataset comparability and reuse potential.