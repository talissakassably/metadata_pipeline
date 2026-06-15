# CA1 metadata-enriched meta-analysis summary

## Aim

Use automatically extracted metadata as an integration layer to compare heterogeneous CA1-related electrophysiology datasets, quantify reuse potential, and identify the variables driving differences between recording/session profiles.

## Main outputs

- Harmonized/enriched sessions: 238
- Datasets: legacy_touchscreen, nwb_ca1, openfield_ca1, touchandsee
- Quantitative PCA: PC1=41.6% and PC2=29.3% of variance
- Quantitative clustering silhouette score: 0.778
- Categorical ACM-like clustering silhouette score: 0.772

## Dataset-level reuse summary

- **legacy_touchscreen**: 25 rows, mean completeness=0.91, mean data-analysis potential=1.00, spike-behavior reusable sessions=25.
- **nwb_ca1**: 53 rows, mean completeness=0.82, mean data-analysis potential=0.67, spike-behavior reusable sessions=53.
- **openfield_ca1**: 114 rows, mean completeness=0.71, mean data-analysis potential=0.88, spike-behavior reusable sessions=93.
- **touchandsee**: 46 rows, mean completeness=0.64, mean data-analysis potential=0.51, spike-behavior reusable sessions=1.

## Quantitative PCA interpretation

The PCA was computed on standardized quantitative metadata / recording-level descriptors, not only binary metadata flags.
- Main PC1 drivers: log_spikes_per_minute, log_recording_duration_s, log_best_unit_count, log_spikes_per_unit, log_n_position_samples
- Main PC2 drivers: log_n_lfp_channels, openminds_readiness_score, log_n_electrodes, log_n_trials, log_n_event_times_total

## Categorical ACM-like interpretation

The categorical analysis was run separately on metadata categories and reuse flags. Format/source variables were excluded from the default analysis to avoid a trivial dataset-format separation.
- Main categorical drivers: has_recording_duration, has_event_metadata, context_open_field, has_brain_region_metadata, can_do_cross_dataset_comparison, has_standardized_format

## One-sentence interpretation

The analysis shows that even among biologically related CA1 datasets, metadata completeness, quantitative recording structure, and available behavioral/electrophysiological descriptors strongly determine whether sessions can be reused for cross-dataset analyses.