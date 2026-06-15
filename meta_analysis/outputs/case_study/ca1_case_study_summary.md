# CA1 metadata-guided cross-dataset case study

## Case study question

Can metadata-enriched quantitative recording features identify which CA1-related sessions are reusable for downstream analyses, and which recording properties drive differences between reusable profiles?

## Dataset overview

- Total sessions/files in enriched table: 238
- Sessions with at least one reuse possibility: 238
- Datasets included: legacy_touchscreen, nwb_ca1, openfield_ca1, touchandsee

## Reuse potential

- spike analysis: 193 sessions/files
- lfp analysis: 185 sessions/files
- behavior analysis: 217 sessions/files
- position analysis: 171 sessions/files
- spike behavior analysis: 172 sessions/files
- spike position analysis: 171 sessions/files
- lfp behavior analysis: 164 sessions/files
- cross dataset comparison: 78 sessions/files

## Quantitative features differing across datasets

- log_events_per_trial: Kruskal-Wallis H=236.58, BH-adjusted p=8.9e-50, effect size≈1.00
- log_n_event_times_total: Kruskal-Wallis H=233.97, BH-adjusted p=1.55e-49, effect size≈0.99
- log_n_trials: Kruskal-Wallis H=233.25, BH-adjusted p=1.55e-49, effect size≈0.98
- log_n_electrodes: Kruskal-Wallis H=232.65, BH-adjusted p=1.57e-49, effect size≈0.98
- log_electrodes_per_unit: Kruskal-Wallis H=232.07, BH-adjusted p=1.68e-49, effect size≈0.98
- log_cross_dataset_reuse_score: Kruskal-Wallis H=223.01, BH-adjusted p=1.27e-47, effect size≈0.94
- log_n_lfp_channels: Kruskal-Wallis H=217.72, BH-adjusted p=1.52e-46, effect size≈0.92
- log_lfp_channels_per_unit: Kruskal-Wallis H=211.47, BH-adjusted p=2.98e-45, effect size≈0.89

## Quantitative features differing across analysis suitability groups

- log_data_analysis_potential_score: Kruskal-Wallis H=169.58, BH-adjusted p=2.55e-36, effect size≈0.71
- log_cross_dataset_reuse_score: Kruskal-Wallis H=162.40, BH-adjusted p=4.63e-35, effect size≈0.68
- log_n_lfp_channels: Kruskal-Wallis H=119.23, BH-adjusted p=7.28e-26, effect size≈0.50
- log_n_trials: Kruskal-Wallis H=95.46, BH-adjusted p=7.91e-21, effect size≈0.40
- log_events_per_trial: Kruskal-Wallis H=93.92, BH-adjusted p=1.3e-20, effect size≈0.39
- log_n_event_times_total: Kruskal-Wallis H=93.66, BH-adjusted p=1.3e-20, effect size≈0.39
- log_recording_duration_s: Kruskal-Wallis H=90.96, BH-adjusted p=4.3e-20, effect size≈0.38
- log_best_spike_count: Kruskal-Wallis H=76.57, BH-adjusted p=5.03e-17, effect size≈0.32

## Variables predicting analysis suitability

- Cross-validated balanced accuracy: 1.00
- log_data_analysis_potential_score: importance=0.207
- log_cross_dataset_reuse_score: importance=0.205
- log_n_lfp_channels: importance=0.099
- log_recording_duration_s: importance=0.089
- log_events_per_trial: importance=0.065
- log_n_event_times_total: importance=0.063
- log_n_trials: importance=0.054
- log_best_behavior_count: importance=0.041

## Dataset similarity

- Closest median quantitative profiles: nwb_ca1 and touchandsee (distance=5.47)
- Most distant median quantitative profiles: legacy_touchscreen and nwb_ca1 (distance=7.54)

## Interpretation

This case study moves beyond descriptive metadata completeness. Metadata is used to define which sessions are reusable, while quantitative recording-level features are used to compare datasets and analysis-suitability profiles. The result is a metadata-guided data reuse analysis rather than a purely administrative metadata summary.