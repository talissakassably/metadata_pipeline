# Biological cross-dataset case study

## Question
Across CA1-related electrophysiology datasets, how do experimental context and recording structure affect available neural activity profiles and reuse potential?

## Datasets
- legacy_touchscreen: 25 sessions, median units=0.00, median spikes=0.00, median trials=0.00, median completeness=0.00
- nwb_ca1: 53 sessions, median units=1357.00, median spikes=nan, median trials=0.00, median completeness=0.33
- openfield_ca1: 114 sessions, median units=8.00, median spikes=0.00, median trials=0.00, median completeness=0.50
- touchandsee: 46 sessions, median units=1644.00, median spikes=0.00, median trials=52.00, median completeness=0.67

## Main dataset-level differences
- n_events_total: H=200.12, BH-adjusted p=3.98e-42, effect≈0.84
- n_trials: H=197.61, BH-adjusted p=6.92e-42, effect≈0.83
- n_lfp_channels: H=191.69, BH-adjusted p=8.77e-41, effect≈0.81
- n_units: H=185.02, BH-adjusted p=1.82e-39, effect≈0.78
- biological_table_completeness_score: H=147.72, BH-adjusted p=1.64e-31, effect≈0.62
- events_per_trial: H=50.75, BH-adjusted p=1.75e-12, effect≈1.00
- spikes_per_trial: H=43.11, BH-adjusted p=7.41e-11, effect≈0.84
- spikes_per_unit: H=42.84, BH-adjusted p=6.22e-10, effect≈0.25

## Main behavioral-context differences
- n_trials: H=152.74, BH-adjusted p=1.63e-29, effect≈0.64
- n_events_total: H=139.26, BH-adjusted p=5.74e-27, effect≈0.58
- n_lfp_channels: H=102.89, BH-adjusted p=1.67e-19, effect≈0.42
- n_units: H=86.77, BH-adjusted p=2.84e-16, effect≈0.35
- biological_table_completeness_score: H=68.86, BH-adjusted p=1.12e-12, effect≈0.27
- mean_firing_rate_hz_filled: H=23.18, BH-adjusted p=1.97e-06, effect≈0.19
- n_spikes_total: H=7.76, BH-adjusted p=0.00609, effect≈0.04
- spikes_per_unit: H=3.73, BH-adjusted p=0.0535, effect≈0.02

## PCA
- PC1: 46.7% variance explained
- PC2: 29.9% variance explained
Top PCA drivers: log_spikes_per_unit, log_n_spikes_total, log_events_per_trial, log_n_events_total, log_recording_duration_s.

## Interpretation
This case study compares biologically meaningful extracted variables: unit yield, spike yield, firing-rate summaries, trial structure, events, LFP availability and recording duration. Metadata is used to harmonize and contextualize the recording descriptors.