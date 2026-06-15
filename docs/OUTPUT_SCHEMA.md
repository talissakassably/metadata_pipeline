# Output table schema

## harmonized_biological_sessions.csv

One row = one session/file/recording.

Important columns:
- dataset_short_name
- session_id
- subject_id
- session_date
- behavioral_context
- recording_system
- brain_region
- n_units
- n_spikes_total
- n_raw_spike_events
- recording_duration_s
- mean_firing_rate_hz
- mean_firing_rate_hz_filled
- n_trials
- n_events_total
- n_lfp_channels
- n_position_samples
- spikes_per_unit
- spikes_per_trial
- events_per_trial
- biological_table_completeness_score

## harmonized_biological_units.csv

One row = one unit/spiketrain when available.

Important columns:
- dataset_short_name
- session_id
- subject_id
- unit_id
- brain_region
- behavioral_context
- n_spikes
- firing_rate_hz
- recording_duration_s
- cell_type
- quality_label
- is_preview_record

## harmonized_biological_trials.csv

One row = one trial/segment when available.

Important columns:
- dataset_short_name
- session_id
- subject_id
- trial_id
- trial_type
- trial_outcome
- trial_side
- object_id
- n_spikes_total
- n_spiketrains
- n_events
- is_preview_record
