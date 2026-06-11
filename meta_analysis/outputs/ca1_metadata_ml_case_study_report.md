# CA1 metadata ML case study report

## Research question

Can automatically extracted metadata be used to compare and cluster heterogeneous CA1-related electrophysiology recording sessions?

## Input overview

- Total rows in harmonized table: 238
- Datasets included: legacy_touchscreen, nwb_ca1, openfield_ca1, touchandsee

## Dataset-level summary

- **legacy_touchscreen**: 25 rows, 25 successful extractions, mean completeness=0.91, mean ephys richness=0.83
- **nwb_ca1**: 53 rows, 53 successful extractions, mean completeness=0.82, mean ephys richness=0.83
- **openfield_ca1**: 114 rows, 114 successful extractions, mean completeness=0.71, mean ephys richness=0.83
- **touchandsee**: 46 rows, 46 successful extractions, mean completeness=0.64, mean ephys richness=0.34

## ML outputs

- PCA coordinates were computed and saved.
- KMeans clusters were computed and saved.
- Silhouette score: 0.864

## Important limitation

TouchAndSee may be represented mainly by file/session metadata if the Neo/pickle loading failed in the provided extraction JSON. Once the legacy pickle compatibility issue is fixed and the extraction is rerun, the same pipeline will include its internal electrophysiology metadata automatically if present in the JSON.

## Interpretation angle

This analysis treats metadata as a scientific integration layer: because the datasets are CA1-related but heterogeneous in format and acquisition system, session-level metadata features can reveal different experimental/recording profiles such as open-field navigation, task/event-rich recordings, standardized NWB unit-rich sessions, and metadata-limited sessions.