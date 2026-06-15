# Actual cross-dataset CA1 unit-activity case study

## Case study question

Can CA1-related datasets be compared through actual extracted unit/spike-train activity summaries such as spike counts, firing rates, unit yield and recording duration?

## Data extracted

- Unit-level rows extracted: 124313
- Session-level rows extracted: 53
- nwb_ca1: 124313 units from 53 sessions/files

## Extraction log

- Successful files: 53
- Non-success/failed files: 48

## PCA of session activity summaries

- PC1: explains 57.1% of variance
- PC2: explains 16.3% of variance

## Interpretation

This analysis is a real data-level case study: it extracts unit/spike-train summaries from the underlying electrophysiology files when readable, then compares firing-rate and unit-yield profiles across datasets. Metadata is used as context and for harmonization, but the central variables come from actual spike/unit data rather than metadata completeness alone.