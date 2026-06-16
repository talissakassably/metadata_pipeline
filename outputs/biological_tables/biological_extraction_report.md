# Biological table export report

This file confirms that the extraction pipeline produced analysis-ready biological tables.

## Generated tables
- Sessions: 238 rows
- Units: 535 rows
- Trials: 245 rows

## Coverage
- legacy_touchscreen: 25 sessions, 18 unit rows (preview_only), 18 trial rows (preview_only), median completeness=0.00
- nwb_ca1: 53 sessions, 0 unit rows (not_available), 0 trial rows (not_available), median completeness=0.33
- openfield_ca1: 114 sessions, 517 unit rows (full_or_dictionary_units), 0 trial rows (not_available), median completeness=0.50
- touchandsee: 46 sessions, 0 unit rows (not_available), 227 trial rows (preview_only), median completeness=0.67

## Notes
- Openfield unit rows come from MClust sorted unit spike-count dictionaries.
- NWB currently contributes session-level unit/electrode/context summaries; full per-unit spike-times require a raw NWB table exporter.
- Legacy touchscreen contributes session summaries plus preview units/trials because the compact extractor did not export full records.
- TouchAndSee contributes session summaries plus preview trial segments because the compact extractor did not export all segments.

These tables are the correct input for the biological cross-dataset case study.