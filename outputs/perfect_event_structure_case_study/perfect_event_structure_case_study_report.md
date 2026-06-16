# Perfect event-structure cross-dataset case study

## Biological question

Does the structure of experience shape neural population organization across hippocampal-entorhinal and object-recognition electrophysiology datasets?

## Core idea

The datasets are not forced into a single artificial metric. Instead, each dataset contributes the biologically valid analysis supported by its structure:

- NWB hippocampal-entorhinal recordings: true spike-time population temporal drift.
- TouchAndSee object-memory recordings: trial-level activity modulation by memory demand and outcome.
- Legacy touchscreen multisensory recordings: trial/task modulation when full records are available.
- Openfield CA1 recordings: continuous spatial-foraging reference, sorted unit yield, duration and LFP availability.

## Evidence table

### NWB hippocampal-entorhinal
- Biological context: LEC/MEC/CA1 event structure, sequence, object, sleep, open field
- Contribution: Tests true spike-time population temporal drift by region and context
- Metric: TOI from binned spike-time population vectors
- Usable sessions/rows: 53
- Evidence strength: strong

### TouchAndSee object-memory
- Biological context: trial-based object/memory behavior
- Contribution: Tests whether trial type/outcome is reflected in trial activity summaries
- Metric: Cliff's delta for memory vs normal, correct vs incorrect, and trial-order drift
- Usable sessions/rows: 38
- Evidence strength: moderate

### Legacy touchscreen multisensory object
- Biological context: visual/tactile/multisensory object and outcome task
- Contribution: Tests whether extracted trial records support outcome/modality activity analysis
- Metric: Trial-level contrast effects if activity records are present
- Usable sessions/rows: 0
- Evidence strength: task metadata only / not usable for activity

### Openfield CA1 continuous spatial foraging
- Biological context: continuous open-field CA1 spatial foraging
- Contribution: Provides continuous spatial reference and contrasts absence of trial/event structure
- Metric: recording duration, sorted unit yield, LFP availability
- Usable sessions/rows: 35
- Evidence strength: contextual_reference

## Main results summary

- NWB usable region/session analyses: 53 rows across 53 sessions.
- TouchAndSee usable trial-effect analyses: 48 rows across 38 sessions.
- Legacy usable trial-effect analyses: 0 rows across 0 sessions.
- Openfield reference sessions: 114 sessions.

## Recommended interpretation

This case study contributes biologically by showing how the structure of experience determines what can be measured from reused electrophysiology datasets. NWB supports population temporal drift analysis across hippocampal-entorhinal regions and behavioral contexts. TouchAndSee supports object/memory trial-level modulation. Legacy touchscreen supports multisensory object-task reuse when trial records are available. Openfield CA1 provides the continuous spatial reference condition.

## What not to claim

Do not claim that all datasets were analyzed with identical raw spike-time methods. The correct claim is that a single biological axis — continuous versus event/trial-structured experience — was tested using the strongest valid representation available for each dataset.