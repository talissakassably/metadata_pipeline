# All-datasets biological case study

## Biological question

How does the structure of experience shape what neural population analyses can be recovered across public electrophysiology datasets?

## Why this version uses all datasets

This pipeline does not force the same metric onto incompatible datasets. Instead, it uses a common biological axis — continuous versus event/trial-structured experience — and assigns each dataset a valid role.

## Dataset contributions

### NWB hippocampal-entorhinal
- Role: true spike-time temporal drift across context
- Evidence strength: strong (5/5)
- Usable sessions/rows: 53
- Main result type: TOI / distance-lag population drift

### TouchAndSee object-memory
- Role: trial-level object/memory/outcome modulation
- Evidence strength: moderate (4/5)
- Usable sessions/rows: 38
- Main result type: Cliff's delta and trial-order activity drift

### Legacy touchscreen multisensory object
- Role: event-rich multisensory task structure; limited activity reuse
- Evidence strength: task-structure-only (1/5)
- Usable sessions/rows: 6
- Main result type: task/event structure + optional trial activity

### Openfield CA1 continuous spatial foraging
- Role: continuous spatial reference
- Evidence strength: contextual_reference (3/5)
- Usable sessions/rows: 35
- Main result type: duration, sorted units, spikes, LFP availability

## Main results

- NWB: 53 usable spike-time temporal-drift analyses across 53 sessions.
- TouchAndSee: 48 usable trial-level effect analyses across 38 sessions.
- Legacy touchscreen: 6 sessions with trial records and 0.0 median event types.
- Openfield CA1: 114 continuous sessions, 35 with sorted-unit metadata.

## Biological interpretation

The NWB dataset provides the strongest direct neural evidence: spike-time population states increasingly diverge with temporal lag, and the strength of this temporal organization varies by behavioral context. TouchAndSee contributes an object-memory/task dimension by testing trial-level activity modulation. Legacy touchscreen contributes event-rich multisensory task structure, even when full activity reuse is limited. Openfield CA1 provides the continuous spatial reference condition, contrasting trial/event-structured datasets with continuous foraging.

## Claim to use

Metadata-guided reuse can identify which biological questions each public electrophysiology dataset can realistically answer. Across these datasets, experience structure emerges as the organizing axis: NWB supports temporal-context drift analysis, TouchAndSee supports object-memory trial analysis, legacy touchscreen supports multisensory event-structure reuse, and openfield CA1 supports continuous spatial-reference analysis.

## Do not overclaim

Do not claim that all datasets were analyzed with identical raw spike-time methods. The correct claim is that the same biological axis was evaluated using the strongest valid representation available for each dataset.