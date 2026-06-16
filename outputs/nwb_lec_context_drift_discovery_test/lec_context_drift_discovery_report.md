# Discovery layer: context-dependent temporal drift in LEC

## Biological question

Does behavioral/event context modulate temporal drift in lateral entorhinal cortex population activity?

## Method summary

- Region: LEC only.
- Spike-time bin size: 10.0 s.
- Unit-count matched subsampling: 50 units per session.
- Repeats per session: 5.
- Metric: Temporal Organization Index (TOI), defined as the slope of mean population distance as a function of time lag.

## Included sessions

- open_field: 4 sessions
- object_context: 4 sessions

## Median TOI by context

- object_context: 0.0010499
- open_field: 0.000713812

## Global test

Kruskal-Wallis across contexts: H=1.333, p=0.2482.

## Controls

- unit_count_available_vs_TOI: Spearman r=0.05988, p=0.888
- duration_vs_TOI: Spearman r=0.04762, p=0.9108
- n_bins_vs_TOI: Spearman r=0.04762, p=0.9108

## Suggested interpretation

This analysis provides the discovery-style layer of the project. It tests a concrete neuroscience hypothesis within the strongest raw dataset identified by the cross-dataset reuse pipeline: whether temporal organization of LEC population activity depends on behavioral context. If sequence/object contexts show higher TOI than open-field sessions after unit-count matching, the result supports the idea that event-structured experience is associated with stronger temporal-context drift in LEC.

## Caution

This does not prove causality and does not replace the original full analysis pipeline of the source paper. It is a simplified, reusable spike-time analysis designed to demonstrate that metadata-guided dataset selection can lead to a biologically meaningful secondary analysis.