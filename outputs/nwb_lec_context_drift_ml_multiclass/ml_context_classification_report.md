# ML layer: decoding behavioral context from LEC temporal organization

## Question

Can LEC temporal-organization features predict behavioral context?

## Why this matters

This adds a machine-learning validation layer to the discovery analysis. If a simple classifier predicts context above permutation baseline, then behavioral context leaves a measurable signature in LEC temporal drift features.

## ML mode

`multiclass_no_sequence`

## Classes

object_context, open_field, sleep

## Results

### logistic_regression
- CV: StratifiedKFold n_splits=5
- Samples: 38
- Balanced accuracy: 0.694
- CV balanced accuracy mean ± SD: 0.711 ± 0.108
- Permutation p-value: 0.004975

### random_forest
- CV: StratifiedKFold n_splits=5
- Samples: 38
- Balanced accuracy: 0.800
- CV balanced accuracy mean ± SD: 0.800 ± 0.083
- Permutation p-value: 0.004975

## Recommended interpretation

Use this as a validation analysis, not as the main biological claim. The main biological result remains the context-dependent TOI difference. The ML result asks whether temporal-drift features are informative enough to recover behavioral context.

## Best sentence for the report

A simple machine-learning classifier trained on LEC temporal-organization features was used as a validation step. Above-chance context prediction would indicate that behavioral context is encoded in the temporal structure of population activity, supporting the interpretation that event-structured experience modulates LEC temporal drift.