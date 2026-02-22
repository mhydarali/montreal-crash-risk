# Executive Summary

## Business Question
How can Montréal proactively identify crash scenarios with high probability of severe outcomes and prioritize interventions where they matter most?

## What Was Built
A full machine learning workflow was developed to classify severe/fatal collision risk using pre-crash and contextual features (time, location, environment, road and weather conditions). The workflow includes data cleaning, leakage control, class balancing, feature selection, model comparison, hyperparameter tuning, and recommendation generation.

## Why This Matters
Severe and fatal crashes are a small minority of total events, so a standard accuracy-first model can miss them. This project intentionally uses a recall-first objective to catch more high-risk cases and support prevention-focused operations.

## Key Results
- Severe/fatal rate in modeled data: **1.5617%** (2,049 cases).
- Final selected model from notebook outputs: **Tuned Logistic Regression**.
- Final selected severe/fatal recall: **0.9398**.
- Cross-validation summary (reported in notebook):
  - Accuracy: **0.5990 ± 0.0023**
  - Precision: **0.5909 ± 0.0023**
  - Recall: **0.6436 ± 0.0030**
  - F1-score: **0.6161 ± 0.0021**

## Strategic Insights
- High-risk patterns concentrate by spatio-temporal context and location features.
- Severity risk is tied to conditions that can be operationally targeted (time windows, corridors, vulnerable-user zones).
- The analysis section in the notebook links model findings to interventions such as targeted speed management, signal upgrades, and vulnerable-user infrastructure prioritization.

## Decision Use
This model is best used as a **risk triage layer** for planning and prioritization, not as an autonomous enforcement decision-maker.

## Artifact References
- Notebook with executed outputs: `notebooks/road_collisions_classification.ipynb`
- Parsed metrics: `results/metrics_summary.json`
- Full visual gallery: `docs/visual-gallery.md`
