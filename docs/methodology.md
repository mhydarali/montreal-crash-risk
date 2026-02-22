# Methodology

## 1. Data and Scope
- Source data was prepared during class workflow and analyzed in `notebooks/road_collisions_classification.ipynb`.
- The portfolio includes a packaged CSV at `data/raw/mtl-road-collision-dataset-2012-2021.csv` and the executed notebook outputs.

## 2. Pipeline Steps
1. Data cleaning and quality checks.
2. Leakage prevention by removing post-collision outcome fields during model training.
3. Feature engineering for time, context, and location patterns.
4. Class-imbalance handling with SMOTENC.
5. Triple validation of feature importance:
   - ANOVA F-test
   - Random Forest importance
   - Permutation importance
6. Model training and comparison:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
7. Hyperparameter tuning with RandomizedSearchCV and GridSearchCV.
8. Final model selection using recall as primary objective.
9. Geographic and severity recommendation analysis.

## 3. Evaluation Framing
Because severe/fatal cases are rare, model quality was interpreted through the class-imbalance lens. The notebook prioritizes recall to reduce false negatives on severe outcomes.

## 4. Parsed Results (from executed notebook outputs)
- Baseline model metrics are captured in `results/model_comparison.csv`.
- Final selected model metrics and CV summary are captured in `results/metrics_summary.json`.

## 5. Rebuild Utility Scripts
- `scripts/extract_notebook_images.py`: exports all embedded notebook PNG outputs.
- `scripts/generate_results_summary.py`: parses key textual metrics from notebook outputs.
- `scripts/build_visual_gallery.py`: builds `docs/visual-gallery.md` from figure manifest.
- `scripts/prepare_portfolio_assets.sh`: runs all steps above.

## 6. Limitations
- The portfolio presents executed notebook outputs from the class workflow.
- Reproducing the exact original training run may require the same enriched data preparation context used in class.
