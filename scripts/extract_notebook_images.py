#!/usr/bin/env python3
"""Extract embedded image outputs from a Jupyter notebook.

This exports existing output figures (already executed in the notebook) so they can be
versioned and displayed in a README/docs gallery without rerunning heavy ML cells.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


CELL_NAME_MAP: Dict[int, List[Tuple[str, str]]] = {
    10: [("part3_target_distribution", "Target severity distribution")],
    11: [("part3_eda_overview", "EDA multi-panel severity patterns")],
    13: [("part4_feature_selection", "Feature selection and ranking")],
    15: [("part5_model_comparison", "Baseline model comparison")],
    17: [("part6_logistic_tuning", "Logistic regression tuning")],
    18: [("part6_tree_tuning", "Decision tree tuning")],
    20: [
        ("part6_final_model_comparison", "Final model selection comparison"),
        ("part6_logistic_coefficients", "Logistic regression coefficients"),
        ("part6_shap_analysis", "SHAP feature importance"),
        ("part6_performance_radar", "Final model performance radar"),
    ],
    22: [("part7_cv_validation", "Cross-validation stability")],
    24: [("part8_geographic_risk", "Geographic risk analysis")],
    26: [("part9_severity_recommendations", "Collision class severity analysis")],
    28: [("part10_exec_summary_dashboard", "Executive summary dashboard")],
    30: [
        ("part10_recommendation_01", "Recommendation 1 card"),
        ("part10_recommendation_02", "Recommendation 2 card"),
        ("part10_recommendation_03", "Recommendation 3 card"),
        ("part10_recommendation_04", "Recommendation 4 card"),
        ("part10_recommendation_05", "Recommendation 5 card"),
        ("part10_recommendation_06", "Recommendation 6 card"),
        ("part10_recommendation_07", "Recommendation 7 card"),
        ("part10_recommendation_08", "Recommendation 8 card"),
        ("part10_recommendation_09", "Recommendation 9 card"),
        ("part10_recommendation_10", "Recommendation 10 card"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        default="notebooks/road_collisions_classification.ipynb",
        help="Path to source notebook",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/images/notebook",
        help="Directory where PNG files will be written",
    )
    parser.add_argument(
        "--manifest",
        default="results/figure_manifest.csv",
        help="CSV manifest for exported images",
    )
    return parser.parse_args()


def decode_png(data: object) -> bytes:
    if isinstance(data, list):
        payload = "".join(data)
    else:
        payload = str(data)
    return base64.b64decode(payload)


def main() -> None:
    args = parse_args()

    notebook_path = Path(args.notebook)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    rows = []
    image_counter = 0

    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        mapped_names = CELL_NAME_MAP.get(cell_index, [])
        mapped_idx = 0
        png_in_cell = 0

        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if "image/png" not in data:
                continue

            png_in_cell += 1
            image_counter += 1

            if mapped_idx < len(mapped_names):
                base_name, description = mapped_names[mapped_idx]
                mapped_idx += 1
            else:
                base_name = f"cell_{cell_index:02d}_image_{png_in_cell:02d}"
                description = "Notebook output figure"

            filename = f"{image_counter:02d}_{base_name}.png"
            out_path = output_dir / filename
            out_path.write_bytes(decode_png(data["image/png"]))

            rows.append(
                {
                    "image_id": image_counter,
                    "cell_index": cell_index,
                    "output_index": output_index,
                    "filename": filename,
                    "relative_path": str((output_dir / filename).as_posix()),
                    "description": description,
                }
            )

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "cell_index",
                "output_index",
                "filename",
                "relative_path",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {image_counter} images to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
