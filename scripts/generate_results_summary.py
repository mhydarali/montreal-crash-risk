#!/usr/bin/env python3
"""Parse key metrics from executed notebook outputs into machine-readable files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


BASELINE_MODEL_LINE = re.compile(
    r"^\s*(Logistic Regression|Decision Tree|Random Forest|XGBoost)\s+"
    r"Acc:\s*([0-9.]+)\s*\|\s*Precision:\s*([0-9.]+)\s*\|\s*"
    r"Recall:\s*([0-9.]+)\s*\|\s*F1:\s*([0-9.]+)\s*\|\s*ROC-AUC:\s*([0-9.]+)",
    re.MULTILINE,
)

SELECTED_MODEL_BLOCK = re.compile(
    r"Logistic Regression \(Tuned \(GridSearchCV\)\):\s*\n"
    r"\s*Accuracy:\s*([0-9.]+)\s*\n"
    r"\s*Precision:\s*([0-9.]+)\s*\n"
    r"\s*Recall:\s*([0-9.]+)[^\n]*\n"
    r"\s*F1-Score:\s*([0-9.]+)\s*\n"
    r"\s*ROC-AUC:\s*([0-9.]+)",
    re.MULTILINE,
)

CV_METRIC_LINE = re.compile(
    r"\s*(Accuracy|Precision|Recall|F1-Score):\s*([0-9.]+)\s*[±\+\-]\s*([0-9.]+)",
)

SEVERE_COUNT_LINE = re.compile(
    r"Severe/Fatal accidents:\s*([0-9,]+)\s*\(([0-9.]+)% of total\)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        default="notebooks/road_collisions_classification.ipynb",
        help="Path to source notebook",
    )
    parser.add_argument(
        "--json-out",
        default="results/metrics_summary.json",
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--csv-out",
        default="results/model_comparison.csv",
        help="Output CSV comparison path",
    )
    return parser.parse_args()


def collect_output_text(nb: Dict) -> List[str]:
    chunks: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if "text" in output:
                chunks.append("".join(output["text"]))
            elif output.get("output_type") == "execute_result":
                text_plain = output.get("data", {}).get("text/plain")
                if text_plain:
                    if isinstance(text_plain, list):
                        chunks.append("".join(text_plain))
                    else:
                        chunks.append(str(text_plain))
    return chunks


def to_float(value: str) -> float:
    return float(value.strip())


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook)
    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    output_text = "\n".join(collect_output_text(nb))

    baseline_models = []
    for match in BASELINE_MODEL_LINE.finditer(output_text):
        baseline_models.append(
            {
                "model": match.group(1),
                "accuracy": to_float(match.group(2)),
                "precision": to_float(match.group(3)),
                "recall": to_float(match.group(4)),
                "f1_score": to_float(match.group(5)),
                "roc_auc": to_float(match.group(6)),
                "stage": "baseline",
            }
        )

    selected_match = SELECTED_MODEL_BLOCK.search(output_text)
    final_selected = None
    if selected_match:
        final_selected = {
            "model": "Logistic Regression (Tuned GridSearchCV)",
            "accuracy": to_float(selected_match.group(1)),
            "precision": to_float(selected_match.group(2)),
            "recall": to_float(selected_match.group(3)),
            "f1_score": to_float(selected_match.group(4)),
            "roc_auc": to_float(selected_match.group(5)),
            "stage": "final_selected",
        }

    cv_summary = {}
    for metric, mean, std in CV_METRIC_LINE.findall(output_text):
        key = metric.lower().replace("-", "_")
        cv_summary[key] = {"mean": to_float(mean), "std": to_float(std)}

    severe_match = SEVERE_COUNT_LINE.search(output_text)
    class_imbalance = None
    if severe_match:
        severe_count = int(severe_match.group(1).replace(",", ""))
        severe_pct = float(severe_match.group(2))
        class_imbalance = {
            "severe_or_fatal_count": severe_count,
            "severe_or_fatal_pct": severe_pct,
        }

    summary = {
        "source_notebook": str(notebook_path.as_posix()),
        "baseline_model_metrics": baseline_models,
        "final_selected_model": final_selected,
        "cross_validation_summary": cv_summary,
        "class_imbalance": class_imbalance,
    }

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = baseline_models.copy()
    if final_selected:
        rows.append(final_selected)

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote JSON summary: {json_out}")
    print(f"Wrote model CSV: {csv_out}")


if __name__ == "__main__":
    main()
