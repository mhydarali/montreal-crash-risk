#!/usr/bin/env bash
set -euo pipefail

./scripts/extract_notebook_images.py
./scripts/generate_results_summary.py
./scripts/build_visual_gallery.py

echo "Portfolio assets refreshed."
