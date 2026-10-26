#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:?Usage: bash run_experiments.sh /path/to/cybench-data/maize/CN}"
CONFIG="${2:-configs/default.yaml}"
for EXP in xgboost lstm transformer proposed ablate_no_lag ablate_no_spatial ablate_no_llm; do
  python -m src.train --config "$CONFIG" --data-dir "$DATA_DIR" --experiment "$EXP"
done
python -m src.report --artifacts artifacts
