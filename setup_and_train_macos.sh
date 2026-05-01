#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
DATASET_FOLDER="${1:-$HOME/Downloads/Task07_Pancreas}"
PROJECT_ROOT="${2:-$SCRIPT_DIR/run_pancreas_m3}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-mps}"

echo "Project folder: $SCRIPT_DIR"
echo "Dataset folder: $DATASET_FOLDER"
echo "Output folder:  $PROJECT_ROOT"
echo "Epochs:         $EPOCHS"
echo "Device:         $DEVICE"

if [[ ! -f "$SCRIPT_DIR/main.py" ]]; then
  echo "Error: main.py was not found in $SCRIPT_DIR"
  exit 1
fi

if [[ ! -d "$DATASET_FOLDER/imagesTr" || ! -d "$DATASET_FOLDER/labelsTr" ]]; then
  echo "Error: dataset folder is invalid."
  echo "Expected:"
  echo "  $DATASET_FOLDER/imagesTr"
  echo "  $DATASET_FOLDER/labelsTr"
  echo
  echo "Usage:"
  echo "  ./setup_and_train_macos.sh /absolute/path/to/Task07_Pancreas"
  exit 1
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install numpy nibabel tqdm matplotlib scikit-image scikit-learn opencv-python albumentations

python - <<'PY'
import sys
import torch

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")

if not mps_available:
    print("MPS is not available. The script can still run on CPU, but not on the Apple GPU.")
    sys.exit(2)
PY

mkdir -p "$PROJECT_ROOT"

python "$SCRIPT_DIR/main.py" \
  --dataset-folder "$DATASET_FOLDER" \
  --project-root "$PROJECT_ROOT" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --force-preprocessing
