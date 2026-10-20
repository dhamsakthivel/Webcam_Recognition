#!/usr/bin/env bash
set -euo pipefail

# Downloads are intentionally excluded from Git by .gitignore.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data/datasets"
FRUITS360_DIR="$DATA_DIR/fruits-360-100x100"
GROCERY_DIR="$DATA_DIR/GroceryStoreDataset"
UNILEON_DIR="$DATA_DIR/unileon-yolo-detection"
UNILEON_ZIP="$UNILEON_DIR/AgricultureDataset.zip"
UNILEON_URL="https://zenodo.org/api/records/18904638/files/AgricultureDataset.zip/content"

mkdir -p "$DATA_DIR"

if [[ ! -d "$FRUITS360_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/fruits-360/fruits-360-100x100.git "$FRUITS360_DIR"
else
  echo "Fruits-360 already exists at $FRUITS360_DIR; skipping clone."
fi

if [[ ! -d "$GROCERY_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/marcusklasson/GroceryStoreDataset.git "$GROCERY_DIR"
else
  echo "GroceryStoreDataset already exists at $GROCERY_DIR; skipping clone."
fi

if [[ ! -f "$UNILEON_ZIP" ]]; then
  command -v curl >/dev/null || { echo "curl is required for the Zenodo download." >&2; exit 1; }
  mkdir -p "$UNILEON_DIR"
  echo "Downloading the Universidad de León YOLO dataset (~1 GB)…"
  curl --fail --location --continue-at - --output "$UNILEON_ZIP" "$UNILEON_URL"
fi
if [[ ! -d "$UNILEON_DIR/train" && ! -d "$UNILEON_DIR/valid" ]]; then
  command -v unzip >/dev/null || { echo "unzip is required for the Zenodo archive." >&2; exit 1; }
  unzip -q -n "$UNILEON_ZIP" -d "$UNILEON_DIR"
fi

if ! command -v python3 >/dev/null || ! python3 -m pip --version >/dev/null; then
  echo "Python 3 with pip is unavailable; skipping optional Kaggle downloads." >&2
  exit 0
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  cat >&2 <<'EOF'
Kaggle API credentials are missing; skipping optional Kaggle downloads.
Create ~/.kaggle/kaggle.json with your Kaggle API token, then run this script again.
See: https://www.kaggle.com/docs/api
EOF
  exit 0
fi

python3 -m pip install --upgrade kaggle
chmod 600 "$HOME/.kaggle/kaggle.json"
python3 -m kaggle datasets download -d moltean/fruits -p "$DATA_DIR/kaggle-fruits" --unzip
python3 -m kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p "$DATA_DIR/kaggle-fruit-vegetable" --unzip

echo "Datasets are available under $DATA_DIR"
