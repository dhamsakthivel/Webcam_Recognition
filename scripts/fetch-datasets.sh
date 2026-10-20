#!/usr/bin/env bash
set -euo pipefail

# Downloads are intentionally excluded from Git by .gitignore.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data/datasets"
FRUITS360_DIR="$DATA_DIR/fruits-360-100x100"

mkdir -p "$DATA_DIR"

if [[ ! -d "$FRUITS360_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/fruits-360/fruits-360-100x100.git "$FRUITS360_DIR"
else
  echo "Fruits-360 already exists at $FRUITS360_DIR; skipping clone."
fi

if ! command -v python3 >/dev/null || ! python3 -m pip --version >/dev/null; then
  echo "Python 3 with pip is required for Kaggle downloads." >&2
  exit 1
fi

python3 -m pip install --upgrade kaggle

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  cat >&2 <<'EOF'
Kaggle API credentials are missing.
Create ~/.kaggle/kaggle.json with your Kaggle API token, then run this script again.
See: https://www.kaggle.com/docs/api
EOF
  exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json"
python3 -m kaggle datasets download -d moltean/fruits -p "$DATA_DIR/kaggle-fruits" --unzip
python3 -m kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p "$DATA_DIR/kaggle-fruit-vegetable" --unzip

echo "Datasets are available under $DATA_DIR"
