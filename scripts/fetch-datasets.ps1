$ErrorActionPreference = "Stop"

# Downloads are intentionally excluded from Git by .gitignore.
$Root = Split-Path -Parent $PSScriptRoot
$DataDirectory = Join-Path $Root "data\datasets"
$Fruits360Directory = Join-Path $DataDirectory "fruits-360-100x100"
New-Item -ItemType Directory -Path $DataDirectory -Force | Out-Null

if (-not (Test-Path (Join-Path $Fruits360Directory ".git"))) {
    git clone --depth 1 https://github.com/fruits-360/fruits-360-100x100.git $Fruits360Directory
}
else {
    Write-Host "Fruits-360 already exists at $Fruits360Directory; skipping clone."
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is required for Kaggle downloads. Install Python 3, then run this script again."
}

python -m pip install --upgrade kaggle

$KaggleCredentials = Join-Path $HOME ".kaggle\kaggle.json"
if (-not (Test-Path $KaggleCredentials)) {
    throw "Kaggle credentials are missing. Create $KaggleCredentials with your Kaggle API token. See https://www.kaggle.com/docs/api"
}

$KaggleFruits = Join-Path $DataDirectory "kaggle-fruits"
$KaggleFruitVegetable = Join-Path $DataDirectory "kaggle-fruit-vegetable"
New-Item -ItemType Directory -Path $KaggleFruits -Force | Out-Null
New-Item -ItemType Directory -Path $KaggleFruitVegetable -Force | Out-Null
python -m kaggle datasets download -d moltean/fruits -p $KaggleFruits --unzip
python -m kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p $KaggleFruitVegetable --unzip

Write-Host "Datasets are available under $DataDirectory"
