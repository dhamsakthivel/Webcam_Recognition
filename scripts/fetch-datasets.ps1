$ErrorActionPreference = "Stop"

# Downloads are intentionally excluded from Git by .gitignore.
$Root = Split-Path -Parent $PSScriptRoot
$DataDirectory = Join-Path $Root "data\datasets"
$Fruits360Directory = Join-Path $DataDirectory "fruits-360-100x100"
$GroceryStoreDirectory = Join-Path $DataDirectory "GroceryStoreDataset"
$UnileonDirectory = Join-Path $DataDirectory "unileon-yolo-detection"
$UnileonArchive = Join-Path $UnileonDirectory "AgricultureDataset.zip"
$UnileonPartialArchive = "$UnileonArchive.part"
$UnileonUrl = "https://zenodo.org/api/records/18904638/files/AgricultureDataset.zip/content"
New-Item -ItemType Directory -Path $DataDirectory -Force | Out-Null

if (-not (Test-Path (Join-Path $Fruits360Directory ".git"))) {
    git clone --depth 1 https://github.com/fruits-360/fruits-360-100x100.git $Fruits360Directory
}
else {
    Write-Host "Fruits-360 already exists at $Fruits360Directory; skipping clone."
}

if (-not (Test-Path (Join-Path $GroceryStoreDirectory ".git"))) {
    git clone --depth 1 https://github.com/marcusklasson/GroceryStoreDataset.git $GroceryStoreDirectory
}
else {
    Write-Host "GroceryStoreDataset already exists at $GroceryStoreDirectory; skipping clone."
}

if (-not (Test-Path $UnileonArchive)) {
    New-Item -ItemType Directory -Path $UnileonDirectory -Force | Out-Null
    Write-Host "Downloading the Universidad de León YOLO dataset (~1 GB)…"
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe --fail --location --retry 3 --continue-at - --output $UnileonPartialArchive $UnileonUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Zenodo download failed. Re-run this script to resume the partial download."
        }
        Move-Item -Path $UnileonPartialArchive -Destination $UnileonArchive -Force
    }
    else {
        Write-Warning "curl.exe is unavailable; download progress may not be shown."
        Invoke-WebRequest -Uri $UnileonUrl -OutFile $UnileonPartialArchive
        Move-Item -Path $UnileonPartialArchive -Destination $UnileonArchive -Force
    }
}
if (-not (Test-Path (Join-Path $UnileonDirectory ".extracted"))) {
    Expand-Archive -Path $UnileonArchive -DestinationPath $UnileonDirectory -Force
    New-Item -ItemType File -Path (Join-Path $UnileonDirectory ".extracted") | Out-Null
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Warning "Python is unavailable; skipping optional Kaggle downloads."
    exit 0
}

$KaggleCredentials = Join-Path $HOME ".kaggle\kaggle.json"
if (-not (Test-Path $KaggleCredentials)) {
    Write-Warning "Kaggle credentials are missing; skipping optional Kaggle downloads. Create $KaggleCredentials with your Kaggle API token. See https://www.kaggle.com/docs/api"
    exit 0
}

python -m pip install --upgrade kaggle
$KaggleFruits = Join-Path $DataDirectory "kaggle-fruits"
$KaggleFruitVegetable = Join-Path $DataDirectory "kaggle-fruit-vegetable"
New-Item -ItemType Directory -Path $KaggleFruits -Force | Out-Null
New-Item -ItemType Directory -Path $KaggleFruitVegetable -Force | Out-Null
python -m kaggle datasets download -d moltean/fruits -p $KaggleFruits --unzip
python -m kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p $KaggleFruitVegetable --unzip

Write-Host "Datasets are available under $DataDirectory"
