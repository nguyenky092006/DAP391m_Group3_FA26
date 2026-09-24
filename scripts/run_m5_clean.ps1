param(
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [switch]$PreflightOnly,
    [switch]$ForceFeatures,
    [switch]$ForceTraining
)
$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv-gpu\Scripts\python.exe"
$Kaldi = Join-Path $RepositoryRoot ".kaldi-win\Library\bin\compute-kaldi-pitch-feats.exe"
$LogDirectory = Join-Path $RepositoryRoot "logs\paper_reproduction"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path $LogDirectory ("run_m5_clean_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
if (-not (Test-Path -LiteralPath $Python)) { throw "GPU environment not found: $Python" }
if (-not (Test-Path -LiteralPath $Kaldi)) { throw "Kaldi backend not found: $Kaldi" }
$Arguments = @(
    "src\pitchfusion\models\run_m5_clean.py",
    "--device", $Device,
    "--kaldi-executable", $Kaldi
)
if ($PreflightOnly) { $Arguments += "--preflight-only" }
if ($ForceFeatures) { $Arguments += "--force-features" }
if ($ForceTraining) { $Arguments += "--force-training" }
Push-Location $RepositoryRoot
try {
    "Run log: $LogPath"
    "M5 reuses the existing five-channel M4 feature cache unless -ForceFeatures is supplied."
    & $Python @Arguments 2>&1 | Tee-Object -LiteralPath $LogPath
    if ($LASTEXITCODE -ne 0) { throw "M5 clean run failed. See $LogPath" }
}
finally {
    Pop-Location
}
