param(
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [ValidateSet("v1", "regularized", "direct5", "full5")][string]$Variant = "v1",
    [switch]$PreflightOnly,
    [switch]$PrepareOnly,
    [switch]$ForceFeatures
)
$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv-gpu\Scripts\python.exe"
$Kaldi = Join-Path $RepositoryRoot ".kaldi-win\Library\bin\compute-kaldi-pitch-feats.exe"
$LogDirectory = Join-Path $RepositoryRoot "logs\paper_reproduction"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path $LogDirectory ("run_m4_clean_{0}_{1}.log" -f $Variant, (Get-Date -Format "yyyyMMdd_HHmmss"))
if (-not (Test-Path -LiteralPath $Python)) { throw "GPU environment not found: $Python" }
if (-not (Test-Path -LiteralPath $Kaldi)) { throw "Kaldi backend not found: $Kaldi" }
$UseLegacyCudnn = $Variant -eq "v1"
$PreviousCudnnV8ApiDisabled = [Environment]::GetEnvironmentVariable(
    "TORCH_CUDNN_V8_API_DISABLED", "Process"
)
if ($UseLegacyCudnn) {
    $env:TORCH_CUDNN_V8_API_DISABLED = "1"
}
else {
    Remove-Item Env:\TORCH_CUDNN_V8_API_DISABLED -ErrorAction SilentlyContinue
}
$Arguments = @("src\pitchfusion\models\run_m4_clean.py", "--device", $Device, "--variant", $Variant, "--kaldi-executable", $Kaldi)
if ($PreflightOnly) { $Arguments += "--preflight-only" }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($ForceFeatures) { $Arguments += "--force-features" }
Push-Location $RepositoryRoot
try {
    "Run log: $LogPath"
    if ($UseLegacyCudnn) {
        "Runtime: TORCH_CUDNN_V8_API_DISABLED=1 (resume-only acceleration)"
    }
    else {
        "Runtime: default cuDNN v8 (fresh-training correctness)"
    }
    & $Python @Arguments 2>&1 | Tee-Object -LiteralPath $LogPath
    if ($LASTEXITCODE -ne 0) { throw "M4 clean run failed. See $LogPath" }
}
finally {
    Pop-Location
    if ($null -eq $PreviousCudnnV8ApiDisabled) {
        Remove-Item Env:\TORCH_CUDNN_V8_API_DISABLED -ErrorAction SilentlyContinue
    }
    else {
        $env:TORCH_CUDNN_V8_API_DISABLED = $PreviousCudnnV8ApiDisabled
    }
}
