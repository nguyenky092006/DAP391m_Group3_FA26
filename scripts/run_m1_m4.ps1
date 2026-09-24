param(
    [ValidateSet("m1", "m2", "m3", "m4", "all")]
    [string]$Model = "all",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$AllowDownload,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$DeprecatedModels = @("m3", "m4", "all")
if ($Model -in $DeprecatedModels) {
    throw "This wrapper contains the retired speaker-disjoint M3/M4 pipeline. Use run_paper_reproduction.ps1 for M3 and run_m4_clean.ps1 for M4."
}
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv-gpu\Scripts\python.exe"
$LogDirectory = Join-Path $RepositoryRoot "logs\m1_m4"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$env:NUMBA_CACHE_DIR = Join-Path $RepositoryRoot "data\interim\numba_cache"
New-Item -ItemType Directory -Path $env:NUMBA_CACHE_DIR -Force | Out-Null
$LogPath = Join-Path $LogDirectory ("run_{0}_{1}.log" -f $Model, (Get-Date -Format "yyyyMMdd_HHmmss"))
$TranscriptStarted = $false

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

Push-Location $RepositoryRoot
try {
    Start-Transcript -LiteralPath $LogPath -Force | Out-Null
    $TranscriptStarted = $true
    Write-Output "Run log: $LogPath"

    if ($Model -eq "m1" -or $Model -eq "all") {
        $FeatureArguments = @("src\pitchfusion\features\prepare_m1_features.py")
        if ($Force) { $FeatureArguments += "--force" }
        & $Python @FeatureArguments
        if ($LASTEXITCODE -ne 0) { throw "M1 feature preparation failed with exit code $LASTEXITCODE" }
        $M1Arguments = @("src\pitchfusion\models\run_m1_svm.py")
        if ($Force) { $M1Arguments += "--force" }
        & $Python @M1Arguments
        if ($LASTEXITCODE -ne 0) { throw "M1 failed with exit code $LASTEXITCODE" }
    }

    if ($Model -in @("m3", "m4", "all")) {
        $ProsodyArguments = @("src\pitchfusion\features\prepare_multi_prosody.py")
        if ($Force) { $ProsodyArguments += "--force" }
        & $Python @ProsodyArguments
        if ($LASTEXITCODE -ne 0) { throw "Prosody preparation failed with exit code $LASTEXITCODE" }
    }

    $DeepModels = switch ($Model) {
        "m2" { @("m2_wav2vec2") }
        "m3" { @("m3_pitch_fusion") }
        "m4" { @("m4_multi_prosody") }
        "all" { @("m2_wav2vec2", "m3_pitch_fusion", "m4_multi_prosody") }
        default { @() }
    }
    if ($DeepModels.Count -gt 0) {
        $Arguments = @(
            "src\pitchfusion\models\run_m2_m4.py",
            "--device", $Device,
            "--models"
        ) + $DeepModels
        if ($AllowDownload) { $Arguments += "--allow-download" }
        if ($Force) { $Arguments += "--force" }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Deep-model run failed with exit code $LASTEXITCODE" }
    }
}
finally {
    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
    Pop-Location
}
