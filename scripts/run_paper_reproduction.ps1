param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$SplitOnly,
    [switch]$PrepareOnly,
    [switch]$PreflightOnly,
    [switch]$AllowDownload,
    [switch]$ForceSplit,
    [switch]$ForcePitch,
    [switch]$ForceTraining
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv-gpu\Scripts\python.exe"
$KaldiExecutable = Join-Path $RepositoryRoot ".kaldi-win\Library\bin\compute-kaldi-pitch-feats.exe"
$LogDirectory = Join-Path $RepositoryRoot "logs\paper_reproduction"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path $LogDirectory ("run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

if (-not (Test-Path -LiteralPath $Python)) {
    throw "GPU environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $KaldiExecutable)) {
    throw "Windows Kaldi backend not found. Run .\scripts\setup_paper_reproduction.ps1 -InstallMiniforge first."
}

$Arguments = @(
    "src\pitchfusion\models\run_paper_reproduction.py",
    "--device", $Device,
    "--kaldi-executable", $KaldiExecutable
)
if ($SplitOnly) { $Arguments += "--split-only" }
if ($PrepareOnly) { $Arguments += "--prepare-only" }
if ($PreflightOnly) { $Arguments += "--preflight-only" }
if ($AllowDownload) { $Arguments += "--allow-download" }
if ($ForceSplit) { $Arguments += "--force-split" }
if ($ForcePitch) { $Arguments += "--force-pitch" }
if ($ForceTraining) { $Arguments += "--force-training" }

Push-Location $RepositoryRoot
try {
    "Run log: $LogPath"
    & $Python @Arguments 2>&1 | Tee-Object -LiteralPath $LogPath
    $RunExitCode = $LASTEXITCODE
    if ($RunExitCode -ne 0) {
        throw "Paper reproduction failed with exit code $RunExitCode. See $LogPath"
    }
}
finally {
    Pop-Location
}
