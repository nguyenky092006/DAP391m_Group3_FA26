param([switch]$Force)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv-gpu\Scripts\python.exe"
$LogDirectory = Join-Path $RepositoryRoot "logs\foundation"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path $LogDirectory ("b1_b5_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python" }

Push-Location $RepositoryRoot
try {
    Start-Transcript -LiteralPath $LogPath -Force | Out-Null
    Write-Output "B1-B5 log: $LogPath"
    & $Python "src\pitchfusion\prepare_foundation.py"
    if ($LASTEXITCODE -ne 0) { throw "B1-B5 pipeline failed with exit code $LASTEXITCODE" }
}
finally {
    try { Stop-Transcript | Out-Null } catch { }
    Pop-Location
}
