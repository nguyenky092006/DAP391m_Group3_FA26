param(
    [switch]$InstallMiniforge
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$MiniforgeRoot = Join-Path $RepositoryRoot ".miniforge-win"
$KaldiEnvironment = Join-Path $RepositoryRoot ".kaldi-win"
$KaldiExecutable = Join-Path $KaldiEnvironment "Library\bin\compute-kaldi-pitch-feats.exe"

function Find-Conda {
    $Command = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $Command) { return $Command.Source }
    $Candidates = @(
        (Join-Path $MiniforgeRoot "Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniforge3\Scripts\conda.exe"),
        (Join-Path $env:ProgramData "miniforge3\Scripts\conda.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    }
    return $null
}

$Conda = Find-Conda
if ($null -eq $Conda -and $InstallMiniforge) {
    # Miniforge rejects the default user path when the Windows account name
    # contains a space. /D must be the final, unquoted NSIS argument.
    $InstallerArguments = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$MiniforgeRoot"
    & winget install --exact --force --id CondaForge.Miniforge3 --source winget `
        --accept-package-agreements --accept-source-agreements `
        --override $InstallerArguments
    if ($LASTEXITCODE -ne 0) { throw "Miniforge installation failed" }
    $Conda = Find-Conda
}
if ($null -eq $Conda) {
    throw "Conda/Miniforge was not found. Run .\scripts\setup_paper_reproduction.ps1 -InstallMiniforge."
}

Push-Location $RepositoryRoot
try {
    if (-not (Test-Path -LiteralPath $KaldiExecutable)) {
        & $Conda create --yes --prefix $KaldiEnvironment --channel conda-forge `
            --strict-channel-priority "kaldi=5.5.1172=cpu_*"
        if ($LASTEXITCODE -ne 0) { throw "Windows Kaldi environment creation failed" }
    }
    if (-not (Test-Path -LiteralPath $KaldiExecutable)) {
        throw "Kaldi package installed without expected executable: $KaldiExecutable"
    }
    & $KaldiExecutable --help 2>&1 | Select-Object -First 1
    if ($LASTEXITCODE -ne 0) { throw "Kaldi executable validation failed" }
    "Windows Kaldi backend ready: $KaldiExecutable"
}
finally {
    Pop-Location
}
