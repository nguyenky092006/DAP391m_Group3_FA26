$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$gpuEnvironment = Join-Path $repositoryRoot ".venv-gpu"
$gpuPython = Join-Path $gpuEnvironment "Scripts\python.exe"
$requirements = Join-Path $repositoryRoot "requirements.txt"

Write-Output "[1/4] Checking NVIDIA GPU and driver"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Install or repair the NVIDIA driver first."
}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

Write-Output "[2/4] Creating isolated Python 3.14 GPU environment"
if (-not (Test-Path -LiteralPath $gpuPython)) {
    py -3.14 -m venv $gpuEnvironment
}
& $gpuPython -m pip install --upgrade pip

Write-Output "[3/4] Installing CUDA-enabled PyTorch and B6 dependencies"
& $gpuPython -m pip install -r $requirements

Write-Output "[4/4] Verifying CUDA from the new environment"
& $gpuPython -c "import json, torch; result={'status':'PASS' if torch.cuda.is_available() else 'BLOCKED','torch':torch.__version__,'cuda_build':torch.version.cuda,'device_count':torch.cuda.device_count(),'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}; print(json.dumps(result, indent=2)); raise SystemExit(0 if result['status']=='PASS' else 1)"

Write-Output "GPU environment ready: $gpuPython"
