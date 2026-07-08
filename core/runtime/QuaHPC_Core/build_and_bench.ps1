# =============================================================================
# build_and_bench.ps1
# -----------------------------------------------------------------------------
# Windows / PowerShell: build libqhpc + Python module, then run the
# head-to-head VQE benchmark against PennyLane-Lightning-GPU.
#
# Usage (in a Developer PowerShell for VS, or plain PowerShell w/ MSVC on PATH):
#
#   .\build_and_bench.ps1                    # default: sm_89, n=12, layers=4
#   .\build_and_bench.ps1 -SmArch 90         # Hopper
#   .\build_and_bench.ps1 -N 14 -Layers 6 -Epochs 500
#   .\build_and_bench.ps1 -SkipBuild         # rerun bench only
#   .\build_and_bench.ps1 -SkipPennylane     # measure QHPC alone
#
# Requirements:
#   - Visual Studio 2022 Build Tools (cl.exe on PATH or run from VS Dev shell)
#   - CUDA Toolkit 12.x  (nvcc on PATH; cuSOLVER, NVRTC come with it)
#   - CMake >= 3.20
#   - Python 3.10+ with: pip install pybind11 numpy pennylane pennylane-lightning-gpu
# =============================================================================
[CmdletBinding()]
param(
    [string]$SmArch       = "89",
    [int]   $N            = 12,
    [int]   $Layers       = 4,
    [int]   $Epochs       = 200,
    [double]$Lr           = 0.05,
    [int]   $Seed         = 42,
    [switch]$SkipBuild,
    [switch]$SkipPennylane,
    [switch]$SkipQhpc,
    [string]$Config       = "Release",
    [string]$Generator    = "Visual Studio 17 2022"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Build = Join-Path $Root "build"

function Section($msg){
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Require-Tool {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Exe
    )
    $cmd = Get-Command $Exe -ErrorAction SilentlyContinue
    if(-not $cmd){
        Write-Error "$Name not found on PATH. Install / add to PATH and retry."
    }
    Write-Host "[ok] $Name ($($cmd.Source))" -ForegroundColor Green
}

# ------------------------------------------------------------------ VS env
# VS Build Tools 2022 does NOT ship Launch-VsDevShell.ps1 (only VS
# Community/Pro/Enterprise does). Use VsDevCmd.bat which IS shipped, and
# import its env vars into the current PowerShell session.
function Load-VsBuildToolsEnv {
    if(Get-Command cl.exe -ErrorAction SilentlyContinue){
        Write-Host "[ok] VS env already loaded (cl.exe on PATH)" -ForegroundColor Green
        return
    }
    # Discover via vswhere first.
    $vsRoot = $null
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if(Test-Path $vswhere){
        $vsRoot = & $vswhere -latest -products "*" -property installationPath
    }
    # Fallback: brute-search the two canonical roots.
    if(-not $vsRoot){
        foreach($probe in @(
            "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
            "C:\Program Files\Microsoft Visual Studio\2022\Community",
            "C:\Program Files\Microsoft Visual Studio\2022\Professional",
            "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
        )){
            if(Test-Path (Join-Path $probe "Common7\Tools\VsDevCmd.bat")){
                $vsRoot = $probe; break
            }
        }
    }
    if(-not $vsRoot){
        Write-Error "Visual Studio 2022 (Build Tools / Community / Pro / Enterprise) not found."
    }
    $devcmd = Join-Path $vsRoot "Common7\Tools\VsDevCmd.bat"
    if(-not (Test-Path $devcmd)){
        Write-Error "VsDevCmd.bat not found at $devcmd"
    }
    Write-Host "[load] VS env from: $devcmd" -ForegroundColor Cyan
    cmd /c "`"$devcmd`" -arch=amd64 -host_arch=amd64 -no_logo && set" | ForEach-Object {
        if($_ -match "^([^=]+)=(.*)$"){
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
    if(-not (Get-Command cl.exe -ErrorAction SilentlyContinue)){
        Write-Error "Loaded VsDevCmd.bat but cl.exe still not on PATH. Re-run installer with C++ workload."
    }
    Write-Host "[ok] cl.exe: $((Get-Command cl.exe).Source)" -ForegroundColor Green
}

# ------------------------------------------------------------------ Preflight
Section "1. Preflight"

Require-Tool "cmake" "cmake"
Require-Tool "nvcc"  "nvcc"
Require-Tool "python" "python"

# Auto-load VS Build Tools env into this session.
Load-VsBuildToolsEnv

# pybind11 cmake path
$pybindCmakeDir = $null

# Check whether the active python is the Microsoft Store sandbox build, which
# has well-known PATH issues for --user installed scripts and stderr noise
# that breaks PowerShell pipelines.
$pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if($pyExe -and $pyExe -like "*WindowsApps*"){
    Write-Warning "Detected Microsoft Store Python at: $pyExe"
    Write-Warning "This build has PATH issues with --user packages."
    Write-Warning "Recommended: install python.org distribution and disable Store alias."
    Write-Warning "Quick fix:  winget install Python.Python.3.11   then reopen PowerShell."
}

function Try-PyImport($mod){
    # Bullet-proof against PowerShell's $ErrorActionPreference='Stop' + Python's
    # traceback-to-stderr behavior. We isolate the call inside a try block AND
    # locally override ErrorActionPreference so a non-zero exit from python
    # never poisons the parent script.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & python -c "import $mod" 2>&1
        $ok = ($LASTEXITCODE -eq 0)
    } catch {
        $ok = $false
    } finally {
        $ErrorActionPreference = $prev
    }
    return $ok
}

function Safe-PipInstall {
    param([string[]]$Packages)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python -m pip install --user --no-warn-script-location @Packages 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $prev
    }
}

try {
    if(Try-PyImport "pybind11"){
        $pybindCmakeDir = (& python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>&1 | Select-Object -Last 1)
    }
} catch { }

if(-not $pybindCmakeDir){
    Write-Host "[..] Installing pybind11..." -ForegroundColor Yellow
    Safe-PipInstall -Packages @("pybind11") | Out-Null
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        $pybindCmakeDir = (& python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>&1 | Select-Object -Last 1)
    } catch { }
    $ErrorActionPreference = $prev
}
if($pybindCmakeDir -and (Test-Path $pybindCmakeDir)){
    Write-Host "[ok] pybind11 cmake dir: $pybindCmakeDir" -ForegroundColor Green
} else {
    Write-Error "pybind11 still not importable. If using Store Python, switch to python.org."
}

if(-not (Try-PyImport "numpy")){
    Write-Warning "numpy missing. Installing..."
    Safe-PipInstall -Packages @("numpy") | Out-Null
}
if(-not $SkipPennylane){
    if(-not (Try-PyImport "pennylane")){
        Write-Warning "pennylane missing. Installing..."
        Safe-PipInstall -Packages @("pennylane","pennylane-lightning-gpu") | Out-Null
    }
    if(-not (Try-PyImport "pennylane_lightning_gpu")){
        Write-Warning "lightning.gpu unavailable; bench will fall back to lightning.qubit (CPU)."
    }
}

# ------------------------------------------------------------------ Build
if(-not $SkipBuild){
    Section "2. CMake configure (SM_$SmArch, $Generator)"

    if(-not (Test-Path $Build)){ New-Item -ItemType Directory -Path $Build | Out-Null }

    $cmakeArgs = @(
        "-S", $Root,
        "-B", $Build,
        "-G", $Generator,
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DQHPC_SM_ARCH=$SmArch",
        "-DQHPC_BUILD_PYTHON=ON",
        "-Dpybind11_DIR=$pybindCmakeDir"
    )
    & cmake @cmakeArgs
    if($LASTEXITCODE -ne 0){ throw "cmake configure failed" }

    Section "3. CMake build ($Config)"
    & cmake --build $Build --config $Config --parallel
    if($LASTEXITCODE -ne 0){ throw "build failed" }

    # Find the produced Python module (.pyd on Windows)
    $pyd = Get-ChildItem -Path $Build -Recurse -Filter "qhpc*.pyd" | Select-Object -First 1
    if(-not $pyd){
        $pyd = Get-ChildItem -Path $Build -Recurse -Filter "qhpc*.dll" |
               Where-Object { $_.Name -notlike "qhpc_core*" } | Select-Object -First 1
    }
    if(-not $pyd){
        throw "Could not find built Python module under $Build. Look for qhpc*.pyd."
    }
    Copy-Item $pyd.FullName (Join-Path $Root "qhpc.pyd") -Force
    Write-Host "[ok] Python module staged: $($pyd.FullName) -> $Root\qhpc.pyd" -ForegroundColor Green

    # Also copy the core .dll next to it (Windows DLL search path needs this)
    $core = Get-ChildItem -Path $Build -Recurse -Filter "qhpc_core*.dll" |
            Select-Object -First 1
    if($core){
        Copy-Item $core.FullName $Root -Force
        Write-Host "[ok] Core DLL staged: $($core.Name) -> $Root\$($core.Name)" -ForegroundColor Green
    } else {
        Write-Warning "qhpc_core.dll not found in build tree; module import may fail at runtime."
    }
} else {
    Section "2-3. Build skipped (-SkipBuild)"
}

# ------------------------------------------------------------------ Bench
Section "4. Benchmark"

$benchArgs = @(
    "--n",      $N,
    "--layers", $Layers,
    "--epochs", $Epochs,
    "--lr",     $Lr,
    "--seed",   $Seed
)
if($SkipPennylane){ $benchArgs += "--skip_pl"   }
if($SkipQhpc)     { $benchArgs += "--skip_qhpc" }

$env:PYTHONPATH = "$Root;$($env:PYTHONPATH)"
# On Windows the loader must see qhpc_core.dll next to qhpc.pyd; we already
# staged both into $Root. Also add CUDA bin (cuSOLVER, NVRTC runtime DLLs).
$cudaBin = $null
if($env:CUDA_PATH){ $cudaBin = Join-Path $env:CUDA_PATH "bin" }
if($cudaBin -and (Test-Path $cudaBin)){
    $env:PATH = "$cudaBin;$($env:PATH)"
    Write-Host "[ok] CUDA bin on PATH: $cudaBin" -ForegroundColor Green
}

Push-Location $Root
try {
    & python "bench_vs_pennylane.py" @benchArgs
    if($LASTEXITCODE -ne 0){ throw "bench script returned non-zero" }
} finally {
    Pop-Location
}

Section "Done"
Write-Host "Artifacts:" -ForegroundColor Green
Write-Host "  $Root\qhpc.pyd     (Python module)"
Write-Host "  $Root\qhpc.dll     (core, if built)"
Write-Host "  $Root\build\       (CMake out tree)"
