
#---

## **2. PowerShell Build Script (`build_project.ps1`)**

#This script will fully automate the build process.

#```powershell
# build_project.ps1
# PowerShell script to build py_network and prepare for testing

Write-Host "=== Cleaning old build ===" -ForegroundColor Yellow
if (Test-Path -Path ".\build") {
    Remove-Item -Recurse -Force ".\build"
}
New-Item -ItemType Directory -Path ".\build" | Out-Null

Write-Host "=== Running CMake configure ===" -ForegroundColor Yellow
cmake -S . -B .\build -A x64 -DCMAKE_PREFIX_PATH="C:/Users/$env:USERNAME/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11/share/cmake/pybind11"

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Building project (Release mode) ===" -ForegroundColor Yellow
cmake --build .\build --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Copying output files to tests folder ===" -ForegroundColor Yellow
$releaseFolder = ".\build\Release"

# Find the .pyd file (Python extension module)
$pydFile = Get-ChildItem -Path $releaseFolder -Filter "py_network*.pyd" | Select-Object -First 1
if ($pydFile) {
    Copy-Item $pydFile.FullName -Destination ".\tests\"
    Write-Host "Copied $($pydFile.Name) to tests folder."
} else {
    Write-Host "ERROR: No .pyd file found!" -ForegroundColor Red
    exit 1
}

# Copy CUDA runtime DLL
$cudaDllPath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/cudart64_12.dll"
if (Test-Path $cudaDllPath) {
    Copy-Item $cudaDllPath -Destination ".\tests\" -Force
    Write-Host "Copied cudart64_12.dll to tests folder."
} else {
    Write-Host "WARNING: cudart64_12.dll not found at expected path!"
}

Write-Host "=== Build complete! ===" -ForegroundColor Green
