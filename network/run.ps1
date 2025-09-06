# run.ps1
# Run from 'network' directory

# Go to build folder
$buildDir = "../build/network"
if (!(Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}
Set-Location -Path $buildDir

# Optional: clean previous build (uncomment if needed)
# Remove-Item -Path * -Recurse -Force -ErrorAction SilentlyContinue

# Configure CMake
cmake ../../network -G "Visual Studio 17 2022"

# Build the project (Release)
cmake --build . --config Release

# Run the executable
Write-Host "`n=== Running neural_network.exe ===`n"
.\Release\neural_network.exe

# Keep console open
Write-Host "`nPress Enter to exit..."
[void][System.Console]::ReadLine()

#return back to network
cd ../../network