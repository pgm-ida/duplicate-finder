#!/usr/bin/env pwsh
# Build the DuplicateFinder .exe with tesseract OCR + rembg u2net bundled.
#
# Tesseract is staged into ./_tesseract_bundle then bundled into the .exe via
# PyInstaller's --add-data. The app.py launcher detects sys._MEIPASS and points
# pytesseract at the bundled binary at runtime.
#
# rembg's u2net.onnx weights are staged into ./_rembg_bundle. At runtime,
# find_duplicates_local.py sets U2NET_HOME to the bundled path when frozen so
# rembg uses the embedded model instead of downloading ~176 MB on first launch.

$ErrorActionPreference = "Stop"

# Locate tesseract install (assume scoop layout for now).
$tessSrc = "$env:USERPROFILE\scoop\apps\tesseract\current"
if (-not (Test-Path "$tessSrc\tesseract.exe")) {
    Write-Host "Tesseract not found at $tessSrc"
    Write-Host "Install via:  scoop install tesseract"
    exit 1
}
if (-not (Test-Path "$tessSrc\tessdata\eng.traineddata")) {
    Write-Host "eng.traineddata missing. Download from:"
    Write-Host "  https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata"
    Write-Host "and save to $tessSrc\tessdata\"
    exit 1
}

# Stage a clean tesseract bundle. We copy:
#   - tesseract.exe (the binary the user invokes)
#   - all DLLs (mass-copy is reliable; PyInstaller will only embed what's referenced)
#   - tessdata/eng.traineddata (language model, ~4 MB)
$stage = "_tesseract_bundle"
Write-Host "Staging tesseract bundle at $stage ..."
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item "$tessSrc\tesseract.exe" "$stage\"
Copy-Item "$tessSrc\*.dll" "$stage\"
New-Item -ItemType Directory -Path "$stage\tessdata" | Out-Null
Copy-Item "$tessSrc\tessdata\eng.traineddata" "$stage\tessdata\"
$staged_size = (Get-ChildItem $stage -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Bundle staged: {0:N1} MB" -f $staged_size)

# Stage rembg's u2net.onnx weights. The model is downloaded by rembg on first
# use into $env:USERPROFILE\.u2net\u2net.onnx (~176 MB). Bundling it avoids a
# first-run network dependency in the .exe.
$rembgSrc = "$env:USERPROFILE\.u2net\u2net.onnx"
if (-not (Test-Path $rembgSrc)) {
    Write-Host "u2net.onnx not found at $rembgSrc"
    Write-Host "Trigger the first-run download with:"
    Write-Host '  python -c "from rembg import new_session; new_session(''u2net'')"'
    exit 1
}
$rembgStage = "_rembg_bundle"
Write-Host "Staging rembg u2net bundle at $rembgStage ..."
if (Test-Path $rembgStage) { Remove-Item -Recurse -Force $rembgStage }
New-Item -ItemType Directory -Path $rembgStage | Out-Null
Copy-Item $rembgSrc "$rembgStage\"
$rembg_size = (Get-ChildItem $rembgStage -Recurse | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("rembg bundle staged: {0:N1} MB" -f $rembg_size)

# Regenerate icon.ico from make_icon.py so a designer edit there flows
# into the build without a separate manual step.
Write-Host "Generating icon.ico ..."
& python make_icon.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Icon generation failed."
    exit $LASTEXITCODE
}

# Clean previous build artifacts (.exe may be locked if app is running).
foreach ($d in "build", "DuplicateFinder.spec") {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue }
}
if (Test-Path "dist\DuplicateFinder.exe") {
    try { Remove-Item -Force "dist\DuplicateFinder.exe" } catch {
        Write-Host "(existing exe locked — will overwrite)"
    }
}

# Build.
Write-Host "Running PyInstaller ..."
$args = @(
    "--onefile", "--windowed",
    "--name", "DuplicateFinder",
    "--icon", "icon.ico",
    "--add-data", "app_index.html;.",
    "--add-data", "icon.ico;.",
    "--add-data", "${stage};tesseract",
    "--add-data", "${rembgStage};u2net",
    "--collect-all", "webview",
    "--collect-all", "cv2",
    "--collect-all", "watchdog",
    "--collect-all", "pynput",
    "--collect-all", "rembg",
    "--collect-all", "onnxruntime",
    "--collect-all", "pymatting",
    "--copy-metadata", "rembg",
    "--copy-metadata", "pymatting",
    "--copy-metadata", "pooch",
    "--copy-metadata", "tqdm",
    "--copy-metadata", "onnxruntime",
    "--hidden-import", "find_duplicates_local",
    "--hidden-import", "tether",
    "--hidden-import", "pytesseract",
    "--hidden-import", "cv2",
    "--hidden-import", "watchdog",
    "--hidden-import", "pynput",
    "--hidden-import", "rembg",
    "--hidden-import", "onnxruntime",
    "app.py"
)
& pyinstaller @args
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller failed."
    exit $LASTEXITCODE
}

if (Test-Path "dist\DuplicateFinder.exe") {
    $exe_size = (Get-Item "dist\DuplicateFinder.exe").Length / 1MB
    Write-Host ""
    Write-Host ("Built dist\DuplicateFinder.exe ({0:N1} MB)" -f $exe_size)
} else {
    Write-Host "Build appears to have failed — no exe produced."
    exit 1
}
