$ErrorActionPreference = 'Stop'
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location (Join-Path $RootDir "rust_core")

Write-Host "==> Building native Rust AVX2+FMA DLL (release)..."
cargo build --release
Copy-Item "target\release\glidance_core.dll" (Join-Path $RootDir "backend\vision\glidance_core.dll") -Force
Write-Host "OK: copied backend\vision\glidance_core.dll"

if ((rustup target list --installed) -contains "wasm32-unknown-unknown") {
    Write-Host "==> Building WebAssembly SIMD128 core..."
    cargo build --target wasm32-unknown-unknown --release
    Copy-Item "target\wasm32-unknown-unknown\release\glidance_core.wasm" (Join-Path $RootDir "frontend\public\glidance_core.wasm") -Force
    Write-Host "OK: copied frontend\public\glidance_core.wasm"
} else {
    Write-Host "NOTE: wasm32-unknown-unknown not installed (using prebuilt frontend\public\glidance_core.wasm)."
}
Pop-Location
