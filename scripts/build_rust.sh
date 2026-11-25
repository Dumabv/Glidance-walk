#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/rust_core"

echo "==> Building native Rust core (release)..."
if [[ "$(uname -s)" == "Darwin" ]]; then
  RUSTFLAGS="-C target-cpu=native" cargo build --release
  cp target/release/libglidance_core.dylib "$ROOT_DIR/backend/vision/libglidance_core.dylib"
  echo "OK: copied backend/vision/libglidance_core.dylib"
else
  RUSTFLAGS="-C target-cpu=native" cargo build --release
  cp target/release/libglidance_core.so "$ROOT_DIR/backend/vision/libglidance_core.so"
  echo "OK: copied backend/vision/libglidance_core.so"
fi

if rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
  echo "==> Building WebAssembly SIMD128 core..."
  cargo build --target wasm32-unknown-unknown --release
  cp target/wasm32-unknown-unknown/release/glidance_core.wasm "$ROOT_DIR/frontend/public/glidance_core.wasm"
  echo "OK: copied frontend/public/glidance_core.wasm"
else
  echo "NOTE: wasm32-unknown-unknown target not installed (using prebuilt frontend/public/glidance_core.wasm)."
fi
