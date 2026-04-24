#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi

PROFILE="${1:-debug}"
mkdir -p .ffi/lib

if [[ "$PROFILE" == "release" ]]; then
  cargo build --release
  cp target/release/libai_memory.dylib .ffi/lib/libai_memory.dylib
else
  cargo build
  cp target/debug/libai_memory.dylib .ffi/lib/libai_memory.dylib
fi

echo "Rust FFI library copied to .ffi/lib/libai_memory.dylib"
