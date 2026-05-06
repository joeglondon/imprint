#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."
./scripts/build-rust-ffi.sh
mkdir -p .build/swiftpm-cache .build/clang-module-cache
export CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache"
swift build --disable-sandbox --cache-path "$PWD/.build/swiftpm-cache" --manifest-cache local
