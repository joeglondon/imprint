#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."
./scripts/build-rust-ffi.sh
swift build

