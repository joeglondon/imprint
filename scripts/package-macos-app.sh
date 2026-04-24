#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

APP_DIR="$PWD/dist/imprint.app"
BIN_DIR="$APP_DIR/Contents/MacOS"
RES_DIR="$APP_DIR/Contents/Resources"
FRAMEWORKS_DIR="$APP_DIR/Contents/Frameworks"

rm -rf "$APP_DIR"
mkdir -p "$BIN_DIR" "$RES_DIR" "$FRAMEWORKS_DIR"
cp macos/Info.plist "$APP_DIR/Contents/Info.plist"
cp macos/AppIcon.icns "$RES_DIR/AppIcon.icns"
cp .build/debug/MemoryApp "$BIN_DIR/MemoryApp"
cp .ffi/lib/libai_memory.dylib "$FRAMEWORKS_DIR/libai_memory.dylib"
chmod +x "$BIN_DIR/MemoryApp"
chmod +x "$FRAMEWORKS_DIR/libai_memory.dylib"

install_name_tool \
  -change "$PWD/target/debug/deps/libai_memory.dylib" \
  "@executable_path/../Frameworks/libai_memory.dylib" \
  "$BIN_DIR/MemoryApp"

codesign --force --deep --sign - "$APP_DIR"

echo "Packaged $APP_DIR"
