#!/bin/bash
set -e 

# Setup LLVM tools for Flatpak environment
export PATH="/usr/bin:$PATH"

# If tools aren't found in PATH, use flatpak-spawn to access host system
if ! command -v llc; then
    echo "Using flatpak-spawn to access host LLVM tools..."
    LLC="flatpak-spawn --host llc"
    CLANG="flatpak-spawn --host clang"
else
    echo "Found LLVM tools in PATH, using directly..."
    LLC="llc"
    CLANG="clang"
fi

# Paths
IR_FILE="llvm/strassen_2x2.ll"
BUILD_DIR="build"
OBJ_FILE="$BUILD_DIR/strassen_2x2.o"
DRIVER="strassen_driver.c"
REFERENCE="reference_driver.c"
EXEC="$BUILD_DIR/strassen_driver"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

echo "Generating object file from LLVM IR..."
$LLC -filetype=obj "$IR_FILE" -o "$OBJ_FILE"

echo "Compiling C driver with LLVM object..."
$CLANG -O2 -march=native "$DRIVER" "$REFERENCE" "$OBJ_FILE" -lm -o "$EXEC"

echo "Setting executable permission..."
chmod +x "$EXEC"

echo "Build complete. Run with: $EXEC"
echo "Example usage:"
echo "  $EXEC -n 64 -print"
echo "  $EXEC -n 128 -iterations 10"
echo "  $EXEC -help"