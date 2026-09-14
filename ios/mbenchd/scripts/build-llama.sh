#!/usr/bin/env bash
# Cross-compiles llama.cpp for arm64 iOS on Linux, using the Darwin Swift SDK's iPhoneOS sysroot and the Swift
# toolchain's clang. Metal kernels are embedded as source and compiled by the device at first launch, which is what
# makes a Metal build possible without Xcode's metal compiler.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENDOR=$HERE/Vendor
BUILD=${BUILD_DIR:-$HOME/.cache/mbench/ios-llama}

BUNDLE=$HOME/.swiftpm/swift-sdks/darwin.artifactbundle
export MBENCH_IOS_SDKROOT=${MBENCH_IOS_SDKROOT:-$(ls -d "$BUNDLE"/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS*.sdk | tail -1)}
export MBENCH_IOS_TOOLBIN=${MBENCH_IOS_TOOLBIN:-$(ls -d "$HOME"/.local/share/swiftly/toolchains/*/usr/bin | tail -1)}
export MBENCH_IOS_MIN=${MBENCH_IOS_MIN:-18.0}

cmake -S "$LLAMA_DIR" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/scripts/ios-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_ACCELERATE=ON \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_SUBPROCESS=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DHOST_CXX_COMPILER=/usr/bin/g++

cmake --build "$BUILD" -j "$(nproc)" --target llama-server-impl

rm -rf "$VENDOR"
mkdir -p "$VENDOR/lib" "$VENDOR/include"
find "$BUILD" -name 'lib*.a' -exec cp {} "$VENDOR/lib/" \;
cp "$LLAMA_DIR"/include/llama.h "$VENDOR/include/"
cp "$LLAMA_DIR"/common/common.h "$VENDOR/include/" 2>/dev/null || true
cp "$LLAMA_DIR"/ggml/include/*.h "$VENDOR/include/"
echo "$(cd "$LLAMA_DIR" && git rev-parse --short HEAD)" > "$VENDOR/llama-commit.txt"
ls -la "$VENDOR/lib"
