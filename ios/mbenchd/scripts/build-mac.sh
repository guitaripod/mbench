#!/usr/bin/env bash
# Builds the app on the Mac, where MLX's Metal kernels can be compiled, and installs it from here over the cable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MAC=${MBENCH_MAC:-mac}
REMOTE=${MBENCH_MAC_DIR:-Dev/mbenchd}

rsync -a --delete --exclude 'xtool/' --exclude '.build/' ./ "$MAC:$REMOTE/"
ssh "$MAC" "cd $REMOTE && bash scripts/pack-mac.sh"
mkdir -p xtool
rsync -a --delete "$MAC:$REMOTE/xtool/mbenchd.app/" xtool/mbenchd.app/

if [[ -z "${LD_LIBRARY_PATH:-}" || ! -e "${LD_LIBRARY_PATH%%:*}/libswiftCore.so" ]]; then
  runtime=$(ls -d "$HOME"/.local/share/swiftly/toolchains/*/usr/lib/swift/linux 2>/dev/null | sort -V | tail -1)
  [[ -n "$runtime" ]] || { echo "build-mac.sh: no Swift runtime; xtool cannot start" >&2; exit 1; }
  export LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
(cd ../.. && mbench phone kill) || true
xtool install xtool/mbenchd.app
(cd ../.. && mbench phone launch)
