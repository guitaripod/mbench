#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${LD_LIBRARY_PATH:-}" || ! -e "${LD_LIBRARY_PATH%%:*}/libswiftCore.so" ]]; then
  runtime=$(ls -d "$HOME"/.local/share/swiftly/toolchains/*/usr/lib/swift/linux 2>/dev/null | sort -V | tail -1)
  [[ -n "$runtime" ]] || { echo "install.sh: no Swift runtime; xtool cannot start" >&2; exit 1; }
  export LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ! pymobiledevice3 usbmux list 2>/dev/null | grep -q Identifier; then
  echo "install.sh: no phone on the cable — plug it in, unlock it, and trust this computer" >&2
  exit 1
fi

xtool dev "$@"
want=$(python3 -c "
import plistlib, pathlib
print(plistlib.loads(pathlib.Path('xtool/mbenchd.app/Info.plist').read_bytes())['CFBundleDisplayName'])
")
got=$(cd ../.. && mbench phone installed)
[[ "$got" == "$want "* ]] || { echo "install.sh: the phone reports '$got', not $want" >&2; exit 1; }
echo "on the phone: $got"
