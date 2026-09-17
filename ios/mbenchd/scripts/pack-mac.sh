#!/usr/bin/env bash
# Packs the .app on macOS. xtool looks for resource bundles under .build/<triple>/<config>, which the Xcode build
# system does not write, and it expects a bundle for every target that declares resources even when none was built.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

XTOOL=${XTOOL:-/opt/homebrew/bin/xtool}
PRODUCTS=.build/out/Products/Release-iphoneos

mkdir -p .build/arm64-apple-ios
ln -sfn ../out/Products/Release-iphoneos .build/arm64-apple-ios/release

for attempt in $(seq 1 15); do
  said=$(MBENCHD_MLX=1 "$XTOOL" dev build -c release 2>&1 | tail -4)
  missing=$(printf '%s' "$said" | grep -o '[A-Za-z0-9_.-]*\.bundle' | head -1 || true)
  if [[ -z "$missing" ]]; then
    cp -R "$PRODUCTS/mlx-swift_Cmlx.bundle" xtool/mbenchd.app/
    printf '%s\n' "$said"
    exit 0
  fi
  mkdir -p "$PRODUCTS/$missing"
  python3 - "$missing" "$PRODUCTS/$missing" <<'PY'
import pathlib, plistlib, sys
name = sys.argv[1].removesuffix(".bundle")
plistlib.dump({"CFBundleDevelopmentRegion": "en", "CFBundleIdentifier": name.replace("_", ".") + ".resources",
               "CFBundleInfoDictionaryVersion": "6.0", "CFBundleName": name, "CFBundlePackageType": "BNDL",
               "CFBundleSupportedPlatforms": ["iPhoneOS"], "MinimumOSVersion": "15.0"},
              open(pathlib.Path(sys.argv[2]) / "Info.plist", "wb"))
PY
done
echo "pack-mac.sh: xtool still cannot find its resource bundles" >&2
exit 1
