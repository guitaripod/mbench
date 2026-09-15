#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
page="${1:-}"
if [[ -z "$page" ]]; then
  page="site/index.html"
  [[ -f "$page" ]] || page="${MBENCH_HOME:-$HOME/.local/share/mbench}/leaderboard.html"
fi
[[ -f "$page" ]] || { echo "screenshots.sh: no leaderboard at $page" >&2; exit 1; }
browser=$(command -v chromium || command -v google-chrome-stable || command -v google-chrome \
  || ls -d "$HOME"/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell 2>/dev/null | tail -1 \
  || true)
[[ -n "$browser" && -x "$browser" ]] || { echo "screenshots.sh: no Chromium found" >&2; exit 1; }
mkdir -p docs
common=(--headless --no-sandbox --disable-gpu --hide-scrollbars --virtual-time-budget=5000 --force-device-scale-factor=2 --window-size=1500,2400)
"$browser" "${common[@]}" --screenshot=docs/leaderboard.png "file://$PWD/$page" 2>/dev/null
"$browser" "${common[@]}" --force-dark-mode --blink-settings=preferredColorScheme=0 --screenshot=docs/leaderboard-dark.png "file://$PWD/$page" 2>/dev/null
"$browser" "${common[@]}" --screenshot=docs/phone.png "file://$PWD/$page#phone" 2>/dev/null
out="docs/leaderboard.png docs/leaderboard-dark.png docs/phone.png"
card="$(dirname "$page")/card.html"
if [[ -f "$card" ]]; then
  "$browser" --headless --no-sandbox --disable-gpu --hide-scrollbars --virtual-time-budget=8000 \
    --force-device-scale-factor=2 --window-size=1200,675 --screenshot=docs/card.png "file://$PWD/$card?bare=1" 2>/dev/null
  out="$out docs/card.png"
fi
echo "$out"
