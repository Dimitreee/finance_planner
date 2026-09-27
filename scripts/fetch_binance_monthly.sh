#!/usr/bin/env bash
# Re-fetch the monthly Binance spot kline archives that data/raw/ holds.
#
# data/raw/ is gitignored, so this script plus data/manifests/*.sha256 is the
# reproducible path: fetch, then verify against the committed manifest.
#
#   ./scripts/fetch_binance_monthly.sh                  # BTCUSDT 1h
#   ./scripts/fetch_binance_monthly.sh ETHUSDT 1d
#
# Verify afterwards:
#   shasum -c data/manifests/binance_BTCUSDT_1h_monthly.sha256   # run from the archive dir
set -euo pipefail

SYMBOL="${1:-BTCUSDT}"
INTERVAL="${2:-1h}"
PREFIX="data/spot/monthly/klines/${SYMBOL}/${INTERVAL}/"
BUCKET="https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE="https://data.binance.vision/${PREFIX}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/binance/klines/${SYMBOL}/${INTERVAL}"

mkdir -p "$DEST"

# The bucket listing caps at 1000 keys per request; page with a marker so this
# stays correct as history grows.
list_keys() {
  local marker="" body
  while :; do
    body="$(curl -sS --fail --max-time 60 "${BUCKET}?delimiter=/&prefix=${PREFIX}&marker=${marker}")"
    printf '%s' "$body" | grep -o '<Key>[^<]*</Key>' | sed 's|<Key>'"${PREFIX}"'||; s|</Key>||' | grep '\.zip$' || true
    if ! printf '%s' "$body" | grep -q '<IsTruncated>true</IsTruncated>'; then
      break
    fi
    marker="$(printf '%s' "$body" | grep -o '<Key>[^<]*</Key>' | tail -1 | sed 's|<Key>||; s|</Key>||')"
    [ -n "$marker" ] || break
  done
}

echo "Listing ${SYMBOL} ${INTERVAL} monthly archives..."
mapfile -t MONTHS < <(list_keys | sort -u)
echo "Found ${#MONTHS[@]} archives. Downloading into ${DEST}"

printf '%s\n' "${MONTHS[@]}" \
  | awk -v b="$BASE" '{print b $0; print b $0 ".CHECKSUM"}' \
  | (cd "$DEST" && xargs -P 8 -n 1 curl -sS --fail --max-time 120 -O)

echo "Verifying published checksums..."
fail=0
for z in "$DEST"/*.zip; do
  expected="$(awk '{print $1; exit}' "${z}.CHECKSUM")"
  actual="$(shasum -a 256 "$z" | awk '{print $1}')"
  if [ "$expected" != "$actual" ]; then
    echo "  MISMATCH $(basename "$z")"
    fail=$((fail + 1))
  fi
done

total=$(ls -1 "$DEST"/*.zip | wc -l | tr -d ' ')
if [ "$fail" -ne 0 ]; then
  echo "FAILED: ${fail} of ${total} archives do not match their published .CHECKSUM"
  exit 1
fi
echo "OK: ${total}/${total} archives match their published .CHECKSUM"
echo
echo "Next, compare against the committed snapshot manifest:"
echo "  (cd '${DEST}' && shasum -c \"\$(git rev-parse --show-toplevel)/data/manifests/binance_${SYMBOL}_${INTERVAL}_monthly.sha256\")"
