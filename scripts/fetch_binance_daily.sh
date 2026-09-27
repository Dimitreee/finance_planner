#!/usr/bin/env bash
# Fetch the daily kline archives for one month, with their published checksums.
#
# Used to reconcile the daily archives against the monthly one for a month that both cover; see
# reports/data_audit_market_seam.md.
#
#   ./scripts/fetch_binance_daily.sh 2026-08
set -euo pipefail

MONTH="${1:?usage: fetch_binance_daily.sh YYYY-MM}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/raw/binance/klines/BTCUSDT/1h/daily/${MONTH}"
BASE="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h"

mkdir -p "$DEST"
cd "$DEST"

for day in $(seq -w 1 31); do
  name="BTCUSDT-1h-${MONTH}-${day}.zip"
  # Months are short; a missing day simply does not exist and is skipped.
  if [ ! -f "$name" ]; then
    curl -fsSL -o "${name}.part" "${BASE}/${name}" && mv "${name}.part" "$name" || {
      rm -f "${name}.part"; continue;
    }
  fi
  # The same .part dance as the archive: curl -f does not remove a partially written output file,
  # so a transient failure would leave an empty .CHECKSUM that the guard above never repairs and
  # that poisons the verification below with no hint of the cause.
  if [ ! -f "${name}.CHECKSUM" ]; then
    curl -fsSL -o "${name}.CHECKSUM.part" "${BASE}/${name}.CHECKSUM" \
      && mv "${name}.CHECKSUM.part" "${name}.CHECKSUM" \
      || rm -f "${name}.CHECKSUM.part"
  fi
done

shopt -s nullglob
checksums=(*.CHECKSUM)
archives=(*.zip)
if [ ${#checksums[@]} -eq 0 ]; then
  echo "nothing downloaded for ${MONTH}; is that a month Binance publishes?" >&2
  exit 1
fi
if [ ${#checksums[@]} -ne ${#archives[@]} ]; then
  echo "${#archives[@]} archives but ${#checksums[@]} checksums: some archive is unverifiable" >&2
  exit 1
fi

echo "verifying ${#checksums[@]} archives"
shasum -a 256 -c "${checksums[@]}"
