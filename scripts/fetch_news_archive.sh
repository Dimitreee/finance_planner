#!/usr/bin/env bash
# Fetch and verify the CryptoPanic news archive.
#
# The archive is not tracked in git. This script plus data/manifests/news_cryptopanic.sha256 are the
# reproducible path: it re-fetches from the published repository and checks both the archive and the
# extracted CSV against the recorded digests. A mismatch is a finding, not something to work around.
#
#   ./scripts/fetch_news_archive.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/data/raw/news"
URL="https://github.com/soheilrahsaz/cryptoNewsDataset/raw/main/csvOutput/news_currencies_source_joinedResult.rar"
MANIFEST="${REPO_ROOT}/data/manifests/news_cryptopanic.sha256"

mkdir -p "$DEST"
cd "$DEST"

# Download to a .part file and move it into place only on success: curl leaves the output file
# behind on an interrupted transfer, and a truncated archive would be mistaken for a finished one on
# every later run.
if [ ! -f news_currencies_source_joinedResult.rar ]; then
  echo "fetching $URL"
  trap 'rm -f "${DEST}/news_currencies_source_joinedResult.rar.part"' EXIT
  curl -L --fail -o news_currencies_source_joinedResult.rar.part "$URL"
  mv news_currencies_source_joinedResult.rar.part news_currencies_source_joinedResult.rar
  trap - EXIT
fi

if [ ! -f news_currencies_source_joinedResult.csv ]; then
  echo "extracting"
  unrar x -o+ news_currencies_source_joinedResult.rar
fi

echo "verifying against $MANIFEST"
shasum -a 256 -c "$MANIFEST"
