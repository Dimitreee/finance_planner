#!/usr/bin/env bash
# A throwaway PostgreSQL for the seam tests, without Docker.
#
# The job and API seam tests drop their tables, so they need a database nobody cares about. Docker is
# the documented route (`make db`), but a local PostgreSQL install works and is faster to start. This
# script keeps its cluster inside a temporary directory and never touches a configured database.
#
# Two things are deliberate. The cluster listens on 55432, not 5432, so it cannot be mistaken for the
# one `CRYPTOGUARD_DATABASE_URL` points at. And Unix sockets are disabled: the data directory lives
# under a long temporary path, and a socket path over 103 bytes is a startup failure on macOS.
#
#   ./scripts/dev_postgres.sh start   # prints the URL to export
#   ./scripts/dev_postgres.sh stop
#   ./scripts/dev_postgres.sh url
set -euo pipefail

PORT="${CRYPTOGUARD_DEV_PG_PORT:-55432}"
STATE="${CRYPTOGUARD_DEV_PG_DIR:-${TMPDIR:-/tmp}/cryptoguard-dev-pg}"
DATA="$STATE/data"
LOG="$STATE/postgres.log"
URL="postgresql://cryptoguard:cryptoguard@127.0.0.1:$PORT/cryptoguard"

find_binary() {
  if command -v "$1" >/dev/null 2>&1; then command -v "$1"; return; fi
  for candidate in /opt/homebrew/bin /usr/local/bin /usr/lib/postgresql/*/bin; do
    [ -x "$candidate/$1" ] && { echo "$candidate/$1"; return; }
  done
  echo "no $1 on PATH and none in the usual places; install PostgreSQL or use 'make db'" >&2
  exit 1
}

case "${1:-}" in
  start)
    INITDB="$(find_binary initdb)"
    PG_CTL="$(find_binary pg_ctl)"
    PSQL="$(find_binary psql)"
    if [ ! -d "$DATA" ]; then
      mkdir -p "$STATE"
      "$INITDB" -D "$DATA" -U postgres --auth=trust >/dev/null
    fi
    if "$PG_CTL" -D "$DATA" status >/dev/null 2>&1; then
      echo "already running on $PORT" >&2
    else
      "$PG_CTL" -D "$DATA" -l "$LOG" \
        -o "-p $PORT -c listen_addresses=127.0.0.1 -c unix_socket_directories=''" start >/dev/null
      # `pg_ctl start` returns before the server accepts connections on a cold cluster.
      for _ in $(seq 1 30); do
        "$PSQL" -h 127.0.0.1 -p "$PORT" -U postgres -d postgres -c 'select 1' >/dev/null 2>&1 && break
        sleep 0.3
      done
    fi
    "$PSQL" -h 127.0.0.1 -p "$PORT" -U postgres -d postgres -v ON_ERROR_STOP=0 >/dev/null 2>&1 <<SQL || true
CREATE ROLE cryptoguard LOGIN PASSWORD 'cryptoguard' SUPERUSER;
CREATE DATABASE cryptoguard OWNER cryptoguard;
SQL
    "$PSQL" "$URL" -tAc 'select 1' >/dev/null
    echo "$URL"
    ;;
  stop)
    PG_CTL="$(find_binary pg_ctl)"
    if "$PG_CTL" -D "$DATA" status >/dev/null 2>&1; then
      "$PG_CTL" -D "$DATA" -m fast stop >/dev/null
      echo "stopped" >&2
    else
      echo "not running" >&2
    fi
    ;;
  url)
    echo "$URL"
    ;;
  *)
    echo "usage: $0 {start|stop|url}" >&2
    exit 2
    ;;
esac
