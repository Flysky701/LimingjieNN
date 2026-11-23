#!/usr/bin/env bash
set -euo pipefail

# Setup & launch headless PCRCalculatorServer (Linux x64) on Google Colab with locked CN DB caching.
#
# Usage:
#   bash scripts/colab_setup_simulator.sh \
#     --cache-dir /content/drive/MyDrive/LimingjieNN/cache \
#     --db-path /content/drive/MyDrive/LimingjieNN/master_cn.db \
#     --db-version 202611131613 \
#     --port 8080

CACHE_DIR="${PCRCALC_CACHE_DIR:-/content/pcr_cache}"
WORK_DIR="${PCRCALC_WORK_DIR:-}"
DB_PATH="${PCRCALC_CN_DB_PATH:-}"
DB_SHA256="${PCRCALC_CN_DB_SHA256:-}"
DB_VERSION="${PCRCALC_CN_VERSION:-202611131613}"
CHAR_VERSION_JP="${PCRCALC_CHAR_VER_JP:-10066000}"
BOSS_VERSION_JP="${PCRCALC_BOSS_VER_JP:-10066000}"
PORT="${PCRCALC_PORT:-8080}"
HOST="${PCRCALC_HOST:-127.0.0.1}"
RELEASE_URL="${PCRCALC_RELEASE_URL:-https://github.com/cc004/PCRCalculator/releases/download/v2.11.0/PCRCalculator-2.11.0-server-linux.zip}"
DISCOVER_LATEST="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --db-path) DB_PATH="$2"; shift 2 ;;
    --db-sha256) DB_SHA256="$2"; shift 2 ;;
    --db-version) DB_VERSION="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --release-url) RELEASE_URL="$2"; shift 2 ;;
    --discover-latest) DISCOVER_LATEST="true"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

WORK_DIR="${WORK_DIR:-/tmp/pcr_calc_server_${PORT}}"
PID_FILE="/tmp/pcr_calc_server_${PORT}.pid"
LOG_FILE="/tmp/pcr_calc_server_${PORT}.log"
export PCRCALC_PERSISTENT_DATA_PATH="/tmp/pcr_calc_persistent_${PORT}"
mkdir -p "$CACHE_DIR" "$WORK_DIR" "$PCRCALC_PERSISTENT_DATA_PATH"

# 1) Install minimal Ubuntu shared libs if missing (best-effort on Colab)
if command -v apt-get >/dev/null 2>&1; then
  if ! ldconfig -p 2>/dev/null | grep -q 'libsqlite3\.so'; then
    apt-get update -qq && apt-get install -y -qq libsqlite3-0 libicu-dev unzip curl >/dev/null || true
  fi
fi

# 2) Download & cache server zip
ZIP_NAME="$(basename "$RELEASE_URL")"
ZIP_PATH="$CACHE_DIR/$ZIP_NAME"
if [[ ! -s "$ZIP_PATH" ]]; then
  echo "[setup] Downloading $RELEASE_URL -> $ZIP_PATH"
  ZIP_TEMP="$(mktemp "${ZIP_PATH}.part.XXXXXX")"
  trap 'rm -f "$ZIP_TEMP"' EXIT
  curl -fL --retry 3 --retry-delay 3 -o "$ZIP_TEMP" "$RELEASE_URL"
  mv -f "$ZIP_TEMP" "$ZIP_PATH"
  trap - EXIT
else
  echo "[setup] Reusing cached archive: $ZIP_PATH"
fi

# 3) Unpack server into fast local /tmp workdir
if [[ ! -x "$WORK_DIR/PCRCalculatorServer" ]]; then
  echo "[setup] Unpacking $ZIP_PATH -> $WORK_DIR"
  unzip -q -o "$ZIP_PATH" -d "$WORK_DIR"
  if [[ ! -f "$WORK_DIR/PCRCalculatorServer" ]]; then
    NESTED_BIN="$(find "$WORK_DIR" -maxdepth 3 -type f -name 'PCRCalculatorServer' | head -n 1 || true)"
    if [[ -n "$NESTED_BIN" ]]; then
      NESTED_DIR="$(dirname "$NESTED_BIN")"
      cp -a "$NESTED_DIR/." "$WORK_DIR/"
    fi
  fi
  chmod +x "$WORK_DIR/PCRCalculatorServer"
fi

# 4) Verify & stage offline locked CN DB as prod<BossVersionCN>.db (per AssetManagerCN.DataVer)
DB_NAME="prod${DB_VERSION}.db"
if [[ -z "$DB_PATH" && ! -f "$PCRCALC_PERSISTENT_DATA_PATH/$DB_NAME" && -f "$CACHE_DIR/persistent/$DB_NAME" ]]; then
  DB_PATH="$CACHE_DIR/persistent/$DB_NAME"
fi
if [[ -n "$DB_PATH" ]]; then
  if [[ ! -f "$DB_PATH" ]]; then
    echo "[error] Specified --db-path does not exist: $DB_PATH" >&2
    exit 1
  fi
  if [[ -n "$DB_SHA256" ]]; then
    ACTUAL_SHA="$(sha256sum "$DB_PATH" | awk '{print $1}')"
    if [[ "${ACTUAL_SHA,,}" != "${DB_SHA256,,}" ]]; then
      echo "[error] DB SHA256 mismatch: expected=$DB_SHA256 actual=$ACTUAL_SHA" >&2
      exit 1
    fi
    echo "[setup] Verified CN DB SHA256: $ACTUAL_SHA"
  fi
fi

# Stop any prior server on this port before staging its database.
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ "$OLD_PID" =~ ^[1-9][0-9]*$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" || true
    sleep 1
  fi
fi

LOCAL_DB="$PCRCALC_PERSISTENT_DATA_PATH/$DB_NAME"
if [[ -n "$DB_PATH" && ! "$DB_PATH" -ef "$LOCAL_DB" ]]; then
  cp -f "$DB_PATH" "$LOCAL_DB"
  echo "[setup] Staged offline CN DB as $DB_NAME"
fi
if [[ -f "$LOCAL_DB" && ! "$LOCAL_DB" -ef "$WORK_DIR/$DB_NAME" ]]; then
  cp -f "$LOCAL_DB" "$WORK_DIR/$DB_NAME"
fi

# 5) Configure CN runtime environment (pinned offline version unless --discover-latest)
if [[ "$DISCOVER_LATEST" == "true" ]]; then
  export PCRCALC_VERSION_DATA="{\"CharacterVersionJP\":${CHAR_VERSION_JP},\"BossVersionJP\":${BOSS_VERSION_JP},\"BossVersionCN\":0,\"useQA\":false,\"useJP\":false,\"newAB\":false,\"useLatestProd\":true}"
else
  export PCRCALC_VERSION_DATA="{\"CharacterVersionJP\":${CHAR_VERSION_JP},\"BossVersionJP\":${BOSS_VERSION_JP},\"BossVersionCN\":${DB_VERSION},\"useQA\":false,\"useJP\":false,\"newAB\":false,\"useLatestProd\":false}"
fi

# 6) Start headless server in background and wait until TCP port accepts HTTP
cd "$WORK_DIR"
nohup ./PCRCalculatorServer --server --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "[setup] Waiting for PCRCalculatorServer (pid=$SERVER_PID) at http://$HOST:$PORT ..."
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[error] PCRCalculatorServer exited prematurely. Tail of $LOG_FILE:" >&2
    tail -n 60 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if python3 -c "import urllib.request; urllib.request.urlopen('http://${HOST}:${PORT}/', timeout=2)" >/dev/null 2>&1 ||
     python3 -c "import socket; s=socket.create_connection(('${HOST}', ${PORT}), 2); s.close()" >/dev/null 2>&1; then
    sleep 2
    echo "[setup] PCRCalculatorServer ready at http://$HOST:$PORT (persistent=$PCRCALC_PERSISTENT_DATA_PATH)"
    exit 0
  fi
  sleep 2
done

echo "[error] Timed out waiting for PCRCalculatorServer. Tail of $LOG_FILE:" >&2
tail -n 60 "$LOG_FILE" >&2 || true
exit 1
