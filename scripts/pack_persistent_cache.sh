#!/usr/bin/env bash
set -euo pipefail

PERSISTENT_DIR=""
OUT=""
FORCE=false

while (($#)); do
  case "$1" in
    --persistent-dir|--out)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "[error] $1 requires a value" >&2
        exit 2
      fi
      if [[ "$1" == --persistent-dir ]]; then
        PERSISTENT_DIR="$2"
      else
        OUT="$2"
      fi
      shift 2
      ;;
    --force) FORCE=true; shift ;;
    *) echo "[error] Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PERSISTENT_DIR" || -z "$OUT" ]]; then
  echo "[error] Usage: $0 --persistent-dir DIR --out TAR [--force]" >&2
  exit 2
fi
if [[ ! -d "$PERSISTENT_DIR" ]]; then
  echo "[error] Persistent directory does not exist: $PERSISTENT_DIR" >&2
  exit 1
fi
if [[ ! -e "$PERSISTENT_DIR/Pool" ]] && ! compgen -G "$PERSISTENT_DIR/manifest_*.json" >/dev/null; then
  echo "[error] No Pool or manifest_*.json found in $PERSISTENT_DIR" >&2
  exit 1
fi

if [[ "$FORCE" == false && -f "$OUT" && -f "${OUT}.sha256" ]]; then
  EXPECTED_SHA="$(awk 'NR == 1 { print $1; exit }' "${OUT}.sha256")"
  ACTUAL_SHA="$(sha256sum < "$OUT" | awk '{print $1}')"
  if [[ "$EXPECTED_SHA" =~ ^[[:xdigit:]]{64}$ && "${EXPECTED_SHA,,}" == "${ACTUAL_SHA,,}" ]]; then
    echo "[pack] Reusing verified archive: $OUT"
    exit 0
  fi
fi

mkdir -p "$(dirname "$OUT")"
trap 'rm -f "${TMP_TAR:-}" "${TMP_SHA:-}"' EXIT
TMP_TAR="$(mktemp "${OUT}.part.XXXXXX")"
TMP_SHA="$(mktemp "${OUT}.sha256.part.XXXXXX")"
tar -C "$PERSISTENT_DIR" -cf "$TMP_TAR" .
tar -tf "$TMP_TAR" >/dev/null
TAR_SHA="$(sha256sum < "$TMP_TAR" | awk '{print $1}')"
printf '%s  %s\n' "$TAR_SHA" "$(basename "$OUT")" > "$TMP_SHA"
mv -f "$TMP_TAR" "$OUT"
mv -f "$TMP_SHA" "${OUT}.sha256"
trap - EXIT
echo "[pack] Wrote $OUT and ${OUT}.sha256"
