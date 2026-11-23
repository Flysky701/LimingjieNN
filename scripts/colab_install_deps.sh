#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=.
WHEELHOUSE=""
OFFLINE_ONLY=false

while (($#)); do
  case "$1" in
    --repo-dir|--wheelhouse)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "[error] $1 requires a value" >&2
        exit 2
      fi
      if [[ "$1" == --repo-dir ]]; then
        REPO_DIR="$2"
      else
        WHEELHOUSE="$2"
      fi
      shift 2
      ;;
    --offline-only) OFFLINE_ONLY=true; shift ;;
    *) echo "[error] Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$WHEELHOUSE" || ! -d "$REPO_DIR" ]]; then
  echo "[error] --wheelhouse is required and --repo-dir must be an existing directory: $REPO_DIR" >&2
  exit 2
fi
if [[ "$OFFLINE_ONLY" == true ]]; then
  if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "[error] Offline wheelhouse does not exist: $WHEELHOUSE" >&2
    exit 1
  fi
else
  mkdir -p "$WHEELHOUSE"
fi

python3 -c 'import platform, sys; print(f"Python {platform.python_version()} (cp{sys.version_info.major}{sys.version_info.minor})")'
missing_packages="$(python3 -c '
from importlib.util import find_spec
for package in ("torch", "numpy", "pyarrow", "onnx", "onnxruntime"):
    if find_spec(package) is None:
        print(package)
')"

if [[ -n "$missing_packages" ]]; then
  mapfile -t MISSING <<< "$missing_packages"
  if ! python3 -m pip install -q --no-index --find-links "$WHEELHOUSE" "${MISSING[@]}"; then
    if [[ "$OFFLINE_ONLY" == true ]]; then
      echo "[error] Missing compatible wheels for: ${MISSING[*]} (wheelhouse: $WHEELHOUSE)" >&2
      exit 1
    fi
    python3 -m pip download -q --only-binary=:all: -d "$WHEELHOUSE" "${MISSING[@]}"
    python3 -m pip install -q --no-index --find-links "$WHEELHOUSE" "${MISSING[@]}"
  fi
fi

python3 -m pip install -q --no-build-isolation --no-deps -e "$REPO_DIR"
python3 -c "import labyrinth_nn, torch, numpy, onnx, onnxruntime; print('deps ok:', onnx.__version__, onnxruntime.__version__)"
