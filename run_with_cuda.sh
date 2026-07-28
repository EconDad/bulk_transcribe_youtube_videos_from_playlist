#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VIRTUAL_ENV:-$ROOT/.venv}"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python virtual environment not found at $VENV" >&2
    exit 1
fi

CUDA_LIB_DIRS="$("$PYTHON" - <<'PY'
import nvidia.cublas.lib
import nvidia.cudnn.lib

print(
    ":".join(
        [
            next(iter(nvidia.cublas.lib.__path__)),
            next(iter(nvidia.cudnn.lib.__path__)),
        ]
    )
)
PY
)"

export LD_LIBRARY_PATH="$CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$PYTHON" \
    "$ROOT/bulk_transcribe_youtube_videos_from_playlist.py" \
    "$@"
