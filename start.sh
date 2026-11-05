#!/usr/bin/env bash
# Run CiteLume locally: builds the UI, then serves UI + API on http://127.0.0.1:8000
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/frontend"
if command -v npm >/dev/null 2>&1; then
  [ -d node_modules ] || npm install --no-audit --no-fund
  npm run build
else
  echo "Warning: npm not found; serving existing frontend/dist (may be stale)."
fi
# Python: CITELUME_PYTHON, else a .venv in the project, else python3 on the PATH
PY="${CITELUME_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
  elif [ -x "$ROOT/../.venv/bin/python" ]; then PY="$ROOT/../.venv/bin/python"
  else PY="python3"; fi
fi
cd "$ROOT/backend"
echo "Starting CiteLume on http://127.0.0.1:8000 ..."
"$PY" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
