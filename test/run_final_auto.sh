#!/usr/bin/env bash
# Big City RPAS Task 2 — FULL PRODUCTION autonomous engagement (final_auto).
# LIVE-FLY + LIVE-FIRE, guided-mode check skipped. The drone WILL move and the
# gun WILL fire. Run deliberately, observer on RC override.
# Extra args pass through. Env overrides: MODEL, TEAM, MAVLINK, PY.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# The environment that has cv2 + pymavlink + pycoral (EdgeTPU) — the pyenv 3.9.19
# interpreter. Using it directly IS "switching into" that venv. Override with PY=.
PY="${PY:-/home/pi/.pyenv/versions/3.9.19/bin/python3}"
MODEL="${MODEL:-$HOME/FullDataSetProdV4b_depheavy_edgetpu.tflite}"
TEAM="${TEAM:-UBCO_AEROSPACE}"
MAVLINK="${MAVLINK:-tcp:10.42.0.1:5760}"
export PYENV_VERSION=3.9.19

[ -x "$PY" ] || { echo "[ERROR] interpreter not found: $PY  (set PY=...)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "[ERROR] model not found: $MODEL  (set MODEL=...)" >&2; exit 1; }
cd "$HERE"
echo "[ENV] $("$PY" --version 2>&1)  ->  $PY"
echo "############################################################"
echo "#  FULL PRODUCTION: LIVE-FLY + LIVE-FIRE, no guided check.  #"
echo "#  Drone WILL move, gun WILL fire. Observer on RC override. #"
echo "#  Ctrl-C within 3s to abort...                             #"
echo "############################################################"
sleep 3

set -x
exec "$PY" -B "$HERE/tf_live_inferenceV2_final_auto.py" "$MODEL" \
  --tpu -p --no-output --mavlink "$MAVLINK" --sharpen 0.4 \
  --start-from-current-gimbal --team-name "$TEAM" \
  --live-fly --live-fire --no-guided-check "$@"
