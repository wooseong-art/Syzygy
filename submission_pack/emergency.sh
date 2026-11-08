#!/usr/bin/env bash
# Last-resort forecast if predict.py could not produce a file at all (e.g. the Python runtime is broken).
# Writes 72 x climatology. Usage: emergency.sh <t0> <out>
set -u
T0="$1"; OUT="$2"
[ -s "$OUT" ] && jq -e '.forecast_speed_kms | length == 72' "$OUT" >/dev/null 2>&1 && exit 0
echo "[emergency] predict.py produced no valid forecast; writing climatology"
VALS=$(printf '430.0,%.0s' $(seq 1 71))430.0
printf '{"t0":"%s","valid_from":null,"forecast_speed_kms":[%s],"stage":"emergency"}\n' "$T0" "$VALS" > "$OUT"
