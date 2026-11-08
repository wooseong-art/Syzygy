#!/usr/bin/env python3
"""Syzygy solar-wind forecast entry point.

    python3 predict.py <t0 YYYYMMDDTHHMMSSZ> [out.json]

Always exits 0 with a valid 72-value forecast in out.json. Progressive output: the file is
(re)written atomically after each stage that succeeds, so whatever happens later (exception,
time-out, OOM kill of a sub-step) the best forecast so far is already on disk.
  stage 0  climatology                       written immediately
  stage 1  baseline (persistence+recurrence) from ACE history
  stage 2  model                             model_stage.predict (placeholder until wired in)
A watchdog thread exits the process (status 0) once the time budget is used up.
Logs go to stdout as JSON lines; the forecast goes only to the output file.
"""
import json
import math
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syzygy_live import net  # noqa: E402  (stdlib only, always importable)

H = 72
CLIMATOLOGY_KMS = 430.0
V_CLIP = (200.0, 1200.0)
BUDGET_S = float(os.environ.get("SYZYGY_BUDGET_S", "600"))   # 15 min platform limit incl. image pull
ACE_FETCH_S = 180.0


def parse_t0(s):
    return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def sanitize(values):
    """72 finite clipped floats, or None if the stage output is unusable."""
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if len(vals) != H:
        return None
    bad = sum(not math.isfinite(v) for v in vals)
    if bad > H // 2:
        return None
    return [round(min(max(v if math.isfinite(v) else CLIMATOLOGY_KMS, V_CLIP[0]), V_CLIP[1]), 2) for v in vals]


def write_forecast(path, t0_str, valid_from, values, stage, info=None):
    doc = {"t0": t0_str, "valid_from": valid_from, "forecast_speed_kms": values,
           "stage": stage, "info": info or {}}
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, default=str)
    os.replace(tmp, path)
    net.log("forecast_written", stage=stage, path=path, first=values[0], last=values[-1])


def main(argv):
    t_start = time.monotonic()
    raw_t0 = (argv[1] if len(argv) > 1 else os.environ.get("INFERENCE_TIMESTAMP", "")).strip()
    out = argv[2] if len(argv) > 2 else "/tmp/forecast.json"

    try:
        t0 = parse_t0(raw_t0)
        t0_str = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
        valid_from = (t0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        net.log("bad_t0", raw=raw_t0)
        write_forecast(out, raw_t0, None, [CLIMATOLOGY_KMS] * H, "climatology", {"error": "unparseable t0"})
        return 0

    net.log("start", t0=t0_str, out=out, budget_s=BUDGET_S, submission=os.environ.get("SUBMISSION_ID"))
    write_forecast(out, t0_str, valid_from, [CLIMATOLOGY_KMS] * H, "climatology")

    def watchdog():
        net.log("watchdog_exit", elapsed_s=round(time.monotonic() - t_start, 1))
        sys.stdout.flush()
        os._exit(0)
    timer = threading.Timer(BUDGET_S, watchdog)
    timer.daemon = True
    timer.start()

    # stage 1: baseline from ACE
    ace_hourly = {}
    try:
        from syzygy_live import ace, baseline
        net.set_deadline(t_start + ACE_FETCH_S)
        ace_hourly, _ = ace.load_ace_hourly(t0)
        vals, info = baseline.forecast(ace_hourly, t0)
        vals = sanitize(vals)
        if vals:
            write_forecast(out, t0_str, valid_from, vals, "baseline", info)
    except Exception:
        net.log("stage_failed", stage="baseline", error=traceback.format_exc()[-1500:])

    # stage 2: model
    try:
        import model_stage
        if model_stage.MODEL_AVAILABLE:
            deadline = t_start + BUDGET_S - 30
            net.set_deadline(deadline - 15)
            vals = sanitize(model_stage.predict(t0, ace_hourly, deadline))
            if vals:
                write_forecast(out, t0_str, valid_from, vals, "model")
            else:
                net.log("stage_skipped", stage="model", reason="no usable output")
        else:
            net.log("stage_skipped", stage="model", reason="model not wired in yet")
    except Exception:
        net.log("stage_failed", stage="model", error=traceback.format_exc()[-1500:])

    net.log("done", elapsed_s=round(time.monotonic() - t_start, 1))
    return 0


if __name__ == "__main__":
    try:
        main(sys.argv)
    except BaseException:
        traceback.print_exc()
    sys.exit(0)
