"""Model stage: runs the team's model (model/, synced by tools/sync_model.py) in a child process.

    predict(t0, ace_hourly, deadline_monotonic, ace_source) -> (72 floats or None, info dict or None)
      t0          : timezone-aware UTC datetime
      ace_hourly  : {hour_start (aware UTC): V km/s}, cut off at t0 (syzygy_live.ace)
      ace_source  : {hour_start: "RT" | "K0" | "RTSW"}
      deadline    : time.monotonic() value by which this call must return
The child (python3 -m syzygy_model.runner) is killed at the deadline. Values are None while the
hourly conversion rule is undecided (syzygy_model/hourly.py); info then still carries the model output.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
MODEL_AVAILABLE = (APP / "model" / "MANIFEST.json").exists() and (APP / "syzygy_model" / "runner.py").exists()
MIN_BUDGET_S = 60


def _log(event, **kw):
    from syzygy_live import net
    net.log(event, **kw)


def predict(t0, ace_hourly, deadline_monotonic, ace_source=None):
    budget = deadline_monotonic - time.monotonic() - 5
    if budget < MIN_BUDGET_S:
        _log("stage_skipped", stage="model", reason=f"only {budget:.0f}s left")
        return None, None
    work = Path(tempfile.mkdtemp(prefix="syzygy_model_"))
    ace_path, out_path = work / "ace.json", work / "model.json"
    src = ace_source or {}
    ace_path.write_text(json.dumps({"hours": {h.strftime("%Y-%m-%dT%H:%M:%SZ"): [v, src.get(h, "")]
                                              for h, v in sorted(ace_hourly.items())}}), encoding="utf-8")
    cmd = [sys.executable, "-m", "syzygy_model.runner", t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
           str(ace_path), str(out_path), f"{budget:.0f}"]
    t = time.monotonic()
    try:
        rc = subprocess.run(cmd, cwd=str(APP), timeout=budget + 3).returncode
    except subprocess.TimeoutExpired:
        _log("stage_failed", stage="model", reason="timeout", elapsed_s=round(time.monotonic() - t, 1))
        return None, {"error": "timeout"}
    if rc != 0 or not out_path.exists():
        _log("stage_failed", stage="model", reason=f"exit {rc}", elapsed_s=round(time.monotonic() - t, 1))
        return None, {"error": f"exit {rc}"}
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    return doc.get("forecast"), doc
