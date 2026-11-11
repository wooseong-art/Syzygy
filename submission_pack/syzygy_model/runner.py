"""Run the team's fc_predict.py (model/, unchanged) for one t0 inside the container.

    python3 -m syzygy_model.runner <t0 YYYY-mm-ddTHH:MM:SSZ> <ace.json> <out.json> <budget_s>

Started by model_stage.py as a separate process, so a crash / hang / OOM here never touches the
baseline forecast that predict.py has already written.

Runtime patches (fc_predict.py and sw_utils.py stay byte-identical to git_repo):
  1. fc_predict.LIVE_CACHE      → /tmp/syzygy_cache/live_images  (code folder is read-only for `runner`)
  2. sw_utils.http_get          → never raises; returns None on failure / exhausted budget; short
                                   timeouts; logs url, status, size, sha256, Last-Modified (reproducibility);
                                   serves files already fetched by the prefetcher from memory
  3. sw_utils.load_ace_hourly   → the ACE series predict.py already built with syzygy_live.ace
                                   (cut off at t0 − 5 min) instead of the L2 → K0 → RT chain
  4. predict(..., monitor=False) → no writes to monitoring/ or predictions/
  5. astropy IERS auto-download off (would contact an undeclared host)
  6. sw_utils.list_dir          → hides JSOC files not yet published at t0 (listing mtime), see list_dir()
Prefetch: the candidate FITS for every image slot (t0 − 12k h, k = 0…10, AIA 193 + HMI) and the PFSS
HMI history ([t0 − 28 d, t0 − 2 h], newest first) are downloaded in parallel while the 30 networks load.
fc_predict then runs exactly as written; its own sequential requests hit the in-memory cache.
"""
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
EXP = APP / "model" / "code" / "experiments" / "syzygy_final_cme_free_2026"
CACHE = Path(os.environ.get("SYZYGY_CACHE", "/tmp/syzygy_cache"))
WORKERS = int(os.environ.get("SYZYGY_DL_WORKERS", "12"))
HTTP_TIMEOUT_S = float(os.environ.get("SYZYGY_HTTP_TIMEOUT_S", "40"))
COMPUTE_RESERVE_S = float(os.environ.get("SYZYGY_COMPUTE_RESERVE_S", "150"))   # kept for preprocessing + PFSS + nets

T_START = time.monotonic()
_net_deadline = float("inf")
_prefetched = {}
_lock = threading.Lock()


_log_lock = threading.Lock()


def log(event, **kw):
    line = json.dumps({"event": event, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       "stage": "model", "t_s": round(time.monotonic() - T_START, 2), **kw}, default=str) + "\n"
    with _log_lock:                                  # one write per line: threads must not interleave
        sys.stdout.write(line)
        sys.stdout.flush()


def _fetch(url, retries=2, headers=None):
    import requests
    import sw_utils as U
    for attempt in range(retries + 1):
        left = _net_deadline - time.monotonic()
        if left < 2:
            log("fetch_skipped", url=url, reason="network budget exhausted")
            return None
        t = time.monotonic()
        try:
            r = U._SESSION.get(url, timeout=(min(10.0, left), min(HTTP_TIMEOUT_S, left)), headers=headers)
            if r.status_code == 404:
                log("fetch_error", url=url, status=404)
                return None
            r.raise_for_status()
            log("fetch", url=url, status=r.status_code, bytes=len(r.content), sha256=sha256(r.content).hexdigest(),
                last_modified=r.headers.get("Last-Modified"), elapsed_s=round(time.monotonic() - t, 2))
            return r
        except requests.RequestException as e:
            log("fetch_error", url=url, error=repr(e)[:300], attempt=attempt)
        if attempt < retries:
            time.sleep(min(1.0 + attempt, max(0.0, _net_deadline - time.monotonic() - 2)))
    return None


def http_get(url, retries=3, timeout=None, headers=None):
    """Drop-in for sw_utils.http_get: same return type (requests.Response or None), never raises."""
    with _lock:
        r = _prefetched.pop(url, None)
    return r if r is not None else _fetch(url, min(retries, 2), headers)


_T0 = None                 # aware UTC datetime of the run (information cut-off)
_listings = {}
NRT_FALLBACK_LAG_MIN = {"aia": 30, "hmi": 90}      # NRT row without a listing mtime: obs <= t0 - lag
DEFINITIVE_LAG_MIN = {"aia": 60, "hmi": 180}       # definitive file that appeared after t0: its NRT twin was public by obs + lag


def list_dir(url):
    """Drop-in for sw_utils.list_dir that hides files not yet PUBLISHED at t0.

    fc_predict only checks observation time <= slot. A container that starts some minutes after t0 (or an
    organiser re-run days later) would otherwise pick up files published after t0. Publication time = the
    modification time in the JSOC directory listing (UTC, to the minute; +1 min for the truncated seconds).
      NRT file         keep iff mtime + 1 min <= t0   (no mtime in the row: obs <= t0 - 30 min AIA / 90 min HMI)
      definitive file  keep iff mtime + 1 min <= t0, or obs <= t0 - 60 min (AIA) / 180 min (HMI): the same
                       observation was already public as NRT by then (NRT lag ~15-20 min AIA, ~50 min HMI),
                       and the NRT copy is deleted once the definitive one exists.
    Rows that are not AIA/HMI FITS files are passed through.
    """
    if url in _listings:
        return _listings[url]
    from datetime import timedelta
    from syzygy_live import sdo
    r = http_get(url)
    names, dropped = [], []
    for name, mt in sdo._ROW.findall(r.text if r is not None else ""):
        a, h = sdo._AIA.match(name), sdo._HMI.match(name)
        if not (a or h) or _T0 is None:
            names.append(name)
            continue
        if a:
            kind, nrt = "aia", "/nrt/" in url
            obs = datetime.strptime(a.group(1) + a.group(2).ljust(6, "0"), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            kind, nrt = "hmi", bool(h.group(1))
            obs = datetime.strptime(h.group(2) + h.group(3), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) - sdo.TAI_MINUS_UTC
        mtime = datetime.strptime(mt, "%d-%b-%Y %H:%M").replace(tzinfo=timezone.utc) if mt else None
        ok = mtime is not None and mtime + timedelta(minutes=1) <= _T0
        if not ok and (mtime is None or not nrt):
            lag = NRT_FALLBACK_LAG_MIN[kind] if nrt else DEFINITIVE_LAG_MIN[kind]
            ok = obs <= _T0 - timedelta(minutes=lag)
        (names if ok else dropped).append(name)
    if dropped:
        log("cutoff_filter", url=url, kept=len(names), hidden_not_published_at_t0=len(dropped), example=dropped[-1])
    _listings[url] = tuple(names)
    return _listings[url]


def ace_loader(ace_path):
    import numpy as np
    import pandas as pd
    doc = json.loads(Path(ace_path).read_text(encoding="utf-8"))
    hours = {pd.Timestamp(k).tz_convert(None): v for k, v in doc["hours"].items()}

    def load_ace_hourly(start, end, use_realtime=False):
        idx = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end), freq="h", inclusive="left")
        v = pd.Series({h: x[0] for h, x in hours.items()}, dtype=float).reindex(idx)
        src = pd.Series({h: x[1] for h, x in hours.items()}, dtype=object).reindex(idx).fillna("")
        log("ace_injected", hours=len(idx), valid=int(v.notna().sum()))
        return pd.DataFrame({"V": v.to_numpy(float), "src": src.to_numpy(object)}, index=idx)
    return load_ace_hourly


def prefetch_tasks(t0, C):
    import pandas as pd
    img = []
    for k in range(11):
        T = t0 - pd.Timedelta(hours=12 * k)
        img += [("aia193", T, True), ("hmi", T, True)]
    hmi_img = {T for inst, T, _ in img if inst == "hmi"}
    lo, hi = t0 - pd.Timedelta(days=C.PFSS_MAX_AGE_DAYS), t0 - pd.Timedelta(hours=C.PFSS_LATENCY_H)
    pf = [("hmi", s, False) for s in reversed(pd.date_range(lo.ceil("6h"), hi.floor("6h"), freq="6h")) if s not in hmi_img]
    return img + pf


def warm(task, U, C):
    """Download the first candidate fc_predict will pick for (instrument, slot) into memory."""
    import pandas as pd
    inst, t, fallback = task
    wl = U.aia_wavelength(inst)
    for c in ([t, t - pd.Timedelta(hours=C.IMG_FILL_TOL_HOURS)] if fallback else [t]):
        cands = (U._aia_candidates(c, C.IMG_MATCH_TOL_MIN, c, wl) if wl
                 else U._hmi_candidates(c, C.IMG_MATCH_TOL_MIN, c))
        if not cands:
            continue
        url = cands[0][1]
        r = _fetch(url)
        if r is None:
            return "failed"
        with _lock:
            _prefetched[url] = r
        return "ok" if c == t else "ok_fallback"
    return "none"


def main(argv):
    global _net_deadline
    t0_s, ace_path, out_path, budget_s = argv[1], argv[2], argv[3], float(argv[4])
    deadline = T_START + budget_s
    _net_deadline = deadline - COMPUTE_RESERVE_S
    log("model_start", t0=t0_s, budget_s=budget_s, net_budget_s=round(_net_deadline - T_START, 1), workers=WORKERS)

    try:
        from astropy.utils import iers
        iers.conf.auto_download = False
    except Exception:
        log("iers_config_failed", error=traceback.format_exc()[-300:])

    sys.path.insert(0, str(EXP))
    import pandas as pd
    import fc_common as F  # noqa: F401  (sets sys.path for the experiment folders)
    import fc_predict as FP
    import config as C
    import sw_utils as U
    log("imports_done")

    FP.LIVE_CACHE = CACHE / "live_images"
    global _T0
    _T0 = datetime.strptime(t0_s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    U.http_get = http_get
    U.list_dir = list_dir
    U.load_ace_hourly = ace_loader(ace_path)
    t0 = pd.Timestamp(t0_s).tz_convert(None)

    tasks = prefetch_tasks(t0, C)
    pool = ThreadPoolExecutor(WORKERS)
    futs = [pool.submit(warm, tk, U, C) for tk in tasks]
    t_pf = time.monotonic()

    FP._P = FP.Pipeline("cpu")
    log("pipeline_loaded", init_s=round(FP._P.init_sec, 2))

    pool.shutdown(wait=True)
    stats = {}
    for tk, f in zip(tasks, futs):
        try:
            r = f.result()
        except Exception as e:
            r = f"error {e!r}"[:80]
        key = "image" if tk[2] else "pfss_hmi"
        stats.setdefault(key, {}).setdefault(r, 0)
        stats[key][r] += 1
    prefetch = dict(tasks=len(tasks), seconds=round(time.monotonic() - t_pf, 1), result=stats)
    log("prefetch_done", **prefetch)

    t_pred = time.monotonic()
    res = FP.predict(t0.strftime("%Y-%m-%dT%H:%M:%S"), {"mode": "live", "monitor": False, "allow_download": True})
    log("predict_done", seconds=round(time.monotonic() - t_pred, 1), state=res["status"]["state"],
        final_model=res["final_model"], final_speed_kms=res["final_speed_kms"], missing=res["status"]["missing_inputs"])

    from syzygy_model.hourly import to_hourly
    hours = json.loads(Path(ace_path).read_text(encoding="utf-8"))["hours"]
    forecast, hourly_info = to_hourly(res, hours, t0)
    res["status"]["download_log"] = f"{len(res['status']['download_log'])} entries (see stdout)"
    doc = {"forecast": forecast, "hourly": hourly_info, "result": res, "prefetch": prefetch,
           "total_s": round(time.monotonic() - T_START, 1)}
    tmp = out_path + ".tmp"
    Path(tmp).write_text(json.dumps(doc, default=str), encoding="utf-8")
    os.replace(tmp, out_path)
    log("model_done", total_s=doc["total_s"], hourly=hourly_info,
        first=None if forecast is None else forecast[0], last=None if forecast is None else forecast[-1])
    return 0


if __name__ == "__main__":
    try:
        rc = main(sys.argv)
    except BaseException:
        log("model_failed", error=traceback.format_exc()[-3000:])
        rc = 1
    sys.exit(rc)
