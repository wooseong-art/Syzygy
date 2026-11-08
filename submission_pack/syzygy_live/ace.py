"""ACE/SWEPAM proton speed -> hourly means, cut off at t0.

Sources, highest priority first (a lower source only fills hours still empty):
  1. RT   NOAA ACE real-time 1-min daily files (sohoftp). Covers the whole
          ~29-day window; today's file is updated every few minutes.
  2. K0   CDAWeb HAPI AC_K0_SWE 5-min (lags ~1 week; fills older RT gaps).
  3. RTSW SWPC RTSW 1-min JSON (last ~24 h), source == "ACE" rows only.

Information cut-off: a sample is used only if obs_time <= t0 - PUBLISH_LAG.
The real-time ACE stream is published within minutes of observation, so a
5-minute lag is conservative. The archives also hold data after t0 (when t0
is in the past), which this cut-off removes.
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from . import net

RT_DAILY_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/{d:%Y%m%d}_ace_swepam_1m.txt"
K0_HAPI_URL = ("https://cdaweb.gsfc.nasa.gov/hapi/data?id=AC_K0_SWE&parameters=Vp"
               "&time.min={a:%Y-%m-%dT%H:%M:%SZ}&time.max={b:%Y-%m-%dT%H:%M:%SZ}")
RTSW_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"

PUBLISH_LAG = timedelta(minutes=5)
V_VALID = (150.0, 2000.0)          # km/s; outside = fill / garbage
MIN_SAMPLES = {"RT": 10, "K0": 3, "RTSW": 10}   # valid samples needed for an hourly mean
UTC = timezone.utc


def _valid(v):
    return v is not None and math.isfinite(v) and V_VALID[0] <= v <= V_VALID[1]


def parse_rt_daily(text):
    """'YYYY MM DD HHMM MJD SOD S Np V T' rows; status S == 0 means good."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 10 or not p[0].isdigit():
            continue
        try:
            t = datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3][:-2] or 0), int(p[3][-2:]), tzinfo=UTC)
            if int(p[6]) == 0:
                out.append((t, float(p[8])))
        except ValueError:
            continue
    return out


def parse_hapi_csv(text):
    """HAPI CSV 'ISO,value'. A JSON body means a status message (e.g. 1201 no data)."""
    if text.lstrip().startswith("{"):
        return []
    out = []
    for line in text.splitlines():
        try:
            ts, v = line.split(",")[:2]
            out.append((datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=UTC), float(v)))
        except ValueError:
            continue
    return out


def parse_rtsw(text, source="ACE"):
    out = []
    for r in json.loads(text):
        if r.get("source") != source or r.get("proton_speed") is None:
            continue
        if (r.get("overall_quality") or 0) != 0:
            continue
        try:
            t = datetime.fromisoformat(r["time_tag"]).replace(tzinfo=UTC)
            out.append((t, float(r["proton_speed"])))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def hourly_means(samples, start, cutoff, min_count):
    """{hour_start: mean V} over [start, cutoff) using samples with obs_time <= cutoff."""
    acc = {}
    for t, v in samples:
        if start <= t <= cutoff and _valid(v):
            h = t.replace(minute=0, second=0, microsecond=0)
            s = acc.setdefault(h, [0.0, 0])
            s[0] += v
            s[1] += 1
    return {h: s[0] / s[1] for h, s in acc.items() if s[1] >= min_count}


def _fetch_text(url, timeout=30):
    body = net.get(url, timeout=timeout)
    return body.decode("utf-8", "replace") if body else ""


def load_ace_hourly(t0, days=29):
    """Return (hourly {hour_start: V}, source {hour_start: name}) for [t0 - days, t0 - PUBLISH_LAG]."""
    cutoff = t0 - PUBLISH_LAG
    start = (t0 - timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
    hourly, source = {}, {}

    def merge(name, values):
        n = 0
        for h, v in values.items():
            if h not in hourly:
                hourly[h], source[h] = v, name
                n += 1
        net.log("ace_merge", source=name, hours_added=n, hours_total=len(hourly))

    # 1. RT daily files, fetched in parallel (~100 KB each)
    day, last = start.date(), cutoff.date()
    urls = []
    while day <= last:
        urls.append(RT_DAILY_URL.format(d=day))
        day += timedelta(days=1)
    with ThreadPoolExecutor(max_workers=8) as ex:
        texts = list(ex.map(_fetch_text, urls))
    samples = [s for txt in texts for s in parse_rt_daily(txt)]
    merge("RT", hourly_means(samples, start, cutoff, MIN_SAMPLES["RT"]))

    all_hours = []
    h = start
    while h <= cutoff:
        all_hours.append(h)
        h += timedelta(hours=1)
    missing = [h for h in all_hours if h not in hourly]

    # 2. K0 for gaps older than a day (K0 itself lags ~1 week, so only old gaps can be filled)
    old = [h for h in missing if h < t0 - timedelta(hours=24)]
    if len(old) >= 6:
        url = K0_HAPI_URL.format(a=old[0], b=min(old[-1] + timedelta(hours=1), cutoff))
        merge("K0", hourly_means(parse_hapi_csv(_fetch_text(url, timeout=60)), start, cutoff, MIN_SAMPLES["K0"]))

    # 3. RTSW for gaps within the last day
    recent = [h for h in missing if h >= t0 - timedelta(hours=24) and h not in hourly]
    if recent:
        txt = _fetch_text(RTSW_URL, timeout=45)
        if txt:
            try:
                merge("RTSW", hourly_means(parse_rtsw(txt), start, cutoff, MIN_SAMPLES["RTSW"]))
            except ValueError as e:
                net.log("ace_parse_error", source="RTSW", error=repr(e)[:200])

    last_h = max(hourly) if hourly else None
    net.log("ace_summary", window_hours=len(all_hours), valid_hours=len(hourly),
            coverage=round(len(hourly) / max(len(all_hours), 1), 3),
            last_valid_hour=last_h, last_valid_v=round(hourly[last_h], 1) if last_h else None)
    return hourly, source
