# Syzygy — solar wind speed forecast

Team **Syzygy** (contact: min070118@gmail.com). Track: Operational Podium (all computation in-container).

This version is the **operational baseline**: a persistence → 27-day-recurrence blend computed
from live ACE/SWEPAM data. The trained Syzygy model (SDO AIA 193 Å + HMI + ACE ensemble) plugs
into `model_stage.py`; until it does, the baseline is the forecast. The baseline also remains the
fallback of the full system.

## Workflow

```
run.sh <t0>
 └─ predict.py <t0> /tmp/forecast.json        (always exits 0)
     stage 0  climatology 430 km/s             written immediately
     stage 1  fetch ACE (t0 - 29 d … t0), cut off at t0, baseline forecast
     stage 2  model_stage.predict(...)         placeholder in this version
     each successful stage atomically overwrites /tmp/forecast.json
 └─ emergency.sh                               writes 72 × 430 km/s only if no valid file exists
 └─ upload.sh                                  unchanged platform uploader
```

* Output: `forecast_speed_kms[k]` is the speed at `t0 + (k+1) h`, k = 0…71. Values are clipped to
  200–1200 km/s; non-finite values are replaced by 430 km/s. Extra keys `stage` and `info` record
  which stage produced the forecast.
* Time budget: 600 s for `predict.py` (watchdog exits 0 with the best forecast written so far),
  180 s for ACE downloads; per request 20–60 s timeout, 2 retries.

## Baseline method

```
v(t0 + k h) = w_k · v_now + (1 − w_k) · v_rec(t0 + k h),   w_k = exp(−k / 48 h)
  v_now  mean hourly ACE speed over [t0 − 6 h, t0)  (else last valid hour within 48 h)
  v_rec  mean hourly speed within ±3 h of (t0 + k h − 655 h)   (655 h ≈ 27.2753-day synodic rotation)
```
Missing persistence → recurrence only; missing recurrence → persistence only; neither → 430 km/s.
The 48 h e-folding time was chosen on a backtest over 2026-06 … 2026-11 (512 six-hourly t0s,
ACE RT hourly truth): MAE 51.0 km/s vs 70.2 (persistence) and 64.1 (recurrence).

## Input data and information cut-off

| Source | URL | Use | Cut-off rule |
|---|---|---|---|
| ACE RT 1-min SWEPAM daily files (NOAA) | `https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/YYYYMMDD_ace_swepam_1m.txt` | primary ACE speed, whole 29-day window; status S = 0 only | obs ≤ t0 − 5 min |
| CDAWeb HAPI `AC_K0_SWE` (Vp, 5-min) | `https://cdaweb.gsfc.nasa.gov/hapi/data?id=AC_K0_SWE&parameters=Vp&…` | fills RT gaps older than 24 h | obs ≤ t0 − 5 min |
| SWPC RTSW 1-min JSON | `https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json` | fills RT gaps in the last 24 h; `source == "ACE"`, `overall_quality == 0` only | obs ≤ t0 − 5 min |
| JSOC AIA synoptic / NRT, HMI 1k (model stage) | `https://jsoc1.stanford.edu/data/aia/synoptic[/nrt]/…`, `https://jsoc1.stanford.edu/data/hmi/fits/…` | SDO images for the trained model (not used by the baseline) | obs ≤ slot time and listing mtime + 1 min ≤ t0 |

Publication lag assumptions: the ACE real-time stream is published within a few minutes of
observation, so samples are used only if observed at least 5 minutes before t0. Archives that
also contain data after t0 (e.g. when t0 is in the past) are cut by observation time. For JSOC
files the directory listing's modification time is used as the publication time; if it is
absent, AIA requires obs ≤ t0 − 30 min and HMI obs ≤ t0 − 90 min. HMI file times are TAI and are
converted to UTC (−37 s).

Hourly means need ≥ 10 valid 1-min samples (RT, RTSW) or ≥ 3 valid 5-min samples (K0); speeds
outside 150–2000 km/s are treated as fill. No external forecasts are used.

## Reproducibility

Every download is logged to stdout as a JSON line with URL, fetch time, HTTP status, byte size,
SHA-256 and the server's `Last-Modified` header, followed by an ACE coverage summary and the stage
that produced the forecast.

## Dependencies and assets

`public.ecr.aws/docker/library/python:3.11-slim`; the baseline uses the Python standard library
only (no weights). Runs as non-root user `runner`, writes only to `/tmp`.

## Local test

```
cd local_tests && ./test_local.sh 20261111T060000Z
```
Failure modes checked locally: no network (`--network none`), missing Python runtime, exhausted
time budget, malformed t0 — all exit 0 with 72 valid values.
