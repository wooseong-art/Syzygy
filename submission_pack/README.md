# Syzygy — solar wind speed forecast

Team **Syzygy** (contact: min070118@gmail.com). Track: Operational Podium (all computation in-container).

The forecast comes from the trained **Syzygy CME-free ensemble** (SDO AIA 193 Å + HMI magnetograms,
a PFSS coronal field model computed in the container, and ACE solar wind speed). An ACE-only
persistence → 27-day-recurrence baseline is always computed first and stays as the fallback.

## Workflow

```
run.sh <t0>
 └─ predict.py <t0> /tmp/forecast.json          (always exits 0)
     stage 0  climatology 430 km/s               written immediately
     stage 1  fetch ACE (t0 − 29 d … t0), cut off at t0 → baseline forecast → written
     stage 2  model (child process, syzygy_model/runner.py) → 72 hourly values → written
     each successful stage atomically overwrites /tmp/forecast.json
 └─ emergency.sh                                 writes 72 × 430 km/s only if no valid file exists
 └─ upload.sh                                    unchanged platform uploader
```

* Output: `forecast_speed_kms[j]` is the speed for `t0 + (j+1) h`, j = 0…71. Values are clipped to
  200–1200 km/s; non-finite values are replaced by 430 km/s. Extra keys `stage` and `info` record
  which stage produced the forecast; `info.model` holds the full model output and `info.fallback`
  the baseline summary.
* Fallback order: full ensemble → G01 numeric model (images or PFSS unavailable) → baseline
  (model failed, timed out, or ACE history empty) → climatology.
* Time budget: 600 s for `predict.py` (a watchdog exits 0 with the best forecast written so far),
  180 s for ACE downloads. The model child process is killed at its deadline; image downloads stop
  150 s before it so that preprocessing and inference can finish.

## Model

Trained offline (no retraining, no external computation at run time). Weights and code are copied
unchanged from the team repository `wooseong-art/git_repo`; `model/MANIFEST.json` lists the source
commit and the SHA-256 of every file.

```
v        = 0.5 · mean_seed(G01) + 0.5 · mean_seed(Router R2)            5 seeds each
Router_s = anchor_s + Σ softmax(router_s) · ΔV_s                        anchor, ΔV from RegimeNet-CF (same seed)
v_final  = v + 0.25 · p_occ · (BoxCox_C(v) − v)                         p_occ = HSS occupancy probability (model O)
```
* G01: numeric network on ACE history (48 h, 27-day recurrence windows) and a 4×3 coronal-hole
  grid from AIA 193 + HMI. RegimeNet-CF / Router: CNN on AIA 193 + HMI images at t0 − 0/24/48/72/96 h,
  coronal-hole features, and PFSS expansion-factor features traced from a Carrington map built
  from the last 28 days of HMI magnetograms.
* Native output: ACE 6 h mean speed over `[t0 + h − 6, t0 + h)` for h = 24/48/72/96. No CME inputs.
* If any image or PFSS input is missing the model falls back to G01 alone (with its own Box–Cox map).

### Hourly values

The 72 hourly values are derived inside the container from the model's own output (rule "KI";
hourly RMSE on the validation years 62.2 km/s vs 95.5 for persistence):
linear interpolation along lead time through `v_now` (mean ACE speed of the last 6 h, placed at
−3 h) and the four 6 h means placed at their window centres (21/45/69/93 h).
The scoring truth (SWPC `ace_swepam_1h.json`) tags an hourly mean with the **start** of its hour,
so the value reported for `t0 + k h` is the interpolated mean over `[t0 + k, t0 + k + 1)`.

## Baseline (fallback)

```
v(t0 + k h) = w_k · v_now + (1 − w_k) · v_rec(t0 + k h),   w_k = exp(−k / 48 h)
  v_now  mean hourly ACE speed over [t0 − 6 h, t0)  (else last valid hour within 48 h)
  v_rec  mean hourly speed within ±3 h of (t0 + k h − 655 h)   (655 h ≈ 27.2753-day synodic rotation)
```
Missing persistence → recurrence only; missing recurrence → persistence only; neither → 430 km/s.

## Input data and information cut-off

Only information published by t0 is used. Fetches happen after t0; every source is cut explicitly.

| Source | URL | Use | Cut-off rule |
|---|---|---|---|
| ACE RT 1-min SWEPAM daily files (NOAA) | `https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/YYYYMMDD_ace_swepam_1m.txt` | primary ACE speed, whole 29-day window; status S = 0 only | obs ≤ t0 − 5 min |
| CDAWeb HAPI `AC_K0_SWE` (Vp, 5-min) | `https://cdaweb.gsfc.nasa.gov/hapi/data?id=AC_K0_SWE&parameters=Vp&…` | fills RT gaps older than 24 h | obs ≤ t0 − 5 min |
| SWPC RTSW 1-min JSON | `https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json` | fills RT gaps in the last 24 h; `source == "ACE"`, `overall_quality == 0` only | obs ≤ t0 − 5 min |
| JSOC AIA 193 synoptic 1k (NRT and definitive) | `https://jsoc1.stanford.edu/data/aia/synoptic[/nrt]/YYYY/MM/DD/H{HH}00/` | images at slots t0 − 12k h, k = 0…10 | obs ≤ slot **and** published by t0 (below) |
| JSOC HMI 1k magnetograms `hmi.M_720s[_nrt]` | `https://jsoc1.stanford.edu/data/hmi/fits/YYYY/MM/DD/` | same slots; PFSS map from [t0 − 28 d, t0 − 2 h] | obs ≤ slot **and** published by t0 (below) |

* ACE: the real-time stream is published within a few minutes of observation, so a sample is used
  only if observed at least 5 minutes before t0. Hourly means need ≥ 10 valid 1-min samples
  (RT, RTSW) or ≥ 3 valid 5-min samples (K0); speeds outside 150–2000 km/s are treated as fill.
* SDO images: for each slot the nearest good-quality image observed within the 3 h before the slot
  is used (if none, the slot 6 h earlier). **Publication time** is the modification time shown in
  the JSOC directory listing (UTC, minute resolution; +1 min for the truncated seconds):
  * NRT file — used only if listing time + 1 min ≤ t0 (no listing time: obs ≤ t0 − 30 min for AIA,
    − 90 min for HMI);
  * definitive file — used if listing time + 1 min ≤ t0, or if obs ≤ t0 − 60 min (AIA) /
    − 180 min (HMI). Definitive files replace the NRT files of the same observation days later
    (HMI NRT files are then deleted), so for those the observation was already public as NRT
    within the stated lag (typical NRT lag: 15–20 min AIA, about 50 min HMI).
  HMI file times are TAI and are converted to UTC (−37 s).
* No external forecasts or model products are used.

## Reproducibility

Every download is logged to stdout as one JSON line with URL, fetch time, HTTP status, byte size,
SHA-256 and the server's `Last-Modified` header. Further lines record the ACE coverage, each
directory listing in which files were hidden by the publication cut-off (`cutoff_filter`), the
model status, and the stage that produced the forecast. A re-run long after t0 can differ slightly
because JSOC replaces NRT files with definitive ones; the log shows which files were used.

## Dependencies and assets

`public.ecr.aws/docker/library/python:3.12-slim` (the model was developed on Python 3.12) with
torch 2.4.1 (CPU), numpy 2.2.0, pandas 2.2.3, scipy 1.15.2, h5py 3.16.0, astropy 7.1.0,
sunpy 7.1.2, requests 2.32.5 (pinned in `requirements.txt`). Model weights: 30 files, about 40 MB,
inside the image. Image size about 1.95 GiB. Runs as non-root user `runner`, writes only to `/tmp`.
Measured locally: about 1 minute per run (120+ FITS files, about 130 MB, fetched in parallel),
peak memory about 1.1 GB.

## Local test

```
python tools/sync_model.py <path to git_repo checkout>     # creates model/ (not needed if model/ is present)
cd local_tests && ./test_local.sh 20261111T060000Z
```
Failure modes checked locally: no network (`--network none`), exhausted time budget, missing
Python runtime, malformed t0 — all exit 0 with 72 valid values.
