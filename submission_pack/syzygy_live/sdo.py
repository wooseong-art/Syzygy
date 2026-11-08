"""JSOC SDO image lookup (AIA synoptic 1k, HMI 1k magnetogram) with the t0 cut-off.

An image for slot time T (T <= t0) is eligible only if
  * obs_time <= T and obs_time >= T - tol        (same rule as training: nearest image not after the slot)
  * published by t0: listing mtime + 1 min <= t0 (the JSOC directory listing shows each file's
    modification time to the minute; +1 min covers the truncated seconds). If a listing row
    carries no mtime, fall back to obs_time <= t0 - FALLBACK_LAG.
The definitive archive is preferred when it is eligible (as in training); the publication check
automatically excludes definitive files that appeared after t0, so past-t0 tests do not leak.

URL layouts (verified 2026-11-08):
  AIA  {base}/YYYY/MM/DD/H{HH}00/AIA{YYYYMMDD}_{HHMMSS}_{WWWW}.fits   base = synoptic or synoptic/nrt
  HMI  {base}/YYYY/MM/DD/hmi.M_720s[_nrt].{YYYYMMDD}_{HHMMSS}_TAI.fits
"""
import os
import re
from datetime import datetime, timedelta, timezone

from . import net

AIA_ARCHIVE = "https://jsoc1.stanford.edu/data/aia/synoptic"
AIA_NRT = "https://jsoc1.stanford.edu/data/aia/synoptic/nrt"
HMI_BASE = "https://jsoc1.stanford.edu/data/hmi/fits"
TAI_MINUS_UTC = timedelta(seconds=37)
FALLBACK_LAG = {"aia": timedelta(minutes=30), "hmi": timedelta(minutes=90)}
UTC = timezone.utc

_ROW = re.compile(r'href="([^"?/][^"]*)">[^<]*</a>\s+(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2})?')
_AIA = re.compile(r"AIA(\d{8})_(\d{4}|\d{6})_(\d{4})\.fits$")
_HMI = re.compile(r"hmi\.M_720s(_nrt)?\.(\d{8})_(\d{6})_TAI\.fits$")
_listing_cache = {}


def list_dir(url):
    """[(name, mtime_utc or None)] from a JSOC directory listing ([] if missing)."""
    if url not in _listing_cache:
        body = net.get(url, timeout=20)
        rows = []
        for name, mt in _ROW.findall(body.decode("utf-8", "replace") if body else ""):
            rows.append((name, datetime.strptime(mt, "%d-%b-%Y %H:%M").replace(tzinfo=UTC) if mt else None))
        _listing_cache[url] = rows
    return _listing_cache[url]


def _published(kind, obs, mtime, t0):
    if mtime is not None:
        return mtime + timedelta(minutes=1) <= t0
    return obs <= t0 - FALLBACK_LAG[kind]


def _hours(a, b):
    h = a.replace(minute=0, second=0, microsecond=0)
    while h <= b:
        yield h
        h += timedelta(hours=1)


def find_aia(slot, t0, wavelength=193, tol=timedelta(minutes=60)):
    """Best eligible AIA file for slot -> dict(url, obs, mtime, product) or None."""
    not_after = min(slot, t0)
    for base, product in ((AIA_ARCHIVE, "aia_synoptic"), (AIA_NRT, "aia_nrt")):
        cands = []
        for h in _hours(slot - tol, not_after):
            url = f"{base}/{h:%Y/%m/%d}/H{h:%H}00/"
            for name, mtime in list_dir(url):
                m = _AIA.match(name)
                if not m or int(m.group(3)) != wavelength:
                    continue
                hms = m.group(2).ljust(6, "0")
                obs = datetime.strptime(m.group(1) + hms, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
                if slot - tol <= obs <= not_after and _published("aia", obs, mtime, t0):
                    cands.append((abs((slot - obs).total_seconds()), dict(url=url + name, obs=obs, mtime=mtime, product=product)))
        if cands:
            return min(cands, key=lambda c: c[0])[1]
    return None


def find_hmi(slot, t0, tol=timedelta(minutes=60)):
    """Best eligible HMI 1k magnetogram for slot -> dict(url, obs, mtime, product) or None (definitive preferred)."""
    not_after = min(slot, t0)
    cands = []
    day = (slot - tol).date()
    while day <= not_after.date():
        url = f"{HMI_BASE}/{day:%Y/%m/%d}/"
        for name, mtime in list_dir(url):
            m = _HMI.match(name)
            if not m:
                continue
            obs = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").replace(tzinfo=UTC) - TAI_MINUS_UTC
            if slot - tol <= obs <= not_after and _published("hmi", obs, mtime, t0):
                nrt = bool(m.group(1))
                cands.append((nrt, abs((slot - obs).total_seconds()),
                              dict(url=url + name, obs=obs, mtime=mtime, product="hmi_nrt" if nrt else "hmi_definitive")))
        day += timedelta(days=1)
    return min(cands, key=lambda c: (c[0], c[1]))[2] if cands else None


def download(cand, dest_dir):
    """Download a candidate to dest_dir; returns the local path or None."""
    body = net.get(cand["url"], timeout=60)
    if not body:
        return None
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, cand["url"].rsplit("/", 1)[1])
    with open(path, "wb") as f:
        f.write(body)
    return path
