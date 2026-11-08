"""
sw_utils.py
-----------------------------------
태양풍 예측 파이프라인 공통 유틸.
  (A) HTTP / JSOC 공개 FITS 탐색 + 다운로드
  (B) SDO 영상 전처리 (태양 반지름 정렬 리샘플링, AIA/HMI 정규화)
  (C) DONKI CME 파싱 + Drag-Based Model + per-bin CME feature
  (D) ACE 태양풍 시계열 (L2 / K0 / 실시간) → persistence / recurrence / target
  (E) 데이터 분할, 디바이스 선택
  (F) 코로나 홀 검출·특징 (면적, 중앙자오선 접근, 모양 유지) + 태양풍 상태 라벨 (Slow / Transition / HSS)

시간은 모두 tz-naive UTC pandas.Timestamp 로 다룬다. (HMI 의 TAI-UTC 37초 차이는 무시)
"""
from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates

import config as C

AU_KM   = 1.495978707e8
RSUN_KM = 6.957e5


# ====================================================
# (A) HTTP / JSOC 공개 FITS 탐색
# ====================================================
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "khu-solarwind-forecast/0.1"
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=C.DOWNLOAD_WORKERS + 4))


def http_get(url: str, retries: int = 3, timeout: int = C.HTTP_TIMEOUT,
             headers: Optional[dict] = None) -> Optional[requests.Response]:
    """GET with retry. 404 는 None, 그 외 오류는 재시도 후 예외."""
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=timeout, headers=headers)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


@lru_cache(maxsize=4096)
def list_dir(url: str) -> tuple[str, ...]:
    """Apache 디렉터리 리스팅에서 href 목록을 반환 (없으면 빈 tuple)."""
    r = http_get(url)
    if r is None:
        return ()
    return tuple(re.findall(r'href="([^"?/][^"]*)"', r.text))


_AIA_RE = re.compile(r"AIA(\d{8})_(\d{4}|\d{6})_(\d{4})\.fits$")
_HMI_RE = re.compile(r"hmi\.M_720s(_nrt)?\.(\d{8})_(\d{6})_TAI\.fits$")


def aia_wavelength(instrument: str) -> Optional[int]:
    """'aia193' → 193, 'aia171' → 171, AIA 가 아니면 None (영상 종류 이름 = 'aia' + 파장)."""
    m = re.fullmatch(r"aia(\d+)", instrument)
    return int(m.group(1)) if m else None


def _aia_candidates(t: pd.Timestamp, tol_min: int, not_after=None,
                    wavelength: int = C.AIA_WAVELENGTH) -> list[tuple[float, str]]:
    """목표 시각 t 주변 AIA synoptic 파일 후보 [(|dt| 분, url)] (아카이브 → NRT 순으로 탐색, 파장 기본 193).
    시간 폴더를 t 에 가까운 순서로 보고, 10분 이내 후보가 나오면 더 보지 않는다 (2분 cadence)."""
    lo = t - pd.Timedelta(minutes=tol_min)
    hi = t + pd.Timedelta(minutes=tol_min)
    if not_after is not None:
        hi = min(hi, not_after)
    hours = sorted(pd.date_range(lo.floor("h"), hi.floor("h"), freq="h"),
                   key=lambda h: abs(h + pd.Timedelta(minutes=30) - t))
    for base in (C.AIA_ARCHIVE_URL, C.AIA_NRT_URL):
        out = []
        for h in hours:
            url = f"{base}/{h:%Y/%m/%d}/H{h:%H}00/"
            for name in list_dir(url):
                m = _AIA_RE.match(name)
                if not m or int(m.group(3)) != wavelength:
                    continue
                hms = m.group(2).ljust(6, "0")
                ft = pd.Timestamp(f"{m.group(1)}T{hms[:2]}:{hms[2:4]}:{hms[4:]}")
                if lo <= ft <= hi:
                    out.append((abs((ft - t).total_seconds()) / 60, url + name))
            if out and min(out)[0] <= 10:
                break
        if out:                       # 아카이브에 있으면 NRT 는 보지 않음
            return sorted(out)
    return []


def _hmi_candidates(t: pd.Timestamp, tol_min: int, not_after=None) -> list[tuple[float, str]]:
    """목표 시각 t 주변 HMI 1k magnetogram 후보. 정규(M_720s) 를 NRT 보다 우선."""
    out = []
    days = pd.date_range((t - pd.Timedelta(minutes=tol_min)).floor("D"),
                         (t + pd.Timedelta(minutes=tol_min)).floor("D"), freq="D")
    for d in days:
        url = f"{C.HMI_URL}/{d:%Y/%m/%d}/"
        for name in list_dir(url):
            m = _HMI_RE.match(name)
            if not m:
                continue
            ft = pd.Timestamp(f"{m.group(2)}T{m.group(3)[:2]}:{m.group(3)[2:4]}:{m.group(3)[4:]}")
            dt = abs((ft - t).total_seconds()) / 60
            if dt <= tol_min and (not_after is None or ft <= not_after):
                out.append((dt + (0.5 if m.group(1) else 0.0), url + name))   # nrt 는 살짝 불리하게
    return sorted(out)


def read_fits_bytes(content: bytes):
    """JSOC 1k FITS (Rice 압축 CompImageHDU) → (data float32, header)."""
    from astropy.io import fits
    with fits.open(io.BytesIO(content)) as hdul:
        hdu = hdul[-1]
        return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()


def fetch_nearest_image(instrument: str, t: pd.Timestamp, tol_min: int = C.IMG_MATCH_TOL_MIN,
                        max_tries: int = 3, not_after: Optional[pd.Timestamp] = None):
    """t 에 가장 가까운 정상 품질 영상 → (data, header, url) 또는 None.
    not_after 를 주면 그 시각 이후 파일은 제외 (실시간 재현 시 미래 영상 차단)."""
    wl = aia_wavelength(instrument)
    cands = _aia_candidates(t, tol_min, not_after, wl) if wl else _hmi_candidates(t, tol_min, not_after)
    for _, url in cands[:max_tries]:
        r = http_get(url)
        if r is None:
            continue
        data, hdr = read_fits_bytes(r.content)
        if is_good_quality(instrument, data, hdr):
            return data, hdr, url
    return None


def is_good_quality(instrument: str, data: np.ndarray, hdr) -> bool:
    """QUALITY 의 심각한 비트(config 의 *_BAD_QUALITY_MASK) + 유효 픽셀 검사."""
    q = int(hdr.get("QUALITY", 0) or 0)
    is_aia = aia_wavelength(instrument) is not None
    mask = C.AIA_BAD_QUALITY_MASK if is_aia else C.HMI_BAD_QUALITY_MASK
    if q & mask:
        return False
    finite = np.isfinite(data)
    if is_aia:                                  # AIA 는 디스크 밖도 값이 있어야 정상
        return finite.mean() > 0.9 and np.nanmedian(data) > 0
    return finite.mean() > 0.5                  # HMI 는 디스크 밖이 NaN (정상 ~0.67)


# ====================================================
# (B) SDO 영상 전처리
# ====================================================
def resample_to_sun_grid(data: np.ndarray, hdr, res: int = C.IMG_RES, fov: float = C.IMG_FOV_RSUN):
    """태양 중심/반지름/회전을 header 로 보정해 [-fov, fov] R_sun 정사각 격자로 리샘플링.
    출력 row 0 = 남쪽 (FITS 관례 유지, imshow(origin='lower') 로 보면 북쪽이 위).
    반환: (image[res,res] float32, r[res,res] 태양중심 거리 R_sun 단위)"""
    cdelt = float(hdr["CDELT1"])
    rsun_arcsec = float(hdr["RSUN_OBS"])
    rot = np.deg2rad(float(hdr.get("CROTA2", 0.0) or 0.0))
    xc, yc = float(hdr["CRPIX1"]) - 1, float(hdr["CRPIX2"]) - 1
    crval1, crval2 = float(hdr.get("CRVAL1", 0.0) or 0.0), float(hdr.get("CRVAL2", 0.0) or 0.0)

    g = (np.arange(res) + 0.5) / res * 2 * fov - fov
    X, Y = np.meshgrid(g, g)
    wx, wy = X * rsun_arcsec - crval1, Y * rsun_arcsec - crval2
    cos, sin = np.cos(rot), np.sin(rot)
    col = xc + (cos * wx + sin * wy) / cdelt
    row = yc + (-sin * wx + cos * wy) / cdelt

    filled = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    scale = 2 * fov * rsun_arcsec / cdelt / res          # 입력 px / 출력 px
    if scale > 1.5:                                       # 다운샘플 anti-aliasing
        filled = gaussian_filter(filled, sigma=0.5 * scale)
    img = map_coordinates(filled, [row, col], order=1, mode="constant", cval=0.0)
    return img.astype(np.float32), np.sqrt(X ** 2 + Y ** 2).astype(np.float32)


def normalize_aia(img: np.ndarray, r: np.ndarray) -> np.ndarray:
    """log10(I / 디스크 중앙값) → [0, 1]. 노출시간·장비 열화·태양주기 밝기 변화를 상쇄."""
    disk = img[(r < 0.95) & (img > 0)]
    med = np.median(disk) if disk.size else 1.0
    v = np.log10(np.clip(img / max(med, 1e-3), 10 ** C.AIA_LOG_MIN, 10 ** C.AIA_LOG_MAX))
    v = (v - C.AIA_LOG_MIN) / (C.AIA_LOG_MAX - C.AIA_LOG_MIN)
    v[r > C.IMG_FOV_RSUN] = 0.0                 # 원 밖 모서리 제거 (roll 기동 영상의 잘린 모서리와 일관되게)
    return v.astype(np.float32)


def normalize_hmi(img: np.ndarray, r: np.ndarray) -> np.ndarray:
    """arcsinh(B / s) / arcsinh(clip / s) → [-1, 1], 디스크 밖(r>0.99)은 0."""
    b = np.clip(img, -C.HMI_CLIP, C.HMI_CLIP)
    v = np.arcsinh(b / C.HMI_ASINH_SCALE) / np.arcsinh(C.HMI_CLIP / C.HMI_ASINH_SCALE)
    v[r > 0.99] = 0.0
    return v.astype(np.float32)


def preprocess_sdo(instrument: str, data: np.ndarray, hdr) -> np.ndarray:
    img, r = resample_to_sun_grid(data, hdr)
    return normalize_aia(img, r) if aia_wavelength(instrument) else normalize_hmi(img, r)


def image_path(instrument: str, t: pd.Timestamp):
    return C.IMG_DIR / instrument / f"{t:%Y%m%d_%H%M}.npy"


# ====================================================
# (C) DONKI CME
# ====================================================
def _ts(x) -> pd.Timestamp:
    """DONKI 시각 문자열('2024-05-08T09:30Z') → tz-naive UTC Timestamp (없으면 NaT)."""
    if x is None or x == "":
        return pd.NaT
    t = pd.Timestamp(x)
    return t.tz_convert(None) if t.tzinfo is not None else t


def fetch_donki_cme(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    url = f"{C.DONKI_API_URL}/CME?startDate={start:%Y-%m-%d}&endDate={end:%Y-%m-%d}"
    r = http_get(url, timeout=120)
    return [] if r is None or not r.text.strip() else r.json()


def flatten_donki(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DONKI CME JSON → (분석 테이블, ENLIL 실행 테이블)."""
    rows, enlil = [], []
    for c in records:
        for a in c.get("cmeAnalyses") or []:
            rows.append(dict(
                activityID=c["activityID"], startTime=_ts(c.get("startTime")),
                time21_5=_ts(a.get("time21_5")), latitude=a.get("latitude"),
                longitude=a.get("longitude"), halfAngle=a.get("halfAngle"),
                speed=a.get("speed"), type=a.get("type"),
                isMostAccurate=bool(a.get("isMostAccurate")),
                levelOfData=a.get("levelOfData"),
                measurementTechnique=a.get("measurementTechnique"),
                submissionTime=_ts(a.get("submissionTime")),
            ))
            for e in a.get("enlilList") or []:
                enlil.append(dict(
                    activityID=c["activityID"], link=e.get("link"),
                    modelCompletionTime=_ts(e.get("modelCompletionTime")),
                    estimatedShockArrivalTime=_ts(e.get("estimatedShockArrivalTime")),
                    isEarthGB=bool(e.get("isEarthGB")),
                    isEarthMinorImpact=bool(e.get("isEarthMinorImpact")),
                ))
    a_df = pd.DataFrame(rows)
    e_df = pd.DataFrame(enlil)
    if len(e_df):
        e_df = e_df.drop_duplicates("link")
    return a_df, e_df


def load_donki_tables():
    parse = lambda p, cols: pd.read_csv(p, parse_dates=cols) if p.exists() else pd.DataFrame()
    a = parse(C.CME_CSV, ["startTime", "time21_5", "submissionTime"])
    e = parse(C.ENLIL_CSV, ["modelCompletionTime", "estimatedShockArrivalTime"])
    return a, e


def dbm_arrival(v0: float, r0_km: float = 21.5 * RSUN_KM, r_target_km: float = AU_KM,
                gamma: float = C.DBM_GAMMA, w: float = C.DBM_W) -> tuple[float, float]:
    """Drag-Based Model (Vršnak+ 2013) 해석해로 도착 소요시간(h)과 도착 속도(km/s)."""
    dv = v0 - w
    if abs(dv) < 1e-6:
        return (r_target_km - r0_km) / w / 3600, w
    s = 1.0 if dv > 0 else -1.0
    r = lambda t: s / gamma * np.log1p(s * gamma * dv * t) + w * t + r0_km
    lo, hi = 0.0, 20 * 86400.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if r(mid) < r_target_km else (lo, mid)
    t = 0.5 * (lo + hi)
    v = dv / (1 + s * gamma * dv * t) + w
    return t / 3600, v


def earth_hit_factor(lat, lon, half_angle) -> float:
    """CME 진행 방향(HEEQ lat/lon)과 지구(0,0)의 각거리로 계산한 0~1 영향 가중치."""
    ha = C.CME_DEFAULT_HALF_ANGLE if pd.isna(half_angle) else float(half_angle)
    if pd.isna(lat) or pd.isna(lon):
        return 0.5                                # 방향 미상이면 중간값
    sep = np.degrees(np.arccos(np.clip(np.cos(np.radians(lat)) * np.cos(np.radians(lon)), -1, 1)))
    if sep <= ha:
        return 1.0
    return float(max(0.0, 1.0 - (sep - ha) / C.CME_HIT_MARGIN_DEG))


def select_known_cmes(a_df: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """t0 시점에 '알려져 있던' CME 분석만, CME 당 하나씩 고른다."""
    if a_df.empty:
        return a_df
    lag = pd.Timedelta(hours=C.CME_REPORT_LAG_HOURS)
    m = ((a_df["time21_5"] + lag <= t0) &
         (a_df["time21_5"] >= t0 - pd.Timedelta(days=C.CME_LOOKBACK_DAYS)) &
         a_df["speed"].notna())
    d = a_df[m].copy()
    if d.empty:
        return d
    d["_rt"] = (d["levelOfData"] == 0) if C.CME_PREFER_REALTIME else False
    d = d.sort_values(["activityID", "_rt", "isMostAccurate", "submissionTime"],
                      ascending=[True, False, False, False])
    return d.drop_duplicates("activityID").drop(columns="_rt")


def cme_inputs(t0: pd.Timestamp, a_df: pd.DataFrame, e_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """DONKI CME → 태양풍 속도 예측용 보조 입력.
    cme_bin [N_BINS, 3] : bin 별 [DBM 도착 가중치, DBM 도착속도/1000 x 가중치, ENLIL 도착 가중치]
    cme_vec [CME_VEC_DIM]: [지구향 CME 수/5, 최대속도/1000, 최대 half-angle/90]
                           + 지구 영향 상위 K개 CME 원값 [hit, 경과시간/72h, 속도/1000, lat/90, lon/90, half-angle/90]"""
    centers = np.array([(k + 0.5) * C.BIN_HOURS for k in range(C.N_BINS)])   # t0 기준 시간(h)
    f = np.zeros((C.N_BINS, C.CME_N_PER_BIN), dtype=np.float32)
    top = []
    n_hit, vmax, hamax = 0, 0.0, 0.0
    sig = C.CME_TIME_SIGMA_HOURS

    for _, c in select_known_cmes(a_df, t0).iterrows():
        hit = earth_hit_factor(c["latitude"], c["longitude"], c["halfAngle"])
        if hit <= 0:
            continue
        n_hit += 1
        ha = float(c["halfAngle"]) if pd.notna(c["halfAngle"]) else C.CME_DEFAULT_HALF_ANGLE
        vmax = max(vmax, float(c["speed"]))
        hamax = max(hamax, ha)
        age_h = (t0 - c["time21_5"]).total_seconds() / 3600
        dt_h, v_arr = dbm_arrival(float(c["speed"]))
        g = hit * np.exp(-0.5 * ((centers - (dt_h - age_h)) / sig) ** 2)
        f[:, 0] = np.maximum(f[:, 0], g)
        f[:, 1] = np.maximum(f[:, 1], g * v_arr / 1000.0)
        lat = 0.0 if pd.isna(c["latitude"]) else float(c["latitude"])
        lon = 0.0 if pd.isna(c["longitude"]) else float(c["longitude"])
        top.append((hit, -age_h, [hit, age_h / 72.0, float(c["speed"]) / 1000.0, lat / 90.0, lon / 90.0, ha / 90.0]))

    if not e_df.empty:
        m = ((e_df["modelCompletionTime"] <= t0) &
             e_df["estimatedShockArrivalTime"].notna() &
             (e_df["estimatedShockArrivalTime"] >= t0 - pd.Timedelta(days=1)) &
             (e_df["modelCompletionTime"] >= t0 - pd.Timedelta(days=C.CME_LOOKBACK_DAYS)))
        for _, e in e_df[m].iterrows():
            t_arr = (e["estimatedShockArrivalTime"] - t0).total_seconds() / 3600
            w = 0.5 if (e["isEarthGB"] or e["isEarthMinorImpact"]) else 1.0
            f[:, 2] = np.maximum(f[:, 2], w * np.exp(-0.5 * ((centers - t_arr) / sig) ** 2))

    vec = np.zeros(C.CME_VEC_DIM, dtype=np.float32)
    vec[:C.CME_N_GLOBAL] = [n_hit / 5.0, vmax / 1000.0, hamax / 90.0]
    top.sort(key=lambda x: (x[0], x[1]), reverse=True)          # 영향 가중치 큰 순, 같으면 최근 것
    for i, (_, _, feats) in enumerate(top[:C.CME_TOP_K]):
        s = C.CME_N_GLOBAL + i * C.CME_TOP_FEATS
        vec[s:s + C.CME_TOP_FEATS] = feats
    return f, vec


# ====================================================
# (D) ACE 태양풍 시계열 (정답지)
# ====================================================
def _clean_v(v: pd.Series) -> pd.Series:
    lo, hi = C.ACE_V_VALID_RANGE
    return v.where((v >= lo) & (v <= hi)).sort_index()


def hourly_mean(v: pd.Series, min_count: int) -> pd.Series:
    """고해상도 V → hourly 평균 (라벨 = hour 시작 시각). 유효 점이 min_count 미만이면 NaN."""
    if v.empty:
        return v
    g = _clean_v(v).resample("1h")
    return g.mean().where(g.count() >= min_count)


@lru_cache(maxsize=8)
def hapi_coverage(dataset: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """데이터셋 가용 구간 (HAPI info). info 가 실패하면 넓게 잡는다."""
    try:
        d = http_get(f"{C.ACE_HAPI_URL}/info?id={dataset}").json()
        return _ts(d["startDate"]), _ts(d["stopDate"])
    except Exception:
        return pd.Timestamp("1998-01-01"), pd.Timestamp.utcnow().tz_localize(None)


def _chunks(dataset: str, start, end, months: int):
    """[start, end) 를 데이터셋 가용 구간으로 자르고 months 개월 단위로 나눈다."""
    a, b = hapi_coverage(dataset)
    s, end = max(pd.Timestamp(start), a), min(pd.Timestamp(end), b + pd.Timedelta(seconds=1))
    while s < end:
        e = min(s + pd.DateOffset(months=months), end)
        yield s, e
        s = e


def fetch_hapi_v(dataset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """CDAWeb HAPI 에서 Vp 를 [start, end) 구간으로 받는다.
    HAPI 는 '데이터 없음'(1201) 을 JSON 으로 돌려준다. 2026-10-04 에는 데이터가 있는 2016 년 이후 구간에도
    1201 을 돌려줬기 때문에, 여기서 비는 구간은 load_ace_hourly 가 CDAS 웹서비스로 다시 채운다."""
    parts = []
    for s, e in _chunks(dataset, start, end, months=12):
        r = http_get(f"{C.ACE_HAPI_URL}/data?id={dataset}&parameters=Vp"
                     f"&time.min={s:%Y-%m-%dT%H:%M:%SZ}&time.max={e:%Y-%m-%dT%H:%M:%SZ}", timeout=300)
        if r is None or not r.text.strip() or r.text.lstrip().startswith("{"):
            continue                                  # 빈 응답 또는 HAPI 상태 JSON
        d = pd.read_csv(io.StringIO(r.text), header=None, names=["time", "V"])
        parts.append(pd.Series(pd.to_numeric(d["V"], errors="coerce").to_numpy(float),
                               index=pd.to_datetime(d["time"].str.rstrip("Z"), format="ISO8601")))
    return _clean_v(pd.concat(parts)) if parts else pd.Series(dtype=float)


def fetch_cdas_v(dataset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """CDAWeb CDAS 웹서비스 (text 형식) 에서 Vp 를 [start, end) 구간으로 받는다 — HAPI 대체 경로.
    6개월 이상 요청은 503 이 나서 1개월 단위로 요청하고, 응답 JSON 이 가리키는 임시 text 파일의
    'dd-mm-yyyy hh:mm:ss.ms  값' 행을 읽는다. 계속 실패하는 달은 경고만 남기고 건너뛴다."""
    parts = []
    for s, e in _chunks(dataset, start, end, months=1):
        try:
            r = http_get(f"{C.CDAS_URL}/{dataset}/data/{s:%Y%m%dT%H%M%SZ},{e:%Y%m%dT%H%M%SZ}/Vp?format=text",
                         retries=4, timeout=300, headers={"Accept": "application/json"})
            files = (r.json().get("FileDescription") or []) if r is not None else []
            for f in files:
                rows = [l for l in http_get(f["Name"], timeout=300).text.splitlines() if l[:2].isdigit()]
                if not rows:
                    continue
                d = pd.read_csv(io.StringIO("\n".join(rows)), sep=r"\s+", header=None, names=["date", "time", "V"])
                v = pd.Series(pd.to_numeric(d["V"], errors="coerce").to_numpy(float),
                              index=pd.to_datetime(d["date"] + " " + d["time"], format="%d-%m-%Y %H:%M:%S.%f"))
                parts.append(v[(v.index >= s) & (v.index < e)])
        except (requests.RequestException, ValueError) as ex:
            print(f"  ! CDAS {dataset} {s:%Y-%m} 실패, 건너뜀: {ex}")
    return _clean_v(pd.concat(parts)) if parts else pd.Series(dtype=float)


def fetch_ace_rt_day(day: pd.Timestamp) -> pd.Series:
    """ACE 실시간 1분 SWEPAM 일별 파일 (SWPC 형식, S==0 인 정상 레코드만)."""
    r = http_get(C.ACE_RT_DAILY_URL.format(date=day))
    lines = [l for l in r.text.splitlines() if l[:1].isdigit()] if r is not None else []
    if not lines:
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", header=None, dtype={3: str},
                     names=["yr", "mo", "da", "hhmm", "mjd", "sod", "S", "Np", "V", "T"])
    t = pd.to_datetime(df["yr"].astype(str) + "-" + df["mo"].astype(str) + "-" + df["da"].astype(str)
                       + " " + df["hhmm"].str.zfill(4), format="%Y-%m-%d %H%M")
    return _clean_v(pd.Series(df["V"].where(df["S"] == 0).to_numpy(float), index=t))


def fetch_rtsw_v() -> pd.Series:
    """SWPC RTSW 1분 JSON (최근 ~1일) 에서 ACE 속도. ACE 결측 구간은 설정 시 운영(active) 위성으로 보충."""
    df = pd.DataFrame(http_get(C.RTSW_URL).json())
    df["time"] = pd.to_datetime(df["time_tag"])
    if "overall_quality" in df:
        df = df[df["overall_quality"].fillna(0) == 0]
    pick = lambda m: df[m].set_index("time")["proton_speed"].astype(float).sort_index()
    v = pick(df["source"] == C.RTSW_SOURCE)
    if C.RTSW_FALLBACK_ACTIVE:
        alt = pick(df["active"].astype(bool) & (df["source"] != C.RTSW_SOURCE))
        n_fill = alt.index.difference(v.dropna().index).size
        if n_fill > 60:
            print(f"  ! RTSW: {C.RTSW_SOURCE} 결측 {n_fill} 분을 운영 위성 값으로 보충")
        v = v.dropna().combine_first(alt)
    return _clean_v(v)


def load_ace_hourly(start, end, use_realtime: bool = False) -> pd.DataFrame:
    """[start, end) ACE hourly V 를 우선순위대로 이어 붙인다.
    L2(AC_H2_SWE; HAPI → CDAS) > K0(AC_K0_SWE; HAPI → CDAS) > [use_realtime] RT 1분 일별 아카이브 > RTSW JSON
    반환: DataFrame(index=hour 시작 시각, V, src)"""
    idx = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end), freq="h", inclusive="left")
    out = pd.DataFrame({"V": np.nan, "src": ""}, index=idx)

    def missing_runs(max_gap=pd.Timedelta(days=31)):
        """결측 hour 들을 31일보다 가까운 것끼리 묶은 [시작, 끝) 구간 목록 (결측 구간만 다시 요청)."""
        miss = out.index[out["V"].isna()]
        if len(miss) == 0:
            return []
        cut = np.flatnonzero(np.diff(miss.values) > max_gap.to_timedelta64()) + 1
        return [(miss[g[0]], miss[g[-1]] + pd.Timedelta(hours=1)) for g in np.split(np.arange(len(miss)), cut)]

    def rt_archive(a, b):
        days = pd.date_range(a.floor("D"), (b - pd.Timedelta(seconds=1)).floor("D"), freq="D")
        if len(days) > 62:
            print(f"  ! RT 아카이브는 최근 구간 보충용 — {len(days)} 일 중 마지막 62 일만 받음")
            days = days[-62:]
        parts = [fetch_ace_rt_day(d) for d in days]
        return hourly_mean(pd.concat(parts), C.ACE_MIN_COUNT_1MIN) if parts else pd.Series(dtype=float)

    rtsw_cache = []

    def rtsw(a, b):                                   # 최근 ~1일 JSON 하나라 구간과 무관하게 한 번만 받는다
        if not rtsw_cache:
            rtsw_cache.append(hourly_mean(fetch_rtsw_v(), C.ACE_MIN_COUNT_1MIN))
        return rtsw_cache[0]

    k0 = lambda fetch: lambda a, b: hourly_mean(fetch(C.ACE_K0_DATASET, a, b), C.ACE_MIN_COUNT_5MIN)
    layers = [("L2", lambda a, b: fetch_hapi_v(C.ACE_L2_DATASET, a, b)),
              ("L2", lambda a, b: fetch_cdas_v(C.ACE_L2_DATASET, a, b)),    # HAPI 가 비워 둔 구간
              ("K0", k0(fetch_hapi_v)),
              ("K0", k0(fetch_cdas_v))]
    if use_realtime:
        layers += [("RT", rt_archive), ("RTSW", rtsw)]
    for name, fn in layers:
        runs = missing_runs()
        if not runs:
            break
        got = [g for g in (fn(a, b) for a, b in runs) if not g.empty]
        if not got:
            continue
        s = pd.concat(got)
        s = s[~s.index.duplicated()].reindex(idx)
        fill = out["V"].isna() & s.notna()
        out.loc[fill, "V"] = s[fill]
        out.loc[fill, "src"] = name
    return out


@dataclass
class HourlySeries:
    """정시(hourly) 등간격 시계열을 numpy 로 빠르게 슬라이싱하기 위한 래퍼."""
    base: pd.Timestamp
    values: np.ndarray

    @classmethod
    def from_series(cls, s: pd.Series) -> "HourlySeries":
        s = s.sort_index()
        s = s[~s.index.duplicated()]
        idx = pd.date_range(s.index[0].floor("h"), s.index[-1].floor("h"), freq="h")
        return cls(idx[0], s.reindex(idx).to_numpy(dtype=np.float32))

    def window(self, start: pd.Timestamp, n: int) -> np.ndarray:
        i0 = int((start - self.base) / pd.Timedelta(hours=1))
        out = np.full(n, np.nan, dtype=np.float32)
        lo, hi = max(i0, 0), min(i0 + n, len(self.values))
        if hi > lo:
            out[lo - i0:hi - i0] = self.values[lo:hi]
        return out


def bin_means(hourly: np.ndarray, bin_h: int = C.BIN_HOURS, min_valid: int = C.MIN_VALID_HOURS_PER_BIN):
    a = hourly.reshape(-1, bin_h)
    n = np.isfinite(a).sum(1)
    with np.errstate(invalid="ignore"):
        m = np.nansum(np.nan_to_num(a, nan=0.0), 1) / np.maximum(n, 1)
    m[n < min_valid] = np.nan
    return m.astype(np.float32)


def solar_wind_inputs(v: HourlySeries, t0: pd.Timestamp):
    """→ (persist[24], recur[N_BINS], target[N_BINS])  모두 km/s, 결측 NaN."""
    persist = v.window(t0 - pd.Timedelta(hours=C.PERSIST_HOURS), C.PERSIST_HOURS)
    recur = bin_means(v.window(t0 - pd.Timedelta(hours=C.RECURRENCE_HOURS), C.HORIZON_HOURS))
    target = bin_means(v.window(t0, C.HORIZON_HOURS))
    return persist, recur, target


REC_WINDOW_FEATURES = ("center", "mean", "max", "min", "range", "std", "slope", "left_mean", "right_mean", "delta",
                       "center_ok", "win_ok", "left_ok", "right_ok", "valid_frac")
assert len(REC_WINDOW_FEATURES) == C.REC_WINDOW_FEAT_DIM


def recurrence_ext_window(v: "HourlySeries", t0: pd.Timestamp) -> np.ndarray:
    """기존 recurrence 16 bin 앞뒤로 REC_PAD_HOURS 씩 늘린 27일 전 6 h 평균 [N_BINS + 2·pad_bins] (km/s, 결측 NaN).
    가운데 N_BINS 개 (pad_bins : pad_bins + N_BINS) 가 solar_wind_inputs 의 recur 와 같다.
    마지막 시간도 t0 - (RECURRENCE_HOURS - HORIZON_HOURS - REC_PAD_HOURS) = t0 - 511 h 이전 (미래 정보 없음)."""
    end_offset = C.HORIZON_HOURS + C.REC_PAD_HOURS - C.RECURRENCE_HOURS       # t0 기준 창 끝 (h, 음수여야 함)
    assert end_offset < 0, "recurrence 창이 예보 발행 시각 이후까지 닿음"
    start = t0 - pd.Timedelta(hours=C.RECURRENCE_HOURS + C.REC_PAD_HOURS)
    return bin_means(v.window(start, C.HORIZON_HOURS + 2 * C.REC_PAD_HOURS))


def recurrence_window_features(ext_kms: np.ndarray, window_h: int, use_history: bool = True) -> np.ndarray:
    """recur_ext_kms [N, N_BINS + 2·pad_bins] → bin 별 ±window_h 창 요약 특징 [N, N_BINS, REC_WINDOW_FEAT_DIM].
    값 (정규화): center, mean, max, min (속도와 같은 (v - V_MEAN)/V_STD), range, std, delta (÷V_STD),
                slope (창 안 선형 기울기, V_STD/일), left_mean / right_mean (기준 bin 앞 / 뒤 절반 평균)
    mask: center_ok (기준 bin 유효), win_ok (창 유효 비율 ≥ REC_WINDOW_MIN_FRAC), left_ok, right_ok (각 절반 기준),
          valid_frac (창 유효 비율). mask 가 0 인 값은 0 으로 둔다 (기존 history_inputs 와 같은 방식).
    window_h 가 달라도 특징 구조·차원은 같다."""
    ext = np.asarray(ext_kms, np.float32)
    pad, w = C.REC_PAD_HOURS // C.BIN_HOURS, window_h // C.BIN_HOURS
    if not (0 < w <= pad and window_h % C.BIN_HOURS == 0):
        raise ValueError(f"window_h {window_h} 는 {C.BIN_HOURS} h 배수이고 REC_PAD_HOURS 이하여야 함")
    if ext.shape[-1] != C.N_BINS + 2 * pad:
        raise ValueError(f"recur_ext_kms 길이 {ext.shape[-1]} ≠ {C.N_BINS + 2 * pad}")
    idx = pad + np.arange(C.N_BINS)[:, None] + np.arange(-w, w + 1)[None]        # [N_BINS, 2w+1]
    win = ext[:, idx]                                                             # [N, N_BINS, 2w+1]
    ok = np.isfinite(win)
    n = ok.sum(-1)
    frac = n / (2 * w + 1)
    x = np.nan_to_num(win)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (x * ok).sum(-1) / np.maximum(n, 1)
        vmax = np.where(ok, win, -np.inf).max(-1)
        vmin = np.where(ok, win, np.inf).min(-1)
        std = np.sqrt(((x - mean[..., None]) ** 2 * ok).sum(-1) / np.maximum(n, 1))
        t = np.arange(-w, w + 1, dtype=np.float32) * C.BIN_HOURS / 24.0            # 일 단위
        tm = (t * ok).sum(-1) / np.maximum(n, 1)
        slope = ((t - tm[..., None]) * (x - mean[..., None]) * ok).sum(-1) / np.maximum(
            ((t - tm[..., None]) ** 2 * ok).sum(-1), 1e-6)
        half = lambda sl: ((x[..., sl] * ok[..., sl]).sum(-1) / np.maximum(ok[..., sl].sum(-1), 1),
                           ok[..., sl].sum(-1) >= C.REC_WINDOW_MIN_FRAC * w)
        left, left_ok = half(slice(0, w))
        right, right_ok = half(slice(w + 1, 2 * w + 1))
    center_ok = ok[..., w]
    win_ok = (frac >= C.REC_WINDOW_MIN_FRAC) & (n >= 3)
    if not use_history:
        center_ok = win_ok = left_ok = right_ok = np.zeros_like(center_ok)
        frac = np.zeros_like(frac)
    nz = lambda a, m: np.where(m, a, 0.0)
    with np.errstate(invalid="ignore"):                  # 창 전체 결측이면 max/min 이 ±inf → mask 0 자리라 0 으로 바뀜
        feats = [nz(norm_v(win[..., w]), center_ok), nz(norm_v(mean), win_ok), nz(norm_v(vmax), win_ok),
                 nz(norm_v(vmin), win_ok), nz((vmax - vmin) / C.V_STD, win_ok), nz(std / C.V_STD, win_ok),
                 nz(slope / C.V_STD, win_ok), nz(norm_v(left), left_ok), nz(norm_v(right), right_ok),
                 nz((right - left) / C.V_STD, left_ok & right_ok),
                 center_ok, win_ok, left_ok, right_ok, frac]
    return np.stack([np.asarray(f, np.float32) for f in feats], -1)


def norm_v(x):
    return (x - C.V_MEAN) / C.V_STD


def denorm_v(x):
    return x * C.V_STD + C.V_MEAN


def history_inputs(persist_kms: np.ndarray, recur_kms: np.ndarray, use_history: bool = True):
    """ACE 과거값 → 모델 입력 (정규화, 결측 0 + 유효 mask). 배치 축이 있어도 된다.
    persist [..., 24] → [..., 48] (값 24 | mask 24),  recur [..., N_BINS] → [..., N_BINS, 2] (값, mask)
    use_history=False 면 mask 까지 전부 0 → 모델 기준선도 0 (= V_MEAN) 이 된다."""
    p, r = norm_v(np.asarray(persist_kms, np.float32)), norm_v(np.asarray(recur_kms, np.float32))
    pm, rm = np.isfinite(p).astype(np.float32), np.isfinite(r).astype(np.float32)
    if not use_history:
        pm[:], rm[:] = 0.0, 0.0
    p, r = np.nan_to_num(p) * pm, np.nan_to_num(r) * rm
    return np.concatenate([p, pm], axis=-1), np.stack([r, rm], axis=-1)


def sample_inputs(t0: pd.Timestamp, v: HourlySeries, a_df: pd.DataFrame, e_df: pd.DataFrame) -> dict:
    """t0 하나의 영상 외 입력과 정답지 (04 학습 데이터와 07 실시간이 같은 함수를 쓴다)."""
    persist, recur, target = solar_wind_inputs(v, t0)
    cme_bin, cme_vec = cme_inputs(t0, a_df, e_df)
    return dict(cme_bin=cme_bin, cme_vec=cme_vec, persist_kms=persist, recur_kms=recur, target_kms=target)


def downsample(img: np.ndarray, res: int = C.TRAIN_RES) -> np.ndarray:
    """IMG_RES² → res² 블록 평균 (256 → 128 이면 2x2 평균)."""
    f = img.shape[-1] // res
    return img.reshape(*img.shape[:-2], res, f, res, f).mean(axis=(-3, -1))


# ====================================================
# (E) 분할 / 디바이스
# ====================================================
SYNODIC_DAYS = 27.2753


def make_split(t0: pd.Series) -> pd.Series:
    """각 샘플에 'train' | 'valid' | 'test' | 'purge' 라벨."""
    t0 = pd.to_datetime(t0)
    if C.SPLIT_MODE == "year":
        year_lab = {y: n for n, ys in (("train", C.TRAIN_YEARS), ("valid", C.VALID_YEARS),
                                       ("test", C.TEST_YEARS)) for y in ys}
        label_of = lambda t: year_lab.get(t.year, "purge")
        years = range(t0.min().year, t0.max().year + 2)
        candidates = pd.to_datetime([f"{y}-01-01" for y in years])
    elif C.SPLIT_MODE == "rotation_block":
        epoch = pd.Timestamp("2010-01-01")
        period = pd.Timedelta(days=SYNODIC_DAYS)
        pat = C.ROTATION_BLOCK_PATTERN
        label_of = lambda t: pat[int((t - epoch) // period) % len(pat)]
        b0, b1 = int((t0.min() - epoch) // period), int((t0.max() - epoch) // period) + 1
        candidates = pd.DatetimeIndex([epoch + period * b for b in range(b0, b1 + 1)])
    else:
        raise ValueError(C.SPLIT_MODE)
    lab = t0.map(label_of)

    # 라벨이 바뀌는 경계 근처(입력 이력 ~ 예측 구간이 경계를 가로지르는) 샘플 purge
    eps = pd.Timedelta(seconds=1)
    boundaries = pd.DatetimeIndex([b for b in candidates if label_of(b - eps) != label_of(b)])
    if len(boundaries) == 0:
        return lab
    # 경계 직전: 예측 구간(t0 ~ t0+HORIZON)이 경계를 넘음 / 경계 직후: 입력 영상 이력이 경계 이전
    before = np.timedelta64(C.SPLIT_PURGE_BEFORE_HOURS, "h")
    after = np.timedelta64(C.SPLIT_PURGE_AFTER_HOURS, "h")
    tb = boundaries.values
    pos = np.searchsorted(tb, t0.values, side="right")
    prev_b = tb[np.clip(pos - 1, 0, len(tb) - 1)]
    next_b = tb[np.clip(pos, 0, len(tb) - 1)]
    near = (((t0.values - prev_b) >= np.timedelta64(0, "h")) & ((t0.values - prev_b) < after)) |            (((next_b - t0.values) > np.timedelta64(0, "h")) & ((next_b - t0.values) < before))
    out = lab.copy()
    out[near] = "purge"
    return out


def get_device(pref: str = C.DEVICE):
    import torch
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in ("auto", "directml"):
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            pass
    return torch.device("cpu")


def gpu_mb(device) -> float:
    """이 프로세스가 DirectML 장치에 할당해 둔 메모리 (MB, torch_directml.gpu_memory 합). DirectML 이 아니면 NaN."""
    if getattr(device, "type", "") != "privateuseone":
        return float("nan")
    import torch_directml
    return float(sum(torch_directml.gpu_memory(device.index or 0, 1)))


def ram_peak_mb() -> float:
    """이 프로세스의 RAM 최대 사용량 (MB, Windows PeakWorkingSetSize). 다른 OS 는 resource.ru_maxrss."""
    import ctypes
    import os
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return float("nan")
    return c.PeakWorkingSetSize / 2 ** 20


# ====================================================
# (F) 코로나 홀 (AIA 193 + HMI) 검출·특징, 태양풍 상태 라벨
# ====================================================
CH_FEATURES = ["area_disk", "area_cm", "area_east", "area_west", "lon_mean", "lat_mean",
               "polarity", "unipolarity", "darkness", "n_regions"]
CH_IMG_COLS = CH_FEATURES + ["iou_12h", "iou_24h"]          # dataset.h5 의 ch_img 열 순서


def b0_deg(t) -> float:
    """지구에서 본 태양 자전축 기울기 B0 (deg)."""
    from sunpy.coordinates import sun
    return float(sun.B0(pd.Timestamp(t).to_pydatetime()).deg)


@lru_cache(maxsize=128)
def _disk_geometry(res: int, b0_tenth: int):
    """영상 픽셀 → (디스크 안, μ, 위도, Stonyhurst 경도, 투영 보정한 구면 면적[R_sun²]). b0 는 0.1° 단위 정수.
    영상 관례: row 0 = 남쪽, x > 0 = 서쪽 → 경도는 서쪽이 +, 자전으로 동쪽(-) 에서 중앙(0) 으로 다가온다."""
    b0 = np.radians(b0_tenth / 10)
    g = (np.arange(res) + 0.5) / res * 2 * C.IMG_FOV_RSUN - C.IMG_FOV_RSUN
    x, y = np.meshgrid(g, g)
    rho2 = x ** 2 + y ** 2
    on = rho2 < 1.0
    mu = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
    lat = np.degrees(np.arcsin(np.clip(y * np.cos(b0) + mu * np.sin(b0), -1, 1)))
    lon = np.degrees(np.arctan2(x, mu * np.cos(b0) - y * np.sin(b0)))
    pix = (2 * C.IMG_FOV_RSUN / res) ** 2
    area = np.where(on, pix / np.maximum(mu, C.CH_MIN_MU), 0.0)
    return on, mu, lat, lon, area


def hmi_gauss(v):
    """HMI 정규화 값 → Gauss (normalize_hmi 의 역변환)."""
    s = np.arcsinh(C.HMI_CLIP / C.HMI_ASINH_SCALE)
    return C.HMI_ASINH_SCALE * np.sinh(np.asarray(v, np.float32) * s)


def detect_coronal_holes(aia: np.ndarray, hmi: np.ndarray, t):
    """AIA 193 (정규화) 이 디스크 중앙값의 CH_INTENSITY_RATIO 보다 어둡고 HMI 가 단극성인 영역.
    반환: mask, geometry(on, mu, lat, lon, area), B[Gauss], AIA 임계값(정규화 단위)"""
    geo = _disk_geometry(aia.shape[-1], int(round(b0_deg(t) * 10)))
    on, mu, lat, lon, area = geo
    thr = (np.log10(C.CH_INTENSITY_RATIO) - C.AIA_LOG_MIN) / (C.AIA_LOG_MAX - C.AIA_LOG_MIN)
    b = hmi_gauss(hmi)
    cand = ndimage.binary_opening(on & (mu >= C.CH_MIN_MU) & (aia > 0) & (aia < thr))
    lab, n = ndimage.label(cand)
    if n == 0:
        return np.zeros_like(cand), geo, b, thr
    idx = np.arange(1, n + 1)
    a_reg = ndimage.sum(area, lab, idx)
    unip = np.abs(ndimage.sum(b, lab, idx)) / np.maximum(ndimage.sum(np.abs(b), lab, idx), 1e-6)
    keep = idx[(a_reg >= C.CH_MIN_AREA_FRAC * 2 * np.pi) & (unip >= C.CH_MIN_UNIPOLARITY)]
    return np.isin(lab, keep), geo, b, thr


def ch_map(mask, on, lat, lon) -> np.ndarray:
    """코로나 홀 비율 위경도 지도 (CH_MAP_RES_DEG 격자, 위도·경도 -90~90). 보이지 않는 칸은 NaN."""
    bins = np.linspace(-90, 90, int(180 / C.CH_MAP_RES_DEG) + 1)
    tot = np.histogram2d(lat[on], lon[on], bins=[bins, bins])[0]
    ch = np.histogram2d(lat[on & mask], lon[on & mask], bins=[bins, bins])[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, ch / tot, np.nan).astype(np.float32)


def coronal_hole_features(aia: np.ndarray, hmi: np.ndarray, t, det=None):
    """영상 한 장의 CH 특징 (CH_FEATURES 순서의 dict) 과 회전 보정 비교용 위경도 지도.
    면적은 보이는 반구(2π R_sun²) 대비 비율, 경도·위도는 /90, 극성은 평균 B[G]/10.
    det: 이미 계산한 detect_coronal_holes 결과 (같은 검출을 두 번 하지 않도록, 없으면 여기서 계산)."""
    mask, (on, mu, lat, lon, area), b, thr = det if det is not None else detect_coronal_holes(aia, hmi, t)
    hemi = 2 * np.pi
    band = mask & (np.abs(lat) <= C.CH_MAX_LAT)
    a_band = area * band
    frac = lambda sel: float(a_band[sel].sum() / hemi)
    near = a_band * (np.abs(lon) <= C.CH_SIDE_LON)
    wm = area * mask
    tot_near, tot = near.sum(), wm.sum()
    babs = (wm * np.abs(b)).sum()
    f = dict(
        area_disk=float(tot / hemi),
        area_cm=frac(np.abs(lon) <= C.CH_CM_LON),
        area_east=frac((lon < -C.CH_CM_LON) & (lon >= -C.CH_SIDE_LON)),
        area_west=frac((lon > C.CH_CM_LON) & (lon <= C.CH_SIDE_LON)),
        lon_mean=float((near * lon).sum() / tot_near / 90) if tot_near > 0 else 0.0,
        lat_mean=float((near * np.abs(lat)).sum() / tot_near / 90) if tot_near > 0 else 0.0,
        polarity=float(np.clip((wm * b).sum() / tot / 10, -3, 3)) if tot > 0 else 0.0,
        unipolarity=float(abs((wm * b).sum()) / babs) if babs > 0 else 0.0,
        darkness=float(((thr - aia) * wm).sum() / tot / thr) if tot > 0 else 0.0,
        n_regions=float(ndimage.label(mask)[1]) / 10,
    )
    return f, ch_map(mask, on, lat, lon)


CH_DEPTH_FEATURES = ["depth_mean", "depth_max", "depth_p75", "depth_p90", "cm_depth_mean", "cm_depth_max",
                     "east_depth_mean", "east_depth_max", "cm_deep_frac", "east_deep_frac"]


def ch_boundary_depth(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """코로나 홀 안 각 픽셀에서 가장 가까운 홀 경계 픽셀까지의 태양면 대원 각거리 (deg). 홀 밖은 NaN.
    경계 = 홀 픽셀 중 4-이웃에 홀이 아닌 픽셀이 있는 것 (경계 픽셀의 깊이 = 0).
    보이는 영역 끝 (μ < CH_MIN_MU, 홀 후보에서 빠짐) 과 맞닿은 홀 픽셀도 경계가 된다 — 그 너머는 모르므로
    깊이를 작게 잡는 보수적 정의라서 가장자리에서 깊이가 부풀려지는 이상치가 생기지 않는다."""
    from scipy.spatial import cKDTree
    depth = np.full(mask.shape, np.nan, np.float32)
    if not mask.any():
        return depth
    edge = mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
    la, lo = np.radians(lat), np.radians(lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)     # 태양면 단위 벡터
    chord, _ = cKDTree(xyz[edge]).query(xyz[mask])
    depth[mask] = np.degrees(2 * np.arcsin(np.clip(chord / 2, 0, 1)))
    return depth


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(np.interp(q * c[-1], c, x[o]))


def ch_depth_features(det) -> dict:
    """detect_coronal_holes 결과 → CH_DEPTH_FEATURES (deg / 비율). 통계는 기존 위치 특징과 같은 영역 정의를 쓴다:
    위도 ±CH_MAX_LAT 안 (전체), 그중 중앙자오선 띠 |경도| ≤ CH_CM_LON, 동쪽 -CH_SIDE_LON ≤ 경도 < -CH_CM_LON.
    평균·분위수·깊은 영역 비율은 투영 보정한 구면 면적 가중. 해당 영역에 홀이 없으면 0 (NaN 없음)."""
    mask, (on, mu, lat, lon, area), _, _ = det
    depth = ch_boundary_depth(mask, lat, lon)
    band = mask & (np.abs(lat) <= C.CH_MAX_LAT)
    regions = {"": band, "cm_": band & (np.abs(lon) <= C.CH_CM_LON),
               "east_": band & (lon < -C.CH_CM_LON) & (lon >= -C.CH_SIDE_LON)}
    f = {k: 0.0 for k in CH_DEPTH_FEATURES}
    for pre, sel in regions.items():
        if not sel.any():
            continue
        w, d = area[sel], depth[sel]
        f[f"{pre}depth_mean"] = float((w * d).sum() / w.sum())
        f[f"{pre}depth_max"] = float(d.max())
        if pre == "":
            f["depth_p75"], f["depth_p90"] = _wquantile(d, w, 0.75), _wquantile(d, w, 0.90)
        else:
            f[f"{pre}deep_frac"] = float(w[d > C.CH_DEEP_DEG].sum() / w.sum())
    return f


def ch_depth_sequence(seq: np.ndarray) -> np.ndarray:
    """깊이 특징 시계열 [..., T, 10] (deg / 비율) → Router 입력 [..., T*10]: 깊이는 /CH_DEPTH_SCALE_DEG, 비율은 그대로,
    결측 (영상 없음) 은 0. 시간 변화 특징은 만들지 않는다 (기존 CH 특징의 시점별 원값 블록과 같은 방식)."""
    s = np.nan_to_num(np.array(seq, dtype=np.float32, copy=True), nan=0.0)
    s[..., :8] /= C.CH_DEPTH_SCALE_DEG
    return s.reshape(*s.shape[:-2], -1)


CH_GRID_SHAPE = (len(C.CH_GRID_LAT_EDGES) - 1, len(C.CH_GRID_LON_EDGES) - 1)   # (위도 3, 경도 4)


def grid_bin(x, edges) -> np.ndarray:
    """값 → 칸 번호 (edges[i] ≤ x < edges[i+1], 마지막 칸만 x = edges[-1] 포함), 범위 밖은 -1."""
    x = np.asarray(x)
    i = np.searchsorted(edges, x, side="right") - 1
    i = np.where(x == edges[-1], len(edges) - 2, i)
    return np.where((x >= edges[0]) & (x <= edges[-1]), i, -1)


def ch_grid_area(det) -> np.ndarray:
    """detect_coronal_holes 결과 → CH 면적 격자 [위도 3, 경도 4] (CH_GRID_LAT_EDGES × CH_GRID_LON_EDGES).
    면적은 기존 area_cm / area_east / area_west 와 같은 정의: 위도 ±CH_MAX_LAT 안 홀의 투영 보정 구면 면적 / 보이는 반구 2π.
    행 = 위도 (남 → 북), 열 = 경도 (동쪽 -60 → 서쪽 +60, 음수 = 동쪽). 12칸 합 = area_cm + area_east + area_west."""
    mask, (on, mu, lat, lon, area), _, _ = det
    band = mask & (np.abs(lat) <= C.CH_MAX_LAT)
    ilat, ilon = grid_bin(lat, C.CH_GRID_LAT_EDGES), grid_bin(lon, C.CH_GRID_LON_EDGES)
    sel = band & (ilat >= 0) & (ilon >= 0)
    n_lat, n_lon = CH_GRID_SHAPE
    cell = np.bincount(ilat[sel] * n_lon + ilon[sel], weights=area[sel], minlength=n_lat * n_lon)
    return (cell / (2 * np.pi)).reshape(CH_GRID_SHAPE).astype(np.float32)


def ch_grid_lags() -> np.ndarray:
    """bin k (lead h = 6(k+1) h) 의 target − d 일 시점이 t0 보다 몇 시간 전인지 [N_BINS, len(CH_GRID_LAG_DAYS)] = 24d − h (≥ 0)."""
    h = (np.arange(C.N_BINS) + 1) * C.BIN_HOURS
    return 24 * np.asarray(C.CH_GRID_LAG_DAYS)[None, :] - h[:, None]


def ch_grid_sequence(grid_img: np.ndarray, grid_idx: np.ndarray) -> np.ndarray:
    """영상별 격자 [N_img, 3, 4] + 샘플별 bin·시점 영상 번호 [N, 16, 4] (-1 = 결측) → Router 의 bin 별 입력 [N, 16, 48]:
    bin 마다 target − 4, 5, 6, 7 일 순서로 12칸 (위도 남 → 북, 그 안에서 경도 동 → 서) × CH_GRID_SCALE. 결측은 0."""
    flat = np.nan_to_num(grid_img.reshape(len(grid_img), -1).astype(np.float32), nan=0.0)
    seq = np.where((grid_idx >= 0)[..., None], flat[np.clip(grid_idx, 0, None)], np.float32(0))
    return (seq.reshape(*grid_idx.shape[:2], -1) * np.float32(C.CH_GRID_SCALE)).astype(np.float32)


def synodic_rate(lat_deg):
    """위도별 태양 자전 각속도 (deg/day, 지구에서 본 synodic). 차등 자전 14.71 - 2.39 sin²φ - 1.78 sin⁴φ (sidereal) - 0.9856."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    return 14.71 - 2.39 * s2 - 1.78 * s2 ** 2 - 0.9856


def sun_xy(lat_deg, lon_deg, b0_deg_: float):
    """태양면 (위도, Stonyhurst 경도) → 영상 좌표 (x, y, μ) [R_sun] — _disk_geometry 의 역변환 (x > 0 = 서쪽, y > 0 = 북쪽,
    μ > 0 = 지구 쪽 반구). _disk_geometry: lat = asin(y cos B0 + μ sin B0), lon = atan2(x, μ cos B0 − y sin B0)."""
    la, lo, b = np.radians(lat_deg), np.radians(lon_deg), np.radians(b0_deg_)
    xp, yp, zp = np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)
    return xp, yp * np.cos(b) - zp * np.sin(b), yp * np.sin(b) + zp * np.cos(b)


def derotate_image(prev: np.ndarray, t_prev, t_cur, r_max: float = 0.99, order: int = 1):
    """이전 시각 영상 (res², 디스크 정렬 격자) 을 현재 시각의 태양면 좌표로 옮긴다 (차등 자전 정렬).
    현재 픽셀 (위도 φ, 경도 L) 의 물질은 dt = t_cur − t_prev 전에 경도 L − ω(φ)·dt 에 있었으므로 (ω = synodic_rate, deg/day,
    기존 CH 회전 보정과 같은 식), 그 위치를 이전 시각의 B0 로 영상에 투영해 이전 영상에서 쌍선형 보간한다.
    반환: (정렬된 이전 영상, 유효 mask = 현재·이전 모두 디스크 안 (r < r_max) 이고 이전 시각에 보이던 곳). 유효하지 않은 곳은 0."""
    from scipy.ndimage import map_coordinates
    res = prev.shape[-1]
    on, mu, lat, lon, _ = _disk_geometry(res, int(round(b0_deg(t_cur) * 10)))
    g = (np.arange(res) + 0.5) / res * 2 * C.IMG_FOV_RSUN - C.IMG_FOV_RSUN
    xx, yy = np.meshgrid(g, g)
    dt_days = (pd.Timestamp(t_cur) - pd.Timestamp(t_prev)).total_seconds() / 86400
    x, y, mu_src = sun_xy(lat, lon - synodic_rate(lat) * dt_days, b0_deg(t_prev))
    valid = on & (np.hypot(xx, yy) < r_max) & (mu_src > 0) & (np.hypot(x, y) < r_max)
    col = (x + C.IMG_FOV_RSUN) / (2 * C.IMG_FOV_RSUN) * res - 0.5
    row = (y + C.IMG_FOV_RSUN) / (2 * C.IMG_FOV_RSUN) * res - 0.5
    out = map_coordinates(prev.astype(np.float64), [row, col], order=order, mode="constant", cval=0.0)   # 1 = 쌍선형 (H1 기본)
    return np.where(valid, out, 0.0).astype(np.float32), valid


def ch_iou(map_old: np.ndarray, map_new: np.ndarray, dt_hours: float) -> float:
    """이전 CH 지도를 dt 만큼 (위도별 차등) 자전시켜 지금 지도와 겹친 비율 (soft IoU) → 모양·위치 유지 정도.
    비교 영역(|경도| ≤ CH_SIDE_LON, |위도| ≤ 60°) 에 양쪽 모두 홀이 없으면 1 (변화 없음)."""
    edges = np.linspace(-90, 90, map_old.shape[0] + 1)
    c = (edges[:-1] + edges[1:]) / 2
    rot = np.full_like(map_old, np.nan)
    for i, la in enumerate(c):
        ok = np.isfinite(map_old[i])
        if ok.sum() >= 2:
            rot[i] = np.interp(c, c[ok] + synodic_rate(la) * dt_hours / 24, map_old[i][ok],
                               left=np.nan, right=np.nan)
    region = (np.abs(c)[:, None] <= 60) & (np.abs(c)[None, :] <= C.CH_SIDE_LON)
    a, b = rot[region], map_new[region]
    ok = np.isfinite(a) & np.isfinite(b)
    union = np.maximum(a[ok], b[ok]).sum()
    return float(np.minimum(a[ok], b[ok]).sum() / union) if union > 1e-6 else 1.0


def regime_labels(v: HourlySeries, t0s, target_kms: np.ndarray) -> np.ndarray:
    """정답 bin 별 태양풍 상태 [N, N_BINS]: 0 Slow, 1 Transition, 2 HSS, -1 결측.
    Transition: bin 중심 앞뒤 12 h (각 6 h 평균) 사이 속도가 TRANSITION_DV 이상 상승 — HSS 앞 상승 구간·충격파 포함
    HSS: 상승 중이 아니고 bin 평균 속도 ≥ HSS_V. 나머지는 Slow."""
    vals = np.nan_to_num(v.values, nan=0.0)
    cnt = np.isfinite(v.values).astype(np.int64)
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    ccnt = np.concatenate([[0], np.cumsum(cnt)])

    def wmean(start):                                  # [start, start+6) hourly 평균, 유효 3시간 미만이면 NaN
        s = np.clip(start, 0, len(vals))
        e = np.clip(start + 6, 0, len(vals))
        n = ccnt[e] - ccnt[s]
        return np.where(n >= 3, (csum[e] - csum[s]) / np.maximum(n, 1), np.nan)

    i0 = ((pd.DatetimeIndex(t0s) - v.base) / pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
    centers = i0[:, None] + np.arange(C.N_BINS)[None, :] * C.BIN_HOURS + C.BIN_HOURS // 2
    rise = wmean(centers + 9) - wmean(centers - 15)
    lab = np.where(rise >= C.TRANSITION_DV, 1, np.where(target_kms >= C.HSS_V, 2, 0)).astype(np.int8)
    lab[~np.isfinite(target_kms)] = -1
    return lab


CH_DYN_FEATURES = ["slope_area_disk", "slope_area_cm", "slope_area_east", "slope_area_west", "log_growth",
                   "lon_drift", "days_to_cm", "iou12_mean", "iou12_min", "iou24_last"]


def ch_dynamic_features(seq: np.ndarray, step_hours: float) -> np.ndarray:
    """CH 특징 시계열 [..., T, 12] (오래된 → 최근, step_hours 간격) → 모델 입력 [..., T*12 + 10].
    시계열 원값(면적은 x10) 뒤에 '변화' 특징을 붙인다:
      커짐/작아짐   : 면적 4종(전체·중앙띠·동쪽·서쪽)의 하루당 기울기, 전체 면적의 처음 대비 마지막 log 비
      중앙자오선 접근: 평균 경도 이동 속도 (1 = 정상 자전 13.2°/day), 지금 위치에서 중앙 도달까지 남은 일수 (/7)
      모양 유지     : 12 h 전과의 회전 보정 IoU 평균·최소, 마지막 24 h IoU"""
    s = np.array(seq, dtype=np.float32, copy=True)
    s[..., 10:12] = np.nan_to_num(s[..., 10:12], nan=0.5)          # 전 영상이 없어 IoU 못 구한 경우 중간값
    s = np.nan_to_num(s, nan=0.0)
    t = (np.arange(s.shape[-2]) - (s.shape[-2] - 1)) * step_hours / 24.0
    tc = t - t.mean()
    slope = lambda x: (x * tc).sum(-1) / (tc ** 2).sum()
    area = s[..., :4] * 10
    lon = s[..., 4] * 90
    near_now = (s[..., -1, 1] + s[..., -1, 2]) > 0
    dyn = np.concatenate([
        np.stack([slope(area[..., k]) for k in range(4)], -1),
        np.log((s[..., -1, 0] + 2e-3) / (s[..., 0, 0] + 2e-3))[..., None],
        (slope(lon) / 13.2)[..., None],
        np.where(near_now, np.clip(-lon[..., -1] / 13.2, -3, 7) / 7, 0.0)[..., None],
        s[..., 1:, 10].mean(-1, keepdims=True),
        s[..., 1:, 10].min(-1, keepdims=True),
        s[..., -1:, 11],
    ], -1)
    flat = s.copy()
    flat[..., :4] *= 10
    return np.concatenate([flat.reshape(*flat.shape[:-2], -1), dyn], -1).astype(np.float32)


CH_FEATURE_SETS = ("full", "area", "snapshot", "none", "full_depth", "full_7d")


def ch_feature_columns(feature_set: str) -> list[int]:
    """ch_dynamic_features 출력 [T*12 + 10] 중 쓸 열 (Expert ablation 용).
    full       : 전체 (시점별 원값 + 변화 특징)
    area       : 면적 4종의 시점별 원값 + 면적 변화 특징 (기울기 4, log 증감) — 위치·모양·자기장 제외
    snapshot   : 마지막 시점(t0)의 원값 12개만 — 커짐/작아짐·접근·모양 유지 같은 '변화' 정보 없음
    none       : 코로나 홀 정보 없음
    full_depth : full + 경계 깊이 특징 시점별 원값 (T*10, ch_depth_sequence) — 데이터의 ch_depth(True) 표현과 함께
    full_7d    : full 과 같은 특징을 7일 (CH_LAGS_LONG 15시점) 시계열로 — 데이터의 ch_input(7) 표현과 함께"""
    n_t, nf = len(C.lags_for(C.CH_STEP_HOURS)), len(CH_IMG_COLS)
    base = n_t * nf
    if feature_set == "full":
        return list(range(base + len(CH_DYN_FEATURES)))
    if feature_set == "full_depth":
        return list(range(base + len(CH_DYN_FEATURES) + n_t * len(CH_DEPTH_FEATURES)))
    if feature_set == "full_7d":
        return list(range(len(C.CH_LAGS_LONG) * nf + len(CH_DYN_FEATURES)))
    if feature_set == "area":
        return ([t * nf + CH_IMG_COLS.index(f) for t in range(n_t) for f in CH_FEATURES[:4]]
                + [base + CH_DYN_FEATURES.index(f) for f in CH_DYN_FEATURES[:5]])
    if feature_set == "snapshot":
        return [(n_t - 1) * nf + k for k in range(nf)]
    if feature_set == "none":
        return []
    raise ValueError(f"CH feature set: {feature_set} (가능: {CH_FEATURE_SETS})")
