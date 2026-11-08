"""
build_features.py — 8 개 factorial 조건이 공유하는 입력·정답·라벨을 features/ 에 만든다 (기존 dataset 은 읽기만).

  H (모든 조건 공통, 과거 태양풍 속도 — ACE hourly, 기존 정답지와 같은 출처)
    hist     [N, 48, 2]   t 직전 48 h hourly V (정규화) + 유효 mask
    rec      [N, 24, 2]   [t − 28 d, t − 22 d) 6 h 평균 + mask  (모든 horizon 의 26–28 일 전 주변)
    v27      [N, 5, 2]    v_27(t, h) = 6 h 평균 [t+h−654 h, t+h−648 h), h = 0/24/48/72/96 + mask  (정확히 27 일)
    v_now    [N]          직전 6 h 평균 (없으면 48 h 안 마지막 값, 그것도 없으면 v_27(t, h=0), 그것도 없으면 V_MEAN)
  B  [N, 3]   대표 코로나홀 fs_CH = Σ Br_i / S_CH (signed, Br ≈ B_LOS/μ, S_CH = 유효 Br 픽셀 수; train 표준화), ch_present, mag_valid
  C  [N, 16]  DONKI CME (time21_5 + 6 h ≤ t, time21_5 ≥ t − 120 h) 요약
  G  [N, 4, 13] t, t−24, t−48, t−72 h 의 4×3 격자 CH_fraction 12 + 영상 유효 flag
  y  [N, 4]   +24/48/72/96 h 정답 (dataset target_kms 의 bin 3/7/11/15 = [t+h−6 h, t+h) 평균, km/s)
  regime [N, 4]  기존 상태 라벨 (0 Slow / 1 Transition / 2 HSS / −1 결측)

부산물: features/slot_features.npz (영상 슬롯별 격자·대표 CH), features/ch_components.csv.gz (CH 별 fs_CH),
features/cme_used.csv.gz, features/ips_earth.csv, features/leakage_checks.json, features/features_meta.json

    python build_features.py            # 전체 (영상 CH 검출은 병렬, ~10 분)
    python build_features.py --skip-images   # 영상 슬롯 특징은 저장된 것 재사용
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

import common as K
from common import C

import sw_utils as U

SLOT_FILE = K.DIRS["features"] / "slot_features.npz"
COMP_FILE = K.DIRS["features"] / "ch_components.csv.gz"
IPS_FILE = K.DIRS["features"] / "ips_earth.csv"


# ====================================================
# 영상 슬롯별: CH 검출 → 격자 (B·G 의 원천)
# ====================================================
def _grid_index(lat, lon):
    ilat = U.grid_bin(lat, K.G_LAT_EDGES)
    ilon = U.grid_bin(lon, K.G_LON_EDGES)
    n_lon = len(K.G_LON_EDGES) - 1
    return np.where((ilat >= 0) & (ilon >= 0), ilat * n_lon + ilon, -1)


def slot_features(t_iso: str):
    """한 영상 슬롯 → (격자 CH 픽셀 수 12, 격자 유효 픽셀 수 12, 격자 CH 투영보정 면적 12, CH 성분 목록).
    셀 순서: 위도 남 → 북 (G_LAT_EDGES) × 경도 동 → 서. 성분: 픽셀 수, 유효 Br 픽셀 수, fs_CH (signed 평균 Br),
    면적 가중 평균 Br, 중심 위경도, 지구 직하점 (B0, 0) 과의 각거리."""
    t = pd.Timestamp(str(t_iso))
    pa, ph = U.image_path("aia193", t), U.image_path("hmi", t)
    if not (pa.exists() and ph.exists()):
        return None
    aia, hmi = np.load(pa).astype(np.float32), np.load(ph).astype(np.float32)
    mask, (on, mu, lat, lon, area), b, _ = U.detect_coronal_holes(aia, hmi, t)
    valid = on & (mu >= C.CH_MIN_MU)
    cell = _grid_index(lat, lon)
    sel_v = valid & (cell >= 0)
    n_valid = np.bincount(cell[sel_v], minlength=12)
    n_ch = np.bincount(cell[sel_v & mask], minlength=12)
    a_ch = np.bincount(cell[sel_v & mask], weights=area[sel_v & mask], minlength=12)
    comps = []
    if mask.any():
        b0 = np.radians(U.b0_deg(t))
        lab, n = ndimage.label(mask)
        la, lo = np.radians(lat), np.radians(lon)
        xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
        earth = np.array([np.cos(b0), 0.0, np.sin(b0)])                       # 지구 직하점 (위도 B0, 경도 0)
        br = np.where(mu >= K.BR_MIN_MU, b / np.maximum(mu, 1e-6), np.nan)  # Br ≈ B_LOS / μ (radial field 가정)
        for i in range(1, n + 1):
            pix = lab == i
            brv = br[pix]
            ok = np.isfinite(brv)
            nv = int(ok.sum())
            c = xyz[pix].mean(0)
            c = c / max(np.linalg.norm(c), 1e-12)
            comps.append(dict(n_pix=int(pix.sum()), n_br_valid=nv,
                              fs_CH=float(brv[ok].sum() / nv) if nv else np.nan,
                              fs_CH_areaw=float((brv[ok] * area[pix][ok]).sum() / area[pix][ok].sum()) if nv else np.nan,
                              lat=float(np.degrees(np.arcsin(np.clip(c[2], -1, 1)))),
                              lon=float(np.degrees(np.arctan2(c[1], c[0]))),
                              dist_deg=float(np.degrees(np.arccos(np.clip(c @ earth, -1, 1)))),
                              mean_mu=float(mu[pix].mean())))
    return n_ch.astype(np.int32), n_valid.astype(np.int32), a_ch.astype(np.float32), comps


def build_slots(times):
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(slot_features, times, chunksize=64))
    M = len(times)
    ok = np.array([r is not None for r in res])
    n_ch = np.zeros((M, 12), np.int32)
    n_valid = np.zeros((M, 12), np.int32)
    a_ch = np.zeros((M, 12), np.float32)
    rep = {k: np.full(M, np.nan, np.float64) for k in ("fs_CH", "fs_CH_areaw", "lat", "lon", "dist_deg", "n_pix", "n_br_valid")}
    n_comp = np.zeros(M, np.int32)
    rows = []
    for m, r in enumerate(res):
        if r is None:
            continue
        n_ch[m], n_valid[m], a_ch[m], comps = r
        n_comp[m] = len(comps)
        if comps:
            # 대표 CH: 지구 직하점에 가장 가까운 CH (각거리 최소, 같으면 큰 것) — 정답을 보지 않는 고정 물리 규칙
            j = min(range(len(comps)), key=lambda q: (comps[q]["dist_deg"], -comps[q]["n_pix"]))
            for k in rep:
                rep[k][m] = comps[j][k]
            for q, cp in enumerate(comps):
                rows.append(dict(slot_time=times[m], ch_id=q + 1, is_representative=q == j, **cp))
    np.savez(SLOT_FILE, times=np.array(times, dtype="S19"), ok=ok, n_ch=n_ch, n_valid=n_valid, a_ch=a_ch, n_comp=n_comp,
             **{f"rep_{k}": v for k, v in rep.items()})
    pd.DataFrame(rows).to_csv(COMP_FILE, index=False, float_format="%.5g")
    print(f"  영상 슬롯 {M}: 영상 있음 {ok.sum()}, CH 있음 {(n_comp > 0).sum()}, CH 성분 {len(rows)}", flush=True)


# ====================================================
# 과거 속도 (H)
# ====================================================
def speed_inputs(v: U.HourlySeries, t0s):
    N = len(t0s)
    hist = np.full((N, K.HIST_HOURS), np.nan, np.float32)
    rec = np.full((N, K.REC_BINS), np.nan, np.float32)
    for i, t in enumerate(t0s):
        hist[i] = v.window(t - pd.Timedelta(hours=K.HIST_HOURS), K.HIST_HOURS)
        rec[i] = U.bin_means(v.window(t - pd.Timedelta(hours=K.REC_START_H), K.REC_BINS * C.BIN_HOURS))
    v27 = rec[:, list(K.V27_REC_IDX)]
    last6 = hist[:, -K.NOW_HOURS:]
    n6 = np.isfinite(last6).sum(1)
    with np.errstate(invalid="ignore"):
        now = np.where(n6 > 0, np.nansum(np.nan_to_num(last6), 1) / np.maximum(n6, 1), np.nan)
    src = np.where(n6 > 0, 0, -1)
    lastv = np.array([h[np.isfinite(h)][-1] if np.isfinite(h).any() else np.nan for h in hist])
    m = ~np.isfinite(now) & np.isfinite(lastv)
    now[m], src[m] = lastv[m], 1
    m = ~np.isfinite(now) & np.isfinite(v27[:, 0])
    now[m], src[m] = v27[m, 0], 2
    m = ~np.isfinite(now)
    now[m], src[m] = K.V_MEAN, 3
    return hist, rec, v27, now.astype(np.float32), src.astype(np.int8)


def pack(x):
    """km/s (NaN 결측) → [.., 2] (정규화 값 결측 0, mask)."""
    ok = np.isfinite(x)
    return np.stack([np.where(ok, (x - K.V_MEAN) / K.V_STD, 0.0), ok.astype(np.float32)], -1).astype(np.float32)


# ====================================================
# CME (C)
# ====================================================
def cme_table():
    a = pd.read_csv(C.CME_CSV, parse_dates=["startTime", "time21_5", "submissionTime"])
    a = a[a["speed"].notna() & a["time21_5"].notna()].copy()
    tech = a["measurementTechnique"].fillna("")
    a["kind"] = np.where(tech.isin(["SWPC_CAT", "StereoCAT"]), 0, np.where(tech == "Plane-of-sky", 1, 2))  # 0 cone 3D, 1 POS, 2 미상
    # CME 당 분석 하나: isMostAccurate → 3D cone 우선 → 가장 늦게 제출된 것 (사후 catalog 기준; 실시간 가용성은 따로 집계)
    a = a.sort_values(["activityID", "isMostAccurate", "kind", "submissionTime"], ascending=[True, False, True, False])
    a = a.drop_duplicates("activityID").sort_values("time21_5").reset_index(drop=True)
    return a


def cme_inputs(t0s, a: pd.DataFrame, cat_start: pd.Timestamp, cat_end: pd.Timestamp):
    from astropy.time import Time
    from sunpy.coordinates import sun
    N = len(t0s)
    b0_all = np.radians(np.asarray(sun.B0(Time(list(t0s.to_pydatetime()))).deg, np.float64))   # 지구의 태양 위도
    X = np.zeros((N, 16), np.float32)
    retro = np.zeros(N, np.int8)          # 쓴 분석 중 하나라도 t 이후 제출 (= 실시간에는 없던 값)
    used = []
    t21 = a["time21_5"].values
    for i, t in enumerate(t0s):
        if not (cat_start <= t <= cat_end):
            continue                                   # catalog 밖: 모두 0 (catalog flag 0)
        X[i, 0] = 1.0
        lo = np.searchsorted(t21, np.datetime64(t - pd.Timedelta(hours=K.CME_WINDOW_H)), "left")
        hi = np.searchsorted(t21, np.datetime64(t - pd.Timedelta(hours=K.CME_LAG_H)), "right")
        d = a.iloc[lo:hi]
        if d.empty:
            continue
        b0 = b0_all[i]
        lat, lon = np.radians(d["latitude"].astype(float).values), np.radians(d["longitude"].astype(float).values)
        sep = np.degrees(np.arccos(np.clip(np.sin(lat) * np.sin(b0) + np.cos(lat) * np.cos(b0) * np.cos(lon), -1, 1)))
        ha = d["halfAngle"].astype(float).values
        dir_ok = np.isfinite(sep)
        earth = dir_ok & np.isfinite(ha) & (sep <= ha)
        margin = np.where(dir_ok, sep - np.nan_to_num(ha, nan=C.CME_DEFAULT_HALF_ANGLE), np.inf)
        spd = d["speed"].astype(float).values
        age = (t - d["time21_5"]).dt.total_seconds().values / 3600
        # 대표 CME: 지구향 중 최고 속도, 없으면 지구향 여유 (각거리 − half-angle) 가 가장 작은 것
        j = int(np.argmax(np.where(earth, spd, -np.inf))) if earth.any() else int(np.argmin(margin))
        X[i, 1:6] = [1.0, len(d) / 5.0, earth.sum() / 3.0, spd.max() / 1000.0, spd[earth].max() / 1000.0 if earth.any() else 0.0]
        X[i, 6:15] = [spd[j] / 1000.0, (sep[j] if dir_ok[j] else 90.0) / 90.0, (ha[j] if np.isfinite(ha[j]) else 0.0) / 90.0,
                      age[j] / K.CME_WINDOW_H, float(earth[j]), float(d["kind"].values[j] == 1), float(not dir_ok[j]),
                      (np.degrees(lat[j]) if dir_ok[j] else 0.0) / 90.0, (np.degrees(lon[j]) if dir_ok[j] else 0.0) / 90.0]
        X[i, 15] = age.min() / K.CME_WINDOW_H
        sub = d["submissionTime"]
        retro[i] = int((sub > t).any())
        used.append(dict(t0=t, n=len(d), n_earth=int(earth.sum()), rep_activityID=d["activityID"].values[j],
                         rep_speed=spd[j], rep_sep=sep[j], rep_halfAngle=ha[j], rep_age_h=age[j], rep_kind=int(d["kind"].values[j]),
                         any_submitted_after_t=bool(retro[i]), max_time21_5=d["time21_5"].max()))
    return X, retro, pd.DataFrame(used)


C_COLS = ["catalog_available", "any_cme", "n_cme/5", "n_earth_directed/3", "vmax_all/1000", "vmax_earth/1000",
          "rep_speed/1000", "rep_sep/90", "rep_halfangle/90", "rep_age/120h", "rep_earth_directed", "rep_speed_is_POS",
          "rep_direction_unknown", "rep_lat/90", "rep_lon/90", "min_age/120h"]


def fetch_ips():
    """평가 전용 (모델 입력 아님): DONKI IPS (지구 위치 행성간 충격파) — CME 영향 구간의 사후 정의."""
    if IPS_FILE.exists():
        return pd.read_csv(IPS_FILE, parse_dates=["eventTime", "submissionTime"])
    import requests
    rows = []
    s = pd.Timestamp(C.DATE_START) - pd.Timedelta(days=10)
    end = pd.Timestamp(C.DATE_END) + pd.Timedelta(days=10)
    while s <= end:
        e = min(s + pd.Timedelta(days=59), end)
        url = f"{C.DONKI_API_URL}/IPS?startDate={s:%Y-%m-%d}&endDate={e:%Y-%m-%d}&location=Earth"
        for _ in range(3):
            try:
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                break
            except Exception as ex:
                print("   IPS 재시도", url, ex, flush=True)
        else:
            raise SystemExit(f"IPS 받기 실패: {url}")
        for d in (r.json() if r.text.strip() else []):
            linked = [x.get("activityID", "") for x in (d.get("linkedEvents") or [])]
            rows.append(dict(activityID=d["activityID"], eventTime=U._ts(d.get("eventTime")), location=d.get("location"),
                             cme_linked=any("CME" in x for x in linked), linked=";".join(linked),
                             submissionTime=U._ts(d.get("submissionTime"))))
        s = e + pd.Timedelta(days=1)
    df = pd.DataFrame(rows).drop_duplicates("activityID").sort_values("eventTime")
    df.to_csv(IPS_FILE, index=False)
    return df


# ====================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    K.ensure_dirs()
    with h5py.File(C.HDF5_PATH, "r") as h:
        t0 = pd.to_datetime(h["t0"][:].astype(str))
        img_times = h["image_times"][:].astype(str)
        img_idx = h["img_idx"][:]
        lags = list(h.attrs["lags_hours"])
        target = h["target_kms"][:]
        regime = h["regime"][:]
    split = pd.read_csv(C.SPLIT_CSV)
    assert (pd.to_datetime(split["t0"]).values == t0.values).all(), "splits.csv 와 dataset t0 순서가 다름"
    split = split["split"].to_numpy()
    N = len(t0)
    print(f"[features] 샘플 {N} (train {np.sum(split == 'train')}, valid {np.sum(split == 'valid')}, test {np.sum(split == 'test')}, "
          f"purge {np.sum(split == 'purge')}), 영상 슬롯 {len(img_times)}", flush=True)

    # ── 영상 → 슬롯 특징 ──
    if not (args.skip_images and SLOT_FILE.exists()):
        build_slots([str(x) for x in img_times])
    s = np.load(SLOT_FILE)
    assert (s["times"].astype(str) == img_times).all()

    # ── H ──
    ace = pd.read_csv(C.ACE_CSV, parse_dates=["time"])
    v = U.HourlySeries.from_series(ace.set_index("time")["V"])
    hist, rec, v27, v_now, now_src = speed_inputs(v, t0)
    y = target[:, list(K.KH)].astype(np.float32)
    reg = regime[:, list(K.KH)].astype(np.int8)

    # ── B ──
    tr = split == "train"
    col = {L: lags.index(L) for L in K.G_LAGS}
    i0 = img_idx[:, col[0]]
    has_img = (i0 >= 0) & s["ok"][np.clip(i0, 0, None)]
    fs = np.where(has_img, s["rep_fs_CH"][np.clip(i0, 0, None)], np.nan)
    ch_present = has_img & (s["n_comp"][np.clip(i0, 0, None)] > 0)
    mag_valid = ch_present & np.isfinite(fs)
    fs_mean, fs_std = float(np.mean(fs[tr & mag_valid])), float(np.std(fs[tr & mag_valid]))
    B = np.stack([np.where(mag_valid, (fs - fs_mean) / fs_std, 0.0), ch_present, mag_valid], -1).astype(np.float32)

    # ── G ──
    G = np.zeros((N, len(K.G_LAGS), 13), np.float32)
    G_raw = np.zeros((N, len(K.G_LAGS), 2, 12), np.int32)
    for li, L in enumerate(K.G_LAGS):
        ii = img_idx[:, col[L]]
        ok = (ii >= 0) & s["ok"][np.clip(ii, 0, None)]
        nc, nv = s["n_ch"][np.clip(ii, 0, None)], s["n_valid"][np.clip(ii, 0, None)]
        frac = np.where(nv > 0, nc / np.maximum(nv, 1), 0.0)
        frac = frac.reshape(N, 4, 3)[:, ::-1].reshape(N, 12)               # 저장 순서 A1 … A12: 북쪽 행부터, 행 안은 동 → 서
        G[:, li, :12] = np.where(ok[:, None], frac, 0.0)
        G[:, li, 12] = ok
        G_raw[:, li, 0] = np.where(ok[:, None], nc.reshape(N, 4, 3)[:, ::-1].reshape(N, 12), 0)
        G_raw[:, li, 1] = np.where(ok[:, None], nv.reshape(N, 4, 3)[:, ::-1].reshape(N, 12), 0)

    # ── C ──
    a = cme_table()
    files = sorted((C.DONKI_DIR / "raw").glob("CME_*.json")) if hasattr(C, "DONKI_DIR") else []
    cat_start = pd.Timestamp("2010-05-10")
    cat_end = pd.Timestamp(files[-1].stem.split("_")[-1]) if files else a["time21_5"].max()
    Cx, retro, used = cme_inputs(t0, a, cat_start, cat_end)
    used.to_csv(K.DIRS["features"] / "cme_used.csv.gz", index=False)
    ips = fetch_ips()

    # ── 누출 검사 ──
    chk = {}
    win_end = {"hist": 0, "rec": -(K.REC_START_H - K.REC_BINS * C.BIN_HOURS), "v27_h96": 96 - K.DAY27_H}
    chk["speed_window_end_offsets_h"] = win_end
    chk["speed_inputs_end_before_t"] = all(x <= 0 for x in win_end.values())
    rs = np.random.default_rng(0).choice(N, 300, replace=False)
    ok27 = []
    for i in rs:
        for j, hh in enumerate(K.V27_LEADS):
            st = t0[i] + pd.Timedelta(hours=hh - K.DAY27_H - C.BIN_HOURS)
            ref = U.bin_means(v.window(st, C.BIN_HOURS))[0]
            ok27.append(np.isclose(ref, v27[i, j], equal_nan=True))
    chk["v27_alignment_recomputed_300x5"] = bool(np.all(ok27))
    okt = [np.allclose(U.bin_means(v.window(t0[i], C.HORIZON_HOURS))[list(K.KH)], y[i], equal_nan=True) for i in rs]
    chk["target_recomputed_300"] = bool(np.all(okt))
    lag_ok = True
    for L in K.G_LAGS:
        ii = img_idx[:, col[L]]
        m = ii >= 0
        ts = pd.to_datetime(img_times[ii[m]])
        lag_ok &= bool((ts <= t0[m]).all())
        chk[f"image_slot_lag{L}_max_offset_h"] = float(((t0[m] - ts) / pd.Timedelta(hours=1)).max())
    chk["image_slots_not_after_t"] = lag_ok
    if len(used):
        chk["cme_all_used_time21_5_plus_6h_le_t"] = bool(((used["max_time21_5"] + pd.Timedelta(hours=K.CME_LAG_H)) <= used["t0"]).all())
    chk["cme_samples_using_post_t_submitted_analysis"] = {sp: float(retro[split == sp].mean()) for sp in ("train", "valid", "test")}
    chk["normalization_from_train_only"] = {"fs_CH_mean_G": fs_mean, "fs_CH_std_G": fs_std}
    # 분할 경계: 다른 split 의 target 시각 겹침
    tb = {}
    for sp in ("train", "valid", "test"):
        m = split == sp
        tt = (t0[m].values[:, None] + (np.array(K.HORIZONS) - C.BIN_HOURS)[None] * np.timedelta64(1, "h")).ravel()
        tb[sp] = set(tt.astype("datetime64[h]").astype(np.int64).tolist())
    chk["target_time_overlap_train_test"] = len(tb["train"] & tb["test"])
    chk["target_time_overlap_train_valid"] = len(tb["train"] & tb["valid"])
    chk["target_time_overlap_valid_test"] = len(tb["valid"] & tb["test"])
    # train 입력 창이 test target 시각을 덮는 경우 (입력으로 본 것이지 학습 target 은 아님 — 보고만)
    test_hours = set()
    for t in t0[split == "test"]:
        st = int(np.datetime64(t, "h").astype(np.int64))
        test_hours.update(range(st, st + 96))
    trh = t0[split == "train"]
    n_hist, n_rec = 0, 0
    for t in trh:
        st = int(np.datetime64(t, "h").astype(np.int64))
        n_hist += any((st - q) in test_hours for q in range(1, K.HIST_HOURS + 1))
        n_rec += any((st - K.REC_START_H + q) in test_hours for q in range(0, K.REC_BINS * C.BIN_HOURS, 6))
    chk["train_samples_whose_hist48_covers_test_target_hours"] = int(n_hist)
    chk["train_samples_whose_rec27_covers_test_target_hours"] = int(n_rec)
    vv = v.values
    ac = lambda lag: float(pd.Series(vv[:-lag]).corr(pd.Series(vv[lag:])))
    chk["ace_autocorr"] = {"lag_24h": ac(24), "lag_96h": ac(96), "lag_27d": ac(K.DAY27_H), "lag_27.27d": ac(655), "lag_13.5d": ac(324)}
    (K.DIRS["features"] / "leakage_checks.json").write_text(json.dumps(chk, indent=2, default=str), encoding="utf-8")

    np.savez_compressed(K.FEATURES_FILE, t0=np.array([str(t) for t in t0], dtype="S19"), split=split.astype("S5"),
                        y=y, regime=reg, hist=pack(hist), rec=pack(rec), v27=pack(v27), v_now=v_now, now_src=now_src,
                        v27_kms=v27, B=B, C=Cx, G=G, G_raw=G_raw, cme_retro=retro, fs_CH_raw=fs.astype(np.float32))
    meta = dict(horizons=K.HORIZONS, target_bins=K.KH, n=N, fs_CH_mean=fs_mean, fs_CH_std=fs_std, C_cols=C_COLS,
                G_cells="A1..A12 = 위도 [45,90]/[0,45]/[-45,0]/[-90,-45] (북→남) × 경도 [-90,-30)/[-30,30)/[30,90] (동→서), Stonyhurst, B0 보정",
                G_lags_h=K.G_LAGS, B_cols=["fs_CH_std", "ch_present", "mag_valid"], speed_norm=[K.V_MEAN, K.V_STD],
                now_src_counts={int(k): int(v_) for k, v_ in zip(*np.unique(now_src, return_counts=True))},
                cme_catalog=[str(cat_start), str(cat_end)], ips_events=int(len(ips)), ips_cme_linked=int(ips["cme_linked"].sum()),
                coverage={"B_ch_present": float(ch_present.mean()), "B_mag_valid": float(mag_valid.mean()), "B_image": float(has_img.mean()),
                          "G_images_all4": float((G[:, :, 12] > 0).all(1).mean()), "C_any_cme": float(Cx[:, 1].mean()),
                          "C_earth_directed": float((Cx[:, 3] > 0).mean()), "y_finite": float(np.isfinite(y).mean()),
                          "hist_valid": float(np.isfinite(hist).mean()), "rec_valid": float(np.isfinite(rec).mean()),
                          "v27_valid": float(np.isfinite(v27).mean())},
                leakage=chk)
    K.atomic_write_json(K.FEATURES_META, meta)
    print(json.dumps(meta["coverage"], indent=1), flush=True)
    print(json.dumps(chk, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
