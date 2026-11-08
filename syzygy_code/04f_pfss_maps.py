"""
04f_pfss_maps.py
-----------------------------------
HMI 전체원반 LOS magnetogram (이미 받은 256² npy) 만으로, 각 시각 T 에 실제로 쓸 수 있었던 정보만 담은
causal 광구 B_r Carrington 지도를 만들고, PFSS source surface (R_ss = 2.5) B_r 지도를 계산해 pfss.h5 에 저장한다.

causal 규칙 (시각 T 의 지도):
  - 관측 시각 (image_index.csv 의 T_OBS, TAI 표기를 UTC 로 보면 실제보다 늦게 잡히므로 보수적) ≤ T - PFSS_LATENCY_H
    인 magnetogram 만 쓴다. 영상을 관측 시각 순으로 넣으면서, 지도 시각도 오름차순으로 그 시점 상태를 떠 둔다.
  - Carrington 칸 (sin 위도 90 × 경도 2°) 마다 '가장 최근에 중앙자오선 ±60°, μ ≥ 0.2 에서 본 값' (B_r = B_los/μ)
  - PFSS_MAX_AGE_DAYS (28일) 넘게 안 본 칸은 결측 → 극관 (|위도| ≥ 60°) 은 같은 반구 60–75° 관측 평균, 나머지는 0
  - 저장하는 지도는 그 시각의 지구 기준 경도 (L0(T) 로 회전, 경도 0 = 지구 방향)
검사: 모든 지도의 '쓰인 가장 늦은 관측 시각' ≤ T - 지연 (위반이면 저장 안 함).
정규화: train 샘플이 쓰는 지도만으로 RMS 를 구해 attrs['train_scale'] 에 저장 (valid·test 통계 안 씀).

    python 04f_pfss_maps.py

결과: pfss.h5 (times, br_ss [N_T, 64, 128], coverage, obs_latest, age_median_days), runs/pfss_sanity.npz
"""
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
import sw_utils as U
from pfss import PFSSGrid

SANITY_TIMES = ["2011-02-15 00:00", "2014-06-01 00:00", "2019-12-01 00:00", "2024-06-01 00:00"]


def obs_times(index_csv) -> pd.DataFrame:
    """image_index.csv 의 HMI 정상 영상: 슬롯 시각, 관측 시각 (T_OBS, TAI 표기 그대로 = 보수적)."""
    d = pd.read_csv(index_csv, parse_dates=["time"])
    d = d[(d.instrument == "hmi") & (d.status == "ok")].copy()
    p = d.t_obs.astype(str).str.extract(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)")
    secs = p[3].astype(float) * 3600 + p[4].astype(float) * 60 + p[5].astype(float)   # '…:59:60' (반올림·윤초) 도 그대로
    d["obs"] = pd.to_datetime(p[0] + "-" + p[1] + "-" + p[2]) + pd.to_timedelta(secs, unit="s")
    if d.obs.isna().any():
        raise SystemExit(f"T_OBS 해석 실패 {int(d.obs.isna().sum())} 건")
    return d.sort_values("obs").reset_index(drop=True)


def l0_deg(t) -> float:
    from sunpy.coordinates import sun
    return float(sun.L0(pd.Timestamp(t).to_pydatetime()).deg)


def project(args):
    """magnetogram 한 장 → (Carrington 칸 번호, 칸 평균 B_r [G]). 중앙자오선 ±60°, μ ≥ 0.2 픽셀만."""
    slot, obs = args
    v = np.load(U.image_path("hmi", pd.Timestamp(slot))).astype(np.float32)
    on, mu, lat, lon, _ = U._disk_geometry(v.shape[-1], int(round(U.b0_deg(obs) * 10)))
    sel = on & (mu >= C.PFSS_MIN_MU) & (np.abs(lon) <= C.PFSS_CMD_MAX_DEG)
    br = U.hmi_gauss(v[sel]) / mu[sel]
    lon_c = np.mod(lon[sel] + l0_deg(obs), 360.0)
    i_s = np.clip(((np.sin(np.radians(lat[sel])) + 1) / 2 * C.PFSS_IN_NSINLAT).astype(int), 0, C.PFSS_IN_NSINLAT - 1)
    i_l = np.clip((lon_c / 360 * C.PFSS_IN_NLON).astype(int), 0, C.PFSS_IN_NLON - 1)
    cell = i_s * C.PFSS_IN_NLON + i_l
    n = np.bincount(cell, minlength=C.PFSS_IN_NSINLAT * C.PFSS_IN_NLON)
    tot = np.bincount(cell, weights=br, minlength=C.PFSS_IN_NSINLAT * C.PFSS_IN_NLON)
    keep = np.flatnonzero(n > 0)
    return keep.astype(np.int32), (tot[keep] / n[keep]).astype(np.float32)


def fill_map(br: np.ndarray, valid: np.ndarray, lat_in: np.ndarray) -> np.ndarray:
    """결측 칸 채우기: 극관 (|위도| ≥ 60°) 은 같은 반구 60–75° 관측 평균, 나머지는 0."""
    out = np.where(valid, br, 0.0)
    lo, hi = C.PFSS_POLAR_FILL_LAT
    for sgn in (1, -1):
        ref = br[(sgn * lat_in >= lo) & (sgn * lat_in <= hi) & valid]
        cap = (sgn * lat_in >= lo) & ~valid
        out[cap] = ref.mean() if ref.size else 0.0
    return out


def main():
    with h5py.File(C.HDF5_PATH, "r") as h:
        t0 = pd.to_datetime(h["t0"][:].astype(str))
    split = pd.read_csv(C.SPLIT_CSV)
    if not (pd.to_datetime(split.t0).values == t0.values).all():
        raise SystemExit("splits.csv 와 dataset.h5 의 t0 순서가 다름")
    lags = pd.to_timedelta(C.PFSS_LAGS_H, unit="h")
    need = np.unique(np.concatenate([(t0 - l).values for l in lags]))
    imgs = obs_times(C.IMAGE_INDEX_CSV)
    print(f"[04f] 지도 시각 {len(need)} 개, HMI magnetogram {len(imgs)} 장 ({imgs.obs.min()} ~ {imgs.obs.max()})")

    grid = PFSSGrid()
    n_in = C.PFSS_IN_NSINLAT * C.PFSS_IN_NLON
    br = np.full(n_in, np.nan)
    seen = np.full(n_in, np.datetime64("NaT"), dtype="datetime64[s]")
    lat_in = grid.in_lat[:, None] * np.ones((1, C.PFSS_IN_NLON))
    latency = np.timedelta64(int(C.PFSS_LATENCY_H * 3600), "s")
    max_age = np.timedelta64(int(C.PFSS_MAX_AGE_DAYS * 86400), "s")
    out_ss = np.zeros((len(need), C.PFSS_OUT_NLAT, C.PFSS_OUT_NLON), np.float32)
    cov, age_med = np.zeros(len(need), np.float32), np.full(len(need), np.nan, np.float32)
    latest = np.full(len(need), np.datetime64("NaT"), dtype="datetime64[s]")
    sanity = {}
    obs_arr = imgs.obs.values.astype("datetime64[s]")
    jobs = list(zip(imgs.time.astype(str), imgs.obs.astype(str)))
    j = 0
    with ProcessPoolExecutor(6) as ex:
        res = ex.map(project, jobs, chunksize=16)
        for k, T in enumerate(tqdm(need.astype("datetime64[s]"), desc="PFSS 지도", mininterval=10)):
            while j < len(jobs) and obs_arr[j] <= T - latency:          # T - 지연 까지 관측된 영상만 넣는다
                cells, vals = next(res)
                br[cells] = vals
                seen[cells] = obs_arr[j]
                j += 1
            valid = ~np.isnat(seen) & (seen >= T - max_age)
            cov[k] = valid.mean()
            if valid.any():
                latest[k] = seen[valid].max()
                age_med[k] = float(np.median((T - seen[valid]) / np.timedelta64(1, "D")))
            phot = fill_map(br.reshape(C.PFSS_IN_NSINLAT, -1), valid.reshape(C.PFSS_IN_NSINLAT, -1), lat_in)
            out_ss[k] = grid.source_surface(phot, l0_deg(pd.Timestamp(T)))
            ts = str(pd.Timestamp(T))[:16]
            if ts in SANITY_TIMES:
                sanity[ts] = dict(phot=phot, valid=valid.reshape(phot.shape), ss=out_ss[k].copy(),
                                  age=np.where(valid, (T - seen) / np.timedelta64(1, "D"), np.nan).reshape(phot.shape))

    ok = ~np.isnat(latest)
    bad = ok & (latest > need.astype("datetime64[s]") - latency)
    if bad.any():
        raise SystemExit(f"지연 조건 위반 지도 {int(bad.sum())} 개 — 저장 안 함")
    if not np.isfinite(out_ss).all():
        raise SystemExit("source surface 지도에 NaN")

    # train 샘플이 쓰는 지도만으로 정규화 상수 (RMS)
    pos = {t: i for i, t in enumerate(need)}
    tr_t0 = t0[(split.split == "train").values]
    tr_maps = np.unique([pos[(t - l).to_datetime64()] for t in tr_t0 for l in lags])
    scale = float(np.sqrt(np.mean(out_ss[tr_maps].astype(np.float64) ** 2)))

    with h5py.File(C.PFSS_FILE, "w") as h:
        h["times"] = np.array([str(pd.Timestamp(t))[:19] for t in need], dtype="S19")
        h["br_ss"] = out_ss
        h["coverage"] = cov
        h["age_median_days"] = age_med
        h["obs_latest"] = np.array(["" if np.isnat(x) else str(x) for x in latest], dtype="S19")
        for k in ("PFSS_LATENCY_H", "PFSS_CMD_MAX_DEG", "PFSS_MIN_MU", "PFSS_MAX_AGE_DAYS", "PFSS_LMAX", "PFSS_RSS",
                  "PFSS_IN_NSINLAT", "PFSS_IN_NLON", "PFSS_OUT_NLAT", "PFSS_OUT_NLON"):
            h.attrs[k] = getattr(C, k)
        h.attrs["train_scale"] = scale
        h.attrs["n_train_maps"] = len(tr_maps)
    np.savez(C.RUN_DIR / "pfss_sanity.npz", lat_in=grid.in_lat, lon_in=grid.in_lon, lat_out=grid.out_lat,
             lon_out=grid.out_lon, **{f"{k}|{n}": v for k, d in sanity.items() for n, v in d.items()})
    print(f"  → {C.PFSS_FILE}: 지도 {len(need)} 개, 모든 지도의 가장 늦은 관측 ≤ T - {C.PFSS_LATENCY_H:g} h 확인")
    print(f"  관측 칸 비율 (≤ {C.PFSS_MAX_AGE_DAYS:g}일): 중앙값 {np.median(cov):.2f}, 10% {np.percentile(cov, 10):.2f}, "
          f"0.5 미만 지도 {int((cov < 0.5).sum())} 개; 칸 나이 중앙값의 중앙값 {np.nanmedian(age_med):.1f}일")
    print(f"  train 정규화 RMS = {scale:.4f} G (train 지도 {len(tr_maps)} 개), |B_r,ss| 99% = "
          f"{np.percentile(np.abs(out_ss), 99):.3f} G")


if __name__ == "__main__":
    main()
