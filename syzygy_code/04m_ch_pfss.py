"""
04m_ch_pfss.py
-----------------------------------
CH-masked AIA193 + CH-open-field PFSS 특징 (config 29절, P1–P4 2×2 실험). train·valid 샘플의 t0 영상 슬롯만 (test 미사용).
  A  지도 (--maps): 슬롯 시각 T 마다 04f 와 같은 causal 광구 B_r Carrington 지도 (HMI LOS, B_r = B_los / μ, μ ≥ PFSS_MIN_MU,
     중앙자오선 ±PFSS_CMD_MAX_DEG, 관측 ≤ T − PFSS_LATENCY_H, PFSS_MAX_AGE_DAYS 넘은 칸 결측 → 극관 60–75° 평균·나머지 0)
     → 실수 구면조화 계수 (l ≤ 20, Carrington 틀). pfss.h5 의 같은 시각 source surface 지도와 일치 확인
  B  슬롯마다 (--trace, 병렬): 기존 CH 검출 (256² AIA193 + HMI, detect_coronal_holes 그대로) → CH-masked AIA193 128²
     (256² 에서 AIA × mask 후 2×2 평균, mask 밖 정확히 0). 2° 경도 × sin(위도) 등간격 (같은 면적) heliographic 격자 중
     보이는 반구·μ ≥ CH_MIN_MU 마디를 PFSS 로 추적 (pfss_trace, R_ss 2.5, 발자국 r = 1, ds 0.02) → open / closed.
     CH 안 마디 = seed. open seed 의 |B|(r_k), r_k = 1.0 … 2.5 (0.1) → 반지름마다 중앙값 B_CH(r) [16],
     f_s = |B(1)| / (2.5² |B(2.5)|) → P10/25/50/75/90 [5]. 진단: N_seed / N_open / N_closed / open_fraction,
     보이는 원반의 PFSS-open 면적·CH 와 겹침 (모델 입력 아님)
  C  특징 (--features): train 분포만 보고 변환 (log10) 결정·clipping (train p0.5 / p99.5)·표준화 (train 평균·표준편차)
     → 샘플별 feat_br [N, 17] = B 16 + missing flag, feat_fs [N, 6] = f_s 5 + missing flag (N_open = 0 이면 0 + flag 1)
저장: ch_pfss_coef.h5 (A), ch_pfss.h5 (B·C)

    python 04m_ch_pfss.py --maps
    python 04m_ch_pfss.py --trace
    python 04m_ch_pfss.py --features
"""
import argparse
import importlib
import os
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd

import config as C
import sw_utils as U

F = importlib.import_module("04f_pfss_maps")
RADII = np.round(np.arange(1.0, C.PFSS_RSS + 1e-9, 0.1), 1)                 # 1.0 … 2.5 (16)
FS_Q = (10, 25, 50, 75, 90)
SEED_DLON = 2.0                                                             # 격자 경도 간격 (deg), sin(위도) 간격 = 같은 값의 라디안
COEF_FILE = C.BASE_DIR / "ch_pfss_coef.h5"
OUT_FILE = C.CH_PFSS_FILE


def slots():
    """train·valid 샘플의 t0 영상 슬롯 (dataset img_idx 의 마지막 = lag 0, 슬롯 ≤ t0) → (고유 슬롯 번호, 시각, 샘플 → 행)."""
    with h5py.File(C.HDF5_PATH, "r") as h:
        t0 = pd.to_datetime(h["t0"][:].astype(str))
        times = pd.to_datetime(h["image_times"][:].astype(str))
        idx0 = h["img_idx"][:, -1]
    split = pd.read_csv(C.SPLIT_CSV)["split"].to_numpy()
    use = np.isin(split, ["train", "valid"])
    assert (times.values[idx0] <= t0.values).all()
    uniq = np.unique(idx0[use])
    row = np.full(len(t0), -1, np.int64)
    pos = pd.Series(np.arange(len(uniq)), index=uniq)
    row[use] = pos.reindex(idx0[use]).to_numpy()
    return uniq, times[uniq], row, t0, split


# ── A. causal 광구 지도 → 구면조화 계수 ──
def build_maps():
    from pfss import PFSSGrid
    uniq, T, row, t0, split = slots()
    need = np.sort(T.values.astype("datetime64[s]"))
    imgs = F.obs_times(C.IMAGE_INDEX_CSV)
    print(f"[04m A] 슬롯 시각 {len(need)} 개, HMI magnetogram {len(imgs)} 장", flush=True)
    grid = PFSSGrid()
    n_in = C.PFSS_IN_NSINLAT * C.PFSS_IN_NLON
    br = np.full(n_in, np.nan)
    seen = np.full(n_in, np.datetime64("NaT"), dtype="datetime64[s]")
    lat_in = grid.in_lat[:, None] * np.ones((1, C.PFSS_IN_NLON))
    latency = np.timedelta64(int(C.PFSS_LATENCY_H * 3600), "s")
    max_age = np.timedelta64(int(C.PFSS_MAX_AGE_DAYS * 86400), "s")
    K = (C.PFSS_LMAX + 1) ** 2
    coef = np.zeros((len(need), K))
    cov = np.zeros(len(need), np.float32)
    latest = np.full(len(need), np.datetime64("NaT"), dtype="datetime64[s]")
    obs_arr = imgs.obs.values.astype("datetime64[s]")
    jobs = list(zip(imgs.time.astype(str), imgs.obs.astype(str)))
    j, t1 = 0, time.time()
    with ProcessPoolExecutor(6) as ex:
        res = ex.map(F.project, jobs, chunksize=16)
        for k, Tk in enumerate(need):
            while j < len(jobs) and obs_arr[j] <= Tk - latency:
                cells, vals = next(res)
                br[cells] = vals
                seen[cells] = obs_arr[j]
                j += 1
            valid = ~np.isnat(seen) & (seen >= Tk - max_age)
            cov[k] = valid.mean()
            if valid.any():
                latest[k] = seen[valid].max()
            phot = F.fill_map(br.reshape(C.PFSS_IN_NSINLAT, -1), valid.reshape(C.PFSS_IN_NSINLAT, -1), lat_in)
            coef[k] = grid.coefficients(phot)
            if k % 2000 == 0:
                print(f"    {k}/{len(need)} 지도 ({time.time() - t1:.0f} s)", flush=True)
    ok = ~np.isnat(latest)
    if (ok & (latest > need - latency)).any():
        raise SystemExit("지연 조건 위반 지도 — 저장 안 함")
    l0 = np.array([F.l0_deg(pd.Timestamp(t)) for t in need])
    b0 = np.array([U.b0_deg(pd.Timestamp(t)) for t in need])
    # pfss.h5 (04f) 의 같은 시각 source surface 지도와 비교
    from pfss import rotate_azimuth
    with h5py.File(C.PFSS_FILE, "r") as hp:
        ptimes = pd.to_datetime(hp["times"][:].astype(str))
        ppos = pd.Series(np.arange(len(ptimes)), index=ptimes)
        common = [k for k, t in enumerate(pd.DatetimeIndex(need)) if t in ppos.index]
        pick = common[:: max(1, len(common) // 50)]
        diffs = []
        for k in pick:
            ss = (grid.y_out @ rotate_azimuth(coef[k] * grid.g, C.PFSS_LMAX, np.radians(l0[k]))).reshape(grid.out_shape)
            ref = hp["br_ss"][int(ppos[pd.Timestamp(need[k])])]
            diffs.append(float(np.abs(ss - ref).max() / max(np.abs(ref).max(), 1e-9)))
    print(f"  pfss.h5 와 같은 시각 {len(pick)} 개 비교: source surface 최대 상대 차 {max(diffs):.2e}", flush=True)
    with h5py.File(COEF_FILE, "w") as ho:
        ho["times"] = np.array([str(pd.Timestamp(t))[:19] for t in need], dtype="S19")
        ho["coef"] = coef
        ho["l0_deg"], ho["b0_deg"], ho["coverage"] = l0, b0, cov
        ho["obs_latest"] = np.array(["" if np.isnat(x) else str(x) for x in latest], dtype="S19")
        ho.attrs["frame"] = "Carrington (계수), 지구 기준은 rotate_azimuth(coef, L0)"
        ho.attrs["max_rel_diff_vs_pfss_h5"] = max(diffs)
    print(f"[04m A] → {COEF_FILE} (관측 칸 비율 중앙값 {np.median(cov):.2f})", flush=True)


# ── B. 슬롯마다 CH 검출 + PFSS 추적 ──
_P = None


def _init():
    global _P
    from pfss_trace import PFSS3D
    _P = PFSS3D()


def grid_nodes():
    from pfss_trace import equal_area_seeds
    return equal_area_seeds(SEED_DLON)


def trace_slot(args):
    """슬롯 한 개 → (masked 128², mask 128², 통계 dict). 실패하면 masked = None (ok = 0, 오류 문자열)."""
    try:
        return _trace_slot(*args)
    except Exception as e:                                                      # 슬롯 하나 때문에 전체가 멈추지 않게
        return None, None, dict(ok=0, err=repr(e))


def _trace_slot(t_iso, coef_c, l0, b0, keep):
    from pfss import rotate_azimuth
    T = pd.Timestamp(t_iso)
    pa, ph = U.image_path("aia193", T), U.image_path("hmi", T)
    if not (pa.exists() and ph.exists()):
        return None, None, dict(ok=0)
    aia, hmi = np.load(pa).astype(np.float32), np.load(ph).astype(np.float32)
    mask, (on, mu, lat, lon, area), bgauss, thr = U.detect_coronal_holes(aia, hmi, T)
    res = aia.shape[-1]
    m256 = mask.astype(np.float32)
    masked128 = U.downsample(aia * m256)
    mask128 = U.downsample(m256) > 0
    ch_area = float((area * mask).sum() / (2 * np.pi))
    inside = masked128[mask128]
    full = U.downsample(m256) == 1.0
    # 격자 마디 → 영상 (256²) 픽셀
    glat, glon, dom = grid_nodes()
    x, y, mu_g = U.sun_xy(glat, glon, b0)
    vis = (mu_g >= C.CH_MIN_MU) & (np.abs(glon) < 90)
    col = np.clip(((x + C.IMG_FOV_RSUN) / (2 * C.IMG_FOV_RSUN) * res).astype(int), 0, res - 1)
    rowp = np.clip(((y + C.IMG_FOV_RSUN) / (2 * C.IMG_FOV_RSUN) * res).astype(int), 0, res - 1)
    in_ch = vis & mask[rowp, col]
    G = _P.field_grid(rotate_azimuth(np.asarray(coef_c), C.PFSS_LMAX, np.radians(l0)))
    sel = np.flatnonzero(vis)
    out = _P.trace(G, glat[sel], glon[sel], radii=RADII, ds=0.02, max_steps=600, keep_path=keep)
    st = out["state"]
    ch = in_ch[sel]
    op = st == 1
    n_seed, n_open, n_closed, n_und = int(ch.sum()), int((ch & op).sum()), int((ch & (st == 0)).sum()), int((ch & (st == -1)).sum())
    bprof = np.full(len(RADII), np.nan)
    fsq = np.full(len(FS_Q), np.nan)
    n_fs_bad = 0
    if n_open > 0:
        bb = out["babs"][ch & op]                                               # [n_open, 16]
        bprof = np.nanmedian(bb, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            fs = bb[:, 0] / (C.PFSS_RSS ** 2 * bb[:, -1])
        good = np.isfinite(fs) & (fs > 0)
        n_fs_bad = int((~good).sum())
        if good.any():
            fsq = np.percentile(fs[good], FS_Q)
    stats = dict(ok=1, ch_area=ch_area, n_seed=n_seed, n_open=n_open, n_closed=n_closed, n_undet=n_und,
                 open_frac=n_open / n_seed if n_seed else np.nan, n_fs_bad=n_fs_bad,
                 disk_nodes=int(vis.sum()), disk_open=int(op.sum()), overlap=int((ch & op).sum()),
                 union=int((ch | op).sum()), node_sr=dom,
                 in_mean=float(inside.mean()) if inside.size else np.nan,
                 in_median=float(np.median(inside)) if inside.size else np.nan,
                 in_p10=float(np.percentile(inside, 10)) if inside.size else np.nan,
                 in_p90=float(np.percentile(inside, 90)) if inside.size else np.nan,
                 n_full=int(full.sum()), bprof=bprof, fsq=fsq)
    if keep:                                                                     # 그림용
        stats.update(trace=out, sel=sel, in_ch=ch, glat=glat[sel], glon=glon[sel], aia=aia, hmi=hmi, mask=mask,
                     mu=mu, G_l0=l0, b0=b0)
    return masked128, mask128, stats


def run_trace(workers: int = 8):
    uniq, T, row, t0, split = slots()
    with h5py.File(COEF_FILE, "r") as hc:
        ctimes = pd.to_datetime(hc["times"][:].astype(str))
        coef, l0, b0 = hc["coef"][:], hc["l0_deg"][:], hc["b0_deg"][:]
        obs_latest = pd.to_datetime(pd.Series(hc["obs_latest"][:].astype(str)), errors="coerce")
    cpos = pd.Series(np.arange(len(ctimes)), index=ctimes)
    ci = cpos.reindex(T).to_numpy()
    assert not np.isnan(ci).any(), "계수 없는 슬롯 — --maps 다시"
    ci = ci.astype(int)
    lat_ok = obs_latest.to_numpy()[ci]
    assert (pd.isna(lat_ok) | (lat_ok <= (T - pd.Timedelta(hours=C.PFSS_LATENCY_H)).values)).all(), "PFSS 지도에 슬롯 − 지연 이후 관측"
    M = len(uniq)
    print(f"[04m B] 슬롯 {M} 개 추적 ({workers} 프로세스, 격자 {SEED_DLON}° 등면적)", flush=True)
    r = C.TRAIN_RES
    keys = ["ok", "ch_area", "n_seed", "n_open", "n_closed", "n_undet", "open_frac", "n_fs_bad", "disk_nodes", "disk_open",
            "overlap", "union", "in_mean", "in_median", "in_p10", "in_p90", "n_full"]
    stat = {k: np.full(M, np.nan) for k in keys}
    bprof = np.full((M, len(RADII)), np.nan)
    fsq = np.full((M, len(FS_Q)), np.nan)
    tmp = OUT_FILE.with_suffix(".tmp.h5")
    t1 = time.time()
    with h5py.File(tmp, "w") as ho:
        dm = ho.create_dataset("masked193", (M, r, r), dtype="float16", chunks=(1, r, r), compression="lzf")
        dk = ho.create_dataset("mask128", (M, r, r), dtype="uint8", chunks=(1, r, r), compression="lzf")
        jobs = [(str(t), coef[c], float(l0[c]), float(b0[c]), False) for t, c in zip(T, ci)]
        with ProcessPoolExecutor(workers, initializer=_init) as ex:
            for k, (m128, k128, s) in enumerate(ex.map(trace_slot, jobs, chunksize=8)):
                if m128 is not None:
                    dm[k] = m128                                                # float32 → HDF5 float16 (dataset.h5 와 같은 반올림)
                    dk[k] = k128.astype(np.uint8)
                    for kk in keys:
                        stat[kk][k] = s[kk]
                    bprof[k], fsq[k] = s["bprof"], s["fsq"]
                else:
                    stat["ok"][k] = 0
                    if "err" in s:
                        print(f"    ! 슬롯 {T[k]} 실패: {s['err']}", flush=True)
                if k % 1000 == 0:
                    print(f"    {k}/{M} 슬롯 ({time.time() - t1:.0f} s)", flush=True)
        ho["slot_img_idx"] = uniq.astype(np.int64)
        ho["slot_times"] = np.array([str(t)[:19] for t in T], dtype="S19")
        ho["sample_slot"] = row
        ho["t0"] = np.array([str(t)[:19] for t in t0], dtype="S19")
        for kk in keys:
            ho[kk] = stat[kk]
        ho["bprof_raw"] = bprof
        ho["fsq_raw"] = fsq
        ho["radii"] = RADII
        ho["fs_q"] = np.array(FS_Q)
        ho["pfss_obs_latest"] = np.array([str(x)[:19] for x in lat_ok], dtype="S19")
        ho.attrs["seed_grid"] = f"경도 {SEED_DLON}° × sin(위도) 간격 {np.radians(SEED_DLON):.4f} (같은 면적), μ ≥ {C.CH_MIN_MU}, |경도| < 90°"
        ho.attrs["trace"] = "pfss_trace.PFSS3D (ln r 61 × 여위도 120 × 경도 240 격자, 삼선형), RK4 ds = 0.02 R_sun, 최대 600 step"
        ho.attrs["br_definition"] = f"B_r = B_los / μ (μ ≥ {C.PFSS_MIN_MU}, 중앙자오선 ±{C.PFSS_CMD_MAX_DEG}°), 04f 와 같은 causal 지도"
    os.replace(tmp, OUT_FILE)
    print(f"[04m B] → {OUT_FILE} ({(time.time() - t1) / 60:.1f} 분)", flush=True)


# ── C. 특징 ──
def skew(x):
    x = x[np.isfinite(x)]
    return float(((x - x.mean()) ** 3).mean() / max(x.std() ** 3, 1e-12))


def build_features():
    with h5py.File(OUT_FILE, "r") as h:
        row = h["sample_slot"][:]
        bprof, fsq = h["bprof_raw"][:], h["fsq_raw"][:]
        n_open, n_seed = h["n_open"][:], h["n_seed"][:]
    split = pd.read_csv(C.SPLIT_CSV)["split"].to_numpy()
    have = row >= 0
    rs = np.where(have, row, 0)
    nopen = np.where(have, n_open[rs], 0)
    bp, fq = bprof[rs], fsq[rs]
    miss_b = ~(have & (nopen > 0) & np.isfinite(bp).all(1))
    miss_f = ~(have & (nopen > 0) & np.isfinite(fq).all(1))
    miss = miss_b | miss_f                                                     # 같은 flag 를 두 표현에 (P1–P4 같은 규칙)
    tr = (split == "train") & ~miss
    rep = {}
    feats = {}
    for name, raw, eps in (("br", bp, 1e-3), ("fs", fq, 0.0)):
        sk = [skew(raw[tr, j]) for j in range(raw.shape[1])]
        use_log = max(sk) > 1.0                                                 # 사전 규칙: train 왜도 > 1 이면 log10
        x = np.log10(np.maximum(raw, 1e-12) + eps) if use_log else raw.copy()
        lo, hi = np.percentile(x[tr], 0.5, axis=0), np.percentile(x[tr], 99.5, axis=0)
        n_out = int(((x[tr] < lo) | (x[tr] > hi)).sum())
        x = np.clip(x, lo, hi)
        mu_, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        z = (x - mu_) / sd
        z[miss] = 0.0
        z = np.nan_to_num(z)
        feats[name] = np.concatenate([z, miss[:, None].astype(np.float32)], 1).astype(np.float32)
        rep[name] = dict(train_skew_raw=[round(v, 2) for v in sk], log10=use_log, eps=eps, clip_lo=lo, clip_hi=hi,
                         mean=mu_, std=sd, n_clipped_train=n_out)
    with h5py.File(OUT_FILE, "a") as h:
        for k in ("feat_br", "feat_fs", "missing"):
            if k in h:
                del h[k]
        h["feat_br"], h["feat_fs"] = feats["br"], feats["fs"]
        h["missing"] = miss.astype(np.uint8)
        for name, d in rep.items():
            for kk, v in d.items():
                h["feat_" + name].attrs[kk] = np.asarray(v) if not isinstance(v, bool) else v
    for name, d in rep.items():
        print(f"  [{name}] train 왜도 (raw) {d['train_skew_raw']} → log10 {d['log10']} (eps {d['eps']}), clip train p0.5/p99.5 "
              f"(train 값 중 clip {d['n_clipped_train']}), 표준화 train 평균·표준편차")
    for s in ("train", "valid"):
        m = split == s
        print(f"  {s}: 샘플 {int(m.sum())}, missing (N_open = 0 또는 CH 없음) {int(miss[m].sum())} ({100 * miss[m].mean():.2f}%)")
    print(f"[04m C] → {OUT_FILE}: feat_br [N, {feats['br'].shape[1]}], feat_fs [N, {feats['fs'].shape[1]}]")


def report():
    """학습 전 분포 보고 (지시서 9·40·43·45절)."""
    with h5py.File(OUT_FILE, "r") as h:
        row = h["sample_slot"][:]
        d = {k: h[k][:] for k in ("ch_area", "n_seed", "n_open", "n_closed", "n_undet", "open_frac", "n_fs_bad", "disk_nodes",
                                  "disk_open", "overlap", "union", "in_mean", "in_median", "in_p10", "in_p90", "ok")}
        bprof, fsq = h["bprof_raw"][:], h["fsq_raw"][:]
    split = pd.read_csv(C.SPLIT_CSV)["split"].to_numpy()
    q = lambda x: " / ".join(f"{v:.3g}" for v in np.nanpercentile(x, [0, 10, 25, 50, 75, 90, 100]))
    for s in ("train", "valid"):
        rr = row[(split == s) & (row >= 0)]
        print(f"\n  [{s}] 샘플 {len(rr)} (최소 / p10 / p25 / p50 / p75 / p90 / 최대):")
        print(f"    CH 면적 (보이는 반구 비) {q(d['ch_area'][rr])}")
        print(f"    N_CH_seed {q(d['n_seed'][rr])}")
        print(f"    N_open {q(d['n_open'][rr])} | N_open = 0: {int((d['n_open'][rr] == 0).sum())} "
              f"({100 * (d['n_open'][rr] == 0).mean():.2f}%), 그 중 CH 없음 (N_seed = 0) {int((d['n_seed'][rr] == 0).sum())}")
        print(f"    open_fraction (N_seed > 0) {q(d['open_frac'][rr][d['n_seed'][rr] > 0])}, undetermined 합 {int(np.nansum(d['n_undet'][rr]))}")
        for j, rk in enumerate(RADII):
            if j in (0, 1, 2, 5, 10, 15):
                print(f"    B_CH({rk:.1f}) [G] {q(bprof[rr, j])}")
        for j, qq in enumerate(FS_Q):
            print(f"    f_s P{qq} {q(fsq[rr, j])}")
        print(f"    NaN: B {int(np.isnan(bprof[rr]).any(1).sum())} 샘플, f_s {int(np.isnan(fsq[rr]).any(1).sum())} 샘플, "
              f"f_s 비유한 장선 합 {int(np.nansum(d['n_fs_bad'][rr]))}, 영상 실패 {int((d['ok'][rr] == 0).sum())}")
        ov = d["overlap"][rr] / np.maximum(d["n_seed"][rr], 1)
        jac = d["overlap"][rr] / np.maximum(d["union"][rr], 1)
        print(f"    PFSS-open 원반 면적 비 {q(d['disk_open'][rr] / d['disk_nodes'][rr])}, CH ∩ open / CH {q(ov[d['n_seed'][rr] > 0])}, "
              f"Jaccard {q(jac[d['union'][rr] > 0])}, 심한 불일치 (CH ≥ 20 seed 이고 open 0) "
              f"{int(((d['n_seed'][rr] >= 20) & (d['n_open'][rr] == 0)).sum())}")
        m = d["n_seed"][rr] > 0
        print(f"    masked 영상 CH 안 화소 평균 {np.nanmedian(d['in_mean'][rr][m]):.3f}, 중앙값 {np.nanmedian(d['in_median'][rr][m]):.3f}, "
              f"p10 {np.nanmedian(d['in_p10'][rr][m]):.3f}, p90 {np.nanmedian(d['in_p90'][rr][m]):.3f} (샘플 중앙값)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.maps:
        build_maps()
    if a.trace:
        run_trace(a.workers)
    if a.features:
        build_features()
    if a.trace or a.features or a.report:
        report()


if __name__ == "__main__":
    main()
