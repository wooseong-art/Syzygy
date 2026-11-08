"""
04b_ch_features.py
-----------------------------------
dataset.h5 에 코로나 홀(CH) 특징과 태양풍 상태 라벨을 추가한다 (04_build_dataset.py 다음에 실행).

  ch_img  [N_img, 12] float32  영상별 CH 특징 (sw_utils.CH_IMG_COLS 순서)
          area_disk/cm/east/west  CH 면적 (보이는 반구 대비): 디스크 전체 / 중앙자오선 ±15° / 동쪽 -60~-15° / 서쪽
          lon_mean, lat_mean     CH 면적 가중 평균 경도(음수 = 동쪽, 다가오는 중) / |위도|  (/90)
          polarity, unipolarity  CH 안 평균 자기장 (G/10), 단극성 |ΣB|/Σ|B|
          darkness, n_regions    임계값 대비 어두운 정도, 조각 수 (/10)
          iou_12h, iou_24h       12 h / 24 h 전 CH 지도를 자전시켜 겹친 비율 (모양·위치 유지)
  regime  [N, 16] int8         정답 bin 별 상태 0 Slow / 1 Transition / 2 HSS / -1 결측

    python 04b_ch_features.py
"""
from concurrent.futures import ProcessPoolExecutor

import argparse

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
import sw_utils as U


def _load(t):
    pa, ph = U.image_path("aia193", t), U.image_path("hmi", t)
    if not (pa.exists() and ph.exists()):
        return None
    return np.load(pa).astype(np.float32), np.load(ph).astype(np.float32)


def image_features(t_iso: str) -> list[float]:
    """영상 한 장의 CH 특징 + 12 h / 24 h 전 영상과의 IoU (전 영상이 없으면 ±6 h 대체, 그래도 없으면 NaN)."""
    t = pd.Timestamp(str(t_iso))
    cur = _load(t)
    if cur is None:
        return [np.nan] * len(U.CH_IMG_COLS)
    f, m_now = U.coronal_hole_features(*cur, t)
    out = [f[k] for k in U.CH_FEATURES]
    for dh in (12, 24):
        iou = np.nan
        for tp in (t - pd.Timedelta(hours=dh), t - pd.Timedelta(hours=dh + 6), t - pd.Timedelta(hours=dh - 6)):
            prev = _load(tp)
            if prev is not None:
                _, m_old = U.coronal_hole_features(*prev, tp)
                iou = U.ch_iou(m_old, m_now, (t - tp).total_seconds() / 3600)
                break
        out.append(iou)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true", help="코로나 홀 특징은 그대로 두고 상태 라벨만 다시 만든다")
    args = ap.parse_args()
    with h5py.File(C.HDF5_PATH, "r") as h:
        times = h["image_times"][:].astype(str).tolist()
        t0 = pd.to_datetime(h["t0"][:].astype(str))
        target = h["target_kms"][:]
        ch = h["ch_img"][:] if args.labels_only else None

    if ch is None:
        with ProcessPoolExecutor(6) as ex:
            feats = list(tqdm(ex.map(image_features, times, chunksize=32), total=len(times),
                              desc="CH features", mininterval=10))
        ch = np.asarray(feats, dtype=np.float32)

    ace = pd.read_csv(C.ACE_CSV, parse_dates=["time"])
    v = U.HourlySeries.from_series(ace.set_index("time")["V"])
    regime = U.regime_labels(v, t0, target)

    with h5py.File(C.HDF5_PATH, "a") as h:
        for k in (("regime",) if args.labels_only else ("ch_img", "regime")):
            if k in h:
                del h[k]
        if not args.labels_only:
            h["ch_img"] = ch
            h["ch_img"].attrs["columns"] = ",".join(U.CH_IMG_COLS)
        h["regime"] = regime
        h["regime"].attrs["transition_dv"] = C.TRANSITION_DV
        h["regime"].attrs["hss_v"] = C.HSS_V

    has = ch[:, 0] > 0
    print(f"[04b] 영상 {len(ch)} 장: CH 검출 {100 * has.mean():.1f}%, 평균 면적 {np.nanmean(ch[:, 0]):.4f} "
          f"(중앙 띠 {np.nanmean(ch[:, 1]):.4f}), IoU 12h 중앙값 {np.nanmedian(ch[:, 10]):.2f}, "
          f"결측 {np.isnan(ch[:, 0]).sum()} 장")
    lab = regime[regime >= 0]
    print("  상태 비율: " + ", ".join(f"{n} {100 * np.mean(lab == i):.1f}%" for i, n in enumerate(C.REGIMES)))
    print(f"  → {C.HDF5_PATH}")


if __name__ == "__main__":
    main()
