"""
fc_predict.py — SYZYGY FINAL (CME-free) 운영 추론: predict(issue_time_utc, source_config) → dict (JSON 직렬화 가능).

발행 시각 → t0 = 발행 시각을 6 h 격자 (00/06/12/18 UT) 로 내림 (학습 표본과 같은 시각 정의). 예측 대상 = ACE 6 h 평균 [t0 + h − 6, t0 + h), h = 24/48/72/96.

source_config:
  {"mode": "archive"}  저장된 전처리 입력 (features.npz / dataset.h5 / ch_pfss.h5) 의 그 t0 행 — 학습·평가와 비트 단위 같은 입력 (재현·replay)
  {"mode": "live"}     원 자료에서 입력을 만든다 (main pipeline 의 실시간 함수 재사용):
                       ACE  : sw_utils.load_ace_hourly(use_realtime=True)  (L2 > K0 > SWPC ACE RT 1 분 아카이브 > RTSW JSON)
                       AIA193 / HMI : 로컬 영상 (images/) → 없으면 JSOC (synoptic / hmi.M_720s(_nrt)) 에서 slot 시각 이전 영상만 받아 전처리
                                      (fc 캐시 features/live_images 에 저장), 특징은 학습 때와 같은 함수 (slot_features, 04b image_features,
                                      detect_coronal_holes) 로 계산
                       PFSS : 실시간 causal 지도 생성은 이 파이프라인에 없음 → Router 가 PFSS (R1/R2) 를 쓰면 영상 경로 불가 → G01 fallback
  선택 키: "device" ("cpu"), "monitor" (True: monitoring/forecast/events.jsonl 에 기록), "allow_download" (live, 기본 True)

fallback: 영상·PFSS 입력이 하나라도 없으면 RegimeNet/Router 를 쓰지 않고 G01 기반 (최종 보정이 G01 에 정의돼 있으면 같은 종류의 G01 보정) 으로 낸다.
CME (DONKI) 는 어떤 경로에서도 읽지 않는다. HELIOLAB 사이트 API 가 없으므로 upload_status = LOCAL_ONLY (업로드 안 함).

    python fc_predict.py --issue 2024-05-10T12:00 --mode archive
    python fc_predict.py --issue now --mode live
"""
import argparse
import importlib
import json
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

import fc_common as F
import config as C
import sw_utils as U
import cr_train as CT
import run_phase as RP
import train as PT
import oc_predict as OP
import oc_calibrate as CAL
import oc_onset as ON
import build_features as BF
from oc_common import K
from fc_model import load_regime, load_router

KH = list(F.KH)
RLAGS = C.lags_for(24)                       # RegimeNet 영상 시점 [96, 72, 48, 24, 0]
CHLAGS = C.lags_for(C.CH_STEP_HOURS)          # CH 특징 시점 [96, 84, …, 0]
LIVE_CACHE = F.EXP / "features" / "live_images"
_P = None


def _sha16(path):
    return F.sha256_file(path)[:16]


class Pipeline:
    def __init__(self, device="cpu"):
        t = time.time()
        torch.set_num_threads(4)
        self.device = device
        self.sel = F.read_json(F.SELECTION)
        assert self.sel, "final_selection.json 없음 — fc_analysis.py --split valid 먼저"
        self.variant, self.w, self.base, self.corr = self.sel["router_variant"], float(self.sel["w"]), self.sel["base"], self.sel["correction"]
        self.meta = F.read_json(F.DIRS["metrics"] / "ensemble_meta_valid.json")
        info = F.read_json(F.COLLIN / "predictions" / "preds_info.json")
        self.ids = {}
        self.g01, self.p2o = [], []
        for s in F.SEEDS:
            for key, lst, nm in (("G4x3", self.g01, "G01"), ("P2o_4x3", self.p2o, "P2o")):
                ck = info[f"{key}|{s}"]["ckpt"]
                lst.append(RP.load(ck)[0])
                self.ids[f"{nm}_s{s}"] = _sha16(ck)
        self.occ, self.t2 = [], []
        minfo = F.read_json(F.HSSON / "predictions" / "model_preds_info.json")
        self.tau = float(np.mean([minfo[f"T2|{s}"]["tau"] for s in F.SEEDS]))
        for s in F.SEEDS:
            self.occ.append(OP.load(f"O_seed{s}")[0])
            n2 = "GRID_T2_lam0.1_beta0.0_seed42" if s == 42 else f"T2_seed{s}"
            self.t2.append(OP.load(n2)[0])
            self.ids[f"O_s{s}"] = _sha16(F.HSSON / "checkpoints" / f"O_seed{s}" / "best.pt")
            self.ids[f"T2_s{s}"] = _sha16(F.HSSON / "checkpoints" / n2 / "best.pt")
        self.regime, self.router = [], []
        for s in F.SEEDS:
            p = F.DIRS["checkpoints"] / "regime" / f"regime_cf_s{s}.pt"
            self.regime.append(load_regime(p, device)[0])
            self.ids[f"RegimeNetCF_s{s}"] = _sha16(p)
            rp = Path(F.read_json(F.DIRS["checkpoints"] / "router" / f"meta_router_{self.variant}_s{s}.json")["checkpoint"])
            self.router.append(load_router(rp, device)[0])
            self.ids[f"Router{self.variant}_s{s}"] = _sha16(rp)
        self.bc = {}
        for b in ("A", self.base):
            p = F.DIRS["configs"] / f"boxcox_maps_{b}.pkl"
            if p.exists():
                with open(p, "rb") as fh:
                    self.bc[b] = pickle.load(fh)
        self.groups = K.groups_of("E001")
        self._feats = self._ds = None
        self.init_sec = time.time() - t

    # ── 입력 (archive) ──
    def feats(self):
        if self._feats is None:
            self._feats = CT.GridFeatures("4x3")
            self._t0 = pd.to_datetime(np.asarray(self._feats.t0).astype(str))
        return self._feats

    def archive_inputs(self, t0):
        f = self.feats()
        hit = np.flatnonzero(self._t0 == t0)
        if not len(hit):
            raise KeyError(f"{t0} 는 저장된 표본 (2010-05-17 … 2026-09-20, 6 h) 에 없음 — live 모드를 쓰거나 다른 시각")
        i = int(hit[0])
        x, _ = f.tensors(np.array([i]), self.groups)
        out = dict(num=x, i=i, obs={}, missing=[])
        with h5py.File(F.CODE.parent / "dataset.h5", "r") as h:
            ds_lags = [int(v) for v in h.attrs["lags_hours"]]
            row = h["img_idx"][i]
            pos = [ds_lags.index(l) for l in RLAGS]
            ii = row[pos]
            u, inv = np.unique(ii, return_inverse=True)
            imgs = h["images"][u][inv].astype(np.float32).reshape(1, -1, C.TRAIN_RES, C.TRAIN_RES)
            cpos = [ds_lags.index(l) for l in CHLAGS]
            cu, cinv = np.unique(row[cpos], return_inverse=True)
            ch_img = h["ch_img"][cu][cinv]
            times = h["image_times"][:][row].astype(str)
            pk, rk = h["persist_kms"][i:i + 1], h["recur_kms"][i:i + 1]
        persist, recur = U.history_inputs(pk, rk, True)
        ch_vec = U.ch_dynamic_features(ch_img[None], C.CH_STEP_HOURS)
        with h5py.File(F.CODE.parent / "ch_pfss.h5", "r") as hc:
            r = int(hc["sample_slot"][i])
            masked = hc["masked193"][r].astype(np.float32)[None, None]
            phys = {"br": hc["feat_br"][i:i + 1].astype(np.float32), "fs": hc["feat_fs"][i:i + 1].astype(np.float32)}
            pfss_latest = hc["pfss_obs_latest"][r]                  # slot 별 (sample_slot)
        out["regime_x"] = (imgs, persist, recur, ch_vec)
        out["router_x"] = (masked, persist, recur, phys.get(F.ROUTERS[self.variant], np.zeros((1, 1), np.float32)))
        out["obs"] = dict(image_slots_used=sorted(set(times.tolist())), latest_image=max(times.tolist()),
                          pfss_obs_latest=str(pfss_latest.decode() if isinstance(pfss_latest, bytes) else pfss_latest),
                          ace_last_valid_hour=self._ace_last(t0))
        return out

    def _ace_last(self, t0, series=None):
        if series is None:
            a = pd.read_csv(F.CODE.parent / "ace" / "ace_hourly.csv", index_col=0, parse_dates=True)
            s = a.loc[(a.index < t0) & (a.index >= t0 - pd.Timedelta(days=3))]
            s = s[s["V"].notna()]
            return dict(hour=str(s.index[-1]) if len(s) else None, source=str(s["src"].iloc[-1]) if len(s) else None)
        s = series[series["V"].notna()]
        return dict(hour=str(s.index[-1]) if len(s) else None, source=str(s["src"].iloc[-1]) if len(s) else None,
                    source_counts_48h={k: int(v) for k, v in s.loc[s.index >= t0 - pd.Timedelta(hours=48), "src"].value_counts().items()})

    # ── 입력 (live) ──
    def _resolve(self, inst, t, allow_download, log):
        """slot 시각 t 의 전처리 영상 경로 (archive → live 캐시 → JSOC 에서 t 이전 영상). 없으면 None."""
        for p in (U.image_path(inst, t), LIVE_CACHE / inst / f"{t:%Y%m%d_%H%M}.npy"):
            if Path(p).exists():
                return Path(p)
        if not allow_download:
            return None
        got = U.fetch_nearest_image(inst, t, not_after=t)
        if got is None:
            log.append(f"{inst} {t} 없음 (JSOC)")
            return None
        data, hdr, url = got
        img = U.preprocess_sdo(inst, data, hdr).astype(np.float16)
        p = LIVE_CACHE / inst / f"{t:%Y%m%d_%H%M}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, img)
        idx = LIVE_CACHE / f"{inst}_index.csv"                 # PFSS causal 지도용 관측 시각 (T_OBS, 04f 와 같은 해석)
        with open(idx, "a", encoding="utf-8") as fh:
            if fh.tell() == 0:
                fh.write("time,t_obs,url\n")
            fh.write(f"{t},{hdr.get('T_OBS', '')},{url}\n")
        log.append(f"{inst} {t} ← {url}")
        return p

    def _live_fs(self, T, allow, log):
        """CH-open PFSS f_s 특징 [1, 6] (04f/04m 와 같은 함수): slot T 의 causal Carrington 지도 (관측 ≤ T − 지연, 28 일 안 HMI)
        → l ≤ 20 계수 → CH seed 장선 추적 → f_s 5 분위 → ch_pfss.h5 feat_fs 의 train 변환 (log10·clip·표준화). → (특징, 최근 관측 시각)"""
        M = importlib.import_module("04m_ch_pfss")
        from pfss import PFSSGrid
        latency = pd.Timedelta(hours=C.PFSS_LATENCY_H)
        lo, hi = T - pd.Timedelta(days=C.PFSS_MAX_AGE_DAYS), T - latency
        arch = M.F.obs_times(C.IMAGE_INDEX_CSV)[["time", "obs"]]
        arch = arch[(arch.obs >= lo) & (arch.obs <= hi)]
        have = set(pd.to_datetime(arch.time))
        extra = []
        for s in pd.date_range(lo.ceil("6h"), hi.floor("6h"), freq="6h"):        # archive 에 없는 HMI slot 은 live 캐시 / JSOC
            if s in have:
                continue
            p = self._resolve("hmi", s, allow, log)
            if p is not None and Path(p).parent == LIVE_CACHE / "hmi":
                extra.append(s)
        li = LIVE_CACHE / "hmi_index.csv"
        if li.exists() and extra:
            d = pd.read_csv(li, parse_dates=["time"]).drop_duplicates("time", keep="last")
            d = d[d.time.isin(extra)]
            p = d.t_obs.astype(str).str.extract(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)")
            d["obs"] = pd.to_datetime(p[0] + "-" + p[1] + "-" + p[2]) + pd.to_timedelta(p[3].astype(float) * 3600 + p[4].astype(float) * 60
                                                                                          + p[5].astype(float), unit="s")
            arch = pd.concat([arch, d[(d.obs >= lo) & (d.obs <= hi)][["time", "obs"]]])
        arch = arch.sort_values("obs", kind="stable")
        if not len(arch):
            return None, None
        grid = PFSSGrid()
        n_in = C.PFSS_IN_NSINLAT * C.PFSS_IN_NLON
        br = np.full(n_in, np.nan)
        seen = np.full(n_in, np.datetime64("NaT"), dtype="datetime64[s]")
        for slot, obs in zip(arch.time.astype(str), arch.obs.astype(str)):
            cells, vals = M.F.project((slot, obs))
            br[cells] = vals
            seen[cells] = np.datetime64(pd.Timestamp(obs).to_datetime64(), "s")
        valid = ~np.isnat(seen) & (seen >= np.datetime64(T - pd.Timedelta(days=C.PFSS_MAX_AGE_DAYS)))
        lat_in = grid.in_lat[:, None] * np.ones((1, C.PFSS_IN_NLON))
        phot = M.F.fill_map(br.reshape(C.PFSS_IN_NSINLAT, -1), valid.reshape(C.PFSS_IN_NSINLAT, -1), lat_in)
        coef = grid.coefficients(phot)
        M._init()
        _, _, st = M._trace_slot(str(T), coef, M.F.l0_deg(T), U.b0_deg(T), False)
        latest = str(seen[valid].max()) if valid.any() else None
        with h5py.File(F.CODE.parent / "ch_pfss.h5", "r") as hc:
            at = dict(hc["feat_fs"].attrs)
        raw = np.asarray(st.get("fsq", np.full(5, np.nan)), float)
        miss = not (st.get("ok") == 1 and st.get("n_open", 0) > 0 and np.isfinite(raw).all())
        x = np.log10(np.maximum(raw, 1e-12) + float(at["eps"])) if bool(at["log10"]) else raw
        z = (np.clip(x, at["clip_lo"], at["clip_hi"]) - at["mean"]) / at["std"]
        z = np.zeros(5) if miss else np.nan_to_num(z)
        return np.concatenate([z, [float(miss)]])[None].astype(np.float32), latest

    def live_inputs(self, t0, cfg):
        allow = cfg.get("allow_download", True)
        log, missing = [], []
        if cfg.get("ace_source") == "local":            # replay 검사용: 학습에 쓴 hourly 파일 (원격 요청 없음)
            a = pd.read_csv(F.CODE.parent / "ace" / "ace_hourly.csv", index_col=0, parse_dates=True)
            ace = a.loc[(a.index >= t0 - pd.Timedelta(hours=K.REC_START_H + 24)) & (a.index < t0)]
        else:
            ace = U.load_ace_hourly(t0 - pd.Timedelta(hours=K.REC_START_H + 24), t0, use_realtime=True)
        v = U.HourlySeries.from_series(ace["V"].astype(float).reindex(pd.date_range(ace.index[0], t0 - pd.Timedelta(hours=1), freq="h")))
        hist, rec, v27, now, src = BF.speed_inputs(v, [t0])
        if not np.isfinite(hist).any():
            missing.append("ACE_48h")
        # 영상 slot: t0 − 12 k h (k = 0…10; CH IoU 가 12/24 h 전 영상을 씀). 학습과 같은 대체 규칙: t, t − 6 h (t0 이후 안 씀)
        paths = {}
        for k in range(11):
            T = t0 - pd.Timedelta(hours=12 * k)
            for inst in ("aia193", "hmi"):
                got = None
                for c in (T, T - pd.Timedelta(hours=C.IMG_FILL_TOL_HOURS)):
                    got = self._resolve(inst, c, allow, log)
                    if got is not None:
                        paths[(inst, T)] = (c, got)
                        break
        orig = U.image_path

        def path_for(inst, t):                         # 학습 특징 함수들이 이 경로로 영상을 읽음 (slot 대체 반영, 그 밖은 archive → live 캐시)
            hit = paths.get((inst, t))
            if hit:
                return hit[1]
            p = Path(orig(inst, t))
            c = LIVE_CACHE / inst / f"{pd.Timestamp(t):%Y%m%d_%H%M}.npy"
            return c if (not p.exists() and c.exists()) else p
        U.image_path = path_for
        try:
            G = np.zeros((1, len(K.G_LAGS), 13), np.float32)
            for li, L in enumerate(K.G_LAGS):
                T = t0 - pd.Timedelta(hours=L)
                r = BF.slot_features(str(T)) if all((ins, T) in paths for ins in ("aia193", "hmi")) else None
                if r is None:
                    missing.append(f"G_lag{L}")
                    continue
                nc, nv = r[0], r[1]
                frac = np.where(nv > 0, nc / np.maximum(nv, 1), 0.0).reshape(4, 3)[::-1].reshape(12)
                G[0, li, :12], G[0, li, 12] = frac, 1.0
            B04 = importlib.import_module("04b_ch_features")
            img_ok = all((ins, t0 - pd.Timedelta(hours=L)) in paths for L in set(RLAGS) | set(CHLAGS) for ins in ("aia193", "hmi"))
            regime_x = router_x = pfss_latest = None
            if img_ok:
                imgs = []
                for L in RLAGS:
                    T = t0 - pd.Timedelta(hours=L)
                    for ins in ("aia193", "hmi"):
                        imgs.append(U.downsample(np.load(path_for(ins, T)).astype(np.float32)))
                imgs = np.stack(imgs)[None].astype(np.float16).astype(np.float32)
                ch_img = np.array([B04.image_features(str(t0 - pd.Timedelta(hours=L))) for L in CHLAGS], np.float32)
                ch_vec = U.ch_dynamic_features(ch_img[None], C.CH_STEP_HOURS)
                pk = hist[:, -C.PERSIST_HOURS:]
                rk = U.bin_means(v.window(t0 - pd.Timedelta(hours=C.RECURRENCE_HOURS), C.HORIZON_HOURS))[None]   # = sw_utils.solar_wind_inputs
                persist, recur = U.history_inputs(pk.astype(np.float32), rk.astype(np.float32), True)
                aia0, hmi0 = (np.load(path_for(ins, t0)).astype(np.float32) for ins in ("aia193", "hmi"))
                mask = U.detect_coronal_holes(aia0, hmi0, paths[("aia193", t0)][0])[0]
                masked = U.downsample(aia0 * mask.astype(np.float32)).astype(np.float16).astype(np.float32)[None, None]
                regime_x = (imgs, persist, recur, ch_vec)
                ph = F.ROUTERS[self.variant]
                if ph == "none":
                    router_x = (masked, persist, recur, np.zeros((1, 1), np.float32))
                elif ph == "fs":
                    fs, pfss_latest = self._live_fs(paths[("aia193", t0)][0], allow, log)
                    if fs is None:
                        missing.append("PFSS_no_HMI_history")
                    else:
                        router_x = (masked, persist, recur, fs)
                else:
                    missing.append("PFSS_Br_live_not_implemented")
            else:
                missing.append("images")
        finally:
            U.image_path = orig
        num = {"hist": torch.from_numpy(BF.pack(hist)), "rec": torch.from_numpy(BF.pack(rec)), "v27": torch.from_numpy(BF.pack(v27)),
               "v_now": torch.from_numpy(((now - K.V_MEAN) / K.V_STD).astype(np.float32)), "B": torch.zeros(1, 3),
               "C": torch.zeros(1, 16), "G": torch.from_numpy(G)}
        slots = sorted({str(c) for (ins, T), (c, p) in paths.items()})
        return dict(num=num, regime_x=regime_x, router_x=router_x, missing=missing, log=log, v_now_source=int(src[0]),
                    obs=dict(image_slots_used=slots, latest_image=slots[-1] if slots else None, pfss_obs_latest=pfss_latest,
                             ace_last_valid_hour=self._ace_last(t0, ace)))

    # ── 모델 ──
    @torch.no_grad()
    def run(self, inp):
        x = inp["num"]
        out = {}
        g = np.stack([PT.forward(n, x).numpy()[0] * K.V_STD + K.V_MEAN for n in self.g01])
        p2o = np.stack([PT.forward(n, x).numpy()[0] * K.V_STD + K.V_MEAN for n in self.p2o])
        occ = np.stack([torch.sigmoid(n(**self._mh(x))["occ_logits"]).numpy()[0] for n in self.occ])
        t2 = [n(**self._mh(x)) for n in self.t2]
        p17 = np.mean([torch.softmax(o["onset_logits"], 1).numpy()[0] for o in t2], 0)
        out.update(g01_per_seed=g, g01=g.mean(0), p2o=p2o.mean(0), pocc16=occ.mean(0), p17=p17,
                   t2_speed=np.mean([(o["speed"].numpy()[0] * K.V_STD + K.V_MEAN) for o in t2], 0))
        out["image_ok"] = inp.get("regime_x") is not None and inp.get("router_x") is not None
        if inp.get("regime_x") is not None:
            imgs, persist, recur, ch_vec = (torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in inp["regime_x"])
            z0 = torch.zeros(1, C.N_BINS, 3)
            z1 = torch.zeros(1, C.CME_VEC_DIM)
            reg = [m(imgs, z0, z1, persist, recur, ch_vec) for m in self.regime]
            out["regime16"] = np.mean([U.denorm_v(r["v"].numpy()[0]) for r in reg], 0)
            if out["image_ok"]:
                masked, persist, recur, chp = (torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in inp["router_x"])
                v16, probs = [], []
                for r, m in zip(reg, self.router):
                    lg = m(masked, z0, z1, persist, recur, chp)
                    p = torch.softmax(lg, -1)
                    v16.append(U.denorm_v((r["anchor"] + (p * r["dv"]).sum(-1)).numpy()[0]))
                    probs.append(p.numpy()[0])
                out["router16"] = np.mean(v16, 0)
                out["router_probs16"] = np.mean(probs, 0)
        return out

    @staticmethod
    def _mh(x):
        return dict(hist=x["hist"], rec=x["rec"], v27=x["v27"], v_now=x["v_now"], B=x["B"], Cx=x["C"], G=x["G"])

    def assemble(self, out, v_now_kms):
        """선택 파일의 base + 보정 하나. 영상 경로가 없으면 G01 기반 fallback."""
        g, pocc = out["g01"][None], out["pocc16"][KH][None]
        base = self.base if out["image_ok"] or self.base == "A" else "A"
        fallback = base != self.base
        if base == "A":
            v = g
        else:
            r4 = out["router16"][KH][None]
            v = {"B": r4, "C": 0.5 * g + 0.5 * r4, "D": (1 - self.w) * g + self.w * r4}[base]
        corr, vc = self.corr, v.copy()
        if corr.startswith("P2o_mix"):
            a = float(corr.split("_a")[1])
            vc = v + np.clip(a * pocc, 0, 1) * (out["p2o"][None] - g)
        elif corr == "C3":
            prm = [np.asarray(p) for p in self.meta["corrections"][base]["C3_params"]]
            vc = CAL.gate_apply(v, pocc, CAL.c3_feats(v, np.array([v_now_kms])), prm, "bounded", 100.0, 1.0)
        elif corr.startswith("HSS_BoxCox"):
            a = float(corr.split("_a")[1])
            bc, _ = RP.apply_boxcox(self.bc[base], v.astype(np.float64))
            vc = v + a * pocc * (bc - v)
        return base, fallback, v[0], vc[0]


def predict(issue_time_utc, source_config=None):
    global _P
    cfg = dict(source_config or {})
    mode = cfg.get("mode", "archive")
    t_all = time.time()
    if _P is None:
        _P = Pipeline(cfg.get("device", "cpu"))
    P = _P
    it = pd.Timestamp.now(tz="UTC").tz_localize(None) if str(issue_time_utc) == "now" else pd.Timestamp(issue_time_utc).tz_localize(None) \
        if pd.Timestamp(issue_time_utc).tzinfo else pd.Timestamp(issue_time_utc)
    t0 = it.floor("6h")
    t1 = time.time()
    inp = P.archive_inputs(t0) if mode == "archive" else P.live_inputs(t0, cfg)
    t_in = time.time() - t1
    t2 = time.time()
    out = P.run(inp)
    v_now = float(inp["num"]["v_now"].numpy()[0] * K.V_STD + K.V_MEAN)
    base, fallback, before, after = P.assemble(out, v_now)
    t_model = time.time() - t2
    onset_bin = int(ON.head_onset(out["p17"][None], P.tau)[0])
    missing = inp.get("missing", [])
    res = {
        "issue_time_utc": str(it), "t0_utc": str(t0), "mode": mode, "horizons_h": list(F.H),
        "target_windows_utc": [[str(t0 + pd.Timedelta(hours=h - 6)), str(t0 + pd.Timedelta(hours=h))] for h in F.H],
        "final_model": P.sel["final"] if not fallback else f"A{'' if P.corr == 'none' else '+' + P.corr} (fallback)",
        "final_speed_kms": after.round(2).tolist(), "before_correction_kms": before.round(2).tolist(),
        "correction": P.corr, "correction_delta_kms": (after - before).round(2).tolist(),
        "ensemble": dict(base=base, w_router=P.w if base == "D" else ({"A": 0.0, "B": 1.0, "C": 0.5}[base]), router_variant=P.variant),
        "g01_kms": out["g01"].round(2).tolist(), "g01_per_seed_kms": out["g01_per_seed"].round(2).tolist(),
        "p2o_kms": out["p2o"].round(2).tolist(), "t2_speed_kms": out["t2_speed"].round(2).tolist(),
        "regimenet_cf_kms": out["regime16"][KH].round(2).tolist() if "regime16" in out else None,
        "regimenet_cf_16bin_kms": out["regime16"].round(2).tolist() if "regime16" in out else None,
        "router_cf_kms": out["router16"][KH].round(2).tolist() if "router16" in out else None,
        "router_cf_16bin_kms": out["router16"].round(2).tolist() if "router16" in out else None,
        "router_regime_probs_16x3": out["router_probs16"].round(4).tolist() if "router_probs16" in out else None,
        "hss_occupancy_prob_16": out["pocc16"].round(4).tolist(),
        "onset_prob_17": out["p17"].round(4).tolist(), "onset_no_event_prob": round(float(out["p17"][-1]), 4),
        "onset_any_96h_prob": round(float(1 - out["p17"][-1]), 4), "onset_tau": round(P.tau, 4),
        "onset_pred_bin": onset_bin if onset_bin < F.NB else None,
        "onset_pred_window_utc": ([str(t0 + pd.Timedelta(hours=6 * onset_bin)), str(t0 + pd.Timedelta(hours=6 * onset_bin + 6))]
                                  if onset_bin < F.NB else None),
        "v_now_kms": round(v_now, 2), "latest_obs": inp.get("obs", {}),
        "models": dict(P.ids, selection_sha16=_sha16(F.SELECTION), preregistration_sha16=_sha16(F.PREREG)),
        "status": dict(state="FALLBACK_G01" if fallback else ("DEGRADED" if missing else "OK"), fallback=fallback, missing_inputs=missing,
                       cme_inputs_used=False, download_log=inp.get("log", [])),
        "latency_s": dict(pipeline_init=round(P.init_sec, 3), inputs=round(t_in, 3), models=round(t_model, 3), total=round(time.time() - t_all, 3)),
        "upload_status": "LOCAL_ONLY",
    }
    if cfg.get("monitor", True):
        _monitor(res)
    return res


def _monitor(res):
    import sys
    if str(F.PHYS) not in sys.path:
        sys.path.insert(0, str(F.PHYS))
    from monitoring.logger import EventLogger
    d = F.DIRS["monitoring"] / "forecast"
    conn = d / "connection.json"
    if not conn.exists():
        d.mkdir(parents=True, exist_ok=True)
        conn.write_text(json.dumps({"api_url": None, "upload_enabled": False, "_comment": "HELIOLAB 사이트 API 없음 → LOCAL_ONLY"}, indent=1))
    lg = EventLogger(d)
    try:
        lg.emit("final_metrics", metrics=dict(t0=res["t0_utc"], final_speed_kms=res["final_speed_kms"], state=res["status"]["state"],
                                              upload_status=res["upload_status"], latency_total_s=res["latency_s"]["total"]),
                message=f"forecast {res['t0_utc']} mode={res['mode']} final={res['final_model']}", split="realtime",
                run_id=f"forecast-{res['t0_utc']}", experiment_id="SYZYGY_FINAL_CME_FREE", stage="inference")
    finally:
        lg.close()
    out = F.DIRS["predictions"] / "realtime"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"forecast_{pd.Timestamp(res['t0_utc']):%Y%m%dT%H%M}_{res['mode']}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True)
    ap.add_argument("--mode", default="archive", choices=["archive", "live"])
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    r = predict(args.issue, {"mode": args.mode, "allow_download": not args.no_download})
    print(json.dumps({k: r[k] for k in ("t0_utc", "final_model", "final_speed_kms", "g01_kms", "router_cf_kms", "status", "latency_s")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
