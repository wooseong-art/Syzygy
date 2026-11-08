"""
pfss_trace.py
-----------------------------------
PFSS 3차원 퍼텐셜 장과 장선 추적 (config 29절, numpy 만). pfss.py 의 구면조화 전개 (실수 직교정규 Y_lm, l ≤ PFSS_LMAX,
단극 제외) 를 그대로 쓰고, source surface (R_ss) 에서 퍼텐셜 0 인 해의 3 성분을 만든다 (R_sun = 1):

  Φ_lm(r) = c_l [r^-(l+1) − R_ss^-(2l+1) r^l] Y_lm,   c_l = a_lm / [(l+1) + l R_ss^-(2l+1)]     (B = −∇Φ)
  B_r = Σ a_lm R_l(r) Y_lm,  R_l(r) = [(l+1) r^-(l+2) + l R_ss^-(2l+1) r^(l−1)] / [(l+1) + l R_ss^-(2l+1)]   (R_l(1) = 1)
  B_θ = −Σ a_lm T_l(r) ∂_θ Y_lm,  B_φ = −Σ a_lm T_l(r) (1/sinθ) ∂_φ Y_lm,
        T_l(r) = [r^-(l+2) − R_ss^-(2l+1) r^(l−1)] / [(l+1) + l R_ss^-(2l+1)]                     (T_l(R_ss) = 0 → 순수 radial)
  R_l(R_ss) 은 pfss.source_surface_factor 와 같다 (기존 source surface 지도와 같은 해).

격자: ln r 균등 n_r (1 … R_ss) × 여위도 칸 중심 n_t × 경도 n_p (경도 0 = 계수의 기준 방향). 계수는 지구 기준 경도
(Stonyhurst, pfss.rotate_azimuth 로 L0 회전) 로 넣어 영상의 위경도와 같은 틀에서 추적한다.
장선: 데카르트 좌표에서 RK4, 발자국 극성 방향 (B_r > 0 이면 +B, < 0 이면 −B) 으로 호 길이 ds 씩. r ≥ R_ss → open,
r < 1 → closed, max_steps 넘김 → undetermined. |B| 는 격자 값을 삼선형 보간 (r 는 ln r 축).
반지름 표본 r_k 의 |B| 는 장선이 r_k 를 처음 지나는 구간에서 r 에 대해 선형 보간 (r = 1 은 발자국 값).
"""
from __future__ import annotations

import numpy as np

import config as C
from pfss import lm_index, real_sph_harm


class PFSS3D:
    def __init__(self, lmax: int = C.PFSS_LMAX, rss: float = C.PFSS_RSS, n_r: int = 61, n_t: int = 120, n_p: int = 240):
        self.lmax, self.rss, self.n_r, self.n_t, self.n_p = lmax, rss, n_r, n_t, n_p
        self.lnr = np.linspace(0.0, np.log(rss), n_r)
        r = np.exp(self.lnr)
        self.theta = (np.arange(n_t) + 0.5) * np.pi / n_t                     # 여위도 칸 중심
        self.phi = np.arange(n_p) * 2 * np.pi / n_p                            # 경도 (0 … 2π)
        TH, PH = np.meshgrid(self.theta, self.phi, indexing="ij")
        lm = lm_index(lmax)
        self.ls = np.array([l for l, _ in lm])
        y = real_sph_harm(lmax, TH.ravel(), PH.ravel())                        # [n_t·n_p, K]
        d = 1e-5
        dth = (real_sph_harm(lmax, TH.ravel() + d, PH.ravel()) - real_sph_harm(lmax, TH.ravel() - d, PH.ravel())) / (2 * d)
        idx = {k: i for i, k in enumerate(lm)}
        dph = np.zeros_like(y)                                                  # ∂φ Y_{l,m} = −m Y_{l,−m} (m > 0), |m| Y_{l,|m|} (m < 0)
        for (l, m), i in idx.items():
            if m > 0:
                dph[:, i] = -m * y[:, idx[(l, -m)]]
            elif m < 0:
                dph[:, i] = -m * y[:, idx[(l, -m)]]
        dph /= np.sin(TH.ravel())[:, None]
        shp = (n_t, n_p, -1)
        self.Y, self.Yt, self.Yp = (a.reshape(shp).astype(np.float32) for a in (y, dth, dph))
        den = np.array([(l + 1) + l * rss ** -(2 * l + 1) for l in range(lmax + 1)])
        L = np.arange(lmax + 1)[:, None]
        self.Rl = ((L + 1) * r[None] ** -(L + 2) + L * rss ** -(2.0 * L + 1) * r[None] ** (L - 1)) / den[:, None]   # [l, n_r]
        self.Tl = (r[None] ** -(L + 2) - rss ** -(2.0 * L + 1) * r[None] ** (L - 1)) / den[:, None]
        self.Rl[0], self.Tl[0] = 0.0, 0.0                                       # 단극 제외 (자속 균형, pfss.py 와 같음)

    def field_grid(self, coef: np.ndarray):
        """계수 [K] → B_r, B_θ, B_φ 격자 [n_r, n_t, n_p] (float32, Gauss)."""
        a = np.asarray(coef, np.float64).copy()
        a[self.ls == 0] = 0.0
        L = self.lmax + 1
        A = np.zeros((3, L, self.n_t, self.n_p), np.float32)
        for l in range(1, L):
            s = self.ls == l
            al = a[s].astype(np.float32)
            A[0, l] = self.Y[..., s] @ al
            A[1, l] = self.Yt[..., s] @ al
            A[2, l] = self.Yp[..., s] @ al
        Br = np.einsum("lr,ltp->rtp", self.Rl.astype(np.float32), A[0])
        Bt = -np.einsum("lr,ltp->rtp", self.Tl.astype(np.float32), A[1])
        Bp = -np.einsum("lr,ltp->rtp", self.Tl.astype(np.float32), A[2])
        return np.stack([Br, Bt, Bp])                                           # [3, n_r, n_t, n_p]

    def exact(self, coef, r, theta, phi):
        """구면조화 합으로 점마다 정확한 (B_r, B_θ, B_φ) (격자 보간 검증용)."""
        a = np.asarray(coef, np.float64).copy()
        a[self.ls == 0] = 0.0
        th, ph, r = (np.atleast_1d(np.asarray(v, np.float64)) for v in (theta, phi, r))
        y = real_sph_harm(self.lmax, th, ph)
        d = 1e-6
        yt = (real_sph_harm(self.lmax, th + d, ph) - real_sph_harm(self.lmax, th - d, ph)) / (2 * d)
        yp = (real_sph_harm(self.lmax, th, ph + d) - real_sph_harm(self.lmax, th, ph - d)) / (2 * d) / np.sin(th)[:, None]
        den = (self.ls + 1) + self.ls * self.rss ** -(2.0 * self.ls + 1)
        R = ((self.ls + 1) * r[:, None] ** -(self.ls + 2.0) + self.ls * self.rss ** -(2.0 * self.ls + 1)
             * r[:, None] ** (self.ls - 1.0)) / den
        T = (r[:, None] ** -(self.ls + 2.0) - self.rss ** -(2.0 * self.ls + 1) * r[:, None] ** (self.ls - 1.0)) / den
        R[:, self.ls == 0] = 0
        T[:, self.ls == 0] = 0
        return (y * R) @ a, -(yt * T) @ a, -(yp * T) @ a

    def _interp(self, G, r, th, ph):
        """격자 [3, n_r, n_t, n_p] 의 삼선형 보간 → [3, N] (r: ln r 축, θ: 칸 중심·극에서 고정, φ: 주기)."""
        fr = np.clip(np.log(np.clip(r, 1.0, self.rss)) / self.lnr[-1] * (self.n_r - 1), 0, self.n_r - 1 - 1e-9)
        ft = np.clip(th / np.pi * self.n_t - 0.5, 0, self.n_t - 1 - 1e-9)
        fp = np.mod(ph, 2 * np.pi) / (2 * np.pi) * self.n_p
        i0, j0, k0 = fr.astype(int), ft.astype(int), np.floor(fp).astype(int)
        wr, wt, wp = fr - i0, ft - j0, fp - k0
        k0 = np.mod(k0, self.n_p)
        k1 = np.mod(k0 + 1, self.n_p)
        out = 0.0
        for di, wi in ((0, 1 - wr), (1, wr)):
            for dj, wj in ((0, 1 - wt), (1, wt)):
                for kk, wk in ((k0, 1 - wp), (k1, wp)):
                    out = out + G[:, i0 + di, j0 + dj, kk] * (wi * wj * wk)
        return out

    def _bcart(self, G, x):
        r = np.linalg.norm(x, axis=0)
        th = np.arccos(np.clip(x[2] / r, -1, 1))
        ph = np.arctan2(x[1], x[0])
        br, bt, bp = self._interp(G, r, th, ph)
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        b = np.stack([br * st * cp + bt * ct * cp - bp * sp,
                      br * st * sp + bt * ct * sp + bp * cp,
                      br * ct - bt * st])
        return b, r

    def trace(self, G, lat_deg, lon_deg, radii=np.round(np.arange(1.0, 2.51, 0.1), 1), ds: float = 0.02,
              max_steps: int = 600, keep_path: bool = False):
        """발자국 (위도, 경도 deg, 계수와 같은 경도 틀) 에서 장선 추적.
        반환 dict: state [N] (1 open, 0 closed, −1 undetermined), babs [N, len(radii)] (open 만 채움, 나머지 NaN),
                  b_foot [N], b_end [N] (open 의 r = R_ss 값), end_lat/end_lon [N] (끝점), path (keep_path 면 점 목록)."""
        n = len(lat_deg)
        th = np.radians(90.0 - np.asarray(lat_deg, np.float64))
        ph = np.radians(np.asarray(lon_deg, np.float64))
        x = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])   # r = 1
        b0, _ = self._bcart(G, x)
        br0 = (b0 * x).sum(0)
        sgn = np.where(br0 >= 0, 1.0, -1.0)
        babs = np.full((n, len(radii)), np.nan)
        bmag0 = np.linalg.norm(b0, axis=0)
        babs[:, 0] = bmag0                                                       # r = 1.0
        xcross = np.full((n, len(radii), 3), np.nan)                            # r_k 를 지나는 위치 (검증용)
        xcross[:, 0] = x.T
        state = np.full(n, -1, np.int8)
        active = np.ones(n, bool)
        r_prev, b_prev = np.ones(n), bmag0.copy()
        end = x.copy()
        paths = [x.copy()] if keep_path else None

        def f(xx, idx):
            b, _ = self._bcart(G, xx)
            nb = np.linalg.norm(b, axis=0)
            return sgn[idx] * b / np.maximum(nb, 1e-12)
        for step in range(max_steps):
            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break
            xa = x[:, idx]
            k1 = f(xa, idx)
            k2 = f(xa + 0.5 * ds * k1, idx)
            k3 = f(xa + 0.5 * ds * k2, idx)
            k4 = f(xa + ds * k3, idx)
            xn = xa + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            bn, rn = self._bcart(G, xn)
            bm = np.linalg.norm(bn, axis=0)
            rp, bp_ = r_prev[idx], b_prev[idx]
            for kk in range(1, len(radii)):                                      # 처음 지나는 r_k 에서 r 선형 보간
                rk = radii[kk]
                hit = np.isnan(babs[idx, kk]) & (rp < rk) & (rn >= rk)
                if hit.any():
                    w = (rk - rp[hit]) / (rn[hit] - rp[hit])
                    babs[idx[hit], kk] = bp_[hit] + w * (bm[hit] - bp_[hit])
                    pc = xa[:, hit] + w * (xn[:, hit] - xa[:, hit])                # 그 위치 (검증용, r = r_k 로 맞춤)
                    xcross[idx[hit], kk] = (pc / np.linalg.norm(pc, axis=0) * rk).T
            x[:, idx] = xn
            r_prev[idx], b_prev[idx] = rn, bm
            opened = rn >= self.rss
            closed = rn < 1.0
            state[idx[opened]] = 1
            state[idx[closed & ~opened]] = 0
            done = opened | closed
            end[:, idx[done]] = xn[:, done]
            active[idx[done]] = False
            if keep_path:
                paths.append(x.copy())
        babs[state != 1] = np.nan
        end[:, active] = x[:, active]
        rr = np.linalg.norm(end, axis=0)
        out = dict(state=state, babs=babs, b_foot=bmag0, b_end=babs[:, -1], xcross=xcross,
                   end_lat=90 - np.degrees(np.arccos(np.clip(end[2] / rr, -1, 1))),
                   end_lon=np.degrees(np.arctan2(end[1], end[0])), sign=sgn)
        if keep_path:
            out["path"] = np.stack(paths)                                         # [steps+1, 3, N]
        return out


def exact_field(coef, r, theta, phi, lmax: int = C.PFSS_LMAX, rss: float = C.PFSS_RSS):
    """격자 없이 구면조화 합으로 (B_r, B_θ, B_φ) [N] — scipy sph_harm_y_all (도함수 포함) 한 번 호출.
    실수 기저: m > 0 → √2 Re Y_l^m, m < 0 → √2 Im Y_l^|m|, m = 0 → Re Y_l^0 (pfss.real_sph_harm 과 같음)."""
    from scipy.special import sph_harm_y_all
    th, ph, r = (np.atleast_1d(np.asarray(v, np.float64)) for v in (theta, phi, r))
    y, dy = sph_harm_y_all(lmax, lmax, th, ph, diff_n=1)                     # [l, m, N], [l, m, N, 2]
    a = np.asarray(coef, np.float64)
    Br = np.zeros_like(r)
    Bt = np.zeros_like(r)
    Bp = np.zeros_like(r)
    k = 0
    for l in range(lmax + 1):
        den = (l + 1) + l * rss ** -(2 * l + 1)
        R = ((l + 1) * r ** -(l + 2.0) + l * rss ** -(2.0 * l + 1) * r ** (l - 1.0)) / den
        T = (r ** -(l + 2.0) - rss ** -(2.0 * l + 1) * r ** (l - 1.0)) / den
        for m in range(-l, l + 1):
            if l == 0:                                                          # 단극 제외
                k += 1
                continue
            mm = abs(m)
            if m == 0:
                Y, Yt, Yp = y[l, 0].real, dy[l, 0, :, 0].real, dy[l, 0, :, 1].real
            elif m > 0:
                Y, Yt, Yp = (np.sqrt(2) * v.real for v in (y[l, mm], dy[l, mm, :, 0], dy[l, mm, :, 1]))
            else:
                Y, Yt, Yp = (np.sqrt(2) * v.imag for v in (y[l, mm], dy[l, mm, :, 0], dy[l, mm, :, 1]))
            Br += a[k] * R * Y
            Bt -= a[k] * T * Yt
            Bp -= a[k] * T * Yp / np.sin(th)
            k += 1
    return Br, Bt, Bp


def trace_exact(coef, lat_deg, lon_deg, ds: float = 0.005, max_steps: int = 3000, rss: float = C.PFSS_RSS, radii=None):
    """기준 (reference) 추적: 매 RK4 단계에서 격자 대신 exact_field. 발자국 |B| (r = 1) 와 r = R_ss 를 처음 지나는 점
    (두 step 사이 선형 보간 위치를 정확히 r = R_ss 로 맞춤) 의 exact |B| → f_s. 반환: state, b1, b_ss, end (xyz)."""
    n = len(lat_deg)
    th = np.radians(90.0 - np.asarray(lat_deg, np.float64))
    ph = np.radians(np.asarray(lon_deg, np.float64))
    x = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    def bcart(xx):
        rr = np.linalg.norm(xx, axis=0)
        t = np.arccos(np.clip(xx[2] / rr, -1, 1))
        p = np.arctan2(xx[1], xx[0])
        br, bt, bp = exact_field(coef, rr, t, p, rss=rss)
        st, ct, sp, cp = np.sin(t), np.cos(t), np.sin(p), np.cos(p)
        return np.stack([br * st * cp + bt * ct * cp - bp * sp, br * st * sp + bt * ct * sp + bp * cp, br * ct - bt * st]), rr
    b0, _ = bcart(x)
    sgn = np.where((b0 * x).sum(0) >= 0, 1.0, -1.0)
    b1 = np.linalg.norm(b0, axis=0)
    babs = np.full((n, len(radii)), np.nan) if radii is not None else None
    if babs is not None:
        babs[:, 0] = b1
    state = np.full(n, -1, np.int8)
    b_ss = np.full(n, np.nan)
    end = np.full((3, n), np.nan)
    active = np.ones(n, bool)
    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        xa = x[:, idx]
        f = lambda z: (lambda b: sgn[idx] * b / np.maximum(np.linalg.norm(b, axis=0), 1e-12))(bcart(z)[0])
        k1 = f(xa)
        k2 = f(xa + 0.5 * ds * k1)
        k3 = f(xa + 0.5 * ds * k2)
        k4 = f(xa + ds * k3)
        xn = xa + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        rp, rn = np.linalg.norm(xa, axis=0), np.linalg.norm(xn, axis=0)
        if radii is not None:                                                    # 반지름 r_k 처음 지나는 점의 정확한 |B|
            for kk in range(1, len(radii) - 1):
                hit = np.isnan(babs[idx, kk]) & (rp < radii[kk]) & (rn >= radii[kk])
                if hit.any():
                    w = (radii[kk] - rp[hit]) / (rn[hit] - rp[hit])
                    pc = xa[:, hit] + w * (xn[:, hit] - xa[:, hit])
                    bc, _ = bcart(pc / np.linalg.norm(pc, axis=0) * radii[kk])
                    babs[idx[hit], kk] = np.linalg.norm(bc, axis=0)
        opened, closed = rn >= rss, rn < 1.0
        if opened.any():
            w = (rss - rp[opened]) / (rn[opened] - rp[opened])
            pe = xa[:, opened] + w * (xn[:, opened] - xa[:, opened])
            pe = pe / np.linalg.norm(pe, axis=0) * rss                         # 정확히 r = R_ss
            be, _ = bcart(pe)
            b_ss[idx[opened]] = np.linalg.norm(be, axis=0)
            end[:, idx[opened]] = pe
            state[idx[opened]] = 1
        state[idx[closed & ~opened]] = 0
        x[:, idx] = xn
        active[idx[opened | closed]] = False
    if babs is not None:
        babs[:, -1] = b_ss
        babs[state != 1] = np.nan
    return dict(state=state, b1=b1, b_ss=b_ss, end=end, babs=babs)


def equal_area_seeds(dlon: float = 2.0):
    """경도 dlon° × sin(위도) 균등 (같은 면적) 격자 → (위도 deg, 경도 deg ∈ (−180, 180]). sin 간격 = dlon° 의 라디안."""
    ds_ = np.radians(dlon)
    n_s = int(round(2 / ds_))
    s = -1 + (np.arange(n_s) + 0.5) * 2 / n_s
    lon = -180 + (np.arange(int(round(360 / dlon))) + 0.5) * dlon
    S, Lo = np.meshgrid(s, lon, indexing="ij")
    return np.degrees(np.arcsin(S.ravel())), Lo.ravel(), 4 * np.pi / S.size      # 칸 하나의 입체각 (sr)
