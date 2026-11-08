"""
pfss.py
-----------------------------------
PFSS (potential field source surface) — 구면조화 전개로 source surface 의 radial field 를 계산 (numpy / scipy 만 사용).

  광구 B_r(θ, φ) = Σ_lm a_lm Y_lm  (실수 직교정규 구면조화, l ≤ LMAX, 단극 a_00 = 0 으로 자속 균형)
  퍼텐셜 장 + r = R_ss 에서 퍼텐셜 0 (source surface) 조건이면 l 성분의 radial field 가
      B_r,lm(R_ss) = a_lm · (2l+1) R_ss^-(l+2) / (l + 1 + l R_ss^-(2l+1))          (R_sun = 1)
  이 되므로, 광구 계수에 이 비를 곱해 source surface 지도를 다시 합성한다.

  입력 격자: Carrington 경도 2° × sin(위도) 90칸 (같은 면적), 출력 격자: 지구 쪽 경도 (Stonyhurst, 0 = 지구 방향)
  128칸 × 위도 64칸. 경도 회전 (Carrington → 지구 기준) 은 계수의 방위각 회전으로 정확히 한다.
"""
from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y

import config as C


def lm_index(lmax: int):
    """실수 구면조화 순서 (l, m), m = -l … l."""
    return [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]


def real_sph_harm(lmax: int, colat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """[N] 개 점 (여위도·경도, rad) → 실수 직교정규 구면조화 [N, (lmax+1)²].
    m > 0: √2 Re Y_l^m (cos mφ 성분), m < 0: √2 Im Y_l^|m| (sin |m|φ 성분), m = 0: Y_l^0."""
    out = np.empty((len(colat), (lmax + 1) ** 2))
    k = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), colat, lon)
            out[:, k] = y.real if m == 0 else np.sqrt(2) * (y.real if m > 0 else y.imag)
            k += 1
    return out


def source_surface_factor(lmax: int, rss: float) -> np.ndarray:
    """l 별 B_r(R_ss) / B_r(R_sun) 비 [ (lmax+1)² ] (단극 l = 0 은 0 → 자속 균형)."""
    g = []
    for l, m in lm_index(lmax):
        g.append(0.0 if l == 0 else (2 * l + 1) * rss ** -(l + 2) / (l + 1 + l * rss ** -(2 * l + 1)))
    return np.asarray(g)


def rotate_azimuth(coef: np.ndarray, lmax: int, angle_rad: float) -> np.ndarray:
    """f(φ) = Σ a Y(φ) 를 f'(φ_s) = f(φ_s + angle) 로 바꾼 계수 (Carrington 경도 = 지구 기준 경도 + L0)."""
    out = coef.copy()
    idx = {lm: i for i, lm in enumerate(lm_index(lmax))}
    for l in range(1, lmax + 1):
        for m in range(1, l + 1):
            ic, is_ = idx[(l, m)], idx[(l, -m)]
            c, s = coef[..., ic], coef[..., is_]
            ca, sa = np.cos(m * angle_rad), np.sin(m * angle_rad)
            out[..., ic] = c * ca + s * sa
            out[..., is_] = -c * sa + s * ca
    return out


class PFSSGrid:
    """고정 격자·고정 설정 (LMAX, R_ss) 의 PFSS 계산기. 행렬은 한 번만 만든다."""

    def __init__(self, lmax: int = C.PFSS_LMAX, rss: float = C.PFSS_RSS,
                 n_sinlat: int = C.PFSS_IN_NSINLAT, n_lon: int = C.PFSS_IN_NLON,
                 out_shape=(C.PFSS_OUT_NLAT, C.PFSS_OUT_NLON)):
        self.lmax, self.rss = lmax, rss
        s = -1 + (np.arange(n_sinlat) + 0.5) * 2 / n_sinlat                  # 칸 중심 sin(위도)
        phi = (np.arange(n_lon) + 0.5) * 2 * np.pi / n_lon                   # 칸 중심 Carrington 경도
        S, P = np.meshgrid(s, phi, indexing="ij")
        self.in_lat = np.degrees(np.arcsin(s))
        self.in_lon = np.degrees(phi)
        y_in = real_sph_harm(lmax, np.arccos(S.ravel()), P.ravel())          # [n_sinlat*n_lon, K]
        self.pinv = np.linalg.pinv(y_in)                                      # 최소제곱 전개 [K, n_in]
        nlat, nlon = out_shape
        lat = -90 + (np.arange(nlat) + 0.5) * 180 / nlat
        lon = -180 + (np.arange(nlon) + 0.5) * 360 / nlon                     # 0 = 지구 방향 (Stonyhurst)
        La, Lo = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
        self.out_lat, self.out_lon = lat, lon
        self.y_out = real_sph_harm(lmax, np.pi / 2 - La.ravel(), Lo.ravel())  # [nlat*nlon, K]
        self.g = source_surface_factor(lmax, rss)
        self.out_shape = out_shape

    def coefficients(self, br_phot: np.ndarray) -> np.ndarray:
        """광구 B_r 지도 [n_sinlat, n_lon] (Gauss, 결측 없음) → 실수 구면조화 계수 [K]."""
        return self.pinv @ br_phot.ravel()

    def source_surface(self, br_phot: np.ndarray, l0_deg: float) -> np.ndarray:
        """광구 B_r (Carrington) → source surface B_r 지도 [nlat, nlon] (Gauss), 경도 0 = 그 시각 지구 방향."""
        a = self.coefficients(br_phot) * self.g
        a = rotate_azimuth(a, self.lmax, np.radians(l0_deg))
        return (self.y_out @ a).reshape(self.out_shape)
