"""
fc_model.py — CME-free RegimeNet (RegimeNetCF) 과 CME-free 데이터 묶음.

RegimeNetCF = model.RegimeNet 에서 CME 입력만 뺀 것 (Adapted from D:\\syzygy\\code\\model.py RegimeNet, 같은 층·같은 폭):
  images [B, 10, 128, 128] (AIA193·HMI × 5 시점) + 좌표 2 → ImageEncoder (B0 ResCNN) → z_img 128
  vec    = ACE 직전 24 h 값 24 + mask 24 (= persist 48) → Linear 128 GELU Dropout → Linear 128 GELU      (기존: [cme_vec 21 | persist 48])
  ch     = CH 특징 118 → 64
  trunk  : bin 마다 [z_img | z_vec | z_ch | lead 16 | recur 2] → 128 GELU Dropout → 64 GELU               (기존: + cme_bin 3)
  heads  : regime 3 (Slow / Transition / HSS), ΔV 3   →   V_k = anchor + Σ_c softmax_c · ΔV_k,c
forward 인자는 데이터 묶음의 위치 (images, cme_bin, cme_vec, persist, recur, ch_vec) 를 맞추려고 CME 자리를 받지만 읽지 않는다.
그 자리에는 CFGPUData 가 0 텐서만 넣는다 (원본 DONKI 배열은 로드 직후 메모리에서 지움). 불변성 시험: fc_checks.py.
"""
import numpy as np
import torch
import torch.nn as nn

import fc_common as F  # noqa: F401  (sys.path)
import config as C
from model import ImageEncoder, RegimeNet, _ContiguousGrad


class RegimeNetCF(nn.Module):
    def __init__(self, ch_dim: int, widths=C.CNN_WIDTHS, pool_ch: int = C.CNN_POOL_CH, d_img: int = C.IMG_EMBED,
                 d_vec: int = C.TAB_EMBED, d_lead: int = C.LEAD_EMBED, d_ch: int = C.CH_EMBED, dropout: float = C.DROPOUT,
                 sdrop: float = C.SPATIAL_DROPOUT, lags=C.HISTORY_LAGS_HOURS, n_regime: int = len(C.REGIMES)):
        super().__init__()
        self.kwargs = dict(ch_dim=ch_dim, widths=list(widths), pool_ch=pool_ch, d_img=d_img, d_vec=d_vec, d_lead=d_lead,
                           d_ch=d_ch, dropout=dropout, sdrop=sdrop, lags=list(lags), n_regime=n_regime)
        self.encoder = ImageEncoder(in_ch=len(lags) * len(C.IMG_CHANNELS), widths=widths, pool_ch=pool_ch, d=d_img,
                                    dropout=dropout, sdrop=sdrop)
        self.vec = nn.Sequential(nn.Linear(2 * C.PERSIST_HOURS, d_vec), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_vec, d_vec), nn.GELU())
        self.lead = nn.Parameter(0.1 * torch.randn(C.N_BINS, d_lead))
        self.ch = nn.Sequential(nn.Linear(ch_dim, d_ch), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ch, d_ch), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(d_img + d_vec + d_ch + d_lead + 2, 128), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(128, 64), nn.GELU())
        self.regime_head = nn.Linear(64, n_regime)
        self.dv_head = nn.Linear(64, n_regime)

    anchor = staticmethod(RegimeNet.anchor)

    def forward(self, images, cme_bin_unused=None, cme_vec_unused=None, persist=None, recur=None, ch_vec=None):
        b = images.shape[0]
        z_img = _ContiguousGrad.apply(self.encoder(images))
        z_vec = _ContiguousGrad.apply(self.vec(persist))
        z_ch = _ContiguousGrad.apply(self.ch(ch_vec))
        ctx = torch.cat([z_img, z_vec, z_ch], dim=1)[:, None].expand(-1, C.N_BINS, -1)
        lead = self.lead[None].expand(b, -1, -1)
        h = torch.cat([ctx, lead, recur], dim=2)
        h = self.trunk(h.reshape(-1, h.shape[-1]))
        logits = self.regime_head(h).view(b, C.N_BINS, -1)
        dv = self.dv_head(h).view(b, C.N_BINS, -1)
        anchor = self.anchor(persist, recur)
        v = anchor + (torch.softmax(logits, dim=-1) * dv).sum(-1)
        return {"v": v, "logits": logits, "dv": dv, "anchor": anchor}


def save_regime(path, model, extra: dict):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "model_kwargs": model.kwargs, "model_type": "regime_cf", **extra}, path)


def load_regime(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["model_type"] == "regime_cf", ck.get("model_type")
    m = RegimeNetCF(**ck["model_kwargs"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def load_router(path, device="cpu"):
    from model import MoENet
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = MoENet(**ck["model_kwargs"])
    m.load_state_dict(ck["state_dict"])
    assert not m.use_cme, "Router 가 CME 를 씀"
    return m.to(device).eval(), ck


def strip_cme(data):
    """데이터 묶음 (SWData / GPUData) 의 DONKI CME 배열을 0 으로 바꾸고 원본을 지운다. → 지운 원본의 요약 (기록용)."""
    import hashlib
    info = {}
    for name in ("cme_bin", "cme_vec"):
        a = getattr(data, name)
        info[name] = dict(shape=list(a.shape), nonzero_frac=float((a != 0).mean()),
                          sha16=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16])
        setattr(data, name, np.zeros(a.shape, np.float32))
        g = "g_" + name
        if hasattr(data, g):
            old = getattr(data, g)
            setattr(data, g, torch.zeros(tuple(old.shape), dtype=old.dtype, device=old.device))
            del old
    data.cme_free = True
    return info
