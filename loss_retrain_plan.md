# 대회 지표(S) 기준 loss 재학습 제안

## 1. 요약

- 현재 모델은 학습 loss와 최종 선택이 **RMSE 중심**입니다. 대회 점수는 RMSE와 **빠른 바람에 크게 가중한 TRMSE**의 기하평균입니다.
- 백테스트에서 오차를 키우는 쪽은 TRMSE입니다(RMSE 54.8, TRMSE 94.8).
- 제안: **구조, 입력, 출력(4개 + KI 보간)은 그대로 두고 loss와 선택 기준만 바꿉니다.**
- MSE와 TRMSE 사이의 비율(α)은 손으로 정하지 않습니다. **학습 중에 S의 기울기에서 자동으로 계산**합니다(4.3절).
- **G01과 Router의 결합 비율(현재 0.5 고정)도 S 기준으로 다시 정합니다**(4.5절). 백테스트에서는 Router 비중을 높일수록 S가 좋아지는 경향이 보였습니다.
- 출력 형태 변경(H96 직접 출력)은 RMSE 기준 비교에서 KI와 동률이었으므로(valid 62.02 vs 62.15, test 63.41 vs 63.18) 이번에는 다루지 않습니다.

## 2. 배경: 대회 채점 방식

공식 채점 코드: starter kit `scoring_demo/swp_scoring/core/metric.py`

lead τ = 1…72h마다, 전체 예보 주기의 (관측, 예측) 쌍으로:

```
RMSE_τ  = √ mean( (v − v̂)² )
TIMSE_τ = 1/(v_max − v_min) · Σ_{k=0}^{99} (q_{k+1} − q_k) · M_k
          q_k = 관측 v의 k% 분위수 (numpy.quantile, linear),  M_k = 관측이 q_k 이상인 쌍들의 MSE
TRMSE_τ = √ TIMSE_τ
S_τ     = √( RMSE_τ · TRMSE_τ )
S_final = mean_τ S_τ            (72개 lead 동일 가중, 낮을수록 좋음)
```

- 정답: ACE 1시간 평균 속도, **status 0인 시간만**.
- 정렬: lead τ 예보는 구간 **[t0+τ, t0+τ+1)** 관측과 비교. 우리 제출본은 `hourly_speed_kms[1:73]`을 씁니다.
- 분위수 기준은 **관측값으로만** 정합니다. 빠른 바람을 느리게 예측(놓침)하면 크게 벌점을 받고, 느린 바람을 빠르게 예측(헛경보)하면 상대적으로 덜 받습니다.

### 백테스트 결과 (제출 이미지, 2026-10-16 ~ 11-12, 112주기, 공식 채점기)

| | S_final | RMSE | TRMSE |
|---|---|---|---|
| 제출본 (C+HBC0.25, KI) | **71.64** | 54.8 | 94.8 |
| 공식 recurrence 27d | 130.07 | 96.3 | 177.1 |
| 공식 persistence | 131.60 | 96.8 | 180.3 |
| 공식 climatology | 149.54 | 107.9 | 207.3 |

lead별 S: 1h 22.9 / 6h 42.4 / 12h 56.5 / 24h 71.1 / 48h 85.4 / 72h 86.6

## 3. 현재 학습 설정 (레포 확인 결과)

| 모델 | loss | 빠른 바람 가중 | checkpoint 선택 |
|---|---|---|---|
| G01 (`physical_factorial_ablation`, E001) | `rmse_loss` (train.py 179행) | 없음 | validation RMSE |
| RegimeNetCF | 고속 가중 MSE + 0.3·class-weighted CE + 0.5·상태 ΔV MSE | 관측 500→700 km/s에서 1→2배 (`config.py` `HS_V0/HS_V1/HS_WEIGHT`) | validation 16 bin RMSE |
| Router R2 | class-weighted CE (power 0.5) | 상태 빈도 기준 | HSS 16 bin RMSE (Slow +2 km/s 조건) |
| O | masked BCE | — | — |
| 결합 비율 w = 0.5, 보정 강도 0.25 | — | — | validation RMSE |

대회 TRMSE와 비교하면 빠른 바람 가중이 훨씬 약하고(최대 2배 vs 상위 1%는 10배 이상), 최종 선택은 RMSE 기준입니다.

## 4. 제안 loss

### 4.1 TIMSE는 샘플별 가중 MSE로 바꿔 쓸 수 있음

분위수 q_k를 **train 관측으로 미리 고정**하면:

```
TIMSE ≈ mean_i( w(v_i) · (v_i − v̂_i)² )
w(v)  = 1/(q_K − q_0) · Σ_{k : q_k ≤ v} (q_{k+1} − q_k) · K/(K − k)        (K = 100)
```

- 가중치는 **관측 속도만의 함수**라 표로 미리 계산합니다. 미분 문제가 없습니다.
- train 분포에서 mean(w) = 1입니다.
- 기준선을 같은 표본으로 만들면 공식 `timse_percentile`과 값이 **정확히 같습니다**(합성 자료로 확인, n = 500 / 5000).
- 맨 위 가중치는 매우 큽니다(합성 자료 기준 최대 37~52, 99% 분위수 9~12). **상한 클리핑이 필요합니다.**

```python
import numpy as np, torch

def make_w_table(v_train_kms, K=100, w_cap_q=0.99):
    """train 관측(km/s, 결측 제외)으로 TRMSE 가중치 표를 만든다."""
    q = np.quantile(v_train_kms, np.arange(K + 1) / K)          # 공식과 같은 linear
    span = q[-1] - q[0]
    cum = np.cumsum(np.diff(q) * K / (K - np.arange(K)) / span)  # 기준선 k 이상에서 누적 가중
    w_train = cum[np.clip(np.searchsorted(q[:-1], v_train_kms, side="right") - 1, 0, K - 1)]
    cap = np.quantile(w_train, w_cap_q)                          # 맨 위 가중치 상한
    q_t = torch.tensor(q[:-1], dtype=torch.float32)
    cum_t = torch.tensor(np.minimum(cum, cap), dtype=torch.float32)
    def w(v_kms):                                                # torch 텐서 → 가중치
        i = torch.searchsorted(q_t, v_kms.contiguous(), right=True) - 1
        return cum_t[i.clamp(0, K - 1)]
    return w
```

### 4.2 α를 학습 파라미터로 두면 안 되는 이유

`loss = (1−α)·MSE + α·TMSE`에서 α를 `nn.Parameter`로 두고 같이 학습하면, optimizer는 **loss가 작은 쪽 항으로 α를 몰아갑니다.** 항상 TMSE > MSE이므로 α → 0이 되고, 결국 기존 RMSE 학습으로 돌아갑니다. α는 "loss를 줄이는 값"이 아니라 "**대회 점수 S를 줄이는 값**"이어야 하므로 다른 방법으로 정해야 합니다.

### 4.3 권장: S의 기울기에서 α를 자동 계산 (adaptive α)

S_final의 기울기를 풀어 쓰면:

```
S_τ = (MSE_τ · TMSE_τ)^¼
∇S_final = mean_τ ∇S_τ = mean_τ [ a_τ · ∇MSE_τ + b_τ · ∇TMSE_τ ]
    a_τ = S_τ / (4 · MSE_τ),     b_τ = S_τ / (4 · TMSE_τ)
```

- 즉 S를 최소화하는 방향은 **horizon별로 계수 a_τ, b_τ를 붙인 MSE + 가중 MSE**와 같습니다.
- 이때 α_τ = b_τ / (a_τ + b_τ) = MSE_τ / (MSE_τ + TMSE_τ)입니다. 현재 오차 수준에서는 α ≈ 0.25입니다(MSE/TMSE ≈ (54.8/94.8)² ≈ 0.33).
- **계수는 학습이 진행되면서 자동으로 바뀝니다.** 빠른 바람 오차가 줄어 TMSE가 작아지면 b_τ가 커지는 식으로, 그때그때 S를 가장 많이 줄이는 비율로 맞춰집니다.
- MSE_τ와 TMSE_τ는 미니배치 값이 아니라 **지수이동평균(EMA)으로 추정한 데이터셋 수준 값**을 씁니다. 그래서 작은 배치에서도 S loss를 직접 쓰는 것보다 안정적입니다.
- horizon별 가중도 같이 맞춰집니다. 대회처럼 각 lead의 S를 동일 가중으로 평균하는 효과가 납니다.

```python
class AdaptiveSLoss(torch.nn.Module):
    """대회 S_final 의 기울기를 따르는 loss. a_τ, b_τ 는 EMA 통계로 매 step 갱신 (기울기 없음).
    gamma = 1 이면 S 그대로. gamma > 1 은 TRMSE 쪽을 더 강조 (검증용 탐색 변수)."""

    def __init__(self, w_fn, n_horizon, ema=0.99, gamma=1.0, v_mean=430.0, v_std=100.0):
        super().__init__()
        self.w_fn, self.ema, self.gamma = w_fn, ema, gamma
        self.v_mean, self.v_std = v_mean, v_std
        self.register_buffer("mse", torch.ones(n_horizon))
        self.register_buffer("tmse", torch.ones(n_horizon))
        self.register_buffer("started", torch.zeros((), dtype=torch.bool))

    def forward(self, pred, y):
        """pred, y: 정규화 단위 [B, H]. y 결측은 NaN. 기존 rmse_loss 자리에 그대로 넣는다."""
        m = torch.isfinite(y)
        yk = torch.nan_to_num(y) * self.v_std + self.v_mean     # 가중치는 km/s 관측으로
        e2 = torch.where(m, (pred - torch.nan_to_num(y)) ** 2, torch.zeros_like(pred))
        w = torch.where(m, self.w_fn(yk), torch.zeros_like(pred))
        n = m.sum(0).clamp(min=1)
        mse_b, tmse_b = e2.sum(0) / n, (w * e2).sum(0) / n       # 배치의 horizon별 값

        if self.training:                                        # EMA 갱신 (기울기 없음)
            with torch.no_grad():
                if not bool(self.started):
                    self.mse.copy_(mse_b.clamp(min=1e-6)); self.tmse.copy_(tmse_b.clamp(min=1e-6))
                    self.started.fill_(True)
                else:
                    self.mse.mul_(self.ema).add_((1 - self.ema) * mse_b)
                    self.tmse.mul_(self.ema).add_((1 - self.ema) * tmse_b)

        with torch.no_grad():
            s = (self.mse * self.tmse).clamp(min=1e-12) ** 0.25
            a = s / (4 * self.mse)
            b = self.gamma * s / (4 * self.tmse)
            scale = (a + b).mean()                               # 학습률 크기를 기존과 비슷하게 유지
        return ((a * mse_b + b * tmse_b).sum() / scale) / pred.shape[1]

    def alpha(self):                                             # 기록용: horizon별 실효 α
        with torch.no_grad():
            a = 1 / self.mse; b = self.gamma / self.tmse
            return (b / (a + b)).cpu().numpy()
```

- `n_horizon`은 출력 개수입니다. G01은 4(24/48/72/96h), RegimeNetCF는 16(6h bin)입니다.
- **epoch마다 `alpha()`를 로그로 남겨 주세요.** 학습 중 α가 어떻게 움직였는지 확인용입니다.
- 학습 초반에는 오차가 커서 계수가 불안정할 수 있습니다. 처음 1~2 epoch는 기존 `rmse_loss`로 warmup 후 전환하거나, 기존 checkpoint에서 fine-tuning으로 시작해도 됩니다.

### 4.4 남는 하이퍼파라미터는 validation S로 탐색

α는 자동으로 정해지고, 사람이 고를 값은 두 개만 남습니다.

| 변수 | 후보 | 의미 |
|---|---|---|
| `gamma` | 0.5 / 1 / 2 | 1 = S 그대로. train과 live의 속도 분포 차이를 보정할 여지 |
| `w_cap_q` | 0.98 / 0.99 / 0.995 | 맨 위 가중치 상한 |

3 × 3 = 9개 조합을 seed 42 하나로 먼저 돌려 validation S가 가장 낮은 조합을 고르고, 그 조합만 seed 5개로 학습합니다. 조합이 더 많아지면 Optuna 같은 탐색 도구를 써도 됩니다(목적 함수 = validation S_final).

**비교 기준선:** 같은 조건에서 고정 α 0.25(`(1−α)·MSE + α·TMSE`)도 한 번 돌려 주세요. adaptive α가 고정값보다 나은지 확인하기 위해서입니다.

### 4.5 G01 + Router 결합 비율 학습

현재 최종 모델은 base **C** = `0.5·G01 + 0.5·Router`(고정)입니다. 사전등록에는 학습형 base **D** = `(1 − w)·G01 + w·Router`(train OOF RMSE 최소, w = 0.75)도 있었지만, 제약(Slow RMSE ≤ G01 + 2, V1 FAR ≤ G01 + 0.05)을 넘지 못해 C가 선택됐습니다(`final_selection.json` `log.step1`: D 전체 RMSE 63.46, C 64.10).

**백테스트 참고 (2026-10-16 ~ 11-12, 112주기, 공식 채점기)**: 결합 비율만 바꾸고 나머지(보정량, KI, `[1:73]`)는 실행 결과 그대로 둔 근사 계산입니다. w = 0.5에서 제출본을 0.013 km/s 안으로 재현합니다.

| w_router | 0.0 | 0.3 | 0.5 (현재) | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| S_final | 77.32 | 73.40 | 71.63 | 70.57 | 70.31 | **70.23** | 70.33 |
| RMSE | 57.52 | 55.33 | 54.77 | 54.99 | 55.37 | 55.95 | 56.69 |
| TRMSE | 104.85 | 98.46 | 94.81 | 91.73 | 90.43 | 89.29 | 88.33 |
| bias | +1.3 | +3.4 | +4.8 | +6.1 | +6.8 | +7.5 | +8.2 |

- Router 비중을 높이면 **TRMSE가 꾸준히 줄고**, RMSE는 0.5~0.7까지 거의 같다가 그 뒤로 나빠집니다. S 기준 최적은 0.8~0.9 근처입니다.
- 방향은 팀원의 D(w = 0.75, RMSE 기준)와 같습니다. 대회 지표에서는 Router 쪽이 빠른 바람을 더 잘 잡는다는 뜻으로 보입니다.
- 112주기는 짧고 보정량을 고정한 근사라, **선택 근거가 아니라 방향 확인용**입니다.

**학습 방법 (권장 순서)**

1. **단일 w**: w ∈ {0, 0.05, …, 1}, 목적 = train OOF의 S(대회 형식). validation S로 확인. 보정 강도 α_corr ∈ {0, 0.25, 0.5, 1}과 **같이** 탐색합니다(보정은 결합 뒤에 적용되므로 따로 고르면 최적이 어긋남).
2. **horizon별 w_h (4개)**: 같은 방식으로 horizon마다. 먼 horizon일수록 Router 비중이 커질 가능성이 있습니다. 파라미터가 4개뿐이라 과적합 위험은 작습니다.
3. **입력 의존 gate (선택)**: `w(x) = sigmoid(a_h + b·p_occ + c·P_HSS(Router) + d·(v_now − 430)/100)`. train OOF로 `AdaptiveSLoss`를 써서 적합, validation S로 확인. 빠른 바람 신호가 강할 때만 Router 쪽으로 기우는 형태입니다. 1·2단계보다 확실히 좋을 때만 채택합니다.

- 구현 위치: 기존 D 탐색은 `fc_analysis.py`에 있습니다. 목적 함수만 RMSE → S(6절 형식)로 바꾸고, 보정 강도 탐색을 같은 반복 안에 넣으면 1·2번이 됩니다.
- 3번 gate 적합 예시 (결합 뒤 보정은 고정해 두고 gate만 학습):

```python
gate = torch.nn.Linear(3, 4)                                   # 입력 3개 → horizon 4개의 로짓
torch.nn.init.zeros_(gate.weight); torch.nn.init.constant_(gate.bias, 0.0)   # 시작점 w = 0.5 (현재 C)
lossf = AdaptiveSLoss(w_fn, n_horizon=4)                       # 4.3절 클래스 그대로
opt = torch.optim.Adam(gate.parameters(), 1e-2)
# x: [N, 3] = [p_occ 평균, Router P_HSS 평균, (v_now − 430)/100],  g, r, y: [N, 4] (km/s, y 결측 NaN)
for step in range(2000):
    i = torch.randint(0, len(x), (512,))
    w = torch.sigmoid(gate(x[i]))
    v = (1 - w) * g[i] + w * r[i]
    loss = lossf((v - 430) / 100, (y[i] - 430) / 100)         # AdaptiveSLoss 는 정규화 단위를 받음
    opt.zero_grad(); loss.backward(); opt.step()
```

- 적합 자료: Router OOF는 seed 42–44만 있으므로(사전등록 `image_oof_seeds`) G01 OOF도 같은 seed 평균으로 맞춥니다.
- 기존 제약(Slow RMSE, V1 FAR)은 계속 보고하되, 채택 기준은 validation S입니다. 제약을 유지할지는 팀에서 결정해 주세요.
- 5절의 재학습 단계(1 G01, 2 RegimeNetCF)를 거친 뒤에는 새 예측으로 **다시** 적합합니다(5절 4단계).

## 5. 실험 계획

출력은 **4 horizon(24/48/72/96h 6시간 평균) + KI 보간 그대로**입니다. 가중치 표는 4개 horizon 정답(train, 6h 평균)을 모두 모아 하나로 만듭니다.

| 단계 | 내용 | 비용 | 진행 조건 |
|---|---|---|---|
| **0** | 재학습 없이 결합 비율(4.5절 1→2→3)과 보정 강도(현재 0.25)를 **같이** S로 다시 선택 | 분 단위 | — |
| **1** | **G01**을 `AdaptiveSLoss`로 재학습. 4.4절 탐색(seed 42) → 최적 조합 seed 42–46. 고정 α 0.25 비교 1회. checkpoint = validation S 최소 | CPU 가능 | — |
| **2** | **RegimeNetCF** 속도 항(고속 가중 MSE)을 `AdaptiveSLoss`(`n_horizon=16`)로 교체: `code/05_train.py` `regime_loss`(807행)의 `loss_v = weighted_mse(out["v"], y, w)`. CE 0.3, ΔV 0.5(`loss_exp`, 같은 고속 가중 `w` 사용)는 처음에는 유지. gamma·w_cap은 1단계 최적값 사용 | GPU | 1단계에서 validation S 개선 |
| **3** | Router: loss는 그대로, checkpoint만 validation S 기준으로 다시 선택 (새 RegimeNetCF head 기준) | GPU 또는 재선택만 | 2단계 진행 시 |
| **4** | BoxCox 맵(A/C)을 새 OOF 예측으로 다시 적합 → 결합 비율·보정 강도를 validation S로 재선택 | 분 단위 | 1~3단계 후 항상 |

## 6. 평가·선택 규칙 (가장 중요)

1. **모든 선택(checkpoint, gamma, w_cap, 결합 비율, 보정 강도)은 validation S_final**로 합니다. RMSE 기준 선택은 쓰지 않습니다.
2. S_final은 **대회와 같은 형식**으로 계산합니다.
   - 모델 4개 출력 → KI → `hourly_speed_kms[1:73]` (lead 1…72)
   - 정답: ACE 1시간 값, lead τ ↔ 구간 [t0+τ, t0+τ+1). status 정보가 있으면 status 0만
   - t0: 00/06/12/18 UTC
   - 지표: 공식 `metric.py`의 `score()`를 그대로 import (분위수 기준은 validation 관측으로 계산되는 것이 정상. 대회도 평가 기간 관측으로 정함)
3. 가중치 표 `w(v)`는 **train 관측으로만** 만듭니다. validation/test 관측을 쓰면 누수입니다.
4. 기존 규칙 유지: seed 5개 평균 앙상블, 27일 블록 bootstrap, test는 선택 고정(SHA-256 기록) 뒤 한 번만.
5. RMSE, TRMSE, bias도 함께 기록합니다. 빠른 쪽으로 기우는 것이 정상이지만 bias가 크게 커지면(예: +15 km/s 이상) gamma를 낮춥니다.

## 7. 주의할 점

- **학습 loss는 4 horizon 6시간 평균에 대한 S이고, 선택 기준은 KI 보간 후 1시간 72개에 대한 S입니다.** 둘은 같지 않으므로, 학습 loss가 줄었다고 끝내지 말고 반드시 6절 형식으로 확인합니다.
- **EMA 계수:** 0.99 기본(약 100 step 평균). 한 epoch의 step 수가 적으면 0.95 정도로 낮춥니다.
- **분포 차이:** 대회 기준선은 live 기간(2026-11 ~ 2027-02) 관측으로 정해집니다. 학습 기간 전체로 만든 표면 큰 문제는 없지만, 결과는 기간에 따라 달라질 수 있습니다.
- **G01은 fallback에도 쓰입니다.** G01을 바꾸면 영상·PFSS가 없을 때의 예측도 바뀝니다. fallback 경로(G01 + BoxCox_A)의 validation S도 따로 보고해 주세요.

## 8. 결과로 받고 싶은 것

1. 표: 단계별 validation / test의 **S_final, RMSE, TRMSE, bias** (lead별 S 곡선 포함하면 더 좋음). 1단계는 adaptive α와 고정 α 0.25 비교 포함
2. 학습 중 horizon별 α 변화 로그
3. 결합 비율 결과: 단일 w 곡선(w별 S·RMSE·TRMSE), horizon별 w_h, gate를 시도했다면 그 결과와 계수
4. 최종 선택값: gamma, w_cap, 결합 비율, 보정 강도, seed별 checkpoint 경로
5. 새 가중치와 설정 파일이 들어간 커밋. 파일 경로가 바뀌면 알려 주세요(Docker 쪽 `tools/sync_model.py`가 `final_selection.json`, `preds_info.json`, router meta에서 경로를 찾습니다). horizon별 w나 gate를 채택하면 `fc_predict.py`의 결합 부분(base 선택, 504행 근처)도 바뀌어야 합니다.
6. 단계 0 결과만이라도 먼저 공유해 주면 효과 크기를 빨리 가늠할 수 있습니다.
