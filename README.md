# final_bundle — 최종 모델 C+HBC0.25 의 학습 파일 묶음 (2026-11-09)

`fc_predict.py` 가 읽는 학습 파일 (가중치 30 개 + 설정) 을 한 폴더에 **복사**한 것이다.
원본은 그대로 있고, `fc_predict.py` 는 여전히 원본 경로를 읽는다 (코드는 바꾸지 않음).
모든 복사본의 sha256 이 원본과 같음을 확인했고 (`MANIFEST.json`), 30 개 가중치 모두 프로젝트 로더로 이 폴더에서 바로 로드되며 원본과 텐서가 같다.

## 구성

| 폴더 | 파일 | 최종 속도에 사용 | 로더 |
|---|---|---|---|
| `weights/G01/` | `G01_s42…46.pt` | O | `collin_hss_reference/run_phase.load` (AblationNet) |
| `weights/RegimeNetCF/` | `RegimeNetCF_s42…46.pt` | O | `fc_model.load_regime` |
| `weights/RouterR2/` | `RouterR2_s42…46.pt` | O | `fc_model.load_router` (model.MoENet) |
| `weights/O_occupancy/` | `O_s42…46.pt` | O | `oc_predict.MultiHeadNet` (`torch.load` → `kwargs`, `model`) |
| `weights/loaded_not_in_final/P2o/` | `P2o_s42…46.pt` | X (출력만) | `run_phase.load` |
| `weights/loaded_not_in_final/T2/` | `T2_s42…46.pt` | X (onset 확률 출력) | `oc_predict.MultiHeadNet` |
| `configs/` | `final_selection.json`, `boxcox_maps_C.pkl` | O | `json`, `pickle` → `run_phase.apply_boxcox` |
| `configs/` | `boxcox_maps_A.pkl` | fallback 때만 | 〃 |
| `configs/` | `pipeline_params.json` | O | 추론 상수 모음 (아래) |
| `configs/` | `ensemble_meta_valid.json`, `preregistration.json` | X | 선정 근거·메타 |
| `provenance/` | 학습 meta, 원래 경로 기록 | X | 기록용 |

`MANIFEST.json` 에 파일마다 원래 경로, 크기, sha256, 역할이 있다.

## final_params.py (JSON·pkl 대신 쓰는 값 모음, 2026-11-09 추가)

Docker 처럼 따로 만드는 추론 코드에서 JSON·pkl 없이 쓰라고 만든 상수 파일이다 (P2o·T2 도 쓰는 기준).
`_build/make_final_params.py` 가 원본 파일에서 값을 자동으로 읽어 만들었다 (손으로 옮겨 적지 않음).

- 들어 있는 값: 조합 (0.5·G01 + 0.5·Router R2), Box–Cox α 0.25, horizon bin, 속도 정규화 430/100,
  f_s 표준화 상수, Box–Cox 사상 C·A 의 horizon 별 파라미터, T2 τ (seed 별·평균), 가중치 30 개 경로와 sha256, 계산식 (docstring)
- Box–Cox 는 `cr_boxcox.BoxCoxDistributionMap.from_params(BOXCOX_C[j]).transform(v)` 로 쓴다 (pkl 을 열 필요 없음, `cr_boxcox.py` 는 필요).
- 검증 (`_build/verify_final_params.py`, 모두 통과):
  1. 값이 원본 (final_selection.json, pkl, ch_pfss.h5 속성, model_preds_info.json, 파일 sha256) 과 정확히 같다.
  2. `from_params` Box–Cox 결과가 pkl 객체 결과와 비트 단위로 같다 (C·A, 150–1200 km/s 4,006 값).
  3. f_s 식 + 상수로 `ch_pfss.h5` 의 저장된 `feat_fs` 22,519 표본을 차이 0 으로 다시 만든다.
  4. `final_params.py` + 이 폴더 가중치만으로 계산한 예보가 `fc_predict.predict` (archive) 와 같다.
     valid t0 8 개, 최대 차이 0.005 km/s (출력 JSON 의 소수 2 자리 반올림), onset bin 일치 (사건 있는 1 개 포함).

## 계산 순서

1. seed 42–46 의 G01, RegimeNet-CF, Router R2, O 를 각각 돌린다.
2. Router 속도 (seed s) = denorm(anchor_s + Σ softmax(Router_s) · ΔV_s), anchor·ΔV 는 **같은 seed** 의 RegimeNet-CF 출력.
3. v = 0.5 · mean(G01) + 0.5 · mean(Router), horizon 24/48/72/96 h (16 bin 중 3, 7, 11, 15).
4. v_final = v + 0.25 · p_occ · (BoxCox_C(v) − v), p_occ = mean(sigmoid(O 의 occ_logits)).
5. 영상·PFSS 입력이 하나라도 없으면 v = mean(G01), 보정은 `boxcox_maps_A.pkl`.

## pipeline_params.json

새로 쓴 파일이며, 값은 원본에서 옮겨 적었다.
- 속도 정규화 (V − 430) / 100
- PFSS f_s 표준화 상수: `D:\syzygy\ch_pfss.h5` 의 `feat_fs` 속성. 이 값을 쓰면 live 추론에서 48 MB 파일을 열 필요가 없다.
- T2 τ 평균, 앙상블·보정 식

## 들어 있지 않은 것

- 코드: 전처리·모델 클래스는 `D:\syzygy\code` 와 각 실험 폴더에 있다 (`sw_utils.py`, `model.py`, `fc_model.py`, `fc_predict.py` 등).
- 학습 기록 (history csv), OOF fold 가중치, smoke 가중치.
