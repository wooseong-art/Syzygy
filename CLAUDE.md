# Benchlab Solar Wind Prediction — Docker 제출 작업 브리프

> Claude Code 세션 컨텍스트용. 2026-11-08 기준 정리.
> 출처: starter-kit 레포(`wooseong-art/Syzygy`, branch `starter-kit`), Slack `#benchlab-support-and-q`, 주최측 공지 PDF(2026-11-03).

## 0. 목표와 마감

- 이미 학습된 태양풍 속도 예측 모델을 **Benchlab 제출용 Docker 이미지**로 패키징한다.
- 트랙: **Operational Podium (100% in-container)** — 상금($20k) 대상. 외부 연산 금지.
- **코드 프리즈: 2026-11-19 23:59:59 UTC (= KST 11-20 08:59)**. 그 전에 실제 제출이 dry run을 통과해 `ELIGIBLE` 상태여야 함.
- 재제출하면 이전 ELIGIBLE 버전이 대체됨 → 새 버전이 실패하면 자격을 잃을 수 있음. 마감 직전 무리한 변경 금지.

## 1. 실행 계약 (플랫폼이 강제)

| 항목 | 내용 |
|---|---|
| 실행 | `<ENTRYPOINT> <t0>` — t0는 argv[1], 예: `20261111T060000Z`. `$INFERENCE_TIMESTAMP`로도 제공 |
| 주기 | 6시간마다 **새 컨테이너**로 실행 (stateless, 캐시/이전 파일 없음) |
| 출력 | t0+1h … t0+72h, **정확히 72개** 시간별 속도(km/s) |
| 출력 형식 | JSON, ≤ 1 MiB. 플랫폼은 `forecast_speed_kms`만 읽음 |
| 업로드 | `$OUTPUT_POST`(presigned S3 POST JSON)로 `upload.sh`가 전송 |
| 시간 제한 | **15분, 이미지 pull 포함** (pull ≈ 1 GB당 10초 추정) |
| 자원 | `submission.json` 선언값. 최대 8 vCPU / 32 GB / 디스크 ~20 GB, **CPU only (GPU 없음)** |
| 네트워크 | 공개 인터넷 사용 가능, 실행 1회당 20 GB fair-use |
| 아키텍처 | `linux/amd64` |
| 이미지 크기 | < 15 GiB (작을수록 좋음) |
| 사용자 | non-root `runner`(uid 1000). 쓰기는 `/tmp` 사용 |
| 성공 조건 | **exit 0 AND 유효한 forecast 업로드** — 둘 다 충족해야 함 |

출력 예시:
```json
{
  "t0": "2026-11-11T06:00:00Z",
  "valid_from": "2026-11-11T07:00:00Z",
  "forecast_speed_kms": [418.2, 419.9, "... 72 values ..."]
}
```
- 로컬 테스트는 값 범위 0–10000, NaN/null 없음을 검사함. 반드시 유한한 float 72개를 출력할 것.

## 2. 제출 패키지 구조 (`submission_pack/` 기반)

ZIP **루트**에 `Dockerfile`, `submission.json`, `README.md`가 필수. 폴더째가 아니라 내용물을 압축할 것.

```
Dockerfile          # EDIT 마커 사이만 수정. 상단 PLATFORM LAYER와 하단 user/ENTRYPOINT 유지
run.sh              # 모델 실행 한 줄만 수정. /tmp/forecast.json에 쓰고 upload.sh 호출
upload.sh           # 수정하지 말 것
submission.json     # 팀명, 이메일, cpu_units, memory_mib, declared_domains
README.md           # 우리 모델 설명으로 교체 (접근법, 입력 데이터, 의존성, 가중치, 재현 방법)
requirements.txt    # 버전 고정(pin)
model 코드 + weights/ + 정규화 통계(scaler 등)
local_tests/        # test_local.sh, mock_platform.py
```

`submission.json` 예시:
```json
{
  "team_name": "<TEAM>",
  "contact_email": "<EMAIL>",
  "cpu_units": 4096,
  "memory_mib": 16384,
  "declared_domains": ["services.swpc.noaa.gov"]
}
```
- 유효한 Fargate 조합만 허용: 1024 → 2048–8192, 2048 → 4096–16384, 4096 → 8192–30720, 8192 → 16384–32768 (MiB, 1024 단위, 8 vCPU는 4096 단위).
- `declared_domains`: **실행 중 접속하는 모든 호스트** (리다이렉트 대상 포함). 빌드 때만 쓰는 pypi 등은 제외.

## 3. Dockerfile 주의사항

- 템플릿은 `FROM ubuntu:22.04`인데 README는 Docker Hub rate limit 때문에 ECR 미러를 권장함 → **`public.ecr.aws/docker/library/python:3.11-slim`** 또는 `public.ecr.aws/ubuntu/ubuntu:22.04` 사용.
- PyTorch는 **CPU 휠** 사용 (`--extra-index-url https://download.pytorch.org/whl/cpu`, `torch==X+cpu`). CUDA 빌드는 약 2.5 GB 낭비. TensorFlow는 `tensorflow-cpu`.
- 가중치와 정규화 통계는 이미지에 COPY. 실행 중 다운로드는 피할 것.
- `.dockerignore`로 학습 데이터, 노트북, 체크포인트 잔여물 제외 → 이미지 최소화.
- 플랫폼 레이어(`ca-certificates curl jq`, `upload.sh` COPY), `useradd runner`, `USER runner`, `ENTRYPOINT ["/app/run.sh"]` 유지.

## 4. 대회 규칙 — 데이터/연산 (11-03 공지)

**허용**
- 공개 관측 데이터와 공개 모델 산출물: SDO/HMI, GONG, ADAPT, OMNIWeb, CDAWeb, NOAA SWPC, UKMO, CCMC, swxforecastlab.org 등. 가입이나 API 키가 필요한 소스도 OK.
- 학습은 외부(GPU 등)에서 해도 됨 (코드 프리즈 전).

**금지 (Operational Podium)**
- 우리 팀 전용으로 컨테이너 밖에서 수행한 연산, 피처 생성, 추론 (공개 URL에 올려도 동일).
- **Pass-through 금지**: 외부 예보를 그대로 넘기거나 단순 보간만 하는 것.
- 라이브 평가 중 재학습, 코드 수정, 수동 개입.

**t0 정보 컷오프 (가장 중요)**
- t0 시점까지 **공개(publish)된** 정보만 사용. 데이터를 받는 시각은 t0 이후여도 되지만, 정보 자체는 t0 이전에 공개됐어야 함.
- 관측 timestamp가 t0 이전이라는 것만으로는 부족. 파일 수정 시각만으로도 증명이 안 될 수 있음.
- 소스가 공개 시각을 제공하지 않으면 **보수적인 공개 지연**을 정해서 적용하고 README에 문서화. 예: `obs_time <= t0 - LAG`.
- 실시간 API는 t0 이후 데이터까지 반환할 수 있으므로 **반드시 직접 잘라낼 것** (과거 t0로 테스트할 때 특히 중요).

**재현성 (권장, 의무 아님)**
- 데이터를 받을 때마다 소스 URL, 받은 시각, 공개 메타데이터(헤더의 Last-Modified, 파일 내 issued time 등), sha256 해시를 **stdout 로그**로 남길 것.
- 주최측이 나중에 재실행(re-check)할 수 있음. 결과가 달라도 원본 데이터가 바뀌었다는 증거가 있으면 실패로 보지 않음.

## 5. 채점

- 정답: **ACE/SWEPAM 월별 1시간 Bulk Speed** (sohoftp, 예: `https://sohoftp.nascom.nasa.gov/sdb/ace/monthly/` — 확인 필요). 실시간 리더보드는 SWPC `ace_swepam_1h.json`.
- 결측과 fill value(-9999.9 등)는 모든 팀에서 제외. 품질 플래그 처리는 추후 공지.
- **예측값이 빠진 시간은 persistence(t0 이전 마지막 유효 ACE 값)로 대체되어 채점됨.** 실패해도 0점은 아니지만 불리함.

## 6. 견고성 요구사항 (구현 필수)

- 다운로드에 timeout과 retry 적용 (전체 15분 예산 고려, 예: 소스당 30–60초, 재시도 2–3회).
- 결측과 지연 처리: 입력 창에 빈 값이 있으면 학습 때와 같은 방식으로 보간.
- **다단계 fallback** — 어떤 예외가 나도 72개 값을 출력하고 exit 0:
  1. 정상 모델 예측
  2. 입력이 부족하면 → 27일 재귀(recurrence) 또는 persistence (t0 이전 마지막 유효 속도)
  3. 데이터를 전혀 못 받으면 → 기후값(예: 400–450 km/s)
- 출력 전 검증: 길이 72, 모두 유한, 물리적 범위로 clip (예: 200–1200 km/s).
- `run.sh`는 `set -euo pipefail`이므로, 모델 스크립트 자체가 내부에서 예외를 잡고 항상 유효한 JSON을 쓰도록 할 것.

## 7. 실시간 입력 데이터 (모델에 맞게 확정 필요)

**TODO: 모델의 학습 입력(변수, 해상도, 창 길이, 소스)을 확인하고, 각각을 실시간으로 받을 수 있는 소스에 매핑할 것.**

참고 후보 (엔드포인트는 실행 전에 직접 확인):
- NOAA SWPC 실시간: `https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json`, `mag-7-day.json`, `https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json`
- ACE 월별 아카이브: `sohoftp.nascom.nasa.gov/sdb/ace/monthly/`
- OMNIWeb / CDAWeb (지연이 큼 — 실시간 입력으로는 부적합할 수 있음)
- SDO, GONG 등 이미지 입력을 쓰면 다운로드 용량과 시간 확인

주의: 학습 데이터(예: OMNI)와 실시간 소스(SWPC, DSCOVR/ACE)는 해상도, 단위, 결측 표기, 위성이 다를 수 있음 → 전처리를 학습과 동일한 분포로 맞출 것.

## 8. 권장 파이프라인

```
run.sh <t0>
 └─ python3 /app/predict.py <t0> > /tmp/forecast.json
      ├─ fetch:     소스별 다운로드 (timeout/retry, 로그: url, time, hash, headers)
      ├─ cutoff:    t0(- 공개 지연) 이후 데이터 제거
      ├─ preprocess: 학습과 동일 (리샘플, 보간, 정규화 통계 로드)
      ├─ infer:     CPU 추론, torch.set_num_threads 조정
      ├─ fallback:  실패 단계별 대체
      └─ validate & write JSON
 └─ /app/upload.sh /tmp/forecast.json
```

## 9. 테스트 절차

1. 템플릿 원본으로 `cd submission_pack/local_tests && ./test_local.sh 20261111T060000Z` → `PASS` 확인.
2. 우리 모델로 교체한 뒤 반복 실행. 이미지 크기와 실행 시간 출력을 확인 (실행은 수 분 이내 목표).
3. **과거 t0 여러 개**로 테스트해서 컷오프 누수가 없는지 확인. 단, 7-day API는 과거 t0를 재현할 수 없으므로 아카이브 기반 테스트 경로를 별도로 마련.
4. 장애 시뮬레이션: 네트워크 차단(`--network none`), 소스 URL 오류, 빈 응답 → fallback으로 PASS하는지 확인.
5. 정답 비교: 과거 구간에서 ACE 실측값 대비 MAE를 계산하고 persistence/recurrence 베이스라인과 비교.

## 10. 제출 전 체크리스트

- [ ] 템플릿 원본 로컬 테스트 PASS
- [ ] 우리 워크플로로 `test_local.sh` PASS
- [ ] `linux/amd64` 빌드, ECR 베이스 이미지, CPU 전용 휠
- [ ] t0 인자 처리, t0 컷오프 적용 (공개 지연 포함)
- [ ] 72개 유한값, fallback 동작 확인
- [ ] `submission.json`: 팀명, 이메일, 유효한 CPU/메모리 조합, 모든 실행 도메인 선언
- [ ] 가중치와 scaler가 이미지 안에 포함됨
- [ ] 15분 대비 충분한 여유 (pull + 실행)
- [ ] `README.md`를 실제 워크플로 설명으로 교체 (데이터 소스, 공개 지연 가정 포함)
- [ ] ZIP 루트에 Dockerfile, submission.json, README.md
- [ ] 실패 시 재제출할 시간을 남기고 업로드 (프리즈: KST 11-20 08:59)

## 11. 미확정 / 확인 필요

- 팀명 정확한 표기, 포털 링크(`.team`의 PORTAL).
- ~~conformance 완료 여부~~ → **완료** (커밋 0534dfd)
- 품질 플래그 처리 방식, live 시작일 (주최측 공지 대기).
- 최종 Participation Terms 전문 (6.2–6.8절).