# 하라는행샤는안하고 — 텍스트로 풀어보는 장면의 재구성 (SNU AI Challenge 2026)

캡션과 무작위로 뒤섞인 4개의 비디오 프레임을 입력받아 원래 시간 순서를 복원하는
과제. 평가 지표는 Exact Match Accuracy(24개 순열 중 완전 일치).

- **팀**: 하라는행샤는안하고 (오수현, 김하람, 권성안)
- **기반 모델**: Qwen/Qwen3-VL-32B-Instruct (NF4 4-bit + LoRA)
- **최종 성적**: Public LB **0.92146**
- **접근 요약**: 문제를 24-way 순열 선택으로 재정의 → SFT와 listwise ranking을
  하나의 학습에 통합(Stage 1) → error analysis가 지목한 순수 시각 추론 병목을
  **vision encoder LoRA**로 해소(Stage 2, 최종 도약).

방법론 상세는 별도 제출한 방법론 보고서(PDF)를 참고.

---

## 1. Repository 구조

```
.
├── README.md              # 이 문서 (환경·설치·실행·재현 안내)
├── requirements.txt       # 고정 라이브러리 버전
├── setup.sh               # 의존성 설치 + 베이스 모델 다운로드
├── download_weights.py    # 최종 LoRA 어댑터 자동 다운로드 (Google Drive)
│
├── common.py              # 모델 로드(NF4), 순열 수학, 24-way scorer,
│                          #   listwise 후보 구성, identity prior,
│                          #   vision LoRA 대상/gradient hook
├── clean_data.py          # [데이터] 검은/중복 프레임 정제 → train_clean/clean_val
├── make_split.py          # [데이터] 최종 학습/검증 분할 (seed 42)
│
├── train.py               # SFT 학습 유틸(collate/monitor/checkpoint) — 재사용 모듈
├── train_unified.py       # [Stage 1] SFT+listwise 통합 학습 (베이스 → LB 0.90924)
├── train_vision.py        # [Stage 2] vision encoder LoRA (Stage1 → LB 0.92146, 최종)
├── infer.py               # 24-way 추론 → 제출 CSV
│
├── test_perms.py          # 순열/후보 구성 수학 CPU 단위테스트
├── smoke_unified.py       # [GPU 사전점검] Stage 1 (SFT+listwise 동작)
├── smoke_vision.py        # [GPU 사전점검] Stage 2 (vision LoRA gradient 흐름)
│
├── data_work/
│   └── train_flags.csv    # 정제 플래그(검은/중복) — clean_data.py 재실행 시 재생성됨
└── weights/               # download_weights.py 가 어댑터를 받아 푸는 위치
```

---

## 2. 실행 환경

### 하드웨어

| 용도 | 사양 | 비고 |
|---|---|---|
| **학습** | A100 80GB 또는 H100 80~94GB (단일 GPU) | NF4 QLoRA 학습 피크 약 45~50GB |
| **추론(대회 제출 환경)** | RTX 3090 24GB (단일 GPU) | **측정 피크 VRAM 약 20.6GB** → 24GB 제약 충족 |
| 디스크 | 120GB+ | 베이스 모델 65GB + 데이터 + 체크포인트 |

- 819행 전체 추론은 24시간 제한 안에 완료된다(H100 기준 약 14초/행 실측; 3090 환산 약 11~13시간).

### 소프트웨어

- **OS/런타임**: Linux, Python 3.12, CUDA 12.x~13.x
- **핵심 라이브러리(고정)**: `torch==2.12.0(+cu130)`, `transformers==5.14.1`,
  `peft==0.19.1`, `accelerate==1.14.0`, `bitsandbytes>=0.43`,
  `qwen-vl-utils==0.0.14`, `safetensors`, `pandas`, `numpy`, `pillow`, `gdown`
- 전체 목록·핀은 `requirements.txt` 참조. `transformers 5.14`의 Qwen3-VL
  M-RoPE / KV-cache 동작에 코드가 의존하므로 **해당 버전 고정을 권장**한다.

---

## 3. 설치

```bash
bash setup.sh
# = torch 설치 + requirements.txt 설치 + Qwen3-VL-32B-Instruct 사전 다운로드(약 65GB)
```

수동으로 하려면:

```bash
pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-32B-Instruct')"
```

---

## 4. 데이터 준비

대회 제공 데이터(`train.csv`, `test.csv`, `train/`, `test/`)를 한 폴더에 두고,
그 경로를 환경변수 `SNU_DATA_DIR`로 지정한다(기본값 `data`). 예:

```
data/
├── train.csv   test.csv   sample_submission.csv
├── train/<Id>/*.jpg
└── test/<Id>/*.jpg
```

정제·분할을 재생성한다(제공 데이터만 사용, 규칙 3.4 준수):

```bash
# (1) 검은/중복 프레임 정제 → data_work/train_clean.csv, clean_val.csv, train_flags.csv
SNU_DATA_DIR=data python clean_data.py

# (2) 최종 학습/검증 분할 → data_work/train_full.csv(8,871), val_small.csv(250)
python make_split.py
```

정제 결과: 검은 프레임 260 + 중복 프레임 209 = **414행(4.3%)** 제거
→ `train_clean` 8,121 / `clean_val` 1,000 → 최종 `train_full` 8,871 / `val_small` 250.
(`data_work/train_flags.csv`에 정제 플래그가 포함되어 있어 정제 결과를 바로 검토할 수 있다.)

---

## 5. 재현 방법

두 가지 경로를 제공한다. **A는 빠른 재현(추론만)**, **B는 처음부터 전체 재학습.**

### A. 최종 가중치로 추론만 (권장, 빠름)

```bash
# (1) 최종 LoRA 어댑터 다운로드 (Google Drive → weights/vision_best)
python download_weights.py

# (2) 24-way 추론 → 제출 CSV
SNU_DATA_DIR=data SNU_ADAPTER_DIR=weights/vision_best \
  SNU_SPLIT=test SNU_TTA_K=3 SNU_PRIOR_ALPHA=0.5 \
  SNU_OUT=submission.csv python infer.py
```

> **가중치 다운로드 링크**: 최종 vision-LoRA 어댑터(약 1.1GB, LB 0.92146)는
> Google Drive에 보관한다. `download_weights.py`가 아래 파일을 자동으로 내려받아
> 압축 해제한다.
>
> - **최종 어댑터 (LB 0.92146)**: https://drive.google.com/file/d/1PA8IpD8Z552WIbgMZKEYDkRi9N4rr1tv/view?usp=sharing

### B. 처음부터 전체 재학습

```bash
# (0) GPU 사전점검 (선택이지만 권장 — 몇 분)
python test_perms.py                                   # CPU: 순열/후보 수학
SNU_DATA_DIR=data python smoke_unified.py              # GPU: Stage 1 동작
SNU_DATA_DIR=data python smoke_vision.py \
  SNU_INIT_ADAPTER=ckpts_unified/ckpt-<N>              # GPU: Stage 2 gradient 흐름

# (1) Stage 1 — SFT+listwise 통합 학습 (베이스 → ckpts_unified/ckpt-<N>)
SNU_DATA_DIR=data python train_unified.py

# (2) Stage 2 — Stage 1 최종 체크포인트에서 vision encoder LoRA 이어 학습
SNU_DATA_DIR=data SNU_INIT_ADAPTER=ckpts_unified/ckpt-<N> python train_vision.py
#   → 최고 검증 체크포인트가 ckpts_vision/best/ 에 저장됨

# (3) 추론
SNU_DATA_DIR=data SNU_ADAPTER_DIR=ckpts_vision/best \
  SNU_SPLIT=test SNU_TTA_K=3 SNU_PRIOR_ALPHA=0.5 \
  SNU_OUT=submission.csv python infer.py
```

- `<N>`은 Stage 1의 최종 step(1 epoch, 8,871행, ACCUM 4 → 약 2218). 학습 로그의
  마지막 `ckpt-<N>` 이름을 사용한다.
- 학습은 검증(`val_small`, 실제 24-way 규칙) 기준 best 체크포인트를 자동 저장한다.

---

## 6. 주요 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 베이스 모델 / 양자화 | Qwen3-VL-32B-Instruct / NF4 4-bit double-quant, bf16 연산 (vision tower는 bf16 유지) |
| LoRA | r=32, α=64, dropout=0.05 |
| LoRA 대상 | Stage1: LLM decoder linear (896 tensor) / Stage2: + vision block linear (216 tensor) |
| 통합 학습(Stage 1) | listwise 비율 0.30, LR 1e-4, epoch 1, ACCUM 4, gradient checkpointing ON |
| Listwise 후보 | 정답 + adjacent swap 3 + 중간(Kendall 2–3) 2 + random 1 = 7 |
| Vision LoRA(Stage 2) | Stage1에서 continue, vision LoRA zero-init, LR 3e-5 |
| 추론 | 24-way constrained scoring + TTA k=3 + identity prior α=0.5 |

---

## 7. 환경변수(모든 스크립트 공통 `SNU_*`)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SNU_DATA_DIR` | `data` | 대회 데이터 폴더(train.csv/test.csv/train//test/) |
| `SNU_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | 베이스 모델 |
| `SNU_ADAPTER_DIR` | (없음) | 추론에 사용할 LoRA 어댑터 경로 |
| `SNU_INIT_ADAPTER` | `ckpts_unified/ckpt-2218` | Stage 2가 이어받을 Stage 1 어댑터 |
| `SNU_SPLIT` | `test` | `test`(제출) 또는 `val` |
| `SNU_TTA_K` | `1` | TTA 시그마 개수(최종 설정 3) |
| `SNU_PRIOR_ALPHA` | `0` | identity prior 가중치(최종 설정 0.5) |
| `SNU_OUT` | (스크립트별) | 출력 CSV 경로 |

---

## 8. 규정 준수

| 항목 | 내용 |
|---|---|
| 학습 데이터 | 제공 train 데이터만 사용, 외부 데이터 없음 |
| 데이터 누수 | 평가 데이터 정보 미사용(identity prior는 train 통계만 활용) |
| 앙상블 | 단일 LoRA 어댑터, 앙상블 없음 |
| 모델 공개일 | Qwen3-VL(2025-10) ≤ 2026-05-31 |
| 추론 방식 | 24-way scoring + 허용된 TTA |
| 외부 API | 미사용(비용 0) |
| 재현성 | seed 42 고정, 정제/분할/단위테스트/사전점검 스크립트 제공 |
