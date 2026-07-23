# Trial 4 — 프레임 순서 복원 (최종 제출, LB 0.90924)

뒤섞인 4개의 비디오 프레임을 캡션과 함께 보고 **원래 시간 순서로 재배열**하는
문제. 채점은 24개 순열 중 정확히 일치해야만 정답인 Exact Match.

- **모델**: Qwen3-VL-32B-Instruct (nf4 4-bit 양자화 + LoRA)
- **핵심 아이디어**: "정답을 가르치기(SFT)"와 "헷갈리는 후보들을 이기게 하기
  (listwise)"를 **하나의 학습 안에서 동시에** 시킴
- **최종 성적**: Kaggle Public LB **0.90924**
  (trial3 SFT-only 0.89 → trial4 0.90924, 약 +2%p)

---

## 1. 왜 이렇게 접근했나

이 문제는 겉보기엔 "생성" 문제 같지만 본질은 **24지선다 분류**다. 답의 공간이
4! = 24개 순열로 고정돼 있고, 채점도 그중 하나를 맞히는 것이기 때문이다.
그래서 학습·추론·평가를 전부 "24개 중 정답 고르기"에 맞췄다.

오답을 분석해 보면 두 종류였다:

- **인접 스왑** (전체 오류의 ~36%): 한 쌍의 앞뒤만 뒤바뀜.
  예) 정답 `[1,3,4,2]` ↔ 예측 `[1,4,3,2]`
- **중간 정도 뒤섞임** (~30%): 두세 쌍이 어긋남.

일반적인 SFT는 "정답 문장의 확률"만 올릴 뿐, 배포 시 실제로 경쟁하는
**근소한 2등 후보를 눌러주지 않는다.** 그래서 listwise 랭킹 손실을 더해
"정답이 이 후보들 사이에서 1등이 되도록" 직접 학습시킨다.

---

## 2. 학습 방법 (train_unified.py)

베이스 모델부터 시작해서, **매 데이터 행마다 두 갈래 중 하나를 확률적으로
골라** 학습한다. 순차적으로 "SFT 먼저, listwise 나중"이 아니라 처음부터 섞는다.

```
        ┌─ 70% 확률 → SFT 갈래
한 행 ──┤
        └─ 30% 확률 → listwise 갈래
```

### SFT 갈래 (기본, 70%)

- 정답 순서 하나에 대한 문장을 만들고, 그 완성 토큰에만 손실을 건다
  (masked next-token loss). trial3와 동일한 방식.
- 매 epoch마다 프레임을 다른 순서(sigma)로 보여주는 **프레젠테이션 셔플
  증강** — 모델이 "입력 위치"가 아니라 "내용"으로 판단하게 만든다.

### listwise 갈래 (30%)

한 행에 대해 후보 **7개**를 만들어 같이 채점하고, "정답이 1등"이 되도록
cross-entropy 손실을 건다:

```
정답            [1,3,4,2]   ← 이게 1등이어야 함
인접 스왑 ×3    [1,4,3,2] [3,1,4,2] [1,3,2,4]   (kendall 거리 1)
중간 뒤섞임 ×2  ...                              (kendall 거리 2~3)
랜덤 ×1         ...
```

- 인접 스왑(36% 오류)뿐 아니라 중간 뒤섞임(30% 오류)까지 후보에 넣어
  두 오류 유형을 모두 겨냥한다. (`common.broadened_candidate_set`)
- 후보별 점수는 배치 forward의 logit에서 직접 계산
  (`common.candidate_logprob_sum`).

### 왜 "동시에" 섞나

SFT를 끝낸 뒤 별도로 listwise를 붙이면(순차 방식), 2단계가 1단계가 굳혀놓은
표현을 깎아먹을 수 있다(catastrophic forgetting — 순차 파인튜닝의 알려진
약점). SFT("정답이 그럴듯하다")와 listwise("정답이 경쟁자를 이긴다")는
**서로 싸우는 목표가 아니라 같은 판단을 다른 각도에서 가르치는 것**이라,
처음부터 함께 학습시키면 서로 강화된다는 가설이다.

### 주요 설정

| 항목 | 값 |
|---|---|
| 베이스 | Qwen3-VL-32B-Instruct, nf4 (double-quant, bf16 연산, 비전타워는 bf16 유지) |
| LoRA | r=32, LLM 디코더 선형층만 (비전타워 제외), 학습 파라미터 0.8% |
| listwise 비율 | 30% |
| 후보 구성 | 정답 + 인접스왑 3 + 중간 2 + 랜덤 1 = 7개 |
| 학습률 | 1e-4 (베이스부터라 SFT 수준의 높은 LR) |
| epoch | 1 (train_full 8,871행) |
| gradient checkpointing | 항상 ON (두 갈래 모두 일반 배치 forward) |

---

## 3. 추론 방법 (infer.py)

학습과 똑같은 "24개 중 정답 고르기" 규칙으로 채점한다.

1. **24-way 제약 채점**: 이미지가 포함된 프롬프트를 KV 캐시로 한 번만
   forward하고, 24개 후보 완성 문장의 log-prob 합을 각각 계산.
2. **TTA (k=3)**: 프레임을 서로 다른 3가지 순서(sigma)로 보여주고 점수를
   평균 → 입력 순서에 대한 편향 제거.
3. **identity prior (α=0.5)**: train에서 identity(`[1,2,3,4]`)가 15.5%였고
   (No_ordering 행), 셔플된 행은 identity가 0%다. 셔플 증강이 이 사전확률을
   지워버리므로 추론 시 `점수 + α·log P_train(순열)`로 되살린다. (train 통계만
   사용 — 규칙 위반 아님)
4. 최종 = 24개 점수 argmax → 항상 유효한 순열 (파싱 실패 없음)

**k=3, α=0.5가 sweet spot이었다.** k=5로 늘려봤지만 LB가 오히려 떨어져
(0.90924 → 0.90575), TTA를 무작정 늘리는 게 이득이 아님을 확인했다.

---

## 4. 데이터

- **정제**: 검은 프레임(260) + 중복 프레임(209) = 414행(4.3%) 제거.
  결과가 `data_work/train_clean.csv`(8,121) / `clean_val.csv`(1,000).
  플래그는 `data_work/train_flags.csv`.
- **분할** (`make_split.py`, seed 42 고정, 재현 가능):
  - `val_small.csv` 250행 — 학습 중 실제규칙 모니터용
  - `train_full.csv` 8,871행 — 학습용 (train_clean + clean_val의 나머지 750)
- 규칙 준수: train 데이터만 사용(Rule 3.4), 앙상블 없음(단일 어댑터),
  외부 데이터 없음, TTA는 규칙상 허용된 test-time augmentation.

---

## 5. 파일

| 파일 | 역할 |
|---|---|
| `common.py` | 모델 로드(nf4), 순열 수학, 프롬프트, 24-way 채점, 후보 구성, prior |
| `train.py` | SFT 학습 유틸 (collate/모니터/체크포인트) — train_unified가 재사용 |
| `train_unified.py` | **최종 학습 스크립트** (SFT+listwise 통합) |
| `smoke_unified.py` | GPU 사전 점검 (두 갈래가 한 모델에서 정상 동작하는지) |
| `infer.py` | 추론 → 제출 CSV (24-way 채점 + TTA + prior) |
| `test_perms.py` | 순열/후보 구성 수학 CPU 단위테스트 |
| `make_split.py` | train_full/val_small 분할 재생성 (seed 42) |
| `setup_box.sh` | GPU 박스 프로비저닝 (deps + 모델 다운로드) |
| `outputs/trial4_round2_submission.csv` | **최종 제출본 (LB 0.90924)** |
| `outputs/test_scores_round2.jsonl` | 그 제출을 만든 24-way 점수 덤프 (재사용 가능) |

`../trial4_archive_round1/`에 이전 라운드/기각된 실험(순차 listwise, hard-negative
채굴, pairwise 재순위, TTA k=5 등)을 보관해 뒀다.

---

## 6. 재현 순서

```bash
# 1) GPU 박스 (80GB+ VRAM 권장) 프로비저닝
bash setup_box.sh                       # deps + Qwen3-VL-32B 다운로드
python make_split.py                    # train_full / val_small 생성

# 2) 사전 점검
python test_perms.py                    # CPU, 수학 검증
SNU_DATA_DIR=data python smoke_unified.py   # GPU, 두 갈래 정상 동작 확인

# 3) 학습 (A100 80GB 기준 ~19h, val 모니터 6회)
SNU_DATA_DIR=data python train_unified.py   # -> ckpts_unified/ckpt-<N>

# 4) 추론 → 제출 CSV
SNU_DATA_DIR=data SNU_ADAPTER_DIR=ckpts_unified/ckpt-<N> \
  SNU_SPLIT=test SNU_TTA_K=3 SNU_PRIOR_ALPHA=0.5 \
  SNU_DUMP_SCORES=outputs/test_scores.jsonl \
  SNU_OUT=outputs/submission.csv python infer.py
```

추론은 RTX 3090(24GB) 기준 VRAM 피크 ~20.5GB로 대회 제출 환경 제약을 만족하며,
819행 전체가 24시간 추론 제한 안에 들어온다.
