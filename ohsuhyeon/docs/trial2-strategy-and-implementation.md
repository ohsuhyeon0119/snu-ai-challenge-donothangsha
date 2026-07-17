# Trial 2 — 전략과 구현

**작성일:** 2026-07-12
**과제:** 캡션 + 뒤섞인 4개 비디오 프레임 → 원본 시간 순서 복원 (24-way Exact Match)
**목표:** trial1의 16.77%에서 유의미하게 도약

---

## 1. 문제의 재정의 (핵심 전략)

### 1.1 관찰: 캡션이 이미 정답 순서로 쓰여 있다

학습 데이터의 캡션을 보면 사건이 **시간 순서대로** 서술되어 있다.

> "A girl hula hoops indoors **before** the scene shifts outdoors to a cheering group on rocks;
> **then**, players swim towards the pool's center..." → 정답 `[3, 1, 2, 4]`

즉 이 과제는 "영상의 픽셀에서 모션을 추론하는" 문제라기보다,
**4개 프레임을 캡션이 서술한 사건 순서에 정렬(align)하는** 문제에 가깝다.
이것이 24-way exact match에서 높은 정확도가 원리적으로 도달 가능한 이유다.

### 1.2 그러나 캡션을 4조각으로 쪼개지 않는다

"캡션이 순서를 담고 있다"에서 곧바로 **"캡션을 4개 절로 분할 → 프레임과 1:1 매칭(Hungarian)"**
으로 가는 것은 취약하다. 실제 캡션은 정확히 4개 사건으로 나뉘지 않는다 — 사건이 3개뿐이거나,
한 문장에 여러 동작이 겹치거나, 카메라 이동만 서술되기도 한다.

따라서 **캡션 전체 + 프레임 4장을 통째로 VLM에 입력**하고, 정렬을 end-to-end로 학습시킨다.
분할이라는 취약한 전제를 두지 않으므로 캡션 구조가 어떻든 깨지지 않는다.

> **전략 요약:** "캡션이 순서를 담고 있다"는 *왜 이 과제가 학습 가능한지*에 대한 근거이지,
> *전처리 단계*가 아니다.

### 1.3 Track A(CLIP 분류기)가 왜 실패했는가

이전 시도인 Track A는 CLIP의 **global pooled 임베딩**을 뽑아 24-클래스 분류기를 학습해
15.7%(다수 클래스 바닥 15.5%와 통계적 동률)에 그쳤다. 이는 정렬 구조를 버리고 문제를
"24개 중 하나 고르기"로 뭉갠 **잘못된 formulation** 때문이다. 프레임별 정렬 신호가 전역 풀링에서
사라진다.

---

## 2. trial1 실패의 근본 원인 (16.77%)

trial1은 Qwen2-VL-2B를 LoRA 파인튜닝했으나 **loss masking이 잘못**되어 있었다.

```python
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100   # padding만 무시
```

이 코드는 **padding만** 무시하고, 나머지 전부 — chat template, 지시문, 그리고 **이미지 1장당 수백
개에 달하는 vision 토큰** — 를 loss에 포함시킨다.

- 전체 시퀀스: ~1,183 토큰
- 실제 정답 토큰: **~17개 (약 1.4%)**

즉 모델은 대부분의 gradient를 "고정된 프롬프트와 vision placeholder를 재현하는 데" 쓰고,
정답 예측 신호는 1% 남짓으로 **희석**되었다.

**증상이 정확히 일치한다:**
- loss가 ~6.8에서 plateau (프롬프트 재현은 금방 숙달 → 더 내려갈 데가 없음)
- 테스트 예측이 두 답으로 **붕괴**: `[1,2,3,4]` 28% + `[3,2,4,1]` 21% = **약 49%**
  → 입력별 추론(conditional)이 아니라 **정답의 주변분포(marginal)만 외운** 모델

trial1 문서는 이를 "moderate"로 평가했으나, 실제로는 **치명적**이었다.

---

## 3. trial2 설계

### 3.1 변경 사항과 근거

| # | 변경 | 근거 |
|---|---|---|
| 1 | **Masked loss** — 프롬프트·vision 토큰 전부 `-100`, **정답 토큰에만** loss | trial1의 근본 원인 정면 교정. 최우선 레버 |
| 2 | LoRA를 **전 linear projection**(`q,k,v,o,gate,up,down`), rank 32로 확대 | trial1은 `q,v`/r=16으로 표현력 부족 |
| 3 | **Cosine LR + warmup** | trial1은 flat 1e-4, 스케줄 없음 |
| 4 | **주기적 검증 → best checkpoint 채택** | trial1은 학습 종료 후 1회만 평가 (과/미학습 판단 불가) |
| 5 | `max_pixels`로 vision 토큰 수 제한 | 순서 판별은 coarse task → 속도 확보 |
| 6 | **Constrained permutation scoring** (추론) | 파싱 실패 제거 + 정확도 향상 |
| 7 | 2B로 기법 검증 → 7B로 본승 (순차, 앙상블 아님) | 싸게 검증 후 천장 높이기 |

### 3.2 Masked loss 구현

핵심은 **프롬프트 길이를 정확히 알아내는 것**이다. vision 토큰이 프롬프트 안에서 확장되므로
텍스트 길이로는 계산할 수 없다. 따라서 **프롬프트만 따로 processor에 통과**시켜 확장 후 길이를
구하고, 그 지점까지 마스킹한다.

```python
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
completion  = C.target_text(io) + processor.tokenizer.eos_token

full        = processor(text=[prompt_text + completion], images=image_inputs, return_tensors="pt")
prompt_only = processor(text=[prompt_text],              images=image_inputs, return_tensors="pt")
plen = prompt_only["input_ids"].shape[1]          # vision 확장 포함한 실제 프롬프트 길이

labels = full["input_ids"].clone()
labels[:, :plen] = -100    # 프롬프트 + 모든 vision 토큰 마스킹 → 정답 토큰만 학습
```

**검증:** smoke test에서 `supervised_tokens=17` / `seq_len=1183` 확인 → 의도대로 정답 토큰만 지도.
학습 시작 직후 loss가 **0.48 → 0.19 → 0.07**로 하강 (trial1의 ~6.8 plateau와 질적으로 다름).

> ⚠️ 해석 주의: 정답 문자열 `"The correct order is [3, 1, 4, 2]."` 중 실제 정보는 **숫자 4개**뿐이고
> 나머지 boilerplate는 몇 스텝 만에 숙달된다. 따라서 **낮은 loss ≠ 높은 정확도**이며,
> 판단은 반드시 val exact-match로 한다.

### 3.3 라벨 매핑 (틀리면 전부 오답)

- `Answer[k]` = **Input_(k+1)의 시간상 위치**
- 모델 출력 `image_order[t]` = **시간상 t+1번째에 오는 Input의 번호**

두 표현은 서로의 역치환이며, 제출은 `Answer` 포맷이어야 한다.

```python
def image_order_from_answer(answer):          # Answer -> image_order
    return [answer.index(pos) + 1 for pos in range(1, 5)]

def answer_from_image_order(image_order):     # image_order -> Answer (제출용)
    answer = [0, 0, 0, 0]
    for t, img in enumerate(image_order, start=1):
        answer[img - 1] = t
    return answer
```

### 3.4 Constrained permutation scoring (추론)

자유 생성 + 정규식 파싱(trial1) 대신, **24개 후보 순열 각각의 likelihood를 계산해 argmax**.

```python
full_texts = [prompt_text + target_text(io) for io in ALL_ORDERS]   # 24개
...
comp_logits = logits[:, plen - 1:-1, :].float()      # 정답 구간만
comp_tokens = inputs["input_ids"][:, plen:]
tok_logp = torch.log_softmax(comp_logits, -1).gather(-1, comp_tokens.unsqueeze(-1)).squeeze(-1)
logps.extend(tok_logp.sum(-1).tolist())              # 후보별 총 log-prob
```

**설계상 이점:**
- 24개 후보의 완성 문자열은 **토큰 길이가 모두 동일**(숫자 1~4는 전부 1글자)하고 프롬프트도 동일
  → **padding 불필요**, 프롬프트 길이 `plen`이 상수 → 깔끔하고 정확한 점수화
- 출력이 **항상 유효한 순열** (파싱 실패·무효 출력 0)
- 규칙 3.3의 "추론 전략" 허용 범위, 24h 예산 내

### 3.5 검증 프로토콜

- trial1과 **동일한 split**(seed 42, 12% stratified, n≈1,145) 재사용 → 16.77%와 직접 비교 가능
- 학습 중 모니터는 **greedy generation**(행당 ~0.5-1초)을 사용
  - 7B에서 constrained scoring은 **행당 ~16.5초**라 학습 중 검증에 쓰면 시간을 다 잡아먹음
  - 최종 판정용으로는 constrained scoring 유지

---

## 4. 구현 구조

```
ohsuhyeon/trial2/
├── common.py          # 모델/프로세서 로드, 프롬프트, 라벨 매핑, score_orders, generate_order
├── train.py           # micro-batch=1 QLoRA (24GB 3090용), masked loss, best-ckpt, resume 지원
├── train_batched.py   # padding-collate 배치 QLoRA (A100 등 대용량 VRAM), greedy 모니터
├── eval.py            # held-out val 전체 exact-match (constrained scoring)
├── infer.py           # test.csv -> 제출 CSV (gen|score 모드, 행별 저장 + resume)
├── clean_data.py      # 검은/중복 프레임 리포트
└── smoke_test.py      # masked forward/backward 1회 + 스코어링 몇 행
```

모든 경로·하이퍼파라미터는 **환경변수로 오버라이드**되어, 2B/7B·3090/A100에서 코드 변경 없이 동작한다.

### 4.1 배치 학습 (train_batched.py)

대용량 VRAM을 쓰려면 micro-batch > 1이 필요하고, 그러려면 **길이가 다른 시퀀스의 padding**과
**per-example 프롬프트 마스킹**을 함께 처리해야 한다.

```python
def collate(batch):
    maxL = max(b["input_ids"].size(0) for b in batch)
    batched = {
        "input_ids":      pad("input_ids", pad_id),      # 오른쪽 패딩
        "attention_mask": pad("attention_mask", 0),
        "pixel_values":   torch.cat([b["pixel_values"] for b in batch], 0),     # flat 연결
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], 0),
    }
    labels = pad("labels", -100)   # 패딩 위치도 -100 → 손실에서 제외
    return batched, labels
```

- **오른쪽 패딩** + `attention_mask=0` + `labels=-100` → 배치 스텝이 단일 예제 손실의 합과 수학적으로 동일
- Qwen2-VL은 vision을 `[total_patches, dim]`으로 평탄화하고 `image_grid_thw`로 각 이미지를 지시하므로,
  배치에서는 **행 순서대로 concat**하면 `image_pad` 토큰 위치와 자연히 정합됨
- **유효 배치 8**(micro 8 × accum 1)로 2B 런(micro 1 × accum 8)과 최적화 동역학을 맞춰 **비교 가능성 유지**

### 4.2 실행 중 만난 문제와 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| 학습 중 eval이 GPU 0%인 채 5분+ 정지, 메모리 71GB | 학습 allocator 캐시가 예약·단편화된 상태에서 generate의 KV 캐시 할당이 스래싱 | eval 전후 `gc.collect()` + `torch.cuda.empty_cache()` |
| 7B constrained scoring이 행당 16.5초 | 후보 24개가 **같은 이미지를 24번 재인코딩**(vision 인코더가 병목) | 학습 중 모니터는 greedy로 전환. (근본 해결은 vision 1회 인코딩 후 재사용 — 향후 과제) |
| trial1 배치 시 OOM | Qwen2 어휘 152k → logits가 `batch × seq × vocab`로 폭증 | `max_pixels`로 vision 토큰 축소 + grad checkpointing + accum |

---

## 5. 실험 결과

**greedy-gen val exact-match** (seed-42 12% split)

| 모델 | 결과 |
|---|---|
| 랜덤 (1/24) | 0.042 |
| 다수 클래스 바닥 (`[1,2,3,4]` 고정) | 0.155 |
| Track A (CLIP + 분류기) | 0.157 |
| trial1 (Qwen2-VL-2B, loss 버그) | 0.168 |
| **trial2 — Qwen2-VL-2B** (3090) | 0.258 → 0.375 → **0.400** (~1 에폭) |
| **trial2 — Qwen2.5-VL-7B** (A100) | 0.400 @ step150 → **0.4167 @ step300** (~0.28 에폭, 조기 중단) |

**해석**
- masked loss 하나로 **0.168 → 0.40+** 도약 → 진단이 정확했음을 입증
- 7B는 2B의 최고점(0.40)을 **0.14 에폭**에 도달 (2B는 ~1.1 에폭 소요) → **천장이 훨씬 높고 학습도 빠름**
- 중단 시점에도 **여전히 상승 중**이었음

**예측 분포의 건강성** (test 819행)
- identity `[1,2,3,4]` **20.1%** (train 실제 비율 15.5%와 근접)
- 나머지가 여러 순열에 고루 분포
- → trial1의 붕괴(2개 답이 49%)와 대조적. **입력별 추론이 실제로 일어나고 있음**

---

## 6. 인프라

로컬 개발기(Apple M5)는 CUDA가 없어 학습 불가 → **Vast.ai GPU 대여**.

- **2B:** RTX 3090 24GB (micro-batch=1 + grad accum), ~9.6초/optim step
- **7B:** A100 80GB / 128 CPU (batched micro-batch=8), ~9.6초/optim step
  - 데이터 4.1GB는 **tar-over-ssh 스트리밍**으로 전송 (파일 3.8만 개 → rsync보다 유리)
  - 7B 가중치(16GB)는 데이터 전송과 **병렬로 사전 다운로드**
  - 모든 학습은 **tmux**로 실행 → SSH 연결이 끊겨도 지속

> 규칙상 **학습 GPU에는 제약이 없다.** RTX 3090 제약은 **최종 추론**에만 적용되며,
> 산출물은 4-bit 베이스 + 작은 LoRA 어댑터라 3090 24GB에서 추론 가능하다.

---

## 7. 규칙 준수

| 규칙 | 준수 내용 |
|---|---|
| 단일 모델 (앙상블 금지) | 최종 제출은 7B **단일 모델** 하나. 2B는 순차적 사전 검증이며 결과 결합 없음 |
| 외부 데이터 금지 | 제공된 train.csv만 사용 |
| 생성형 증강 금지 | 데이터 생성/변형 없음 |
| 2026-05-31 이전 공개 가중치 | Qwen2-VL(2024-08), Qwen2.5-VL(2025-01) |
| 외부 API 금지 | 사용 안 함 |
| 경량화 허용 | 4-bit 양자화 + LoRA |
| 최종 추론 3090 24GB / 24h / 80GB | 4-bit 7B ≈ 6GB VRAM, 어댑터 380MB. 추론 예산 내 |
| 데이터 누수 금지 | 검증은 train에서 분리한 held-out만 사용. **파일 메타데이터/타임스탬프 등 비의미적 신호 일절 미사용** |

---

## 8. 한계와 다음 단계

### 현재 한계
- 7B를 **0.28 에폭에서 조기 중단** (크레딧 제약) — 아직 상승 중이었음
- 제출은 **greedy 추론** 기반 (constrained scoring이면 몇 %p 상승 기대되나 행당 16.5초)
- **데이터 정제 미적용** — 주최측이 명시한 "정답 특정 불가/무관 프레임(검은 화면 등)" 노이즈가
  학습과 val 지표를 모두 깎고 있음 (`clean_data.py`는 리포트만 작성한 상태)

### 0.8을 향한 레버 (기대효과 순)
1. **데이터 정제** — 불량/모호 샘플을 학습에서 제외. 본선 평가 항목("전처리 적절성")과도 직결
2. **풀 학습** — 2~3 에폭 완주 (궤적이 아직 우상향)
3. **Constrained scoring 추론** — vision 1회 인코딩 후 24후보 재사용으로 가속(현재 병목) 후 적용
4. **STaR 방식 self-CoT** — 모델이 스스로 추론을 생성 → **정답과 일치하는 트레이스만** 선별 →
   그것으로 재파인튜닝. 외부 데이터·API 없이 제공 데이터와 모델 자신만 사용.
   규칙 3.3이 CoT 추론을 명시 허용하나, *자체 생성 학습 트레이스*가 "생성형 증강"에 해당하는지는
   **운영진 사전 확인 필요**

---

## 부록: 재현 방법

```bash
export HF_HOME=/workspace/.hf_home
export SNU_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export SNU_DATA_DIR=/workspace/data/snuaichallenge_data
export SNU_ADAPTER_DIR=/workspace/trial2_adapter_7b

python smoke_test.py                      # 파이프라인 점검 (supervised_tokens=17 확인)
python train_batched.py                   # 대용량 VRAM 배치 학습 (3090이면 train.py)
python eval.py                            # held-out val 전체 exact-match
SNU_INFER_MODE=gen python infer.py        # 제출 CSV (score = 정밀/느림)
```

주요 환경변수: `SNU_MICRO_BATCH SNU_ACCUM SNU_EPOCHS SNU_LR SNU_LORA_R SNU_EVAL_EVERY
SNU_EVAL_N SNU_EVAL_MODE SNU_MAX_PIXELS SNU_SCORE_CHUNK SNU_GRAD_CKPT SNU_ADAPTER_SUBDIR`
