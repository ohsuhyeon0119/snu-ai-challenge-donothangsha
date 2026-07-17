# Trial 2 학습 가이드 — 우리가 만든 것을 처음부터 이해하기

> **대상:** 딥러닝 기초(모델, 학습, loss 정도)는 알지만 **파인튜닝·LoRA·VLM은 이름만 들어본** 사람
> **목표:** 이번에 구현한 코드가 *무엇을, 왜, 어떻게* 하는지 스스로 설명할 수 있게 되기
> **읽는 법:** 순서대로. 각 장은 앞 장에 의존합니다. 코드는 전부 `ohsuhyeon/trial2/`의 실제 코드입니다.

---

## 목차

1. [큰 그림 — 우리가 푸는 문제](#1-큰-그림--우리가-푸는-문제)
2. [언어모델은 사실 "다음 단어 맞히기" 기계다](#2-언어모델은-사실-다음-단어-맞히기-기계다)
3. [이미지는 어떻게 언어모델에 들어가는가 (VLM)](#3-이미지는-어떻게-언어모델에-들어가는가-vlm)
4. [파인튜닝이란 무엇인가](#4-파인튜닝이란-무엇인가)
5. [LoRA — 1.1%만 학습해서 모델을 바꾸는 법](#5-lora--11만-학습해서-모델을-바꾸는-법)
6. [4-bit 양자화 — 16GB 모델을 6GB에 욱여넣기](#6-4-bit-양자화--16gb-모델을-6gb에-욱여넣기)
7. [우리 문제를 "다음 토큰 맞히기"로 번역하기](#7-우리-문제를-다음-토큰-맞히기로-번역하기)
8. [⭐ 핵심: Loss Masking — 이번 프로젝트의 전부](#8--핵심-loss-masking--이번-프로젝트의-전부)
9. [학습 루프의 부품들](#9-학습-루프의-부품들)
10. [검증 — 우리가 잘하고 있는지 어떻게 아는가](#10-검증--우리가-잘하고-있는지-어떻게-아는가)
11. [추론 — 두 가지 방법과 그 트레이드오프](#11-추론--두-가지-방법과-그-트레이드오프)
12. [결과 읽는 법 — loss가 낮은데 정확도가 낮은 이유](#12-결과-읽는-법--loss가-낮은데-정확도가-낮은-이유)
13. [실전에서 부딪힌 것들](#13-실전에서-부딪힌-것들)
14. [전체 흐름 다시 보기](#14-전체-흐름-다시-보기)

---

## 1. 큰 그림 — 우리가 푸는 문제

### 문제
캡션 한 줄과 **뒤섞인 4장의 비디오 프레임**이 주어진다. 원래 시간 순서를 복원하라.

- 가능한 순서는 4! = **24가지**
- 채점은 **exact match**: 4개 다 맞아야 1점, 하나라도 틀리면 0점. 부분점수 없음
- 그래서 아무렇게나 찍으면 1/24 ≈ **4.2%**

### 우리가 발견한 핵심 통찰
캡션을 자세히 보면 **사건이 시간 순서대로 서술**되어 있다.

> "A girl hula hoops indoors **before** the scene shifts outdoors...; **then**, players swim..."

즉 이 문제는 "픽셀에서 움직임을 추론하라"가 아니라,
**"4장의 사진을 캡션이 말하는 순서에 맞춰 줄 세워라"**에 가깝다.
캡션이 답의 순서를 이미 알려주고 있으니, 모델은 *어떤 사진이 캡션의 어느 대목인지*만 알아내면 된다.

### 그런데 왜 캡션을 4조각으로 쪼개지 않았나?
"캡션이 순서를 담고 있다" → "그럼 캡션을 4개 절로 자르고 사진과 1:1 매칭하면 되겠네?"
라고 생각하기 쉽다. **하지만 이건 깨지기 쉽다.** 실제 캡션은:
- 사건이 3개뿐일 수도 있고 (사진은 4장인데)
- 한 문장에 여러 동작이 겹쳐 있기도 하고
- 카메라 이동만 서술하기도 한다

그래서 우리는 **캡션 전체 + 사진 4장을 통째로** 모델에 넣고, "정렬"을 **모델이 스스로 배우게** 했다.
쪼갠다는 전제 자체를 두지 않으니 캡션 구조가 어떻든 깨지지 않는다.

> 📌 **기억할 것:** "캡션이 순서를 담고 있다"는 *왜 이 문제가 풀릴 수 있는지에 대한 근거*이지,
> *전처리 방법*이 아니다.

---

## 2. 언어모델은 사실 "다음 단어 맞히기" 기계다

파인튜닝을 이해하려면 이것부터 확실히 해야 한다.

### 토큰(token)
모델은 글자나 단어가 아니라 **토큰** 단위로 본다. 토큰은 "자주 나오는 문자 덩어리"다.

```
"The correct order is [3, 1, 4, 2]."
→ ["The", " correct", " order", " is", " [", "3", ",", " 1", ",", " 4", ",", " 2", "]", "."]
```

각 토큰은 사전(vocabulary)의 번호로 바뀐다. Qwen 모델의 사전 크기는 **약 152,000개**다.
(이 숫자는 나중에 메모리 문제에서 다시 등장한다 — 기억해두자.)

### 모델이 하는 일
언어모델은 **"지금까지의 토큰들을 보고, 다음 토큰이 무엇일지"** 확률로 답한다.

```
입력: "The correct order is [3, 1, 4,"
출력: 152,000개 토큰 각각에 대한 확률
      → " 2" : 87%
      → " 3" :  5%
      → " 1" :  3%
      → ...
```

이게 전부다. 번역도, 요약도, 우리의 순서 맞히기도 — 전부 이 "다음 토큰 맞히기"로 표현된다.

### 학습(training)이란
정답 문장을 주고, 모델이 **각 위치에서 실제 다음 토큰에 높은 확률**을 주도록 파라미터를 조정하는 것.

- **loss(손실)**: 모델이 정답 토큰에 준 확률이 낮을수록 커지는 값. "얼마나 틀렸나"의 수치
- **gradient(기울기)**: loss를 줄이려면 각 파라미터를 어느 방향으로 움직여야 하는지
- **학습**: loss 계산 → gradient 계산(backward) → 파라미터 조금 이동(optimizer step) → 반복

> 🔑 **여기가 8장의 복선입니다.** loss는 **"어느 토큰들을 채점할지"**에 따라 완전히 달라진다.

---

## 3. 이미지는 어떻게 언어모델에 들어가는가 (VLM)

우리가 쓴 모델은 **Qwen2.5-VL** — VL은 Vision-Language, 즉 **이미지도 보는 언어모델**이다.

### 원리: 이미지를 "토큰처럼" 만든다
1. 이미지를 격자로 잘라 조각(patch)으로 나눈다
2. **비전 인코더(ViT)**가 각 조각을 벡터로 바꾼다
3. 그 벡터들을 **텍스트 토큰과 같은 공간**에 밀어넣는다
4. 언어모델 입장에선 그냥 "토큰 시퀀스"일 뿐 — 일부는 글자에서, 일부는 픽셀에서 왔을 뿐

```
[이미지1 토큰 ~256개][ "Image 1" ][이미지2 토큰 ~256개][ "Image 2" ] ... [ 캡션 텍스트 ][ 지시문 ]
└─────────────────────── 전부 합쳐서 하나의 긴 토큰 시퀀스 ───────────────────────┘
```

### 실제 숫자 (우리 케이스)
smoke test에서 측정한 값:

```
seq_len = 1183 토큰
```

이 1,183개 중 **대부분(약 1,000개)이 이미지에서 온 vision 토큰**이다. 사진 4장이니까.
텍스트(캡션+지시문)는 100~200개 남짓.

> 📌 **이 비율이 나중에 재앙의 원인이 된다.** (8장)

### max_pixels — 이미지 해상도 조절 손잡이
이미지를 크게 넣을수록 vision 토큰이 많아진다 → 느려지고 메모리를 먹는다.
`max_pixels`로 "이미지당 최대 픽셀 수"를 제한해 토큰 수를 조절한다.

```python
MAX_PIXELS = int(os.environ.get("SNU_MAX_PIXELS", str(256 * 28 * 28)))
```

`28×28`은 패치 하나의 크기라, `256 * 28 * 28`은 **"이미지당 패치 256개까지"**라는 뜻이다.

---

## 4. 파인튜닝이란 무엇인가

### 사전학습 vs 파인튜닝
- **사전학습(pre-training)**: 인터넷 규모 데이터로 "언어와 세상 일반"을 배움. 수백만 달러, 수개월
- **파인튜닝(fine-tuning)**: 그 모델을 **우리 과제에 맞게 조금 더** 학습. 몇 시간, 몇 달러

비유하자면 사전학습은 **대학 교육**, 파인튜닝은 **회사 온보딩**이다.
이미 똑똑한 사람에게 "우리 회사에선 이런 양식으로 보고해"를 가르치는 것.

### 왜 파인튜닝이 필요했나
Qwen2.5-VL은 이미 이미지와 글을 이해한다. 하지만 그냥 물어보면(zero-shot):
- "순서를 `[3,1,4,2]` 형식으로만 답하라"는 걸 잘 안 지키고
- 이 과제 특유의 "캡션↔프레임 정렬" 감각이 없다

실제로 우리 smoke test에서 **파인튜닝 안 한 원본 7B 모델의 정확도는 5개 중 0개**였다.

```
scored 5 rows, untuned acc=0/5
```

파인튜닝 후: **0.4167** (즉 42%). 이게 파인튜닝의 힘이다.

---

## 5. LoRA — 1.1%만 학습해서 모델을 바꾸는 법

### 문제: 전체 파인튜닝은 너무 비싸다
7B 모델 = 파라미터 **70억 개**. 전부 학습하려면:
- 파라미터 자체 + gradient + optimizer 상태(Adam은 파라미터당 2개 더) → **메모리가 파라미터의 4~6배**
- 7B × 2바이트 × 6 ≈ **80GB 이상** → 3090(24GB)은커녕 A100 80GB도 빠듯

### LoRA의 아이디어
> "파인튜닝으로 생기는 변화량은 사실 **아주 단순한(저차원) 패턴**이더라."

원래 가중치 `W`(예: 3584×3584 = 1,280만 개)를 **얼려두고(frozen)**,
옆에 **얇은 행렬 두 개** `A`(3584×32), `B`(32×3584)만 새로 학습한다.

```
출력 = W·x  +  B·A·x
       └─고정─┘  └─학습─┘
```

- `A`, `B`의 파라미터 수: 3584×32 + 32×3584 = **229,376개** (원본의 1.8%)
- `32`가 **rank(r)** — 이 "얇기"를 정하는 값. 우리는 `r=32` 사용

### 실제 숫자 (7B 학습 로그)
```
trainable params: 95,178,752 || all params: 8,387,345,408 || trainable%: 1.1348
```

**83억 개 중 9,500만 개(1.13%)만 학습**했다. 나머지는 얼어 있다.

### 우리 설정
```python
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
```

- `target_modules`: 트랜스포머 안의 **모든 선형 계층**에 LoRA를 붙였다
  - trial1은 `q_proj`, `v_proj` 둘만 붙였다 → 표현력 부족. 그래서 우리는 7개 전부로 확대
- `lora_alpha = 2r`: LoRA 출력의 크기 조절(관례적으로 rank의 2배)
- `lora_dropout`: 과적합 방지

### 보너스: 결과물이 작다
학습 결과 저장되는 건 **LoRA 어댑터뿐 — 380MB**. 원본 7B(16GB)는 그대로 두고,
"차이분"만 들고 다니면 된다. (백업이 쉬웠던 이유)

---

## 6. 4-bit 양자화 — 16GB 모델을 6GB에 욱여넣기

### 원리
모델 가중치는 보통 숫자 하나당 16비트(2바이트)로 저장된다. 7B × 2바이트 = **약 16GB**.
이걸 **4비트**로 줄이면 **약 4~6GB**. 정밀도를 조금 희생하고 메모리를 4배 아낀다.

비유: 사진을 RAW → JPEG로 저장하는 것. 미세하게 손해 보지만 용량이 확 준다.

### 우리 설정
```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # 정규분포에 최적화된 4bit 형식
    bnb_4bit_use_double_quant=True,     # 양자화 상수까지 한 번 더 압축
    bnb_4bit_compute_dtype=torch.bfloat16,  # 계산은 16bit로 (정확도 유지)
)
```

핵심은 **`compute_dtype=bfloat16`**: **저장은 4bit, 계산은 16bit**로 되돌려서 한다.
그래서 정확도 손실이 크지 않다.

### QLoRA = 4bit 양자화 + LoRA
- 원본 가중치: **4bit로 얼려둠** (메모리 절약)
- LoRA 어댑터: **16bit로 학습** (정확도 필요)

이 조합 덕에 **3090(24GB)에서도 7B를 다룰 수 있다.** 실측:
```
7B 모델 로드 후 GPU 메모리: 6,269 MiB  (≈6GB)
```

> 📌 대회 규칙이 "최종 추론은 RTX 3090 24GB"인데, 이 덕분에 통과한다.

---

## 7. 우리 문제를 "다음 토큰 맞히기"로 번역하기

2장에서 "모델은 다음 토큰만 맞힌다"고 했다. 그럼 **순서 맞히기**를 어떻게 표현할까?

### 답: 프롬프트와 정답을 "글"로 쓴다

**프롬프트(질문):**
```
[이미지1] Image 1 [이미지2] Image 2 [이미지3] Image 3 [이미지4] Image 4
Caption: "A girl hula hoops indoors before..."
The 4 images above (Image 1 to Image 4) are frames from a single video, given in
shuffled order. The caption describes what happens in chronological order. Decide
the correct chronological order of the images and output it as a list of the four
image numbers, e.g. [3, 1, 4, 2].
```

**정답(모델이 생성해야 할 글):**
```
The correct order is [2, 3, 1, 4].
```

이렇게 하면 "순서 맞히기"가 그냥 **"이 글 다음에 올 토큰들 맞히기"**가 된다.
언어모델이 원래 할 줄 아는 일로 번역된 것이다.

### 코드
```python
INSTRUCTION = (
    'Caption: "{sentence}"\n'
    "The 4 images above (Image 1 to Image 4) are frames from a single video, "
    "given in shuffled order. The caption describes what happens in chronological "
    "order. Decide the correct chronological order of the images and output it as a "
    "list of the four image numbers, e.g. [3, 1, 4, 2]."
)

def target_text(image_order):
    return f"The correct order is {image_order}."   # str([3,1,4,2]) == "[3, 1, 4, 2]"
```

### ⚠️ 라벨 매핑 — 헷갈리면 전부 오답
여기서 조심할 게 있다. 데이터의 `Answer`와 모델이 출력하는 순서는 **서로 다른 표현**이다.

- **`Answer[k]`** = "Input_(k+1)번 사진이 **시간상 몇 번째**인가"
- **`image_order[t]`** = "시간상 t+1번째에 오는 건 **몇 번 사진**인가"

예: `Answer = [3, 1, 2, 4]`는 "1번 사진은 3번째, 2번 사진은 1번째, 3번 사진은 2번째, 4번 사진은 4번째"
→ 시간순으로 줄 세우면 **2번 → 3번 → 1번 → 4번** → `image_order = [2, 3, 1, 4]`

둘은 **서로의 역치환(inverse permutation)**이다.

```python
def image_order_from_answer(answer):          # 학습 라벨 만들 때
    return [answer.index(pos) + 1 for pos in range(1, 5)]

def answer_from_image_order(image_order):     # 제출 파일 만들 때
    answer = [0, 0, 0, 0]
    for t, img in enumerate(image_order, start=1):
        answer[img - 1] = t
    return answer
```

> 📌 이걸 거꾸로 하면 **모든 라벨이 틀린 채로 학습**된다. 조용히 망하는 종류의 버그라 무섭다.

---

## 8. ⭐ 핵심: Loss Masking — 이번 프로젝트의 전부

여기가 이번 작업에서 **가장 중요한 한 장**이다. 이거 하나로 16.8% → 40%+가 됐다.

### 상황 복습
우리 입력은 이렇게 생겼다:

```
[───────── 프롬프트 1,166개 토큰 ─────────][─ 정답 17개 토큰 ─]
 이미지 토큰 ~1000개 + 지시문 + 캡션         "The correct order is [2, 3, 1, 4]."
```

### 질문: 이 중 **어느 토큰을 채점**해야 할까?

당연히 **정답 부분(17개)만** 채점해야 한다.
"프롬프트를 재현하는 능력"은 우리가 원하는 게 아니니까. 프롬프트는 **주어지는 것**이지,
모델이 맞혀야 할 게 아니다.

### trial1이 한 실수
```python
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100   # padding만 무시
```

> `-100`은 PyTorch에서 **"이 위치는 채점하지 마"**라는 약속된 표식이다.

이 코드는 **padding만** 빼고 **전부 채점**했다. 즉:
- 이미지 토큰 1,000개를 "맞히도록" 학습
- 매번 똑같은 지시문을 "맞히도록" 학습
- 정작 중요한 정답 17개는 **전체의 1.4%**

**결과:** gradient의 98.6%가 "고정된 프롬프트 외우기"에 쓰였다.
정답을 맞히는 신호는 노이즈에 묻혔다.

### 증상 (진단이 맞았다는 증거)
| 증상 | 해석 |
|---|---|
| loss가 ~6.8에서 **plateau** | 프롬프트 재현은 금방 완벽해짐 → 더 내려갈 게 없음 |
| 예측이 `[1,2,3,4]` 28% + `[3,2,4,1]` 21% = **49%로 붕괴** | 입력을 보고 판단(conditional)하는 게 아니라, **정답의 평균적 분포(marginal)만 외움** |

즉 모델은 **"뭘 물어보든 제일 흔한 답을 뱉는"** 상태였다. 공부 안 하고 "3번 찍기"를 배운 셈.

### trial2의 수정
```python
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
completion  = C.target_text(io) + processor.tokenizer.eos_token

full        = processor(text=[prompt_text + completion], images=image_inputs, return_tensors="pt")
prompt_only = processor(text=[prompt_text],              images=image_inputs, return_tensors="pt")
plen = prompt_only["input_ids"].shape[1]     # ← 프롬프트의 실제 토큰 길이

labels = full["input_ids"].clone()
labels[:, :plen] = -100                      # ← 프롬프트+이미지 전부 채점 제외
```

### 왜 프롬프트를 **따로 한 번 더** 통과시키나?
이게 미묘한 포인트다. "프롬프트 글자 수를 세면 되지 않나?" — **안 된다.**

이미지가 **프롬프트 안에서 수백 개 토큰으로 확장**되기 때문이다. 글자 수로는 알 수 없다.
그래서 **프롬프트만 실제로 processor에 통과시켜** 확장된 후의 길이(`plen`)를 얻어야 한다.

`full`은 `prompt + completion`이고 `prompt_only`는 `prompt`뿐인데, 이미지 확장은 동일하므로
**`full`의 앞 `plen`개 = 정확히 프롬프트 부분**이다. 거기까지 `-100`.

### 검증 — 진짜 됐는지 확인하기
말로만 "고쳤다"고 하면 안 된다. 그래서 smoke test에서 **직접 셌다**:

```python
n_supervised = int((labels != -100).sum())
print(f"seq_len={inputs['input_ids'].shape[1]} supervised_tokens={n_supervised}")
```
```
seq_len=1183 supervised_tokens=17    ← 의도한 대로 정답 토큰만!
```

### 효과 (즉시 나타남)
| | trial1 | trial2 |
|---|---|---|
| loss 궤적 | ~17 → 6.8에서 **정체** | 0.48 → 0.19 → **0.07** |
| val 정확도 | 0.168 | **0.40 (2B) / 0.4167 (7B)** |

> 🎓 **이 장의 교훈:** 딥러닝에서 버그는 에러를 내지 않는다. **조용히 성능만 깎는다.**
> "학습이 돌아간다"와 "학습이 올바르다"는 완전히 다른 얘기다.
> 그래서 `supervised_tokens=17`처럼 **의도를 숫자로 확인하는 습관**이 중요하다.

---

## 9. 학습 루프의 부품들

이제 학습 코드를 한 줄씩 뜯어보자.

```python
loss = model(**inputs, labels=labels).loss / ACCUM   # 1. 순전파 + loss
loss.backward()                                       # 2. 역전파 (gradient 계산)
if micro % ACCUM == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 3. gradient 클리핑
    optim.step()      # 4. 파라미터 갱신
    sched.step()      # 5. 학습률 갱신
    optim.zero_grad() # 6. gradient 초기화
```

### 9.1 배치(batch)와 gradient accumulation
**배치**: 한 번에 여러 예제를 처리하는 것. 크면 gradient가 안정적이고 GPU를 잘 쓴다.

문제는 **메모리**. 3090(24GB)에서 7B를 배치 2로 돌리면 OOM(메모리 초과)이 난다.

**gradient accumulation**이 해법이다:
```
예제 1개 처리 → gradient 누적 (파라미터는 아직 안 건드림)
예제 1개 처리 → gradient 누적
... 8번 반복 ...
그제서야 optimizer.step()  → "배치 8"과 동일한 효과
```

- 3090(2B): `micro_batch=1 × accum=8` = **유효 배치 8**
- A100(7B): `micro_batch=8 × accum=1` = **유효 배치 8**

> 📌 **유효 배치를 8로 맞춘 이유**: 두 실험의 최적화 조건을 같게 해서 **결과를 비교 가능**하게 하려고.

`loss / ACCUM`으로 나누는 이유: 8번 더할 거니까 미리 나눠야 평균이 된다.

### 9.2 배치를 쓰려면 padding이 필요하다
예제마다 캡션 길이가 달라 시퀀스 길이가 다르다. 배치로 묶으려면 **길이를 맞춰야** 한다.

```python
def collate(batch):
    maxL = max(s.size(0) for s in seqs)
    batched = {
        "input_ids":      pad("input_ids", pad_id),    # 짧은 건 뒤에 채움(오른쪽 패딩)
        "attention_mask": pad("attention_mask", 0),    # 채운 부분은 0 → "보지 마"
        "pixel_values":   torch.cat([b["pixel_values"] for b in batch], 0),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], 0),
    }
    labels = pad("labels", -100)   # 채운 부분은 -100 → "채점하지 마"
    return batched, labels
```

**세 가지가 짝을 이룬다:**
| | 의미 |
|---|---|
| `attention_mask = 0` | 모델이 이 위치를 **쳐다보지도 않음** |
| `labels = -100` | 이 위치는 **채점 안 함** |
| `input_ids = pad_id` | 자리만 채우는 더미 토큰 |

이 셋이 맞으면 **배치로 계산한 loss는 예제를 하나씩 계산해 합친 것과 수학적으로 같다.**

> 💡 `pixel_values`를 그냥 `cat`으로 붙이는 이유: Qwen2-VL은 이미지를 `[전체 패치 수, 차원]`으로
> **평탄하게** 받고 `image_grid_thw`로 "어디부터 어디까지가 몇 번째 이미지"를 표시한다.
> 그래서 행 순서대로 이어붙이면 `input_ids` 안의 이미지 자리와 자동으로 맞아떨어진다.

### 9.3 학습률(learning rate)과 스케줄
**학습률** = 한 걸음의 보폭. 크면 빨리 가지만 헤매고, 작으면 안정적이지만 느리다.

```python
sched = get_cosine_schedule_with_warmup(optim, int(0.03 * total_optim), total_optim)
```

- **warmup(처음 3%)**: 0에서 시작해 서서히 목표 학습률까지 올림
  - 왜? 시작 직후엔 gradient가 요동친다. 처음부터 큰 보폭이면 모델이 망가진다
  - 실제 로그에서 확인 가능: `lr=1.61e-05` → `3.23e-05` → ... → `9.93e-05`(목표 1e-4에 근접)
- **cosine decay(나머지)**: 코사인 곡선을 따라 서서히 0으로 줄임
  - 왜? 막판엔 미세 조정만 해야 한다. 큰 보폭이면 좋은 답 주변을 계속 튕겨 다닌다

trial1은 **처음부터 끝까지 1e-4 고정**이었다. 이것도 개선 포인트였다.

### 9.4 gradient clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
gradient가 가끔 폭발적으로 커질 때(이상한 예제 등) 파라미터가 날아가는 걸 막는다.
"보폭이 아무리 커도 1.0을 넘지 마" — 안전벨트다.

### 9.5 gradient checkpointing
역전파를 하려면 순전파의 중간 결과들을 **다 기억**해야 한다 → 메모리 폭발.

**gradient checkpointing**: 중간 결과를 **버렸다가, 필요할 때 다시 계산**한다.
- 메모리 ↓↓ (많이 절약)
- 속도 ↓ (20~30% 느려짐)

```python
model.gradient_checkpointing_enable()
```

**시간을 주고 메모리를 사는** 거래다. 24GB 3090에서는 필수였다.

---

## 10. 검증 — 우리가 잘하고 있는지 어떻게 아는가

### 10.1 왜 학습 데이터로 채점하면 안 되나
학습에 쓴 데이터로 채점하면 **외운 걸 확인**하는 꼴이다. 시험 문제를 미리 알려주고 시험 보는 것.
그래서 데이터 일부를 **떼어놓고(held-out)** 학습에 절대 쓰지 않는다.

```python
train_df, val_df = train_test_split(df, test_size=VAL_FRAC, random_state=SEED,
                                    stratify=df["Answer"])
```

- `test_size=0.12`: 12%(약 1,145개)를 검증용으로 분리
- `random_state=42`: **같은 시드 → 항상 같은 분리**. trial1과 동일한 split을 써서 **직접 비교** 가능
- `stratify=df["Answer"]`: 24가지 정답이 train/val에 **골고루** 들어가게

> 📌 이건 대회 규칙의 "데이터 누수 금지"와도 직결된다. 우리는 **테스트셋을 학습에 일절 쓰지 않았다.**

### 10.2 채점 방식
```python
def exact_match(preds_answer, truths_answer):
    correct = sum(1 for p, t in zip(preds_answer, truths_answer) if list(p) == list(t))
    return correct / max(1, len(truths_answer))
```
4개 전부 같아야 1점. 대회 채점과 동일하게 맞췄다.

### 10.3 기준선(baseline)을 알아야 숫자가 의미를 갖는다
| 기준 | 정확도 | 의미 |
|---|---|---|
| 랜덤 찍기 | 4.2% | 1/24 |
| **다수 클래스** | **15.5%** | 무조건 `[1,2,3,4]`만 답하기 |
| trial1 | 16.8% | ← **다수 클래스와 거의 같음 = 사실상 아무것도 못 배움** |

> 🎓 **교훈:** "16.8%"라는 숫자만 보면 랜덤(4.2%)의 4배라 잘한 것 같다.
> 하지만 **올바른 기준선(15.5%)과 비교**하면 "아무것도 못 배웠다"가 드러난다.
> **기준선 없는 숫자는 의미가 없다.**

### 10.4 best checkpoint
```python
if acc > prog["best_acc"]:
    prog["best_acc"] = acc
    model.save_pretrained(best_path)     # 최고 기록을 낸 순간의 모델을 저장
```

학습이 진행될수록 계속 좋아지진 않는다(과적합). 그래서 **주기적으로 채점하고 최고점을 저장**한다.
trial1은 **맨 끝에 딱 한 번만** 평가해서, 2 에폭이 적절했는지조차 알 수 없었다.

---

## 11. 추론 — 두 가지 방법과 그 트레이드오프

학습이 끝났다. 이제 실제 테스트 데이터에 답을 내야 한다. 두 가지 방법이 있다.

### 11.1 방법 A: Greedy Generation (생성)
그냥 모델에게 답을 **쓰게** 한다.

```python
out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
m = _ORDER_RE.search(text)     # "[3, 1, 4, 2]" 를 정규식으로 뽑아냄
```

- `do_sample=False` = **greedy**: 매번 가장 확률 높은 토큰 선택 (무작위성 없음, 재현 가능)
- **빠름**: 행당 ~2.7초
- **단점**: 모델이 이상한 걸 쓰면? `[1,1,2,3]`(중복!)이나 아예 딴소리를 쓸 수도 있다
  → 그래서 파싱 실패 시 `[1,2,3,4]`로 **fallback**해야 한다 (= 그냥 찍는 것)

### 11.2 방법 B: Constrained Scoring (점수화) ⭐
발상을 뒤집는다. **모델에게 쓰게 하지 말고, 24개 후보를 채점시키자.**

```python
full_texts = [prompt_text + target_text(io) for io in ALL_ORDERS]   # 24개 후보 문장
...
comp_logits = logits[:, plen - 1:-1, :].float()     # 정답 구간의 예측만
comp_tokens = inputs["input_ids"][:, plen:]
tok_logp = torch.log_softmax(comp_logits, -1).gather(-1, comp_tokens.unsqueeze(-1)).squeeze(-1)
logps.extend(tok_logp.sum(-1).tolist())            # 후보별 총 log 확률
best = int(max(range(len(logps)), key=lambda i: logps[i]))   # 가장 그럴듯한 후보 채택
```

**"모델아, 이 24개 문장 각각이 얼마나 그럴듯하니?" → 가장 높은 걸 선택.**

**장점:**
- **항상 유효한 순열** (24개 후보가 전부 유효하니까). 파싱 실패 = 0
- 정확도가 보통 조금 더 높다 (fallback으로 찍는 경우가 사라지니까)

**설계상 예쁜 점:** 24개 후보 문장은 **토큰 길이가 전부 같다**.
왜냐면 숫자 1~4는 전부 한 글자라, `[3, 1, 4, 2]`나 `[2, 4, 1, 3]`이나 토큰 수가 동일하기 때문이다.
게다가 프롬프트도 동일하다. 그래서:
- **padding 불필요**
- 프롬프트 길이 `plen`이 **상수** → 정답 구간을 자를 때 인덱스 계산이 깔끔

**단점: 느리다.** 7B에서 **행당 16.5초**.
- 이유: 후보 24개가 **같은 이미지를 24번 다시 인코딩**한다. 비전 인코더가 병목
- 819행 × 16.5초 = **3.8시간**

### 11.3 우리의 선택
| | greedy | constrained |
|---|---|---|
| 819행 소요 | **~37분** | ~3.8시간 |
| 정확도 | 기준 | 몇 %p 높음 |

이번엔 시간·비용 제약 때문에 **greedy**로 제출 파일을 만들었다.
마침 **우리 val 0.4167도 greedy로 측정한 값**이라 일관성도 있다.

> 🔧 **미래 개선:** "비전 인코딩을 1번만 하고 24개 후보가 재사용"하게 만들면
> constrained도 ~1시간 내로 가능하다. (Qwen의 위치 인코딩 처리 때문에 구현이 까다로워 미뤘다)

### 11.4 안전장치: 행마다 저장 + 이어하기
3.8시간짜리 작업이 99%에서 끊기면? 전부 날아간다. 그래서:

```python
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    done = set(prev["Id"].astype(str))      # 이미 한 건 건너뛰기
...
    w.writerow([row["Id"], str(C.answer_from_image_order(io))])
    f.flush()                                # 한 행 끝날 때마다 즉시 디스크에 씀
```

**결과가 나올 때마다 바로 파일에 쓰고**, 다시 실행하면 **하던 데서 이어간다.**

---

## 12. 결과 읽는 법 — loss가 낮은데 정확도가 낮은 이유

학습 로그를 보면:
```
optim_step 20  loss=0.1918
optim_step 300 loss=0.1344
```

loss가 0.13! 거의 0에 가깝다. 그럼 정확도도 90%쯤 되어야 하는 것 아닌가?
**실제로는 41.67%였다.** 왜?

### 이유: loss는 "토큰 평균"이다
정답 문장을 다시 보자:
```
"The correct order is [3, 1, 4, 2]."
  └────── 대부분이 매번 똑같은 boilerplate ──────┘   └ 진짜 정보는 숫자 4개뿐 ┘
```

17개 supervised 토큰 중:
- `The`, `correct`, `order`, `is`, `[`, `,`, `]`, `.` ... → **몇 스텝이면 100% 맞힘** (loss ≈ 0)
- 실제 정보를 담은 **숫자 4개** → 이게 어려운 부분

**17개 토큰의 평균 loss**는 13개의 쉬운 토큰이 끌어내린다. 그래서 낮게 나온다.
게다가 exact match는 **숫자 4개를 전부** 맞혀야 한다.

> 🎓 **교훈:** **loss는 학습이 "돌아가는지" 확인하는 용도**,
> **정확도(우리의 실제 지표)는 별도로 재야 한다.** 둘을 혼동하면 안 된다.

### 더 좋은 신호: 예측 분포
정확도 숫자 하나보다 더 많은 걸 말해주는 게 있다. **모델이 뭘 답하고 있는지의 분포**다.

| | trial1 | **trial2 (우리)** |
|---|---|---|
| `[1,2,3,4]` 비율 | 28% | **20.1%** |
| 2위 답 비율 | 21% (`[3,2,4,1]`) | 7% |
| 상위 2개 합계 | **49%** 😱 | 27% |
| 실제 train의 `[1,2,3,4]` 비율 | 15.5% | 15.5% |

- **trial1**: 두 답이 절반을 차지 → "입력을 안 보고 흔한 답만 뱉음"의 명백한 증거
- **trial2**: `[1,2,3,4]`가 20.1%로 **실제 비율 15.5%에 근접**하고, 나머지는 여러 순열에 고루 퍼짐
  → **입력별로 다르게 판단하고 있다**는 증거

> 🎓 **교훈:** 정확도는 "얼마나 맞았나"만 말해준다.
> **분포는 "어떻게 틀리고 있나"를 말해주고**, 그게 다음에 뭘 고칠지 알려준다.

---

## 13. 실전에서 부딪힌 것들

교과서엔 안 나오지만 실제로 시간을 잡아먹은 것들.

### 13.1 OOM (메모리 초과)
**trial1의 증상:** 배치 2만 해도 24GB 3090에서 터짐.

**원인:** logits 텐서가 `배치 × 시퀀스길이 × 어휘크기`로 커진다.
Qwen의 어휘는 **152,000개**(2장에서 기억해두라던 그 숫자!).

```
8 × 1183 × 152064 × 2바이트 ≈ 2.9 GB   ← logits "하나"가 이만큼
```

**대응:** `max_pixels`로 시퀀스 줄이기 + gradient checkpointing + gradient accumulation

### 13.2 GPU가 0%인데 5분째 멈춤 (메모리 스래싱)
**증상:** 학습 중 검증이 시작되자 GPU 사용률 **0%**, CPU는 210%, 메모리 **71GB**, 5분간 무응답.

**진단 과정:**
1. 죽었나? → `ps`로 보니 **CPU 210% 사용 중** = 살아있고 뭔가 열심히 하는 중
2. 로그 확인 → bitsandbytes 연산 로그 = **생성은 돌고 있음**
3. 메모리 71GB → PyTorch가 학습용으로 **예약해둔 메모리가 조각나 있어**, 생성이 KV 캐시를
   할당하려다 계속 실패·재시도 → GPU는 놀고 메모리 관리만 하느라 CPU만 탐

**해결:**
```python
gc.collect(); torch.cuda.empty_cache()    # 예약 메모리 반환 → 조각 정리
```

> 🎓 **교훈:** "멈춘 것 같다"고 바로 죽이면 안 된다. **GPU/CPU/로그/메모리를 각각 확인**하면
> "죽음"인지 "느림"인지 구분된다. 실제로 이건 느린 거였고, **기다렸더니 0.4167이 나왔다.**

### 13.3 로컬 맥에서 학습이 안 된다
Apple M5에는 CUDA가 없다 → **GPU를 빌렸다**(Vast.ai).

- 2B: RTX 3090 24GB
- 7B: A100 80GB (128 CPU!)

> 📌 **규칙 확인:** 대회의 "RTX 3090" 제약은 **최종 추론에만** 적용된다.
> 학습은 아무 GPU나 써도 된다. 우리 결과물은 4bit 베이스 + 380MB 어댑터라 3090에서 잘 돌아간다.

**실전 팁들:**
- 데이터 4.1GB(파일 3만8천 개)는 `rsync`보다 **tar 스트리밍**이 훨씬 빠름
  (파일마다 왕복하지 않고 한 줄기로 밀어넣음)
  ```bash
  tar cf - -C snuaichallenge_data . | ssh box 'tar xf - -C /workspace/data/...'
  ```
- 학습은 반드시 **tmux**로: SSH가 끊겨도 계속 돈다
- 데이터 전송과 모델 다운로드를 **병렬로** 돌려 시간 절약

---

## 14. 전체 흐름 다시 보기

이제 전체를 한눈에 꿰어보자.

```
[준비]
 train.csv (9,535행)
   └─ seed 42로 88% / 12% 분리 → 학습용 / 검증용(절대 학습에 안 씀)

[한 예제가 모델에 들어가는 과정]
 사진 4장 + 캡션
   └→ 프롬프트로 조립: [이미지토큰 ~1000개] + "Image 1~4" + 캡션 + 지시문
   └→ 정답 문자열 붙임: "The correct order is [2, 3, 1, 4]."
   └→ ⭐ 프롬프트 부분 전부 -100 (채점 제외), 정답 17개 토큰만 채점
   └→ 총 1,183 토큰

[학습]
 4bit로 얼린 7B 모델 + 1.13%짜리 LoRA 어댑터
   └→ 순전파 → loss(정답 17토큰만) → 역전파 → gradient 누적(유효 배치 8)
   └→ cosine 스케줄로 학습률 조절, clipping으로 안전
   └→ 150 스텝마다 검증 → 최고점이면 어댑터 저장

[검증]
 held-out 1,145행 → greedy 생성 → Answer로 변환 → exact match
   └→ 기준선(15.5%)과 비교해서 의미 판단

[추론]
 test.csv 819행 → best 어댑터 로드 → greedy 생성(또는 24후보 점수화)
   └→ image_order → Answer 포맷으로 역변환
   └→ 행마다 즉시 CSV 저장 (중단돼도 이어하기 가능)

[결과]
 819행 제출 파일, 유효하지 않은 순열 0개, 분포 건강
```

### 최종 성적표
| | 정확도 | 비고 |
|---|---|---|
| 랜덤 | 4.2% | |
| 다수 클래스 바닥 | 15.5% | **진짜 기준선** |
| trial1 | 16.8% | loss masking 버그 |
| trial2 — 2B | **40.0%** | masked loss 수정 |
| trial2 — 7B | **41.67%** | 0.28 에폭에서 조기 중단 (아직 상승 중이었음) |

---

## 마무리: 이번 프로젝트가 주는 교훈 5가지

1. **딥러닝 버그는 에러를 내지 않는다. 조용히 성능만 깎는다.**
   trial1은 아무 에러 없이 완벽하게 3.88시간 학습을 마쳤다. 결과가 쓰레기였을 뿐.

2. **"의도"를 숫자로 확인하라.**
   "loss masking 고쳤다"는 말은 무의미하다. `supervised_tokens=17`을 **찍어봐야** 안다.

3. **기준선 없는 숫자는 의미가 없다.**
   16.8%는 랜덤의 4배지만, 다수 클래스(15.5%)와 비교하면 아무것도 못 배운 것이다.

4. **문제를 어떻게 표현하느냐가 모델 크기보다 중요할 수 있다.**
   같은 2B 모델이, loss 채점 범위만 바꿨더니 16.8% → 40%가 됐다.

5. **정확도보다 분포를 봐라.**
   "두 답이 49%"라는 관찰이, 숫자 하나보다 훨씬 많은 것을 알려줬다.

---

## 더 읽을거리 (이 문서 다음)

| 문서 | 내용 |
|---|---|
| `docs/trial2-strategy-and-implementation.md` | 같은 내용의 **기술 문서 버전** (전략·설계 근거 중심) |
| `trial2/SUMMARY.md` | 작업 로그 (무엇을 했나) |
| `docs/superpowers/specs/2026-07-12-frame-ordering-trial2-design.md` | 최초 설계 스펙 |
| `trial2/common.py` | 이 문서의 7·8·11장 코드가 전부 여기 |
| `trial2/train.py`, `trial2/train_batched.py` | 9장의 학습 루프 |

### 직접 해볼 것
```bash
# 1. masked loss가 진짜 되는지 두 눈으로 확인하기 (가장 추천)
python smoke_test.py     # supervised_tokens=17 이 찍히는지 보기

# 2. 일부러 망가뜨려 보기 — labels[:, :plen] = -100 줄을 지우면?
#    → supervised_tokens가 1183으로 뛰고, 그게 바로 trial1의 상태다
```
