# PaliGemma 실험 계획

## 결론

**PaliGemma/PaliGemma 2 같은 VLM**을 사용한다. 이 대회는 이미지 4장과 문장을 함께 봐야 하므로 텍스트 전용 Gemma보다 PaliGemma가 문제 구조에 맞다.

다만 PaliGemma는 보통 `image + text -> text` 형태의 단일 이미지 입력 모델로 쓰인다. 따라서 4개 프레임을 각각 따로 넣는 대신, `Input_1`~`Input_4`를 라벨이 박힌 2x2 contact sheet 한 장으로 합쳐서 입력한다.

권장 시작점:

- `google/paligemma-3b-pt-448`
- 2x2 contact sheet, 전체 448x448
- 각 칸에 `Image 1`, `Image 2`, `Image 3`, `Image 4` 라벨 표시
- 출력은 image order 형식: `[3, 1, 4, 2]`
- 제출 `Answer`는 image order의 inverse로 변환

`pt` 모델을 권장하는 이유는 PaliGemma model card가 pretrained 모델을 특정 태스크에 fine-tuning해서 transfer하는 용도로 설명하기 때문이다. `mix` 모델은 빠른 zero-shot/interactive sanity에는 쓸 수 있지만, 최종 학습은 `pt`가 더 정석이다.

## 왜 448인가

224 contact sheet는 4분할하면 각 프레임이 너무 작다. 448x448 contact sheet는 각 프레임이 대략 224x224라서 PaliGemma 224 단일 이미지 입력과 비슷한 시각 정보를 유지한다. 896은 더 좋을 수 있지만 3090 24GB와 추론 시간 리스크가 커서 1차 실험 대상은 아니다.

## 학습 방식

1. train row에서 4개 이미지를 읽는다.
2. 2x2 contact sheet를 만든다.
3. `Answer`를 image order로 바꾼다.
   - 대회 `Answer`: 각 `Input_i`가 원본에서 몇 번째인지
   - 학습 출력 image order: 시간 순서대로 어떤 `Input_i`가 오는지
4. 프롬프트:

```text
caption: <Sentence>
The image is a 2x2 grid of four shuffled video frames labeled Image 1 to Image 4.
The caption describes the video in chronological order.
Return only the chronological order as a Python list of image numbers.
```

5. completion:

```text
[3, 1, 4, 2]
```

6. loss는 prompt 토큰을 `-100`으로 mask하고 completion 토큰에만 건다.
7. LoRA/QLoRA로 학습한다.
8. inference는 greedy generation만 믿지 말고 24개 순열 completion의 likelihood를 비교한다.

## 리스크

- PaliGemma 3B는 Qwen2.5-VL-7B보다 작아서 reasoning 성능은 낮을 수 있다.
- 4프레임을 한 장으로 합치면 spatial label은 명확하지만, 각 프레임 디테일은 줄어든다.
- PaliGemma가 chat template 중심 모델이 아니므로 Qwen trial2 코드를 그대로 복사하면 안 된다.
- Hugging Face에서 Google license gate 승인이 필요하다. 원격 GPU에서 미리 `huggingface-cli login` 또는 `HF_TOKEN` 설정이 필요할 수 있다.

## 첫 실험 순서

1. contact sheet 생성 함수만 로컬에서 unit-level 확인
2. 원격 GPU에서 model/processor load smoke test
3. 50~200 step overfit/short train
4. validation split exact match
5. constrained scoring inference 속도 측정
6. 3090 24GB에서 로드/추론 가능성 확인

## 환경 변수 초안

```bash
export SNU_DATA_DIR=/workspace/data/snuaichallenge_data
export SNU_MODEL=google/paligemma-3b-pt-448
export SNU_ADAPTER_DIR=/workspace/outputs/paligemma_lora
export SNU_CONTACT_SHEET_SIZE=448
export SNU_EPOCHS=2
export SNU_LR=2e-4
export SNU_LORA_R=16
```

