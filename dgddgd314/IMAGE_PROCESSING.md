# Image Processing Notes

이미지 처리 관련 코드는 항상 남긴다. PaliGemma 실험에서는 4개 프레임을 단일 이미지 입력으로 바꾸는 과정이 모델 성능과 재현성에 직접 영향을 주므로, 생성 방식과 산출물을 추적한다.

현재 방식:

- `Input_1`~`Input_4`를 2x2 contact sheet 한 장으로 합친다.
- 각 칸 상단에 `Image 1`~`Image 4` 라벨을 박는다.
- 기본 contact sheet 크기는 448x448이다.
- 각 프레임은 비율을 유지해 칸 안에 넣고, EXIF orientation을 반영한다.

관련 코드:

- `src/snu_frame_ordering/contact_sheet.py`
- `scripts/paligemma_make_sheet.py`
- `scripts/audit_data.py`

운영 규칙:

- contact sheet 생성 로직을 임시 코드로 두지 말고 repo에 유지한다.
- 이미지 audit 결과는 `outputs/`에 저장한다. `outputs/`는 gitignore 대상이지만, 실험 기록에는 파일명과 생성 명령을 남긴다.
- contact sheet 크기, 라벨 방식, crop/resize 정책을 바꾸면 `PALIGEMMA_PLAN.md`와 이 파일에 이유를 기록한다.
- 학습/추론에서 사용한 이미지 전처리 방식은 보고서에 그대로 옮길 수 있어야 한다.

