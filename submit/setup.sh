#!/usr/bin/env bash
# 환경 구성: 의존성 설치 + 베이스 모델(Qwen3-VL-32B-Instruct) 사전 다운로드.
# GPU 박스(vastai pytorch 이미지 등)에서 한 번 실행한다.
#   bash setup.sh
set -euo pipefail

# 1) torch — 학습·검증에 사용한 정확한 빌드. 환경의 CUDA에 맞는 빌드를 쓸 것.
#    (이미 적절한 torch가 있으면 이 줄은 건너뛰어도 된다.)
pip install -q "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu130 || \
  echo "torch 사전 설치 건너뜀 — 환경의 기존 torch 사용"

# 2) 나머지 의존성(버전 고정)
pip install -q -r requirements.txt

# 3) 베이스 모델 사전 다운로드(약 65GB). 오프라인 실행을 위해 로컬 캐시에 받는다.
python - <<'PY'
from huggingface_hub import snapshot_download
import os
m = os.environ.get("SNU_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
print("downloading base model:", m)
snapshot_download(m)
print("done")
PY

pip freeze | grep -Ei "transformers|peft|torch==|qwen-vl|accelerate|bitsandbytes" \
  > versions.lock.txt || true
echo "--- setup 완료. 고정 버전: versions.lock.txt"
echo "다음: (1) 데이터 준비  (2) python download_weights.py  또는  전체 재학습"
