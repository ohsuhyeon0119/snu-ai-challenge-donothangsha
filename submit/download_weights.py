"""
최종 모델 가중치(LoRA 어댑터) 자동 다운로드 스크립트.

GitHub 100MB 파일 제한 때문에 어댑터(약 1.1GB)는 Google Drive에 보관하며,
이 스크립트가 gdown으로 내려받아 weights/ 아래에 풀어 놓는다.

  python download_weights.py

기본 대상: 최종 제출 모델(Public LB 0.92146)의 vision-LoRA 어댑터.
필요 시 --stage1 로 중간 체크포인트(SFT+listwise 통합, LB 0.90924)도 받는다.

※ 제출자 안내: 아래 FILE_ID 를 실제 Google Drive 파일 ID로 채워야 한다.
  (Drive에서 파일 우클릭 → 링크 복사 → /d/<이 부분>/view 의 <이 부분>이 ID)
"""
import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

# ── 제출자가 채울 값 ────────────────────────────────────────────────
# 최종 vision-LoRA 어댑터 (LB 0.92146). tar.gz 안에 adapter_model.safetensors 등 포함.
FINAL_FILE_ID = "1PA8IpD8Z552WIbgMZKEYDkRi9N4rr1tv"   # vision_best_ckpt1288.tar.gz (LB 0.92146)
# (선택) 중간 통합 모델 (LB 0.90924). 처음부터 재학습 대신 stage-2만 재현할 때의 시작점.
STAGE1_FILE_ID = ""  # ckpt2218.tar.gz  (미제공 — 필요 시 채움)
# ────────────────────────────────────────────────────────────────────

WEIGHTS = Path("weights")


def fetch(file_id, out_name, dest_subdir):
    if not file_id or file_id.startswith("여기에"):
        sys.exit(f"[오류] {out_name}: download_weights.py 상단의 FILE_ID를 "
                 f"실제 Google Drive 파일 ID로 채워 주세요.")
    WEIGHTS.mkdir(exist_ok=True)
    tgz = WEIGHTS / out_name
    print(f"다운로드: {out_name} (gdown)")
    subprocess.run([sys.executable, "-m", "gdown", file_id, "-O", str(tgz)],
                   check=True)
    dest = WEIGHTS / dest_subdir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"압축 해제 → {dest}")
    with tarfile.open(tgz) as t:
        t.extractall(dest)
    print(f"완료: {dest}  (파일: {[p.name for p in dest.iterdir()]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true",
                    help="중간 통합 모델(LB 0.90924)도 받는다(선택)")
    args = ap.parse_args()

    fetch(FINAL_FILE_ID, "vision_best.tar.gz", "vision_best")
    if args.stage1:
        fetch(STAGE1_FILE_ID, "ckpt_unified.tar.gz", "ckpt_unified")

    print("\n최종 추론:")
    print("  SNU_DATA_DIR=<데이터경로> SNU_ADAPTER_DIR=weights/vision_best \\")
    print("    SNU_SPLIT=test SNU_TTA_K=3 SNU_PRIOR_ALPHA=0.5 \\")
    print("    SNU_OUT=submission.csv python infer.py")


if __name__ == "__main__":
    main()
