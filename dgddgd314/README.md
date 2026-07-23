# dgddgd314 SNU AI Challenge workspace

This folder is for the frame-ordering track work owned by `dgddgd314`.

Current direction:
- Use PaliGemma 2 with true multi-image inputs: `images=[[img1, img2, img3, img4]]`, not a 2x2 contact sheet for the main path.
- Default target is `google/paligemma2-10b-pt-224`; `run_paligemma_overnight.sh` downloads it to `SNU_MODEL_DIR` when the local model folder is missing.
- Keep the proven trial3 recipe: clean train/val split, answer-token-only `suffix` labels, presentation-shuffle augmentation, QLoRA, validation checkpointing, and constrained 24-permutation scoring with TTA.
- Local machine work should focus on data audit, validation, packaging, and remote GPU run orchestration. Training/inference speed and VRAM checks need a CUDA box.

Key commands:
- `python scripts/clean_data.py --data-dir <data_dir> --out-dir data_work`
- `python scripts/test_perms.py`
- `python scripts/paligemma_smoke.py --data-dir <data_dir> --no-load-model`
- `python scripts/paligemma_train_skeleton.py --data-dir <data_dir> --epochs 1`
- `python scripts/paligemma_infer.py --data-dir <data_dir> --adapter-dir <adapter> --tta-k 3 --prior-alpha 0.0 --dump-scores outputs/val_scores.jsonl --out outputs/submission.csv`
- `python scripts/validate_submission.py --data-dir <data_dir> --submission outputs/submission.csv`
