# PaliGemma Plan

Use PaliGemma 2 with true multi-image input for the frame-ordering task.

## Default Experiment

- Model: `google/paligemma2-10b-pt-224`
- Input: four separate images per sample, passed as `images=[[img1, img2, img3, img4]]`
- Training labels: processor `suffix=[completion]`, so loss is applied to answer tokens only
- Decision rule: constrained 24-way candidate likelihood scoring
- Robustness: presentation-shuffle augmentation during training and `SNU_TTA_K` presentation TTA during inference
- Data: train-only black/duplicate-frame cleaning into `data_work/train_clean.csv` and `data_work/clean_val.csv`

## Why Not Contact Sheet As Main Path

Contact sheets compress four frames into one image and lose detail. Official Transformers PaliGemma docs support multiple images per sample via nested image lists, so the main path should keep frames separate. Contact sheet code remains available only for debugging or fallback.

## Run Order

```bash
python scripts/clean_data.py --data-dir "$SNU_DATA_DIR" --out-dir data_work
python scripts/test_perms.py
python scripts/paligemma_smoke.py --data-dir "$SNU_DATA_DIR" --no-load-model
python scripts/paligemma_train_skeleton.py --data-dir "$SNU_DATA_DIR" --epochs 1
python scripts/paligemma_infer.py --data-dir "$SNU_DATA_DIR" --adapter-dir outputs/paligemma_lora/best --tta-k 3 --out outputs/submission.csv
python scripts/validate_submission.py --data-dir "$SNU_DATA_DIR" --submission outputs/submission.csv
```

## Next Levers

1. Compare 3B vs 10B with the exact same multi-image pipeline.
2. Compare 224 vs 448 only after 224 is stable; four 448 images are much heavier.
3. Re-rank checkpoints using real 24-way+TTA clean-val scoring.
4. If language LoRA + full projector training is weak, add vision LoRA on the last 2 vision blocks.
