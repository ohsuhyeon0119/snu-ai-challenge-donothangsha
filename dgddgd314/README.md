# dgddgd314 SNU AI Challenge workspace

This folder is for the frame-ordering track work owned by `dgddgd314`.

Current direction:
- Do not use plain Gemma 7B as the main model: it is text-only, while the task requires image + text reasoning.
- If staying in the Google model family, use a vision-language Gemma-family model such as PaliGemma/Gemma 3, or treat plain Gemma only as a text decoder behind a separate image encoder.
- Keep the proven recipe from the sibling `ohsuhyeon/trial2` work: answer-token-only loss masking, QLoRA, validation checkpointing, and constrained 24-permutation scoring.
- Local machine work should focus on data audit, validation, packaging, and remote GPU run orchestration. Training/inference speed and VRAM checks need a CUDA box.

Key docs:
- [docs/PLAN.md]
- [docs/REMOTE_GPU_RUNBOOK.md]

Utility scripts:
- `python scripts/validate_submission.py --data-dir <data_dir> --submission <csv>`
- `python scripts/audit_data.py --data-dir <data_dir> --out outputs/data_audit.csv`

