# Candidate1 A1: punctuation-based event hints

A1 keeps the complete original `Sentence` and appends a variable-length list of
approximate event hints split only at commas and semicolons. The prompt does not
describe the punctuation mechanics. It tells the model that the hints are
approximate, may not correspond one-to-one with the four images, and should be
ignored in favor of the original caption when ambiguous.

The model architecture, completion target, completion-only loss, validation
split, and inference decoder are unchanged from Candidate1 v3. The saved
`run_config.json` records `caption_prompt.mode=punctuation`, so validation and
inference reproduce the training prompt exactly.

## Prompt shape

```text
Original caption:
<complete Sentence>

Approximate event hints:
[Event 1] <first candidate>
[Event 2] <second candidate>
...

The event hints are approximate and may not correspond one-to-one with the four
images. Use the original caption when the hints are ambiguous.

The caption describes events in chronological order...
```

## Tiny smoke test

```powershell
$env:PYTHONPATH="src"
python -m snu_ordering.candidate1.tiny_overfit `
  --config configs\candidate1-a1.json `
  --train-csv C:\Project\data\train.csv `
  --image-root C:\Project\data\train `
  --output-dir runs\candidate1-a1-tiny `
  --base-model C:\Project\models\Qwen2-VL-2B-Instruct `
  --processor C:\Project\models\Qwen2-VL-2B-Instruct `
  --local-files-only
```

## Controlled training run

Use a new output directory; do not resume an A0 checkpoint with an A1 prompt.

```powershell
$env:PYTHONPATH="src"
python -m snu_ordering.candidate1.train `
  --config configs\candidate1-a1.json `
  --train-csv C:\Project\data\train.csv `
  --image-root C:\Project\data\train `
  --output-dir runs\candidate1-a1 `
  --base-model C:\Project\models\Qwen2-VL-2B-Instruct `
  --processor C:\Project\models\Qwen2-VL-2B-Instruct `
  --local-files-only
```

For a 1,200-step screening run, add `--max-steps 1200`. Keep all other settings
identical to the A0 run.
