# Caption Structure EDA

This analysis uses only `train.csv` captions. It does not inspect test labels or
images, and it does not force captions into exactly four events.

## Extractor

`snu_ordering.caption_structure` preserves the original caption and extracts a
variable number of candidate event segments connected by confidence-aware
relations:

- `NEXT`: then, next, afterward, finally, followed by
- `BEFORE_AFTER`: before, after
- `OVERLAP`: while, as, meanwhile
- `STRONG`: a semicolon without an explicit discourse marker
- `WEAK`: a bare comma

Composite markers such as `; then,` are treated as one boundary. Bare commas
have confidence `0.35`; the default confident-event report uses boundaries with
confidence at least `0.7`.

## Full training-set result

Command:

```powershell
$env:PYTHONPATH="src"
python -m snu_ordering.caption_eda --train-csv C:\Project\data\train.csv
```

The 9,535 training captions produced:

| Metric | All boundaries | Confidence >= 0.7 |
|---|---:|---:|
| Mean candidate events | 3.084 | 2.516 |
| Median candidate events | 3 | 3 |
| Exactly 1 | 24.97% | 28.14% |
| Exactly 2 | 12.51% | 17.98% |
| Exactly 3 | 15.70% | 30.71% |
| Exactly 4 | 28.13% | 20.64% |
| 5 or more | 18.69% | 2.54% |

Boundary prevalence by sentence:

| Boundary type | Sentences | Rate |
|---|---:|---:|
| `NEXT` | 5,960 | 62.51% |
| `BEFORE_AFTER` | 793 | 8.32% |
| `OVERLAP` | 4,423 | 46.39% |
| `STRONG` | 152 | 1.59% |
| `WEAK` | 3,972 | 41.66% |

The result supports treating punctuation as soft structure: even after temporal
connectives are added, only 20.64% of captions have exactly four high-confidence
event candidates. The next modeling step should therefore preserve the full
caption and append variable-length event hints rather than truncate, pad, merge,
or split every caption to four events.

`No_ordering=True` rates were similar across confident-event buckets (13.2% to
15.8%), so event-count buckets should be used for error analysis rather than as
a direct shortcut for predicting the identity permutation.

## Auditable outputs

The CLI can save both the aggregate JSON and one row per caption. The row-level
CSV includes event counts, bucket labels, extracted segments, typed boundaries,
and per-relation counts.

```powershell
$env:PYTHONPATH="src"
python -m snu_ordering.caption_eda `
  --train-csv C:\Project\data\train.csv `
  --output-json outputs\caption_eda.json `
  --output-rows-csv outputs\caption_buckets.csv
```

Use `--limit` for a bounded smoke test and `--confidence-threshold` for a
threshold sensitivity analysis.
