# Image Processing Notes

Main path: true PaliGemma multi-image input.

For each row, the four frame files are opened as RGB PIL images and passed to the processor as one sample:

```python
processor(images=[[img1, img2, img3, img4]], text=[prompt], suffix=[answer])
```

The image order inside the nested list defines the displayed labels used by the prompt:

- first image -> Image 1
- second image -> Image 2
- third image -> Image 3
- fourth image -> Image 4

Presentation shuffle augmentation changes that nested-list order with `SIGMAS`; labels are remapped into slot-space with `slot_order`.

## Current Defaults

- Main model: `google/paligemma2-10b-pt-224`
- Override with `SNU_MODEL` for 3B smoke tests or 448 experiments.
- Main inference: 24-candidate likelihood scoring with `SNU_TTA_K` presentation TTA.
- Contact sheet utilities remain only as fallback/debug tooling.

## Relevant Files

- `src/snu_frame_ordering/paligemma_common.py`
- `src/snu_frame_ordering/orders.py`
- `scripts/paligemma_smoke.py`
- `scripts/paligemma_train_skeleton.py`
- `scripts/paligemma_infer.py`
- `scripts/clean_data.py`
