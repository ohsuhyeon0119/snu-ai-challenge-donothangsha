# SNU AI Challenge — Hermes Development Brief

## 1. Purpose

Use this document as the implementation plan for the SNU AI Challenge project.

The task is to predict the correct temporal order of four shuffled video frames using:

- one text caption (`Sentence`)
- four input frame images (`Input_1` to `Input_4`)

The target is a permutation such as:

```text
[2, 4, 3, 1]
```

This means:

- `Input_1` is the 2nd frame in the original sequence
- `Input_2` is the 4th frame
- `Input_3` is the 3rd frame
- `Input_4` is the 1st frame

The competition metric is **Exact Match Accuracy**. A prediction is correct only when all four positions match exactly.

---

## 2. Competition Constraints

All implementation decisions must follow these constraints:

- Use Python.
- Do not use external training data.
- Do not use commercial external APIs during training or inference.
- Use only one final model. Model ensembling is not allowed.
- Model compression methods such as LoRA and quantization are allowed.
- The final model must run offline.
- The final model must run on a single NVIDIA RTX 3090 with 24 GB VRAM.
- Full test-set inference must finish within 24 hours.
- Final code should use relative paths.
- Training and inference code must eventually be available as `.py` files.
- The final model code and weights together must not exceed 80 GB.
- Before using any pretrained model, verify that its weights were publicly released on or before May 31, 2026.
- Do not inspect or manually label the test set in ways that could cause data leakage.

Do not combine predictions from multiple models in the final submission.

---

## 3. Core Modeling Decision

Treat this task as a **24-class classification problem** instead of generating a free-form text answer.

There are exactly:

```text
4! = 24
```

valid permutations.

Create a fixed mapping between the 24 permutations and class IDs.

Example:

```python
import itertools

PERMUTATIONS = list(itertools.permutations([1, 2, 3, 4]))

PERM_TO_CLASS = {
    permutation: index
    for index, permutation in enumerate(PERMUTATIONS)
}

CLASS_TO_PERM = {
    index: permutation
    for index, permutation in enumerate(PERMUTATIONS)
}
```

Training target:

```python
answer = (2, 4, 3, 1)
target_class = PERM_TO_CLASS[answer]
```

Inference output:

```python
predicted_class = logits.argmax(dim=-1).item()
predicted_answer = list(CLASS_TO_PERM[predicted_class])
```

This design is preferred because:

- every prediction is always a valid permutation
- no text parsing is required
- the training objective matches Exact Match Accuracy
- invalid outputs such as `[1, 1, 3, 4]` cannot occur

---

# 4. Shared Development Foundation

Before implementing the three candidate models, build one shared data and evaluation pipeline.

## 4.1 Required shared modules

Recommended structure:

```text
src/
├── data/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── split.py
├── models/
│   ├── permutation.py
│   ├── qwen_classifier.py
│   ├── siglip_transformer.py
│   └── structured_decoder.py
├── training/
│   ├── train.py
│   ├── losses.py
│   └── metrics.py
├── inference/
│   ├── predict.py
│   └── submission.py
└── utils/
    ├── config.py
    ├── seed.py
    └── logging.py

configs/
├── qwen_24class.yaml
├── siglip_transformer.yaml
└── qwen_structured.yaml

notebooks/
├── data_analysis.ipynb
└── smoke_test.ipynb
```

Do not move competition data into the Git repository.

## 4.2 Data validation

Check the following using the training set only:

- all `Answer` values are valid permutations
- distribution of the 24 permutation classes
- proportion of `No_ordering=True`
- class imbalance for `[1, 2, 3, 4]`
- missing or corrupted image files
- black or nearly black frames
- highly similar or duplicated frames within a sample
- very short or unclear captions
- samples that may be noisy or ambiguous

Do not automatically delete noisy samples.

Instead, create a sample-quality score or sample weight.

Example:

```text
normal sample                   weight = 1.0
slightly ambiguous sample       weight = 0.7
black or duplicated frame       weight = 0.4
```

Store the reason for each reduced weight so it can be analyzed later.

## 4.3 Validation split

Do not rely only on a naive random split.

Recommended procedure:

1. Compute image similarity using perceptual hash or pretrained embeddings.
2. Group highly similar samples.
3. Ensure similar samples do not appear in both training and validation sets.
4. Preserve the 24-class distribution as much as possible.
5. Preserve the `No_ordering` ratio.

Recommended initial split:

```text
train: 85%
validation: 15%
```

Use a fixed seed and save the split IDs to disk.

## 4.4 Metrics

Record all of the following:

- Exact Match Accuracy
- Pairwise Accuracy
- Accuracy for `[1, 2, 3, 4]`
- Accuracy for non-identity permutations
- Per-class accuracy for all 24 classes
- confusion matrix
- inference time per sample
- peak GPU memory

Also run the following ablations:

- caption only
- images only
- caption plus images

This will reveal how much each modality contributes.

---

# 5. Candidate Model 1 — Main Model

## Qwen Vision-Language Model + QLoRA + 24-Class Head

### Priority

This is the primary model to develop first.

Use an eligible Qwen vision-language checkpoint that:

- was released before the competition cutoff
- supports multiple images
- fits the RTX 3090 deployment constraint

Before implementation, verify model eligibility.

## 5.1 Architecture

```text
Sentence
  +
Input_1 image
Input_2 image
Input_3 image
Input_4 image
  ↓
Qwen vision-language backbone
  ↓
Selected pooled or final hidden representation
  ↓
Linear(hidden_size, 24)
  ↓
24 permutation logits
```

The model must not generate a string such as `[2, 4, 3, 1]`.

Instead, attach a classification head and predict one of 24 classes.

Use a clearly structured input format:

```text
Caption:
A person enters a room and then sits on a chair.

Frame Input_1:
<image>

Frame Input_2:
<image>

Frame Input_3:
<image>

Frame Input_4:
<image>

Task:
Predict the original temporal position of each input frame.
```

## 5.2 Initial training configuration

Start with:

```yaml
quantization: 4bit
lora_rank: 16
lora_alpha: 32
batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
epochs: 3
gradient_checkpointing: true
loss: cross_entropy
```

Initial trainable components:

```text
vision encoder                 frozen
language-model backbone        QLoRA
multimodal projector           trainable or LoRA
24-class classifier head       trainable
```

Keep image resolution and visual token count conservative at first because four images are processed per sample.

## 5.3 Development phases

### Phase 1 — Smoke test

Use only a very small subset.

Requirements:

- load 10 to 100 samples
- run one forward pass
- compute loss
- run backward
- confirm gradients exist in intended modules
- save and reload a checkpoint
- report peak GPU memory
- verify class-to-permutation conversion

Then test whether the model can overfit 20 to 50 samples.

If it cannot overfit a tiny subset, inspect:

- label mapping
- hidden-state selection
- frozen parameters
- loss computation
- image ordering
- prompt formatting

### Phase 2 — Frozen vision encoder

Train:

- LoRA adapters
- multimodal projector if necessary
- classifier head

Compare at least:

- LoRA rank 8
- LoRA rank 16
- two visual-token or image-resolution settings

Use validation Exact Match Accuracy for model selection.

### Phase 3 — Partial unfreezing

Only after Phase 2 is stable, try one of the following:

- unfreeze the multimodal projector
- unfreeze the final one or two vision blocks

Do not unfreeze the entire vision encoder initially.

### Phase 4 — Inference optimization

Implement:

- `torch.no_grad()`
- BF16 or FP16 inference
- deterministic inference
- batched inference where memory permits
- no free-form generation
- offline model loading
- inference-time measurement
- peak-memory measurement
- submission validation

## 5.4 Advantages

- strong multimodal and temporal-language understanding
- close to the provided baseline
- no output parsing failures
- valid permutation guaranteed
- likely the best balance between performance and implementation effort

## 5.5 Risks

- slow training and inference
- four images may produce high memory usage
- overfitting may occur on a small or noisy dataset
- hidden-state extraction must be implemented carefully

---

# 6. Candidate Model 2 — Efficient Alternative

## SigLIP-Style Image/Text Encoder + Frame Transformer + 24-Class Head

### Priority

Develop this model in parallel as a faster and more interpretable alternative.

Use an eligible image-text encoder released before the competition cutoff.

## 6.1 Architecture

```text
Sentence  ── text encoder  ── text embedding
Input_1   ── image encoder ── frame embedding 1
Input_2   ── image encoder ── frame embedding 2
Input_3   ── image encoder ── frame embedding 3
Input_4   ── image encoder ── frame embedding 4

[CLS, TEXT, FRAME_1, FRAME_2, FRAME_3, FRAME_4]
                     ↓
          small Transformer encoder
                     ↓
              24-class classifier
```

Each frame embedding must include an input-position embedding.

Example:

```python
frame_token = (
    image_embedding
    + frame_type_embedding
    + input_position_embedding
)
```

This is required because `Input_1` and `Input_2` are semantically different positions in the prediction target.

## 6.2 Auxiliary objectives

Use 24-class cross entropy as the main objective:

```text
L_global = CE(global_logits, permutation_class)
```

Also create six pairwise temporal labels:

```text
Input_1 before Input_2?
Input_1 before Input_3?
Input_1 before Input_4?
Input_2 before Input_3?
Input_2 before Input_4?
Input_3 before Input_4?
```

Optional total loss:

```text
L_total =
    1.0 × L_global
  + 0.3 × L_pairwise
  + 0.1 × L_no_ordering
```

`L_no_ordering` predicts whether the current input order is already correct.

Use `No_ordering` only as a training label. Do not expect this field in the test set.

## 6.3 Development phases

### Phase 1 — Frozen encoders

Train only:

- frame Transformer
- global classifier
- pairwise head
- optional no-ordering head

Keep image and text encoders frozen.

### Phase 2 — Architecture comparison

Compare:

```text
A. Concatenate embeddings + MLP
B. Frame Transformer + 24-class head
C. Frame Transformer + global, pairwise, and no-ordering heads
```

Use the same validation split for all comparisons.

### Phase 3 — Partial fine-tuning

If frozen encoders underperform, unfreeze only the last one or two encoder blocks.

Do not immediately fine-tune the full encoder.

### Phase 4 — Noise handling

Experiment with:

- sample weighting
- label smoothing around 0.05
- class-weighted sampling if needed
- reduced weights for black or duplicated-frame samples

## 6.4 Advantages

- faster training and inference
- safer under the RTX 3090 limit
- easy to analyze and explain
- lower implementation risk
- pairwise errors can be diagnosed clearly

## 6.5 Risks

- weaker commonsense temporal reasoning than a large VLM
- frame-level embeddings may lose subtle visual progression
- performance may depend strongly on the quality of the pretrained encoder

---

# 7. Candidate Model 3 — Structured High-Performance Extension

## Qwen Vision-Language Backbone + Global and Pairwise Structured Decoder

### Priority

Implement only after Candidate Model 1 is stable.

This is a single model with one shared backbone and multiple heads. It is not an ensemble.

## 7.1 Architecture

```text
Qwen multimodal representation
        ├── Global head: 24 permutation logits
        └── Pairwise head: 6 before/after logits
```

For every candidate permutation, combine:

```text
final_score(permutation)
=
global_logit(permutation)
+
lambda × pairwise_consistency(permutation)
```

A permutation should receive a lower score when it conflicts with pairwise predictions.

Example logic:

```python
final_scores = global_logits.clone()

for permutation_id, permutation in enumerate(PERMUTATIONS):
    final_scores[:, permutation_id] += (
        pairwise_weight
        * compute_pairwise_consistency(
            permutation=permutation,
            pairwise_logits=pairwise_logits,
        )
    )
```

## 7.2 Training schedule

### Stage 1 — Pairwise curriculum

Start with:

```text
L =
0.5 × L_global
+
1.0 × L_pairwise
```

This encourages learning of local before/after relations.

### Stage 2 — Global permutation emphasis

Then switch to:

```text
L =
1.0 × L_global
+
0.3 × L_pairwise
```

### Stage 3 — Tune structured-decoding weight

Evaluate:

```text
lambda = 0.0
lambda = 0.2
lambda = 0.5
lambda = 1.0
```

`lambda = 0.0` must reproduce Candidate Model 1 behavior.

## 7.3 Error analysis

Measure performance by scenario type where possible:

- human actions
- object movement
- cooking or assembly
- camera movement
- scene transitions
- nearly identical frames
- black or irrelevant frames

Interpretation:

- high pairwise accuracy but low Exact Match suggests a global decoding problem
- low pairwise accuracy suggests weak visual or caption understanding
- strong identity accuracy but weak non-identity accuracy suggests class imbalance

## 7.4 Advantages

- directly models both local and global temporal structure
- better diagnostic value
- strong methodological justification for the final report
- may improve Exact Match over a plain 24-class head

## 7.5 Risks

- highest implementation complexity
- more hyperparameters
- pairwise loss may overpower global sequence understanding
- should not be attempted before Candidate Model 1 is reliable

---

# 8. Recommended Development Order

Use this order:

```text
1. Reproduce the provided baseline and generate one valid submission
2. Build the shared 24-class data and evaluation pipeline
3. Implement Candidate Model 1
4. Evaluate Candidate Model 1 on the fixed validation split
5. Implement Candidate Model 2 as a separate sequential experiment
6. Compare both models using the same validation split
7. If Candidate Model 1 is better, extend it into Candidate Model 3
8. Select one final model only
9. Retrain the selected model on the full training set
10. Create a standalone offline inference pipeline
11. Generate and validate the final submission
```

Do not use public leaderboard results as the only model-selection criterion.

---

# 9. Solo Development Workflow

Assume that one developer is responsible for the entire project.

Do not attempt to implement all candidate models at the same time. Work sequentially and keep the shared pipeline stable so that each model can be compared fairly.

## Stage A — Data and Evaluation Foundation

Complete these tasks first:

- dataset inspection
- 24-class permutation mapping
- sample-quality and noise analysis
- grouped validation split
- evaluation metrics
- submission validation
- experiment logging

Do not begin large-model fine-tuning until this foundation is tested.

## Stage B — Main Qwen Model

Implement Candidate Model 1 first:

- QLoRA configuration
- multimodal hidden-state extraction
- 24-class classification head
- tiny-subset overfitting test
- GPU-memory measurement
- offline inference path

Candidate Model 1 should become the first complete training and inference pipeline.

## Stage C — Efficient Alternative

After Candidate Model 1 is stable, implement Candidate Model 2:

- image and text embedding extraction
- frame Transformer
- pairwise auxiliary objective
- speed and memory benchmarking

Use the exact same validation split and metrics as Candidate Model 1.

## Stage D — Structured Extension

Implement Candidate Model 3 only when:

- Candidate Model 1 trains reliably
- the validation pipeline is trusted
- time and compute budget remain
- pairwise error analysis suggests that structured decoding may help

Do not begin Candidate Model 3 merely because it is more complex.

## Stage E — Finalization

After comparing the candidates:

1. Select one model only.
2. Freeze the final architecture and hyperparameters.
3. Retrain using the approved training data.
4. Create a standalone offline `inference.py`.
5. Measure memory usage and runtime.
6. Validate the final submission format.
7. Document all commands and environment details.

## Time-Management Rule

Use the following priority order when time is limited:

```text
reliable shared pipeline
    ↓
Candidate Model 1
    ↓
Candidate Model 2
    ↓
Candidate Model 3
```

A complete, reproducible Candidate Model 1 is more valuable than three incomplete model prototypes.

# 10. Initial Milestones

## Milestone 1 — Baseline and data pipeline

Acceptance criteria:

- original baseline remains unchanged
- valid debug submission can be generated
- 24-class mapping has unit tests
- training data passes integrity checks
- validation split is saved and reproducible
- submission validator is implemented

## Milestone 2 — Model smoke tests

Acceptance criteria for each model:

- forward pass succeeds
- backward pass succeeds
- gradients exist only where expected
- a checkpoint can be saved and reloaded
- 20 to 50 samples can be overfit
- peak GPU memory is recorded
- no full-dataset training or inference is run yet

## Milestone 3 — First controlled experiments

Acceptance criteria:

- all models use the same train/validation split
- Exact Match Accuracy is reported
- pairwise and identity metrics are reported
- inference speed is measured
- results are recorded in an experiment log
- only one major variable changes per experiment

## Milestone 4 — Final candidate

Acceptance criteria:

- one model is selected
- offline inference works
- RTX 3090 memory feasibility is documented
- estimated full-test runtime is below 24 hours
- relative paths are used
- `inference.py` is separate from training code
- submission format validation passes
- model size is below 80 GB

---

# 11. Experiment Logging

Create a structured experiment log.

Recommended format:

```markdown
## Experiment ID

- Date:
- Model:
- Backbone:
- Training split:
- Validation split:
- Image resolution:
- Quantization:
- LoRA rank:
- Learning rate:
- Epochs:
- Loss configuration:
- Exact Match Accuracy:
- Pairwise Accuracy:
- Identity Accuracy:
- Non-identity Accuracy:
- Peak GPU memory:
- Inference time per sample:
- Notes:
```

Do not overwrite previous results.

---

# 12. Submission Validation

Before writing `submission.csv`, validate:

- columns are exactly `Id` and `Answer`
- row count matches `test.csv`
- ID order matches `test.csv`
- no values are missing
- every answer is a valid permutation of `[1, 2, 3, 4]`
- output uses the expected list-string format
- no debug-only subset is accidentally submitted

Example checks:

```python
assert submission.columns.tolist() == ["Id", "Answer"]
assert len(submission) == len(test_df)
assert submission["Id"].tolist() == test_df["Id"].tolist()
assert submission["Answer"].notna().all()
```

Also parse every answer and verify:

```python
sorted(answer) == [1, 2, 3, 4]
```

---

# 13. Instructions for Hermes

Follow these rules while working:

1. Inspect the repository before modifying files.
2. Explain the proposed change before implementation.
3. Make small, reviewable changes.
4. Do not modify the original baseline notebook unless explicitly instructed.
5. Do not start full training or full test inference unless explicitly instructed.
6. Use a small debug subset for smoke tests.
7. Do not delete datasets, checkpoints, or experiment results.
8. Do not commit or push unless explicitly instructed.
9. After each task:
   - summarize changed files
   - explain the implementation
   - run relevant tests
   - show `git status`
   - show `git diff`
10. Prefer reusable `.py` modules over placing all logic in notebooks.
11. Keep all paths configurable and relative where possible.
12. Keep competition constraints visible in code comments and documentation.
13. Never implement model ensembling.
14. Verify pretrained-model eligibility before adding it as a dependency.
15. Stop and report clearly when required data, hardware, or dependencies are unavailable.

---

# 14. First Task for Hermes

Start with repository analysis only.

Do not modify files yet.

Perform the following:

1. Inspect the current repository structure.
2. Locate:
   - the original baseline notebook
   - training and test data path assumptions
   - existing model code
   - existing dependency files
   - existing Git ignore rules
3. Compare the current repository against the recommended structure in this document.
4. Propose the minimum set of files required for:
   - permutation mapping
   - dataset loading
   - validation splitting
   - metrics
   - submission validation
   - experiment logging
5. Identify implementation risks.
6. Provide a staged development plan.
7. Do not run training or full inference.
8. Do not commit or push.
9. Show the current `git status`.

Wait for approval before creating or modifying files.
