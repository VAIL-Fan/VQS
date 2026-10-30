# Novelty / Contributions (Code-Accurate)

This document summarizes the **key novel components** added on top of the base SAM2 video tracking pipeline in this repo. The goal is to make the novelty **paper-ready** while staying **strictly aligned with the actual code** and config knobs.

> Scope: **two-stage refinement**, **quality-weighted RAM**, **attention gating**, and **parallel (multi-frame) processing** in training and inference.

---

## 1) Two-Stage Memory Refinement (Stage-1 → Stage-2)

### What problem it addresses
Single-pass tracking can lock onto early mistakes. The two-stage design adds a second pass that **rebuilds the memory bank** using **stage-1 predictions**, then re-runs decoding to refine masks.

### How it works (high level)
- **Stage 1**: run video tracking once (query frame conditions the tracker; the rest are predicted).
- **Stage 2**: build an **override memory bank** using stage-1 outputs and run tracking again with:
  - **cached backbone features** (no second backbone forward),
  - a **stronger memory bank** (Query + selected memories),
  - and (optionally) distractor-aware memories.

### What the code actually does
- Training model: `training/model/sam2.py` (`SAM2Train.forward`)
  - Runs `stage1_outputs = self.forward_tracking(..., return_dict=True)`
  - Builds an override memory bank via `_build_stage_two_override_memory(...)`
  - Runs stage 2 via `_run_second_stage(...)`, which calls `forward_tracking(... override_cond_outputs=override_memory, cached_backbone=stage1_cache, checkpoint_sam=...)`
  - **Training** returns both outputs in a merged structure:
    - Per frame: `{ "stage1": ..., "stage2": ... }`
    - This allows the loss to combine stage-1 + stage-2 supervision.

### Two supported “stage-2 memory sources”
Controlled by `two_stage_feature_source`:
- **`maskmem` (recommended)**:
  - Stage 2 memory is built from `maskmem_features` produced by the memory encoder (semantic mask memory).
  - This avoids injecting projected pixel features directly into the memory bank.
- **`pix` (legacy)**:
  - Stage 2 memory is built from `pix_feat_with_mem` via a 1×1 adapter and a merge conv.

These are configured in YAML under:
- `trainer.model.two_stage_feature_source: maskmem | pix`

### Score weighting option (deliberately off by default)
Controlled by:
- `two_stage_score_weighting: true|false`
- `two_stage_detach_scores: true|false`
- `two_stage_normalize_weights: true|false`

Design choice:
- When weighting is enabled, the system can weight selected frames by stage-1 confidence.
- `two_stage_detach_scores=true` prevents stage-2 loss from encouraging the score head to inflate confidence.

---

## 2) DAM Memory Bank: Query + RAM + DRM (Stage 2)

### Core novelty
Instead of a simple “query + merged feature” memory, Stage 2 can use a **structured memory bank**:

```
[ Query ] + [ RAM slots (target) ] + [ DRM slots (distractor) ]
```

This is enabled by:
- `enable_dam: true`

### RAM (Recent Appearance Memory)
RAM stores **reliable target memories** from earlier frames.
- `dam_ram_size`: how many RAM slots per object (default used: 2)
- `dam_ram_min_score`: minimum stage-1 confidence to be eligible (default used: 0.5)

### DRM (Distractor-Resolving Memory)
DRM stores **confusing competing masks** (distractors), encoded into the same memory feature space.
- `dam_drm_size`: maximum DRM slots per object (default used: 1)
- `dam_drm_selection: recent | max_divergence`
- `dam_drm_strength_mode: divergence | score_weighted`

Important implementation detail (novel + practical):
- **DRM slots are sparse / no-padding**: if no valid distractor exists for an object, **no DRM slot is added** (variable length memory bank per object).

### How distractors are represented (learnable tag)
When distractor memory is enabled, the code adds a learnable “type tag” to distractor memory features:
- `distractor_embed_spatial` is a learnable vector of length `mem_dim` (64)
- DRM feature is modified as:

\[
\mathrm{DRMFeat} \leftarrow \mathrm{DRMFeat} + \mathrm{distractor\_embed\_spatial}
\]

This makes the attention mechanism aware that “this memory slot is a distractor”, without changing the attention architecture.

Enable/disable:
- `enable_distractor_memory: true|false`
- `distractor_use_in_stage1: true|false`
- `distractor_use_in_stage2: true|false`

### How distractor masks are obtained (two modes)
Controlled by:
- `distractor_memory_source: gt | pred_top1_low_gt | pred_top1_low_pred`

1) **`gt`**
- Builds a distractor mask on each frame by taking the union of “other objects” in the same frame.
- Used to encode a distractor slot (supervised / clean distractor definition).

2) **`pred_top1_low_gt`** (training-only)
- Uses stage-1 multimask predictions and GT masks to find a high-score competing mask with low GT IoU.
- Good for training because it creates “hard negative” distractors.

3) **`pred_top1_low_pred`** (inference-capable)
- Uses stage-1 multimask predictions only.
- Selects a competing mask that is confident and sufficiently different from the best mask, using a divergence threshold.

Common knobs:
- `distractor_pred_min_score`
- `distractor_pred_max_gt_iou` (for `pred_top1_low_gt`)
- `distractor_pred_min_divergence` (for `pred_top1_low_pred`)
- `distractor_pred_mask_thresh`

---

## 3) Quality-Weighted RAM Selection (Recency + Quality)

### Motivation
Purely recency-based RAM selection may pick recent but **low-quality** frames. This module selects RAM slots using a tunable combination of:
- **quality** (stage-1 confidence),
- **recency** (how close the frame is to the current time).

### Switch (compatibility with original behavior)
- `enable_quality_weighted_ram: false` → **original greedy recency selection**
- `enable_quality_weighted_ram: true` → **quality-weighted selection**

This ensures the feature is backward compatible and can be ablated cleanly.

### What is the “quality score” in code?
The quality signal comes from stage-1 SAM IoU predictions:
- For each frame, the model has `multistep_pred_ious[0]` with shape `[B, M]` (B objects, M multimask candidates).
- The per-object quality used for RAM is:

\[
q_{b,t} = \max_{m} \mathrm{IoU}_{b,t,m}
\]

This is implemented as `ious.max(dim=-1).values`.

### The selection algorithm (code-accurate)
Given frame ids \(t \in \{1,\dots,T\}\) (non-query frames):

1) **Eligibility**:
\[
\text{eligible}_{b,t} = [q_{b,t} \ge \tau]
\]
where \(\tau = \texttt{dam\_ram\_min\_score}\).

2) **Normalize quality per object across eligible frames**:
- If eligible scores have a range:
\[
\hat{q}_{b,t} = \frac{q_{b,t}-\min(q_{b,\cdot})}{\max(q_{b,\cdot})-\min(q_{b,\cdot})}
\]
- If all eligible frames have the same score, the code treats all eligible frames equally.

3) **Recency weighting** (shared across objects):
\[
r_t = \exp\big(-\lambda (T - t)\big)
\]
where \(\lambda = \texttt{ram\_recency\_decay}\).

Then normalize to \([0,1]\):
\[
\hat{r}_t = \frac{r_t - \min(r)}{\max(r)-\min(r)}
\]

4) **Combine quality + recency**:
\[
s_{b,t} = \alpha \hat{r}_t + (1-\alpha)\hat{q}_{b,t}
\]
where \(\alpha = \texttt{ram\_selection\_alpha} \in [0,1]\).

5) **Top-K selection**:
- Select the `dam_ram_size` frames with highest \(s_{b,t}\) per object.
- Ineligible frames are masked out (score set to \(-\infty\)).

### Original (baseline) RAM behavior
If `enable_quality_weighted_ram=false`, RAM selection is:
- Filter by `dam_ram_min_score`,
- pick the most recent eligible frames greedily (per object),
- up to `dam_ram_size` slots.

---

## 4) Attention Gating over Memory Types (Query / RAM / DRM)

### Motivation
Even with distractor tags, attention may not reliably suppress distractor slots early in training. Attention gating adds **explicit, interpretable, learnable scalars** controlling how much each memory type contributes.

### Switch (compatibility)
- `enable_attention_gating: false` → original behavior (no gating)
- `enable_attention_gating: true` → gated memory features

### What gates exist (code-accurate)
Three learnable scalars:
- `gate_query`
- `gate_ram`
- `gate_drm`

Initialized from YAML:
- `gate_query_init`
- `gate_ram_init`
- `gate_drm_init`

### How gating is applied (exact operation)
When building the stage-2 DAM memory bank, for each slot feature tensor \(m\):

\[
\tilde{m} = g_{\text{type}} \cdot m
\]

Where \(g_{\text{type}}\) is one of the three gate parameters depending on whether the slot is `query`, `ram`, or `drm`.

Key properties:
- Gates are **learned end-to-end** with the rest of the model.
- Gates are **interpretable**: you can inspect values to see whether the model prefers RAM vs DRM.
- Gates can be **negative** (the code allows it), so DRM can be explicitly suppressed.

---

## 5) Parallel / Multi-Frame Processing (“Parallel SAM2”)

This repo increases efficiency and temporal modeling by processing multiple frames together inside one segment.

### 5.1 Segment-based parallelism (train + inference)
The system processes a video in **segments (clips)**:
- **1 query frame** (conditioning, has mask input)
- **N video frames** (prediction frames)
- Total frames per segment:
  - Training: `scratch.num_frames`
  - Inference: `--num_frames_per_segment`

This makes “how many frames we process in parallel” a clean ablation knob:
- baseline: `scratch.num_frames=8` → **1 + 7**
- ablation: `scratch.num_frames=16` → **1 + 15**

### 5.2 Parallelism inside the model forward
Inside `forward_tracking(...)` (training model):
- The memory attention step is run on a tensor that stacks frames (dimension involves `B * num_video_frames`).
- Then an optional **SpatioTemporalTransformer** processes `[B*T, C, H, W]` jointly:
  - This allows **explicit cross-frame spatio-temporal self-attention** before SAM decoding.

Enable spatio-temporal transformer in YAML under:
- `trainer.model.spatio_temporal_transformer: ...`

### 5.3 Inference-time parallelism (engineering + scalability)
Inference adds two more “parallel knobs”:

1) **Multi-GPU distribution** (video-level):
- `tools/test_video_segmentation_multi_gpu.sh` splits the video list across GPUs.

2) **Segment batching on a single GPU** (within-video):
- `tools/test_video_segmentation.py` supports:
  - `--segments_per_batch K`
  - which processes multiple segments from the same video in parallel on the same GPU.

---

## 6) Ablation-Friendly Knobs (for experiments)

### Frames per segment
- Training: `scratch.num_frames` (total frames = query + video frames)
- Inference: `--num_frames_per_segment`

### Two-stage refinement
- `enable_two_stage_memory`
- `two_stage_feature_source: maskmem | pix`
- `two_stage_iou_threshold`, `two_stage_topk`
- `two_stage_score_weighting`, `two_stage_detach_scores`, `two_stage_normalize_weights`

### DAM / distractor memory
- `enable_dam`
- `dam_ram_size`, `dam_drm_size`
- `dam_ram_min_score`
- `dam_drm_selection`, `dam_drm_strength_mode`
- `enable_distractor_memory`
- `distractor_memory_source`
- `distractor_use_in_stage1`, `distractor_use_in_stage2`
- `distractor_pred_min_score`, `distractor_pred_min_divergence`, `distractor_pred_max_gt_iou`, `distractor_pred_mask_thresh`

### Quality-weighted RAM (new)
- `enable_quality_weighted_ram`
- `ram_selection_alpha`
- `ram_recency_decay`

### Attention gating (new)
- `enable_attention_gating`
- `gate_query_init`, `gate_ram_init`, `gate_drm_init`

### nf16 stability knobs (important for “parallel frames” ablation)
- `spatio_temporal_transformer.max_temporal_pos_frames` must be ≥ (num_video_frames) = (`scratch.num_frames - 1`)
- `memory_attention.use_activation_checkpointing: true` can reduce VRAM for nf16

---

## 7) What to claim as “novelty” in a paper (suggested wording)

- **Two-stage refinement** that rebuilds memory from stage-1 predictions and re-runs decoding using cached backbone features.
- **Structured memory bank** in stage 2: Query + RAM (target) + DRM (distractor), with sparse per-object DRM.
- **Quality-weighted RAM selection**: a tunable mixture of stage-1 quality and temporal recency.
- **Attention gating**: learnable scalar gates over Query/RAM/DRM that explicitly control memory contribution and improve interpretability.
- **Parallel multi-frame processing**: segment-level inference/training with configurable clip length (supports ablations like 1+7 → 1+15), plus optional spatio-temporal transformer for cross-frame feature mixing.

