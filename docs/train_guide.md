# SAM2 Training Guide: Model Structure and Dataset Flow

This document details the current model structure and dataset flow for SAM2 training, including the **Two-Stage Memory Refinement** mechanism.

## Table of Contents

1. [Training Configuration](#training-configuration)
2. [Model Architecture](#model-architecture)
3. [Two-Stage Memory Refinement](#two-stage-memory-refinement)
4. [Dataset Flow](#dataset-flow)
5. [Loss Functions](#loss-functions)
6. [Execution Guide](#execution-guide)
7. [Visual Illustrations](#visual-illustrations)

---

## Training Configuration

### Current Setup (Feb 3, 2026 run - DRM2 Ablation)
- **Model**: SAM2.1 with Hiera-B+ backbone + Adaptive Gating + **DRM Size 2**
- **Config file**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_DRM2_Jan31.yaml`
- **Key config values**:
  - `enable_adaptive_gating: true`
  - `dam_drm_size: 2` (Ablation: DRM slots = 2)
  - `experiment_log_dir: ./sam2_logs/two_stage_mem_dam_adaptive_gating_DRM2_Jan31`
- **Training**: Started from scratch. Log: `train_dam_adaptive_gating_DRM2_Jan31.log`.

### Previous Setup (Jan 31 - DRM3 Ablation)
- **Model**: SAM2.1 with Hiera-B+ backbone + Adaptive Gating + **DRM Size 3**
- **Config file**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_DRM3_Jan31.yaml`
- **Training Status**: Stopped; inference started with DRM3 checkpoint.
- **Inference (DRM3)**: Jan 3 video list (822 videos). Output: `./test_outputs_2stage_DAM_AdaptiveGate_DRM3_Jan31`. Log: `inference_DRM3_Jan31.log`.

### Previous Setup (Jan 28, 2026 run)
- **Model**: SAM2.1 with Hiera-B+ backbone + Adaptive Gating
- **Current Training Status**: Stopped after ~110 epochs (converged significantly from loss ~291 to ~38.5).

### Current Recommended Run (Two-stage + DAM + Adaptive Gating + Quality-weighted RAM)
- **Config file**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml`
- **Key config values currently enabled**:
  - `enable_two_stage_memory: true`
  - `two_stage_feature_source: maskmem`
  - `enable_dam: true`
  - `enable_distractor_memory: true`
  - `distractor_memory_source: pred_top1_low_pred`
  - `enable_quality_weighted_ram: true`
  - `enable_adaptive_gating: true`
  - `enable_attention_gating: false`
- **Experiment log dir**: `sam2_logs/two_stage_mem_dam_adaptive_gating_Jan28/`
- **Environment**: `sam2_dam_jan28` (Python 3.10.13, PyTorch 2.5.1+cu124)
- **Inference Status**: Running on **Jan 3** video list (822 videos total).
  - **Incremental Update**: Cleaned up Dec 30-only videos and processing 113 remaining Jan 3-specific videos.
  - **Output Dir**: `./test_outputs_2stage_DAM_AdaptiveGate_Jan28`

### Ablation: nf16 (1 query + 15 video frames)
- **Goal**: ablation on “frames processed in parallel” by changing **7 → 15** video frames per segment.
- **Training clip length**:
  - `scratch.num_frames: 16` (total = **1 query + 15 video**)
  - Old default: `scratch.num_frames: 8` (total = **1 query + 7 video**)
- **Training config file (on disk)**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_Gate_QualityWeight_Jan15_nf16.yaml`
  - **Hydra `-c` path note**: when running `training/train.py`, pass `-c configs/...` (NOT `-c sam2/configs/...`).
- **Experiment log dir (nf16)**: `sam2_logs/two_stage_mem_dam_gating_quality_Jan15_nf16_v2/`
- **Inference segment length (must match training)**:
  - `tools/test_video_segmentation_multi_gpu.sh` default: `NUM_FRAMES_PER_SEGMENT=16`
  - `tools/test_video_segmentation.py` flag: `--num_frames_per_segment 16`
- **Two critical knobs for nf16 stability**:
  - **Temporal pos-embed capacity** (SpatioTemporalTransformer):  
    - `spatio_temporal_transformer.max_temporal_pos_frames: 32` (must be **>= (scratch.num_frames - 1)**; nf16 needs >= 15)
  - **VRAM reduction (recommended for nf16)**:
    - `memory_attention.use_activation_checkpointing: true` (reduces VRAM, adds compute)

### Ablation: nf12 (1 query + 11 video frames)
- **Goal**: keep the **nf8 / “frame-7” baseline settings** but increase parallel frames to **11 video frames** (nf12 total).
- **Training clip length**:
  - `scratch.num_frames: 12` (total = **1 query + 11 video**)
  - Baseline: `scratch.num_frames: 8` (total = **1 query + 7 video**)
- **Training config file (on disk)**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_Gate_QualityWeight_Jan15_nf12.yaml`
- **Experiment log dir (nf12)**: `sam2_logs/two_stage_mem_dam_gating_quality_Jan15_nf12/`
- **Inference segment length (must match training)**:
  - `tools/test_video_segmentation.py`: `--num_frames_per_segment 12`
- **Critical knob**:
  - **Temporal pos-embed capacity** (SpatioTemporalTransformer):
    - `spatio_temporal_transformer.max_temporal_pos_frames` must be **>= (scratch.num_frames - 1)**; nf12 needs **>= 11**
    - In the nf12 YAML we set: `max_temporal_pos_frames: 12`

### Two-Stage Refinement Settings
- **Enabled**: `True` (`enable_two_stage_memory`)
- **IoU Threshold**: 0.5
- **Top-K Frames**: 2
- **Checkpointing**: Disabled for Stage 2 (Faster, since memory is sufficient)
- **Stage-2 Memory Source (recommended)**: `two_stage_feature_source: maskmem` (merge true `maskmem_features`; avoids injecting projected pixel features into the memory bank)
- **Score Weighting (recommended)**: `two_stage_score_weighting: false` (turn on only after validating score calibration; if enabled, keep `two_stage_detach_scores: true`)

### Distractor-Aware Memory (DAM-style, optional)
- **Goal**: add a **distractor memory slot** (features encoded from masks of *other objects* in the same video+frame) to help the tracker suppress similar-looking non-target objects.
- **Implementation in this repo**:
  - Compute per-object distractor mask on the query frame: union of masks of other objects in the same video+frame.
  - Encode distractor mask into memory features using the existing `memory_encoder`.
  - Append it as an extra memory slot in the memory bank for `memory_attention` (stage 1 / stage 2 configurable).
- **Config knobs**:
  - `enable_distractor_memory: true|false`
  - `distractor_memory_source: gt | pred_top1_low_gt | pred_top1_low_pred`
    - `gt`: use GT distractor mask (union of other objects in the same video+frame)
    - `pred_top1_low_gt`: **(training-only)** from Stage 1 query-frame multimask outputs, pick **top-1**
      competing mask with **high confidence** (`distractor_pred_min_score`) but **low GT IoU**
      (`distractor_pred_max_gt_iou`), encode it as distractor memory for **Stage 2**
    - `pred_top1_low_pred`: **(inference-capable)** from Stage 1 multimask outputs, treat the highest-score
      candidate as **best**, then pick the **top-1** alternative with high confidence but **high divergence**
      from best, using \( \mathrm{div} = 1-\mathrm{IoU}(\text{best}, \text{alt}) \ge \) `distractor_pred_min_divergence`.
  - `distractor_use_in_stage1: true|false`
  - `distractor_use_in_stage2: true|false`
  - `distractor_pred_min_score: 0.7` (only for `pred_top1_low_gt`)
  - `distractor_pred_max_gt_iou: 0.2` (only for `pred_top1_low_gt`)
  - `distractor_pred_min_divergence: 0.5` (only for `pred_top1_low_pred`)
  - `distractor_pred_mask_thresh: 0.0` (logit threshold; 0.0 means prob 0.5)
  - `dam_drm_selection: recent | max_divergence`
    - `recent`: pick the most recent frames that triggered distractor detection
    - `max_divergence`: pick the top-K frames with the largest divergence (most discriminative) as DRM anchors
  - `dam_drm_strength_mode: divergence | score_weighted`
    - `divergence`: strength = divergence
    - `score_weighted`: strength = score * divergence (closer to "high-confidence + high-ambiguity" anchors)
  - **Quality-Weighted RAM Selection (NEW, optional)**:
    - `enable_quality_weighted_ram: true|false`
      - `false` → use original greedy recency-based RAM selection (current default)
      - `true`  → use quality-weighted RAM selection (see below)
    - `ram_selection_alpha: 0.0–1.0`
      - Weight on **recency** vs **quality** in RAM selection
      - `0.0` → quality-only (Stage 1 IoU scores)
      - `1.0` → recency-only (equivalent to original behavior)
      - Recommended: `0.3–0.5` (e.g., `0.4` = 40% recency, 60% quality)
    - `ram_recency_decay: float`
      - Decay rate λ for exponential recency weights:
      - `recency_weight[t] = exp(-λ × (T - t))`, where T is the most recent frame index
  - **Attention Gating (NEW)**: Learnable gates to explicitly control Query/RAM/DRM contribution
    - `enable_attention_gating: true|false` - Enable explicit attention gating
    - `gate_query_init: 1.0` - Initial value for Query gate (always attend to query)
    - `gate_ram_init: 1.0` - Initial value for RAM gate (attend to RAM memories)
    - `gate_drm_init: 0.5` - Initial value for DRM gate (suppress distractors initially, learn optimal value)
    - **Mechanism**: Memory features are multiplied by gate values before attention computation
    - **Benefits**: Explicit control over memory contribution, better interpretability, more stable training
  - **Adaptive Gating (NEW, Jan 28, 2026)**: Dynamic, input-dependent control over memory contribution
    - `enable_adaptive_gating: true|false` - Enable adaptive gating (replaces attention gating)
    - **Mechanism**: An MLP-based module computes weights $(w_q, w_r, w_d)$ dynamically based on Query, Recent, and Distractor features.
    - **Architecture**:
      - **Input**: Query (1 slot), Recent (variable), Distractor (variable).
      - **Global Pooling**: Spatially pools each slot to get feature vectors.
      - **Slot Reduction**: MLPs reduce Recent and Distractor slots to a single-slot representation.
      - **Adaptive Weighting**: Concatenates representations and uses a final MLP with Sigmoid to get 3 adaptive weights.
    - **Benefits**: Weights adapt to specific video content/difficulty rather than being static parameters.

### Implementation Status (Code-Accurate)
- **Two-stage refinement**: Implemented in `training/model/sam2.py` (`enable_two_stage_memory`, stage 1 + stage 2).
- **Stage-2 memory source**:
  - `two_stage_feature_source: maskmem` (**recommended**) merges true `maskmem_features`.
  - `two_stage_feature_source: pix` keeps the older projected-pixel merge path.
- **DAM (RAM/DRM) stage-2 memory bank**:
  - Enabled by `enable_dam: true` (stage-2 memory bank becomes Query + RAM + DRM).
  - **RAM**: selects recent reliable target memories from frames 1..7 based on `dam_ram_min_score`.
  - **DRM**: uses distractor detection + selection and stores only "triggered" distractor anchors (no padding).
  - **Selection policy**: `dam_drm_selection: max_divergence` (or `recent`).
  - **Strength policy**: `dam_drm_strength_mode: score_weighted` (or `divergence`).
- **Attention Gating (NEW)**:
  - **Purpose**: Explicit control over Query/RAM/DRM contribution via learnable gates
  - **Implementation**: Memory features are multiplied by gate values before attention computation
    - `gate_query`: Controls Query memory contribution (initialized to 1.0)
    - `gate_ram`: Controls RAM memory contribution (initialized to 1.0)
    - `gate_drm`: Controls DRM memory contribution (initialized to 0.5 to suppress initially)
  - **Benefits**: 
    - Explicit control vs. implicit learning in attention weights
    - Better interpretability (can visualize gate values)
    - More stable training (explicit vs implicit learning)
    - Expected improvement: +1-2% IoU + better interpretability
- **Adaptive Gating (NEW, Jan 28, 2026)**:
  - **Purpose**: Input-dependent weight generation for Query/RAM/DRM.
  - **Logic**: Replaces static `gate_query`/`gate_ram`/`gate_drm` parameters with dynamic weights computed via an `AdaptiveGate` MLP module.
  - **Flow**: Spatially pool features → reduce slots to fixed size via MLP → concatenate (Query+Recent+Distractor) → predict weights $(w_q, w_r, w_d)$ via MLP + Sigmoid.
- **No-padding DRM slots**:
  - If divergence does not exceed threshold for an object, **no DRM slot is added** for that object.
  - In DAM mode, `memory_attention` is executed **per-object** to support variable-length memory banks.
- **Known limitation**:
  - In DAM-per-object path, object-pointer tokens (`obj_ptr`) are currently not appended (easy to add later if needed).

### Training Notes (Important)
- If you change the model architecture (e.g., enable DAM/DRM, add new parameters like `distractor_embed_spatial`, enable attention gating, or turn on quality-weighted RAM), **do not resume** from an older run's `checkpoint.pt`.
- Use a **new** `launcher.experiment_log_dir` (or delete the old checkpoints) so training initializes from `./checkpoints/sam2.1_hiera_base_plus.pt`.
- **Attention Gating**: When enabling `enable_attention_gating: true`, this adds new learnable parameters (`gate_query`, `gate_ram`, `gate_drm`). Start a fresh training run to initialize these gates properly.
- **Current Training Run (as of Jan 28, 2026)**: Training was performed with the DAM + Adaptive Gating + quality-weighted RAM configuration (`sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml`) and stopped after ~110 epochs for inference.

### Current Recommended Run (Two-stage + DAM + Adaptive Gating + Quality-weighted RAM)
- **Config file**: `sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml`
- **Key config values currently enabled**:
  - `enable_two_stage_memory: true`
  - `two_stage_feature_source: maskmem`
  - `enable_dam: true`
  - `enable_distractor_memory: true`
  - `distractor_memory_source: pred_top1_low_pred`
  - `enable_quality_weighted_ram: true`
  - `enable_adaptive_gating: true`
  - `enable_attention_gating: false`
- **Experiment log dir**: `sam2_logs/two_stage_mem_dam_adaptive_gating_Jan28/`
- **Environment**: `sam2_dam_jan28` (Python 3.10.13, PyTorch 2.5.1+cu124)

**Copy/paste YAML snippet (matches Jan 15 config with attention gating + quality-weighted RAM)**:

```yaml
trainer:
  model:
    enable_two_stage_memory: true
    two_stage_feature_source: maskmem
    two_stage_score_weighting: false
    two_stage_detach_scores: true
    two_stage_normalize_weights: true

    enable_distractor_memory: true
    distractor_memory_source: pred_top1_low_pred
    distractor_use_in_stage1: false
    distractor_use_in_stage2: true
    distractor_pred_min_score: 0.7
    distractor_pred_min_divergence: 0.5
    distractor_pred_mask_thresh: 0.0

    enable_dam: true
    dam_ram_size: 2
    dam_drm_size: 1
    dam_ram_min_score: 0.5
    dam_drm_selection: max_divergence
    dam_drm_strength_mode: score_weighted

    # Quality-weighted RAM selection (NEW)
    enable_quality_weighted_ram: true
    ram_selection_alpha: 0.4
    ram_recency_decay: 0.2
    
    # Attention Gating (NEW): Explicit control over Query/RAM/DRM contribution
    enable_attention_gating: true
    gate_query_init: 1.0  # Always attend to query
    gate_ram_init: 1.0     # Attend to RAM memories
    gate_drm_init: 0.5     # Suppress DRM initially, learn optimal value

launcher:
  experiment_log_dir: ./sam2_logs/two_stage_mem_dam_gating_quality_Jan15
```

### Dataset Paths
- **Images**: `./data/VQS-4K/images`
- **Masks**: `./data/VQS-4K/masks`
- **Training Video List**: `./splits/train_t-IoU.txt`
- **Dataset Multiplier**: 2 (for data augmentation/repetition)

---

## Model Architecture

The SAM2Train model consists of five main components plus the new Two-Stage Refinement modules:

### 1. Image Encoder
**Location**: `sam2.modeling.backbones.image_encoder.ImageEncoder`
- **Trunk**: Hiera-B+ backbone (112 embed dim)
- **Neck**: FPN outputting 256-dim features
- **Output**: Multi-scale feature maps with positional encodings

### 2. Memory Attention
**Location**: `sam2.modeling.memory_attention.MemoryAttention`
- **Purpose**: Conditions current frame's visual features with memories from past frames
- **Configuration**: 4 Layers, 256 Model Dim, RoPE Attention

### 3. Memory Encoder
**Location**: `sam2.modeling.memory_encoder.MemoryEncoder`
- **Purpose**: Encodes mask predictions into 64-dim memory features
- **Output**: `[1, 64, H, W]` memory tensor

### 4. Spatio-Temporal Transformer
**Location**: `sam2.modeling.spatio_temporal_transformer.SpatioTemporalTransformer`
- **Purpose**: Batches frames 1-7 to share temporal/spatial context
- **Configuration**: 2 Layers, 256 Model Dim, 1/4 scale downsampling

### 5. SAM Mask Decoder
**Location**: Inherited from SAM2Base
- **Purpose**: Generates masks, IoU predictions, and object scores
- **Output**: 3 multi-masks, IoU scores, object logits

### 6. Two-Stage Refinement Modules (NEW)
**Location**: `SAM2Train` (custom modules)
- **Frame Adapter** (`two_stage_frame_adapter`):
  - **Input**: 256-dim backbone features (`pix_feat_with_mem`)
  - **Operation**: Conv2d(256 → 64) + SiLU
  - **Purpose**: Projects high-dim visual features to memory dimension
- **Merge Conv** (`two_stage_merge_conv`):
  - **Input**: Concatenated features from K frames (64 * K channels)
  - **Operation**: Conv2d(128 → 64) + SiLU + Conv2d(64 → 64)
  - **Purpose**: Fuses multiple frame features into a single memory token

---

## Two-Stage Memory Refinement

This novel mechanism improves tracking by running a second pass with an enhanced memory bank derived from high-confidence predictions in the first pass.

### Process Flow

1.  **Stage 1 Forward Pass**:
    - The model processes all 8 frames (1 Query + 7 Video) using the standard flow.
    - The memory bank initially contains only the **Query Frame (Frame 0)**.
    - Output: Initial masks and IoU predictions for all frames.

2.  **Frame Selection** (applies to legacy two-stage path when `enable_dam=false`):
    - The model analyzes Stage 1 IoU predictions for Frames 1-7.
    - **Selection Criteria**: Select top-K (2) frames where `IoU > 0.5`.
    - **Padding**: If fewer than K frames meet the threshold, the Query Frame is used to pad the slots.
    - **Note**: When `enable_dam=true`, frame selection uses different criteria (see below).

3.  **Stage-2 Memory Build with DAM** (`enable_dam=true`):
    
    When DAM is enabled, Stage 2 memory selection is based on **Stage 1 predictions** from frames 1-7:
    
    #### RAM (Recent Appearance Memory) Selection
    
    **Purpose**: Select reliable target memories to help track the object accurately.
    
    **Selection Process (Current Implementation)**:
    1. **Extract Scores**: For each frame 1-7, extract the **maximum IoU score** from Stage 1's multimask predictions (`multistep_pred_ious[0].max(dim=-1).values`).
    2. **Filter by Threshold**: Keep only frames where `score >= dam_ram_min_score` (default: 0.5).
    3. **Greedy Selection by Recency** (when `enable_quality_weighted_ram=false`):
       - Select up to `dam_ram_size` (default: 2) frames using a greedy algorithm:
       - For each RAM slot, pick the **most recent** eligible frame (highest frame index).
       - Mark it as used, then repeat for the next slot.
       - Per-object: selection is done independently for each object in the batch.

    **Quality-Weighted Selection (NEW, when `enable_quality_weighted_ram=true`)**:
    1. **Extract Scores**: Same as above — use Stage 1 IoU scores to get a `[B, F]` score matrix.
    2. **Filter by Threshold**: Same as above — keep only frames where `score >= dam_ram_min_score`.
    3. **Normalize Quality**:
       - For each object, normalize scores across **eligible** frames to `[0,1]`.
       - Highest IoU → quality 1.0, lowest eligible IoU → quality 0.0.
    4. **Compute Recency Weights** (exponential decay):
       - For each frame index `t` (1–7), compute
         \\( w_\\mathrm{recency}(t) = \\exp(-\\lambda \\cdot (T - t)) \\),
         where `T` is the most recent frame index and `λ = ram_recency_decay`.
       - Normalize recency weights to `[0,1]`.
    5. **Combine Quality + Recency**:
       - For each object and frame:
         \\( s(t) = \\alpha \\cdot w_\\mathrm{recency}(t) + (1-\\alpha) \\cdot q_\\mathrm{norm}(t) \\),
         where
         - \\( \\alpha = \\texttt{ram_selection_alpha} \\in [0,1] \\),
         - \\( q_\\mathrm{norm}(t) \\) is the normalized IoU-based quality.
    6. **Top-K Selection**:
       - For each object, select the `dam_ram_size` frames with the highest combined score `s(t)`.
       - These frames form the RAM slots for that object.
    
    **Example**: If frames 1-7 have scores `[0.3, 0.6, 0.4, 0.7, 0.5, 0.8, 0.2]` and `dam_ram_min_score=0.5`, `dam_ram_size=2`:
       - Eligible frames: [1, 3, 4, 5] (scores ≥ 0.5)
       - Selected: Frame 5 (most recent), then Frame 4 (next most recent)
       - RAM slots: `[Frame 5, Frame 4]`
    
    #### DRM (Distractor-Resolving Memory) Selection
    
    **Purpose**: Select competing distractor masks to help the model distinguish similar objects.
    
    **Selection Process** (for `distractor_memory_source=pred_top1_low_pred`, inference-capable):
    
    1. **Identify Best Mask**: For each frame, find the **best predicted mask** (highest IoU score from Stage 1 multimask outputs).
    2. **Compute Divergence**: For each alternative mask candidate:
       - Compute `IoU(best_mask, alternative_mask)`
       - Compute `divergence = 1 - IoU(best, alt)`
    3. **Filter Candidates**: Keep only alternatives that satisfy:
       - `score >= distractor_pred_min_score` (default: 0.7) — high confidence
       - `divergence >= distractor_pred_min_divergence` (default: 0.5) — high divergence from best
    4. **Select Top Alternative**: Pick the alternative with the highest score among valid candidates.
    5. **Compute Strength**: 
       - If `dam_drm_strength_mode == "divergence"`: `strength = divergence` (only considers how different masks are)
       - If `dam_drm_strength_mode == "score_weighted"`: `strength = score * divergence` (default)
       
       **Why Score-Weighted?**
       - **Pure divergence** only measures how different the masks are, but doesn't consider confidence.
       - **Score-weighted** combines both factors:
         - **High score** (e.g., 0.8): Model is confident this is a real object (not noise)
         - **High divergence** (e.g., 0.7): Mask is very different from the best prediction
       - This prefers distractors that are **both confident AND different**, which are more reliable and useful for distinguishing similar objects.
       
       **Example Comparison**:
       - Frame A: score=0.9, divergence=0.6 → strength = 0.9 × 0.6 = **0.54**
       - Frame B: score=0.5, divergence=0.9 → strength = 0.5 × 0.9 = **0.45**
       - With score-weighted, Frame A is preferred (higher confidence distractor)
       - With pure divergence, Frame B would be preferred (higher divergence but lower confidence)
       
    6. **Select Frames Using Strength**:
       - If `dam_drm_selection == "recent"`: 
         - Strength is computed but **not used for ranking**
         - Picks most recent frames with valid distractors (based on `has_mat` only)
       - If `dam_drm_selection == "max_divergence"` (default): 
         - **Uses strength to rank frames**: `top_k(strengths, k=dam_drm_size)`
         - Picks top-K frames with **highest strength values**
         - This selects the most confident and divergent distractors across all frames
    7. **Sparse Slots**: If no valid distractor is found for an object, **no DRM slot is added** (no padding).
    
    **Important: No Strength Threshold**
    - There is **no minimum strength threshold** for selecting DRM slots.
    - As long as a candidate passes the initial filters (`score >= 0.7` AND `divergence >= 0.5`), it will be considered valid regardless of the final strength value.
    - If **all 7 frames fail** to find valid distractors (all candidates fail the score/divergence thresholds), then **no DRM slots are added** for that object (sparse slots, no padding).
    - The strength value is only used for **ranking** when `dam_drm_selection == "max_divergence"`, not as a rejection threshold.
    
    **Example**: Frame 3 has 3 multimask predictions:
       - Mask A: score=0.9 (best)
       - Mask B: score=0.8, IoU(best, B)=0.3 → divergence=0.7 ✓ (valid)
       - Mask C: score=0.6, IoU(best, C)=0.8 → divergence=0.2 ✗ (too low)
       - Selected distractor: Mask B
       - Strength: `0.8 * 0.7 = 0.56` (if score_weighted)
    
    **Note**: The selected distractor mask is then **encoded** using `memory_encoder` to create the DRM slot, and a learnable `distractor_embed_spatial` tag is added to distinguish it from target memories.
    
    **Important**: Only DRM slots have the `distractor_embed_spatial` embedding. Query and RAM slots do NOT have this embedding:
    - **Query slots**: Use standard `maskmem_features` from `memory_encoder` (no special tag)
    - **RAM slots**: Use `maskmem_features` from Stage 1 target predictions (no special tag)
    - **DRM slots**: Use `maskmem_features` from distractor masks + `distractor_embed_spatial` tag
    
    This learnable embedding allows the model to distinguish distractor memories from target memories in the memory bank, helping the attention mechanism differentiate between "what to track" (Query/RAM) and "what to avoid" (DRM).
    
    **Why Distractor Embedding Instead of Feature-Level Include/Exclude?**
    
    You might wonder: why use a distractor embedding to tag DRM features, rather than explicitly including RAM and excluding DRM at the feature level (e.g., `final_features = ram_features - drm_features`)?
    
    **Key Insight**: The current approach uses **learned attention** to implicitly include RAM and exclude DRM, rather than hardcoded feature operations. This is more flexible:
    
    - **Attention learns the combination**: The model learns optimal attention weights through training
      - High attention weights for Query/RAM → include their information
      - Low attention weights for DRM → exclude their information
    - **Soft exclusion**: DRM can still contribute when helpful (e.g., for contrast), but is generally suppressed
    - **Adaptive attention weights**: Attention weights are **computed dynamically** for each frame/video based on feature similarity
    - **Unified architecture**: All memory slots treated uniformly, simpler design
    
    **Important Clarification**: 
    - The **distractor embedding is fixed** after training (same value for all DRM slots, all frames, all videos)
    - But the **attention weights are computed dynamically** for each frame/video based on:
      - Feature content (what the DRM actually looks like)
      - Context (what the current frame looks like)
      - Similarity between query and memory features
    - So the "adaptivity" comes from **dynamically computed attention weights**, not from the distractor embedding changing
    
    The distractor embedding is a **fixed learnable tag** (after training) that helps the attention mechanism learn to distinguish DRM from RAM. The actual "including RAM and excluding DRM" happens **implicitly** through dynamically computed attention weights, not through explicit feature-level operations.
    
    For detailed explanation, see `design_choice_distractor_embedding.md` and `distractor_embedding_adaptivity_clarification.md`.
    
    #### Memory Bank Assembly
    
    For each object, the Stage 2 memory bank is assembled as:
    ```
    [Query Frame] + [RAM slots...] + [DRM slots...]
    ```
    - Query: Always present (Frame 0)
    - RAM: 0 to `dam_ram_size` slots (from frames 1-7 with high scores)
    - DRM: 0 to `dam_drm_size` slots (from frames 1-7 with valid distractors)
    
    **Variable Length**: Each object can have a different number of DRM slots (0 to `dam_drm_size`) depending on whether distractors were detected.
    
    **DRM Embedding Application**:
    - Each DRM slot is created independently by encoding a distractor mask
    - **All DRM slots use the same `distractor_embed_spatial` value**
    - Example with 2 DRM slots:
      ```python
      # DRM Slot 1 (from Frame 2)
      d_feats_1 = memory_encoder(...)  # [1, 64, 64, 64]
      d_feats_1 = d_feats_1 + distractor_embed_spatial.view(1, -1, 1, 1)  # Same embedding
      
      # DRM Slot 2 (from Frame 5)
      d_feats_2 = memory_encoder(...)  # [1, 64, 64, 64]
      d_feats_2 = d_feats_2 + distractor_embed_spatial.view(1, -1, 1, 1)  # Same embedding
      
      # Both slots have the exact same distractor_embed_spatial value added
      ```

    #### RAM/DRM Feature Dimensions and Attention Mechanism

    **Feature Creation from Predictions**:

    1. **RAM Features (Recent Appearance Memory)**:
       - **Source**: Stage 1 target mask predictions from selected frames (frames 1-7 with `score >= dam_ram_min_score`)
       - **Encoding Process**:
         - Stage 1 predictions include `maskmem_features` which are already encoded by `memory_encoder`
         - RAM slots directly use these pre-encoded `maskmem_features` from Stage 1 outputs
         - **No additional encoding step** — features are extracted from `stage1_dict["non_cond_frame_outputs"][t]["maskmem_features"]`
       - **Dimensions**: `[1, C_mem, H_mem, W_mem]` where:
         - `C_mem = 64` (memory dimension, `mem_dim`, from `memory_encoder.out_dim`)
         - `H_mem = W_mem = 64` (memory feature map size for 768×768 input images)
         - Shape: `[1, 64, 64, 64]` per RAM slot
       - **Special Embedding**: None (standard target memory features)

    2. **DRM Features (Distractor-Resolving Memory)**:
       - **Source**: Stage 1 distractor mask predictions (competing masks with high divergence)
       - **Encoding Process**:
         - Selected distractor masks (logits) are passed to `_encode_external_memory_from_mask()`
         - This function uses `memory_encoder` to encode the distractor mask:
           - Input: `mask_for_mem` (distractor mask logits) `[1, 1, H_img, W_img]`
           - Process: Apply sigmoid → scale/bias → `memory_encoder(pix_feat, mask_for_mem)`
           - Output: `maskmem_features` `[1, C_mem, H_mem, W_mem]`
         - **Add Distractor Embedding**: `distractor_embed_spatial` (learnable parameter) is added:
           - **Original Shape**: `[C_mem]` = `[64]` (1D vector, only channel dimension)
           - **Initialization**: `nn.Parameter(torch.zeros(64))` (initialized to zeros)
           - **Reshaping**: `distractor_embed_spatial.view(1, -1, 1, 1)` → `[1, 64, 1, 1]`
           - **Operation**: `d_feats = d_feats + distractor_embed_spatial.view(1, -1, 1, 1)` (element-wise addition with broadcasting)
           - **Shape Flow**:
             - `d_feats`: `[1, 64, 64, 64]` (original DRM features)
             - `distractor_embed_spatial.view(1, -1, 1, 1)`: `[1, 64, 1, 1]` (reshaped embedding)
             - **Broadcasting**: `[1, 64, 1, 1]` broadcasts to `[1, 64, 64, 64]` (spatial dimensions expanded)
             - **Result**: `[1, 64, 64, 64]` (same size as original features)
           - This distinguishes DRM slots from Query/RAM slots
           - **Note**: Uses element-wise addition with broadcasting (same pattern as position embeddings)
           - **Important**: This is a **single learnable vector** that:
             - **During training**: Updated via backpropagation (learns optimal values)
             - **After training**: Becomes a **fixed value** shared across **all frames and all videos**
             - **Same embedding value is added to ALL DRM slots**: If the memory bank has 2 DRM slots, both slots use the **exact same** `distractor_embed_spatial` value
             - Example: Memory bank with `[Query, RAM1, RAM2, DRM1, DRM2]` → Both DRM1 and DRM2 have the same `distractor_embed_spatial` added to them
       - **Dimensions**: `[1, C_mem, H_mem, W_mem]` where:
         - `C_mem = 64` (same as RAM)
         - `H_mem = W_mem = 64` (same as RAM)
         - Shape: `[1, 64, 64, 64]` per DRM slot
       - **Special Embedding**: `distractor_embed_spatial` `[64]` (learnable, distinguishes distractors)

    3. **Memory Bank Assembly (Per-Object)**:
       - **Concatenation**: For each object `b`, memory slots are concatenated along the first dimension:
         ```python
         feats_b = [q_mem[b:b+1]]  # Query: [1, 64, 64, 64]
         # Add RAM slots
         for t in ram_frames_per_obj[b]:
             feats_b.append(mem[b:b+1])  # RAM: [1, 64, 64, 64] each
         # Add DRM slots
         for t in drm_frames_per_obj[b]:
             feats_b.append(d_feats)  # DRM: [1, 64, 64, 64] each
         dam_feats_per_obj[b] = torch.cat(feats_b, dim=0)  # [M_b, 64, 64, 64]
         ```
       - **Final Shape**: `[M_b, C_mem, H_mem, W_mem]` where:
         - `M_b` = number of slots for object `b` (1 Query + RAM_slots + DRM_slots, variable)
         - Example: `[4, 64, 64, 64]` for 1 Query + 2 RAM + 1 DRM

    #### Memory Bank Shape Transformation for Attention

    **Complete Flow from Individual Slots to Attention Tokens:**

    1. **Individual Memory Slots** (before concatenation):
       ```python
       Query:  [1, 64, 64, 64]  # 1 slot
       RAM1:   [1, 64, 64, 64]  # 1 slot
       RAM2:   [1, 64, 64, 64]  # 1 slot
       DRM1:   [1, 64, 64, 64]  # 1 slot
       # Total: 4 slots (example)
       ```

    2. **Concatenation** (along dimension 0):
       ```python
       feats_b = [Query, RAM1, RAM2, DRM1]
       memory_bank = torch.cat(feats_b, dim=0)
       # Result: [4, 64, 64, 64]
       #         [M_b, C_mem, H_mem, W_mem]
       #         M_b = 4 (number of slots)
       ```

    3. **Add Batch Dimension** (for per-object processing):
       ```python
       mem_feats_b = memory_bank.unsqueeze(0)
       # [4, 64, 64, 64] → [1, 4, 64, 64, 64]
       # [M_b, C, H, W] → [1, M_b, C, H, W]
       ```

    4. **Flatten to Tokens** (spatial dimensions → sequence length):
       ```python
       # Method: _flatten_memory_tokens()
       B, num_memories, C, H_mem, W_mem = mem_feats_b.shape
       # B=1, num_memories=4, C=64, H_mem=64, W_mem=64
       
       tokens_per_memory = H_mem * W_mem  # 64 * 64 = 4096
       
       # Reshape: [1, 4, 64, 64, 64] → [1, 4, 64, 4096]
       feats_view = mem_feats_b.view(B, num_memories, C, tokens_per_memory)
       # [1, 4, 64, 4096]
       
       # Permute: [1, 4, 64, 4096] → [4096, 1, 4, 64]
       feats_perm = feats_view.permute(3, 0, 1, 2)
       # [4096, 1, 4, 64]
       
       # Reshape: [4096, 1, 4, 64] → [16384, 1, 64]
       # (4096 tokens per slot × 4 slots = 16384 total tokens)
       memory_tokens = feats_perm.reshape(tokens_per_memory * num_memories, B, C)
       # [M_b * 4096, B, C] = [16384, 1, 64]
       ```

    5. **Repeat for Multiple Video Frames** (if num_video_frames > 1):
       ```python
       # For 7 video frames:
       memory_tokens = memory_tokens.repeat(1, num_video_frames, 1)
       # [16384, 1, 64] → [16384, 7, 64]
       # Each frame attends to the same memory bank
       ```

    6. **Final Shape for Attention**:
       ```python
       # Memory tokens (Keys/Values)
       memory_tokens: [M_b * 4096, num_video_frames, C_mem]
       # Example: [16384, 7, 64] for 4 slots × 7 frames
       
       # Image tokens (Queries)
       image_tokens: [4096, num_video_frames, C_img]
       # Example: [4096, 7, 256] for 7 frames
       
       # Attention computation:
       # Q: [4096, 7, 256]  (image features)
       # K: [16384, 7, 64]  (memory features, all slots)
       # V: [16384, 7, 64]  (memory features, all slots)
       ```

    **Shape Summary Table:**

    | Stage | Shape | Description |
    |-------|-------|-------------|
    | **Individual Slots** | `[1, 64, 64, 64]` each | Query, RAM, DRM slots |
    | **After Concatenation** | `[M_b, 64, 64, 64]` | All slots stacked (M_b = number of slots) |
    | **Add Batch Dim** | `[1, M_b, 64, 64, 64]` | For per-object processing |
    | **Flatten to Tokens** | `[M_b*4096, 1, 64]` | Spatial dims → sequence length |
    | **Repeat for Frames** | `[M_b*4096, 7, 64]` | Each frame gets same memory |
    | **In Attention** | `[M_b*4096, 7, 64]` | Keys/Values for cross-attention |

    **Example with 4 Slots (1 Query + 2 RAM + 1 DRM):**
    ```
    Query:  [1, 64, 64, 64]  ┐
    RAM1:   [1, 64, 64, 64]  ├─→ concat(dim=0) → [4, 64, 64, 64]
    RAM2:   [1, 64, 64, 64]  │
    DRM1:   [1, 64, 64, 64]  ┘
    
    [4, 64, 64, 64] → unsqueeze(0) → [1, 4, 64, 64, 64]
    
    [1, 4, 64, 64, 64] → flatten → [16384, 1, 64]
    (4 slots × 4096 tokens/slot = 16384 tokens)
    
    [16384, 1, 64] → repeat(1, 7, 1) → [16384, 7, 64]
    (7 video frames)
    
    Final: [16384, 7, 64]  (Keys/Values for attention)
    ```

    **Input to Memory Attention Module**:

    1. **Image Features (Queries)**:
       - **Source**: Backbone + FPN output features from frames 1-7
       - **Dimensions**: `[B*num_video_frames, C_img, H_img, W_img]` where:
         - `C_img = 256` (hidden dimension, `hidden_dim`, from backbone FPN)
         - `H_img = W_img = 64` (feature map size for 768×768 input)
         - Shape: `[B*7, 256, 64, 64]` for 7 video frames
       - **Flattening**: Reshaped to tokens for attention:
         - `[B*7, 256, 64, 64]` → `[B*7, 4096, 256]` (HW×C format)
         - Each spatial location (64×64 = 4096 tokens) becomes a query token

    2. **Memory Features (Keys/Values)**:
       - **Source**: Per-object memory bank (Query + RAM + DRM slots)
       - **Dimensions**: `[M_b, C_mem, H_mem, W_mem]` = `[M_b, 64, 64, 64]` per object
       - **Flattening**: Reshaped to tokens for attention:
         - `[M_b, 64, 64, 64]` → `[M_b, 4096, 64]` (HW×C format)
         - Each memory slot contributes 4096 tokens (keys/values)
       - **Per-Object Processing**: When DAM is enabled, `memory_attention` runs **per-object** to handle variable-length memory banks:
         ```python
         for b in range(B):
             # Get memory bank for object b: [M_b, 64, 64, 64]
             mem_feats_b = dam_feats_per_obj[b].unsqueeze(0)  # [1, M_b, 64, 64, 64]
             # Flatten to tokens: [1, M_b, 4096, 64]
             memory_tokens = flatten(mem_feats_b)  # [M_b*4096, 64]
             # Image features for this object: [7, 256, 64, 64]
             curr = batched_vision_feats[-1][:, idxs]  # [HW, 7*B, 256]
             # Cross-attention: image tokens (queries) attend to memory tokens (keys/values)
             pix_feat_with_mem = memory_attention(
                 curr=curr,  # [HW, 7*B, 256] - queries
                 memory=memory_tokens,  # [M_b*4096, 64] - keys/values
                 ...
             )
         ```

    3. **Cross-Attention Mechanism**:
       - **Query (Q)**: Image feature tokens `[4096, 256]` per frame
       - **Key (K)**: Memory feature tokens `[M_b*4096, 64]` (all slots flattened)
       - **Value (V)**: Memory feature tokens `[M_b*4096, 64]` (same as keys)
       - **Attention Operation**: 
         - Image features (256-dim) cross-attend to memory features (64-dim)
         - The attention mechanism learns to:
           - **Attend to Query/RAM slots**: Focus on target object appearances
           - **Attend away from DRM slots**: Suppress distractor information
         - Output: Conditioned image features `[4096, 256]` per frame
       - **Dimension Mismatch Handling**: 
         - **Input dimensions**: Q (256-dim) vs K/V (64-dim) - different feature dimensions
         - **Sequence lengths**: Q (4096 tokens) vs K/V (M_b*4096 tokens) - different sequence lengths
         - **Solution**: The `RoPEAttention` module uses **separate projection layers**:
           ```python
           # Configuration in cross_attention:
           embedding_dim: 256  # For Q (query)
           kv_in_dim: 64       # For K and V (keys/values)
           
           # Projection layers inside RoPEAttention:
           q_proj = nn.Linear(256, internal_dim)  # Projects Q: 256 → internal_dim
           k_proj = nn.Linear(64, internal_dim)    # Projects K: 64 → internal_dim
           v_proj = nn.Linear(64, internal_dim)    # Projects V: 64 → internal_dim
           ```
         - **After projection**: Both Q and K/V are projected to the same `internal_dim` (typically 256)
         - **Attention computation**: `attention_weights = softmax(Q @ K^T / sqrt(d))` where Q and K are now the same dimension
         - **Sequence length difference**: This is fine for cross-attention - Q can attend to K/V with different sequence lengths
         - **Output projection**: `out_proj = nn.Linear(internal_dim, 256)` projects back to 256-dim to match Q dimension

    4. **Output**:
       - **Conditioned Image Features**: `[B*num_video_frames, C_img, H_img, W_img]` = `[B*7, 256, 64, 64]`
       - These features are then passed to the Spatio-Temporal Transformer and SAM Mask Decoder for final predictions

    **Key Points**:
    - **RAM and DRM features have the same dimensions** (`[1, 64, 64, 64]` per slot)
    - **Only DRM features have the distractor embedding** added (`distractor_embed_spatial`)
    - **Memory bank is variable-length per object** (different number of RAM/DRM slots)
    - **Memory attention runs per-object** when DAM is enabled to handle variable-length banks
    - **Image features (256-dim) attend to memory features (64-dim)** via cross-attention
    - **All memory slots (Query + RAM + DRM) are in the same attention space** — the model learns to distinguish them through the distractor embedding and attention weights
    - **Attention Gating (NEW)**: When `enable_attention_gating=true`, memory features are multiplied by learnable gates before attention:
      - Query slots: `feats_query = feats_query * gate_query` (typically 1.0)
      - RAM slots: `feats_ram = feats_ram * gate_ram` (typically 1.0)
      - DRM slots: `feats_drm = feats_drm * gate_drm` (typically 0.5 initially, learns optimal value)
      - This provides explicit control over memory contribution vs. implicit learning in attention weights

    #### Position Embedding for maskmem_features

    **PositionEmbeddingSine** is applied to `maskmem_features` **inside the MemoryEncoder** module to provide spatial context.

    **Application Process**:
    1. **Inside MemoryEncoder** (`sam2/modeling/memory_encoder.py`):
       - After generating `maskmem_features` with shape `[B, 64, 64, 64]`
       - Position embedding is computed: `pos = self.position_encoding(x).to(x.dtype)`
       - Output shape: `[B, 64, 64, 64]` (same as `maskmem_features`)

    2. **PositionEmbeddingSine** (`sam2/modeling/position_encoding.py`):
       - Generates sinusoidal encodings for 2D spatial positions
       - Uses sin/cos functions with different frequencies (temperature=10000)
       - Normalizes coordinates to [0, 2π] range
       - Splits embedding dimension into y and x components (each gets 32 dims for 64-dim output)

    3. **Shape Flow**:
       - **Input to MemoryEncoder**: `pix_feat` `[B, 256, 64, 64]`, `masks` `[B, N, 768, 768]`
       - **After processing**: `maskmem_features` `[B, 64, 64, 64]`
       - **Position embedding**: `maskmem_pos_enc` `[B, 64, 64, 64]` (same shape)
       - **Returned together**: `{"vision_features": maskmem_features, "vision_pos_enc": [maskmem_pos_enc]}`

    4. **Usage in Memory Attention**:
       - Position embeddings are **added** to memory features in cross-attention
       - Operation: `k = memory + pos` (element-wise addition)
       - Provides spatial context: helps attention mechanism understand where features are located in 2D space
       - Shape: `[M*4096, B, 64]` + `[M*4096, B, 64]` = `[M*4096, B, 64]` (after flattening)

    #### Element-wise Addition Pattern

    Both **position embeddings** and **distractor embeddings** use **element-wise addition** to combine with the original features:

    1. **Position Embedding** (for all memory slots: Query, RAM, DRM):
       - **Location**: Added in memory attention cross-attention layer
       - **Operation**: `k = memory + pos` (element-wise addition)
       - **Purpose**: Provides spatial context (where features are located in 2D space)
       - **Applied to**: All memory slots (Query, RAM, DRM) - all get position embeddings
       - **Shape**: `[M*4096, B, 64]` + `[M*4096, B, 64]` = `[M*4096, B, 64]`

    2. **Distractor Embedding** (only for DRM slots):
       - **Location**: Added when creating DRM features (before adding to memory bank)
       - **Operation**: `d_feats = d_feats + distractor_embed_spatial.view(1, -1, 1, 1)` (element-wise addition)
       - **Purpose**: Tags DRM slots as "distractor" to distinguish from target memories
       - **Applied to**: Only DRM slots (Query and RAM slots do NOT get this embedding)
       - **Shape**: `[1, 64, 64, 64]` + `[1, 64, 1, 1]` (broadcast) = `[1, 64, 64, 64]`

    **Key Differences**:
    - **Position embedding**: Applied to ALL memory slots (Query, RAM, DRM) in the attention layer
    - **Distractor embedding**: Applied ONLY to DRM slots when creating the features
    - **Both use element-wise addition**: Simple, efficient, and allows gradients to flow through both paths
    - **Order of application**: 
      1. First: Distractor embedding added to DRM features (`d_feats = d_feats + distractor_embed_spatial`)
      2. Then: Position embedding added to all memory features in attention (`k = memory + pos`)

    **Example Flow for DRM Slot**:
    ```python
    # Step 1: Encode distractor mask
    d_feats = memory_encoder(pix_feat, distractor_mask)  # [1, 64, 64, 64]
    
    # Step 2: Add distractor embedding (element-wise)
    d_feats = d_feats + distractor_embed_spatial.view(1, -1, 1, 1)  # [1, 64, 64, 64]
    
    # Step 3: Add to memory bank
    memory_bank.append(d_feats)  # Now in memory bank
    
    # Step 4: In memory attention, add position embedding (element-wise)
    k = memory + pos  # [M*4096, B, 64] + [M*4096, B, 64] = [M*4096, B, 64]
    ```

    **Key Properties**:
    - **Same shape as features**: Position embedding has identical shape to `maskmem_features` (`[B, 64, 64, 64]`)
    - **Spatial encoding**: Encodes 2D positions using sinusoidal functions (similar to Transformer positional encodings)
    - **Computed per feature map**: Based on spatial dimensions (H, W) of the feature map
    - **Added in attention**: Combined with memory features via element-wise addition in cross-attention keys

    **For detailed technical explanation**, see `position_embedding_maskmem.md`.

    #### Attention Gating Mechanism (NEW)

    **Purpose**: Provide explicit control over Query/RAM/DRM contribution via learnable gates, complementing the implicit attention mechanism.

    **Implementation**:
    1. **Gate Parameters**: Three learnable scalar parameters (initialized per config):
       - `gate_query`: Controls Query memory contribution (default: 1.0)
       - `gate_ram`: Controls RAM memory contribution (default: 1.0)
       - `gate_drm`: Controls DRM memory contribution (default: 0.5, to suppress initially)
    
    2. **Application**: Gates are applied to memory features **before** flattening and attention:
       ```python
       # When building memory bank per object
       for slot_idx, mem_type in enumerate(memory_types_b):
           if mem_type == "query":
               gate = self.gate_query
           elif mem_type == "ram":
               gate = self.gate_ram
           elif mem_type == "drm":
               gate = self.gate_drm
           # Apply gate: multiply memory features by gate value
           feats_b[slot_idx] = feats_b[slot_idx] * gate
       ```
    
    3. **Interaction with Attention**: 
       - Gates modify memory features **before** attention computation
       - Attention weights are still learned dynamically based on feature similarity
       - Gates provide a **global scaling factor** per memory type
       - Attention provides **per-token fine-grained** weighting
    
    4. **Benefits**:
       - **Explicit Control**: Direct control over memory type contribution vs. implicit learning
       - **Interpretability**: Can visualize gate values to understand model behavior
       - **Stability**: More stable training (explicit vs implicit learning)
       - **Flexibility**: Gates can be positive (attend) or negative (suppress)
       - **Expected Improvement**: +1-2% IoU + better interpretability
    
    5. **Training Behavior**:
       - Gates are learnable parameters updated via backpropagation
       - Initial values guide early training (e.g., suppress DRM initially)
       - Model learns optimal gate values during training
       - Final gate values indicate how much each memory type contributes
    
    6. **Example Gate Values After Training**:
       - `gate_query ≈ 1.0`: Query memory is always important
       - `gate_ram ≈ 0.8-1.2`: RAM memories are useful, may be slightly adjusted
       - `gate_drm ≈ 0.3-0.7`: DRM memories are suppressed but still contribute (for contrast)
    
    **Comparison with Implicit Attention**:
    - **Implicit (without gating)**: Attention mechanism learns to attend to Query/RAM and away from DRM through attention weights alone
    - **Explicit (with gating)**: Gates provide global scaling, attention provides fine-grained weighting
    - **Combined**: Both mechanisms work together for optimal performance

    #### Adaptive Gating Mechanism (NEW, Jan 28, 2026)

    **Purpose**: Replace static attention gates with input-dependent weights computed dynamically from the current memory content.

    **Implementation**:
    1.  **Input**:
        - Query features: `[1, 64, 64, 64]`
        - Recent (RAM) features: `list` of `[1, 64, 64, 64]` (variable length)
        - Distractor (DRM) features: `list` of `[1, 64, 64, 64]` (variable length)
    2.  **Global Pooling**: Spatially pool each slot (Global Average Pooling) to get `[1, 64]` vectors.
    3.  **Slot Reduction**:
        - RAM representations are averaged and passed through a `recent_reducer` MLP.
        - DRM representations are averaged and passed through a `distractor_reducer` MLP.
    4.  **Adaptive Weighting**: 
        - Concatenate `[Query_v, Recent_v, Distractor_v]` to get a `[1, 192]` vector.
        - Pass through a `gate_mlp` to produce 3 weights $(w_q, w_r, w_d)$ in range `[0, 1]`.
    5.  **Application**: Each memory type is multiplied by its corresponding weight before concatenation into the final memory bank.

    **Benefits**:
    - **Adaptivity**: The model can prioritize different memory types based on the scene (e.g., trust Query more when distractors are highly similar).
    - **Dynamic**: Weights change every frame based on the specific memory content.

4.  **Memory Augmentation**:
    - The memory bank is expanded from Stage 1's single Query slot:
      - **If DAM disabled** (`enable_dam=false`): `[Query, Merged_Feat]` (2 memory slots)
        - Merged_Feat: fused features from top-K high-confidence frames
      - **If DAM enabled** (`enable_dam=true`): `[Query, RAM_slots..., DRM_slots...]` (1 + RAM_size + DRM_size slots)
        - RAM slots: recent reliable target memories (default: 2 slots)
        - DRM slots: distractor memories (sparse, 0 to `dam_drm_size` slots per object)
        - See "Memory Bank Structure" section below for detailed breakdown

5.  **Stage 2 Forward Pass**:
    - The model re-processes Frames 1-7 using this **augmented memory bank**.
    - Frames 1-7 attend to **all slots** in the memory bank via Cross-Attention:
      - Query frame (always present)
      - RAM slots (if DAM enabled) - helps track the target accurately
      - DRM slots (if DAM enabled) - helps distinguish distractors
      - Merged_Feat (if DAM disabled) - fused high-confidence features
    - **Outputs**: The final predictions come from this second stage.

6.  **Training Logic**:
    - **Shared Model Weights**: Stage 1 and Stage 2 use the **same model instance** with **shared weights** for all modules:
      - `memory_attention` (4 layers, 256 dim)
      - `memory_encoder` (64-dim output)
      - `spatio_temporal_transformer` (2 layers, 256 dim)
      - `mask_decoder` (SAM decoder)
      - All other modules are shared between stages
    - **Only Difference**: The memory bank content differs:
      - **Stage 1**: Memory bank = `[Query Frame]` (1 slot)
      - **Stage 2**: Memory bank = `[Query, RAM..., DRM...]` or `[Query, Merged_Feat]` (multiple slots)
    - **All Modules Are Trainable**: **No modules are frozen** by default:
      - **Backbone + FPN**: `freeze_image_encoder=False` (default) — all parameters trainable
      - **Memory Attention**: Trainable (4 layers)
      - **Memory Encoder**: Trainable (64-dim output)
      - **Spatio-Temporal Transformer**: Trainable (2 layers)
      - **Mask Decoder**: Trainable (SAM decoder)
      - **Two-Stage Modules**: Trainable (`two_stage_frame_adapter`, `two_stage_merge_conv`)
      - **DAM Modules**: Trainable (`distractor_embed_spatial` learnable parameter)
      - **All other parameters**: Trainable by default
    - **End-to-End Gradients**: Gradients flow from Stage 2 outputs back through the `merged_feat`, `two_stage_merge_conv`, `two_stage_frame_adapter`, and into Stage 1's backbone features.
    - **Backbone Training**: The Backbone + FPN are **NOT frozen** (`freeze_image_encoder=False` by default). Gradients flow through the backbone in both Stage 1 and Stage 2:
      - **Stage 1**: Backbone runs forward pass, gradients computed normally.
      - **Stage 2**: Backbone features are cached from Stage 1 (not recomputed), but gradients from Stage 2 losses still flow back through the cached features to the backbone parameters.
    - **Backbone Caching**: Stage 2 reuses the backbone features computed in Stage 1 to save computation (no second forward pass through backbone).
    - **Robustness**: Even if no frames are selected (all padding), the query frame is passed through the adapter and merge conv to ensure all parameters receive gradients (DDP compatibility).

---

## Dataset Flow

The dataset flow remains largely the same, supplying 8-frame sequences.

### Overview
```
Raw Video Dataset -> Frame Sampling (1 Query + 7 Video) -> Transforms -> Batching -> Model
```

### Key Components
1.  **Frame Sampling**: Randomly selects 1 query frame and 7 consecutive video frames.
2.  **Transforms**: Resizes to **768x768** (consistent across frames), applies color jitter and flips.
3.  **Batch Collation**: Stacks frames into `[8, B, 3, 768, 768]` tensors.

---

## Loss Functions

During **training**, the model returns both **Stage 1** and **Stage 2** per-frame outputs, and the loss is computed as a weighted sum:

- **Stage 1 loss weight**: 0.5 (initial guess)
- **Stage 2 loss weight**: 1.0 (refined prediction)

During **evaluation/inference**, only **Stage 2** outputs are returned.

### Components
1.  **Mask Loss** (Weight: 20): Focal loss on mask predictions.
2.  **Dice Loss** (Weight: 1): Measures overlap.
3.  **IoU Loss** (Weight: 1): L1 loss on IoU predictions.
4.  **Classification Loss** (Weight: 1): Focal loss on object scores.
5.  **False Positive Loss** (Weight: 1): Penalizes non-empty predictions when GT is empty.

---

## Execution Guide

### Training

To start training the two-stage model with DAM, attention gating, and quality-weighted RAM on 4 GPUs (recommended minimum):

```bash
cd <REPO_ROOT>
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate sam2_dam_jan10
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_DAM_Gate_QualityWeight_Jan15.yaml \
  --use-cluster 0 --num-gpus 4 \
  > train_DAM_Gate_QualityWeight_Jan15.log 2>&1 &
```

**Note**: This command uses the Jan 15 DAM + gating + quality-weighted configuration. The training will:
- Initialize from scratch (fresh run) with the new architecture and gates
- Save checkpoints to `sam2_logs/two_stage_mem_dam_gating_quality_Jan15/checkpoints/`
- Log training stats to `sam2_logs/two_stage_mem_dam_gating_quality_Jan15/logs/`
- Write TensorBoard logs to `sam2_logs/two_stage_mem_dam_gating_quality_Jan15/tensorboard/`

**Key Parameters**:
- `enable_two_stage_memory=true`
- `resolution=768`
- `batch_size=1` (per GPU)
- `two_stage_feature_source=maskmem` (recommended)
- `two_stage_score_weighting=false` (recommended)
- `enable_dam=true`
- `enable_distractor_memory=true`
- `enable_quality_weighted_ram=true`
- `enable_attention_gating=true`

### Training (nf16 ablation: 1 query + 15 video frames)

```bash
cd <REPO_ROOT>
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate sam2_dam_jan10

# nf16 = total 16 frames per sample (1 query + 15 video)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_DAM_Gate_QualityWeight_Jan15_nf16.yaml \
  --use-cluster 0 --num-gpus 4 \
  > train_DAM_Gate_QualityWeight_Jan15_nf16_v3_ckpt.log 2>&1 &
```

### Training (nf12 ablation: 1 query + 11 video frames)

```bash
cd <REPO_ROOT>
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate sam2_dam_jan10

# nf12 = total 12 frames per sample (1 query + 11 video)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_DAM_Gate_QualityWeight_Jan15_nf12.yaml \
  --use-cluster 0 --num-gpus 4 \
  > train_DAM_Gate_QualityWeight_Jan15_nf12.log 2>&1 &
```

**Quick “is it running?” check**:

```bash
tail -50 <REPO_ROOT>/train_DAM_Gate_QualityWeight_Jan15_nf16_v3_ckpt.log
```

### Training (Adaptive Gating - Jan 28, 2026 run)

To start training with **Adaptive Gating** from scratch on 4 GPUs:

```bash
cd <REPO_ROOT>
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate sam2_dam_jan28
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
  conda run --no-capture-output -n sam2_dam_jan28 \
  python -u training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml \
  --use-cluster 0 --num-gpus 4 \
  > train_dam_adaptive_gating_Jan28.log 2>&1 &
```

**Key Parameters**:
- `enable_adaptive_gating=true`
- `enable_attention_gating=false`
- `experiment_log_dir=./sam2_logs/two_stage_mem_dam_adaptive_gating_Jan28`

### Testing

To evaluate the model on the test set using the two-stage refinement (with DAM + gating if enabled in the config):

```bash
bash tools/test_video_segmentation_multi_gpu.sh
```

**Important (Training vs Inference dataset paths)**:
- **Training** uses dataset paths from the YAML config under `dataset.img_folder` and `dataset.gt_folder`.
- **Inference** (this test script) **does NOT use** `dataset.img_folder` / `dataset.gt_folder` from the YAML.
  - Instead it uses the command-line arguments:
    - `--base-video-dir` (images root)
    - `--input-mask-dir` (masks root)
  - So if you move data to SSD/NVMe, you must pass the SSD paths via these flags (or update the script defaults).

**Test Details**:
- **Test Video List**: `./splits/test_t-IoU.txt` (or the appropriate test split)
- **Checkpoint**: `sam2_logs/two_stage_mem_dam_gating_quality_Jan15/checkpoints/checkpoint.pt` (for the current DAM + gating + quality-weighted run)
- **Output Directory**: `./test_outputs_2stage_DAM_gating_quality_Jan15`
- **Num GPUs**: 4 (or match your training setup)
- **Score Threshold**: 0.5
  - **Inference dataset (SSD/NVMe example)**:
    - Images: `./data/VQS-4K/images`
    - Masks: `./data/VQS-4K/masks`

**Example (run inference on SSD/NVMe paths via command line)**:

```bash
cd <REPO_ROOT>
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate sam2_dam_jan10

bash tools/test_video_segmentation_multi_gpu.sh \
  --num-gpus 4 \
  --video-list-file ./splits/test_t-IoU.txt \
  --checkpoint sam2_logs/two_stage_mem_dam_gating_quality_Jan15/checkpoints/checkpoint.pt \
  --config sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_Gate_QualityWeight_Jan15.yaml \
  --base-video-dir ./data/VQS-4K/images \
  --input-mask-dir ./data/VQS-4K/masks \
  --output-dir ./test_outputs_2stage_DAM_gating_quality_Jan15_SSD \
  --score-thresh 0.5
```

**Example (nf12 inference: must use 12 frames per segment)**:

```bash
cd <REPO_ROOT>
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate sam2_dam_jan10

bash tools/test_video_segmentation_multi_gpu.sh \
  --num-gpus 4 \
  --num-frames-per-segment 12 \
  --video-list-file ./splits/test_t-IoU.txt \
  --checkpoint sam2_logs/two_stage_mem_dam_gating_quality_Jan15_nf12/checkpoints/checkpoint.pt \
  --config sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_Gate_QualityWeight_Jan15_nf12.yaml \
  --base-video-dir ./data/VQS-4K/images \
  --input-mask-dir ./data/VQS-4K/masks \
  --output-dir ./test_outputs_2stage_DAM_gating_quality_Jan15_nf12_SSD \
  --score-thresh 0.5
```

The test script automatically handles:
1.  Loading the config with two-stage refinement enabled.
2.  Splitting the test set across available GPUs.
3.  Running inference where each video is processed in **N-frame chunks** (`--num_frames_per_segment`):
    - **Default**: 16 frames (1 Query + 15 Video) for the nf16 ablation.
    - **Old default**: 8 frames (1 Query + 7 Video).
4.  Merging predictions and saving masks as PNGs.

---

## Visual Illustrations

### Two-Stage Data Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                       TWO-STAGE REFINEMENT FLOW                           │
└───────────────────────────────────────────────────────────────────────────┘

STEP 1: STAGE 1 FORWARD
   Input Frames [0..7]
        │
        ▼
   [Backbone + FPN] (Cached for Stage 2)
        │
        ▼
   [Memory Bank: Query Frame Only]
        │
        ▼
   [Transformer & SAM Decoder]
        │
        ▼
   OUTPUT: Stage 1 Predictions (IoUs, Masks, Features)


STEP 2: MEMORY REFINEMENT
   Analyze Stage 1 Predictions
        │
        ├─────────────────────────────────────┬──────────────────────────────┐
        │                                     │                              │
        ▼                                     ▼                              ▼
   [DAM DISABLED]                    [DAM ENABLED]                  [DAM ENABLED]
   Select Top-2 Frames              Select RAM Frames               Select DRM Frames
   (IoU > 0.5)                      (Recent, Score ≥ 0.5)         (Divergence > 0.5)
        │                                     │                              │
        ▼                                     ▼                              ▼
   Merge Features                    Extract Target Memories         Extract Distractor Memories
   (Adapter + MergeConv)             (from frames 1-7)              (from competing masks)
        │                                     │                              │
        └─────────────────────────────────────┴──────────────────────────────┘
                                        │
                                        ▼
                              Build Augmented Memory Bank


STEP 3: STAGE 2 FORWARD
   Augmented Memory Bank (varies by configuration)
        │
        ▼
   [Transformer & SAM Decoder]
   (Using Cached Backbone Features)
        │
        ▼
   FINAL OUTPUT: Stage 2 Predictions
```

### Memory Bank Structure

The memory bank is a **single unified structure** that all frames attend to via Cross-Attention. The structure differs between Stage 1 and Stage 2, and within Stage 2 it depends on whether DAM is enabled.

#### Stage 1 Memory Bank (Always the Same)

```
┌─────────────────────────────────────────────────────────┐
│              STAGE 1 MEMORY BANK                         │
│  (Used for initial forward pass on all 8 frames)        │
└─────────────────────────────────────────────────────────┘

Slot 0: Query Frame (Frame 0) encoded mask
        └─> [1 slot total]
```

**All frames (0-7) attend to this single Query slot.**

---

#### Stage 2 Memory Bank (Two Configurations)

##### Configuration A: DAM Disabled (`enable_dam=false`)

```
┌─────────────────────────────────────────────────────────┐
│        STAGE 2 MEMORY BANK (Legacy Two-Stage)           │
│  (Used when only two-stage refinement is enabled)       │
└─────────────────────────────────────────────────────────┘

Slot 0: Query Frame (Frame 0) encoded mask
Slot 1: Merged Feature from Top-2 high-confidence frames
        └─> [2 slots total]
```

**Frames 1-7 attend to both slots via Cross-Attention.**

---

##### Configuration B: DAM Enabled (`enable_dam=true`) ⭐ **Current Recommended**

```
┌─────────────────────────────────────────────────────────┐
│        STAGE 2 MEMORY BANK (DAM Mode)                    │
│  (Query + RAM + DRM in a single unified bank)            │
└─────────────────────────────────────────────────────────┘

Slot 0: Query Frame (Frame 0) encoded mask
        │
        ├─> RAM Slots (Recent Appearance Memory)
        │   └─> Slot 1..N: Recent reliable target memories
        │       (Selected from frames 1-7 with score ≥ dam_ram_min_score)
        │       (Default: dam_ram_size=2, so typically 2 slots)
        │
        └─> DRM Slots (Distractor-Resolving Memory)
            └─> Slot M..P: Distractor memories (sparse, conditional)
                (Only added if divergence > threshold, no padding)
                (Default: dam_drm_size=1, but may be 0 if no distractor found)

        └─> [1 + RAM_size + DRM_size slots total]
            (Typical: 1 + 2 + 0~1 = 3~4 slots per object)
```

**Key Points:**
- **All slots are in the SAME memory bank** — Query, RAM, and DRM are not separate banks.
- **Per-object variable length**: Each object in the batch can have a different number of DRM slots (0 to `dam_drm_size`) depending on whether distractors were detected.
- **Frames 1-7 attend to ALL slots** (Query + RAM + DRM) via Cross-Attention.
- **RAM slots** store reliable target appearances to help track the object accurately.
- **DRM slots** store distractor information to help the model distinguish similar objects.

**Example with default config (`dam_ram_size=2`, `dam_drm_size=1`):**
```
Object 0: [Query, RAM_frame3, RAM_frame5, DRM_frame2]  → 4 slots
Object 1: [Query, RAM_frame4, RAM_frame6]              → 3 slots (no DRM, divergence too low)
Object 2: [Query, RAM_frame2, RAM_frame5, DRM_frame1] → 4 slots
```

**Note:** The memory bank is built **per-object** when DAM is enabled, so `memory_attention` runs per-object to handle variable-length banks.
