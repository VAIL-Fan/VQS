# SAM2 Training Guide

This document provides a comprehensive guide to the SAM2 training architecture, model structure, data flow, and training settings used in the current configuration.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Flow](#data-flow)
4. [Training Configuration](#training-configuration)
5. [Frame Sampling Strategy](#frame-sampling-strategy)
6. [Loss Functions](#loss-functions)
7. [Optimization Settings](#optimization-settings)

---

## Overview

The current training setup fine-tunes SAM2.1 (Segment Anything Model 2.1) on a custom video object segmentation dataset. The model is based on the Hiera-B+ backbone and uses a modified frame sampling strategy optimized for query-target matching tasks.

**Key Features:**
- **Model**: SAM2.1 with Hiera-B+ backbone
- **Input Resolution**: 1024×1024
- **Number of Frames**: 8 frames per training sample
- **Frame Strategy**: Query frame (frame 0) + 7 consecutive video frames (frames 1-7)
- **Memory Strategy**: Only query frame (frame 0) is stored in memory
- **Processing**: Query frame first, then all 7 video frames processed in parallel
- **Training Mode**: Fine-tuning from pretrained SAM2.1 checkpoint
- **GPUs**: 4 GPUs (0, 1, 2, 3) with batch size 1 per GPU

---

## Model Architecture

The SAM2Train model consists of four main components:

### 1. Image Encoder

**Location**: `sam2.modeling.backbones.image_encoder.ImageEncoder`

**Components**:
- **Scalp**: Initial processing layer (scalp=1)
- **Trunk**: Hiera-B+ backbone
  - Embedding dimension: 112
  - Number of heads: 2
  - Drop path rate: 0.1
- **Neck**: FPN (Feature Pyramid Network) neck
  - Output dimension: 256
  - Backbone channels: [896, 448, 224, 112]
  - FPN top-down levels: [2, 3]
  - Position encoding: Sine positional encoding
    - Number of position features: 256
    - Normalized: True
    - Temperature: 10000

**Output**: Multi-scale feature maps at different resolutions with positional encodings

### 2. Memory Attention

**Location**: `sam2.modeling.memory_attention.MemoryAttention`

**Purpose**: Conditions current frame's visual features with memories from past frames

**Architecture**:
- **Number of Layers**: 4
- **Model Dimension**: 256
- **Feedforward Dimension**: 2048
- **Dropout**: 0.1
- **Activation**: ReLU

**Attention Mechanisms**:
- **Self-Attention**: RoPE (Rotary Position Embedding) Attention
  - Rope theta: 10000.0
  - Feature sizes: [64, 64]
  - Number of heads: 1
  - Position encoding at attention: False
  
- **Cross-Attention**: RoPE Attention
  - Position encoding at keys: True
  - Position encoding at queries: False
  - Rope K repeat: True
  - KV input dimension: 64

**Function**: Enables the model to attend to information from previous frames stored in memory

### 3. Memory Encoder

**Location**: `sam2.modeling.memory_encoder.MemoryEncoder`

**Purpose**: Encodes previous frame outputs (masks) into memory representations

**Components**:
- **Output Dimension**: 64
- **Position Encoding**: Sine positional encoding
  - Number of position features: 64
  - Normalized: True
  - Temperature: 10000

- **Mask Downsampler**: 
  - Kernel size: 3
  - Stride: 2
  - Padding: 1

- **Fuser**: CXBlock-based fusion
  - Number of layers: 2
  - Dimension: 256
  - Kernel size: 7
  - Padding: 3
  - Layer scale init value: 1e-6
  - Uses depth-wise convolutions: True

**Memory Bank**:
- **Number of Memories**: 7 (num_maskmem) - configured but only 1 used
- **Actual Usage**: Only query frame (frame 0) is stored in memory
- Stores encoded mask representation from query frame for matching

### 4. SAM Mask Decoder

**Location**: Inherited from SAM2Base

**Features**:
- High-resolution feature maps enabled
- Multi-mask output: 3 masks on first click
- IoU prediction with sigmoid activation
- Object score prediction enabled

**Object Pointer Integration**:
- Object pointers from other frames used in encoder
- Temporal positional encoding added to object pointers
- Signed temporal positional encoding
- Maximum 16 object pointers in encoder
- MLP projection for object pointer projection

---

## Data Flow

### Input Processing Pipeline

```
1. Dataset Loading
   ↓
2. Frame Sampling (RandomUniformSampler)
   ↓
3. Image Loading & Transforms
   ↓
4. Collation (BatchedVideoDatapoint)
   ↓
5. Model Forward Pass
```

### Frame Sampling Strategy

**Current Implementation** (Modified for query-target matching):

1. **Frame Selection**:
   - Randomly selects 1 query frame from the video
   - Selects 7 consecutive video frames starting from a random position (after query frame if possible)

2. **Frame Distribution**:
   - **Frame 0**: Query frame (randomly selected from video)
   - **Frames 1-7**: 7 consecutive video frames from the same video

3. **Purpose**:
   - Query frame (frame 0) is encoded into memory
   - All 7 video frames (frames 1-7) use the same query frame memory
   - Enables learning query-target matching: match query frame to each of the 7 video frames
   - Leverages SAM2's pretrained matching capabilities for fine-tuning

### Forward Pass Flow

```
Input: BatchedVideoDatapoint
  ├─ flat_img_batch: [B*T, 3, H, W]  # All frames flattened
  ├─ masks: [T, N, H, W]              # Ground truth masks per frame
  └─ objects_identifier: [T, N, 3]    # [video_id, obj_id, frame_idx]

Step 1: Image Encoding
  ├─ forward_image(flat_img_batch)
  ├─ Image Encoder (Hiera-B+ + FPN)
  └─ Output: backbone_fpn, vision_pos_enc

Step 2: Prompt Preparation
  ├─ prepare_prompt_inputs(backbone_out, input)
  ├─ Sample initial conditioning frames (1-2 frames)
  ├─ Sample correction points/boxes
  └─ Prepare prompts for SAM decoder

Step 3: Tracking Forward Pass
  ├─ forward_tracking(backbone_out, input)
  │
  ├─ Step 3a: Process Query Frame (Frame 0)
  │   ├─ Prepare memory-conditioned features (no previous memory)
  │   ├─ SAM Mask Decoder → Generate masks
  │   └─ Encode memory: _encode_new_memory() → Store query frame in memory
  │
  ├─ Step 3b: Process Video Frames (Frames 1-7) in Parallel
  │   ├─ For each frame t (1 to 7):
  │   │   ├─ Prepare memory-conditioned features
  │   │   │   ├─ _prepare_memory_conditioned_features()
  │   │   │   ├─ Use ONLY query frame (frame 0) memory
  │   │   │   ├─ Memory Attention (self-attn + cross-attn with query memory)
  │   │   │   └─ Output: pix_feat_with_mem [B, C, H, W]
  │   │   │
  │   │   ├─ SAM Mask Decoder
  │   │   │   ├─ Prompt Encoder (points/boxes/masks)
  │   │   │   ├─ Mask Decoder
  │   │   │   └─ Output: masks, ious, object_scores
  │   │   │
  │   │   └─ Skip memory encoding (frames 1-7 don't contribute to memory)
  │   │
  │   └─ All 7 frames processed independently using same query memory
  │
  └─ Output: previous_stages_out
      ├─ masks: [T, N, H, W]  # Predictions for all 8 frames
      ├─ ious: [T, N]
      └─ object_scores: [T, N]
```

### Memory Mechanism

**Memory Bank Structure**:
- **Size**: 7 memories (num_maskmem) - configured but only 1 used
- **Content**: Only query frame (frame 0) mask representation
- **Temporal Encoding**: Learnable temporal positional encodings

**Memory Update**:
1. **Query Frame (Frame 0)**:
   - Processed first as initial conditioning frame
   - Receives initial prompts (points/boxes)
   - Mask prediction is encoded via Memory Encoder
   - Encoded memory is stored (this is the ONLY memory)

2. **Video Frames (Frames 1-7)**:
   - All use the SAME query frame memory (from frame 0)
   - Processed in parallel (independent of each other)
   - Do NOT encode their own memory (skip memory encoding)
   - All attend to query frame memory via Memory Attention

**Key Modification**:
- **Override**: `_prepare_memory_conditioned_features()` ensures frames 1-7 only use frame 0 memory
- **Memory Encoding**: Only frame 0 encodes memory; frames 1-7 skip encoding
- **Parallel Processing**: Frames 1-7 are independent and can be processed simultaneously

---

## Training Configuration

### Dataset Settings

```yaml
Dataset:
  - Type: PNGRawDataset (DAVIS-style video dataset)
  - Image folder: Custom dataset path
  - Ground truth folder: Mask annotations path
  - File list: Optional subset of videos
  - Multiplier: 2 (dataset repetition factor)

Frame Configuration:
  - Number of frames: 8 (1 query + 7 consecutive video frames)
  - Max objects per video: 3
  - Batch size: 1 per GPU
  - Effective batch size: 4 (across 4 GPUs)
  - Number of workers: 10
  - Number of GPUs: 4 (GPUs 0, 1, 2, 3)
```

### Data Augmentation

**Transforms Applied** (in order):
1. **RandomHorizontalFlip**: Consistent across frames
2. **RandomAffine**: 
   - Degrees: 25
   - Shear: 20
   - Consistent across frames
3. **RandomResize**: 
   - Target size: 1024×1024
   - Square: True
   - Consistent across frames
4. **ColorJitter** (consistent):
   - Brightness: 0.1
   - Contrast: 0.03
   - Saturation: 0.03
5. **RandomGrayscale**: Probability 0.05
6. **ColorJitter** (per-frame):
   - Brightness: 0.1
   - Contrast: 0.05
   - Saturation: 0.05
7. **ToTensor**: Convert to tensor
8. **Normalize**: 
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

### Training Parameters

```yaml
Training:
  - Mode: train_only
  - Max epochs: 40
  - Phases per epoch: 1
  - Accelerator: CUDA
  - Seed: 123
  - Mixed precision: bfloat16
```

### Prompt Sampling (Training)

```yaml
Point Input:
  - Probability: 0.5
  - Correction points per frame: 7
  - Frames to correct: 2 (random 1-2, always includes first frame)
  - Sample from GT probability: 0.1

Box Input:
  - Probability: 0.5 (when point input is not used)
  - Combined with point input: 0.25 total probability

Initial Conditioning Frames:
  - Number: 2 (random 1-2)
  - Always includes first frame
  - Additional frames added when receiving corrections
```

---

## Frame Sampling Strategy

### Modified RandomUniformSampler

**File**: `training/dataset/vos_sampler.py`

**Algorithm**:
```python
1. Randomly select 1 query frame from video
2. Select 7 consecutive video frames starting from random position
   (preferably after query frame to avoid overlap)
3. Create frame sequence:
   - Frame 0: Query frame (randomly selected)
   - Frames 1-7: 7 consecutive video frames
```

**Rationale**:
- **Query Frame (0)**: Encoded into memory, serves as reference for matching
- **Video Frames (1-7)**: All use the same query frame memory for prediction
- **Benefits**:
  - Leverages SAM2's pretrained matching capabilities
  - Enables fine-tuning for query-target matching tasks
  - Maintains model structure (8 frames) without architectural changes
  - All 7 video frames learn to match against the same query
  - Parallel processing: frames 1-7 are independent

**Example**:
```
Video has frames: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
Query frame: frame_2 (randomly selected)
Start position: frame_5 (randomly selected after query)

Training sequence:
  Frame 0: frame_2 (query) → Encoded into memory
  Frame 1: frame_5 (video) → Uses query memory
  Frame 2: frame_6 (video) → Uses query memory
  Frame 3: frame_7 (video) → Uses query memory
  Frame 4: frame_8 (video) → Uses query memory
  Frame 5: frame_9 (video) → Uses query memory
  Frame 6: frame_10 (video) → Uses query memory
  Frame 7: frame_11 (video) → Uses query memory
```

---

## Loss Functions

**Location**: `training/loss_fns.py`

**Loss Class**: `MultiStepMultiMasksAndIous`

### Loss Components

1. **Mask Loss** (Binary Cross-Entropy)
   - Weight: 20.0
   - Applied to all predicted masks
   - Uses sigmoid activation

2. **Dice Loss**
   - Weight: 1.0
   - Measures overlap between predicted and ground truth masks
   - Formula: `1 - (2*intersection + 1) / (union + 1)`

3. **IoU Loss**
   - Weight: 1.0
   - Supervised on all frames: True
   - Uses L1 loss: True
   - Measures intersection over union

4. **Classification Loss** (Object Score)
   - Weight: 1.0
   - Predicts object presence/absence
   - Focal loss parameters:
     - Gamma: 0.0
     - Alpha: -1.0 (no weighting)

### Loss Computation

```python
For each frame t and object n:
  total_loss = (
      20.0 * mask_loss(pred_mask[t,n], gt_mask[t,n]) +
      1.0 * dice_loss(pred_mask[t,n], gt_mask[t,n]) +
      1.0 * iou_loss(pred_iou[t,n], gt_iou[t,n]) +
      1.0 * class_loss(pred_obj_score[t,n], gt_obj_present[t,n])
  )
```

---

## Optimization Settings

### Optimizer

**Type**: AdamW

**Learning Rates**:
- **Base LR**: 5.0e-6 (for most parameters)
- **Vision LR**: 3.0e-6 (for image_encoder.*)

**Schedulers**:
- **Type**: Cosine annealing
- **Base LR**: 5.0e-6 → 5.0e-7 (10x reduction)
- **Vision LR**: 3.0e-6 → 3.0e-7 (10x reduction)

**Weight Decay**:
- **Default**: 0.1
- **Exemptions**: 
  - Bias parameters: 0.0
  - LayerNorm parameters: 0.0

**Layer Decay**:
- **Image Encoder Trunk**: 0.9 (exponential decay per layer)
- **Position Embeddings**: 1.0 (no decay)

### Gradient Clipping

- **Type**: Gradient norm clipping
- **Max norm**: 0.1
- **Norm type**: L2

### Mixed Precision

- **Enabled**: True
- **Dtype**: bfloat16
- **Benefits**: Faster training, reduced memory usage

### Distributed Training

- **Backend**: NCCL
- **Find unused parameters**: True
- **Number of GPUs**: 4 (GPUs 0, 1, 2, 3)
- **CUDA_VISIBLE_DEVICES**: 0,1,2,3
- **Memory Optimization**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Multi-node**: Supported via SLURM

---

## Checkpointing

### Initialization

- **Source**: Pretrained SAM2.1 checkpoint
- **Path**: `./checkpoints/sam2.1_hiera_base_plus.pt`
- **Strict loading**: True
- **State dict key**: `['model']`

### Saving

- **Directory**: `${experiment_log_dir}/checkpoints`
- **Frequency**: Last checkpoint only (save_freq: 0)
- **Format**: PyTorch state dict

---

## Logging

### TensorBoard

- **Directory**: `${experiment_log_dir}/tensorboard`
- **Flush interval**: 120 seconds
- **Logs**: Training losses, metrics

### Text Logs

- **Directory**: `${experiment_log_dir}/logs`
- **Log frequency**: Every 10 iterations
- **Content**: Training progress, errors, warnings

---

## Key Training Features

### 1. Query-Target Matching

The modified frame sampling and memory mechanism enables the model to learn:
- Matching query frame (frame 0) to each of 7 video frames (frames 1-7)
- All video frames use the same query memory for consistent matching
- Leveraging SAM2's pretrained matching capabilities
- Fine-tuning for specific query-target matching tasks

### 2. Memory-Only-Query Strategy

- **Query Frame (Frame 0)**: Only frame encoded into memory
- **Video Frames (Frames 1-7)**: All use the same query frame memory
- **No Sequential Memory**: Video frames don't use each other's memory
- **Parallel Processing**: Frames 1-7 are independent and can be processed simultaneously

### 3. Parallel Frame Processing

- Query frame processed first to create memory
- All 7 video frames processed in parallel (not sequentially)
- Each video frame independently matches against query memory
- Enables efficient batch processing

### 4. Multi-Mask Tracking

- Outputs 3 masks on first click
- Object pointer integration from query frame
- Temporal consistency through query memory mechanism

### 5. Iterative Refinement

- Correction points sampled on error regions
- Multiple correction iterations (up to 7 points per frame)
- Query frame is always an initial conditioning frame

---

## Model Components Summary

| Component | Location | Key Parameters | Purpose |
|-----------|----------|----------------|---------|
| Image Encoder | `sam2.modeling.backbones.image_encoder` | Hiera-B+, FPN, d=256 | Extract visual features |
| Memory Attention | `sam2.modeling.memory_attention` | 4 layers, RoPE, d=256 | Condition on past frames |
| Memory Encoder | `sam2.modeling.memory_encoder` | 2 layers, d=64 | Encode mask memories |
| SAM Decoder | Inherited from SAM2Base | Multi-mask, IoU pred | Generate masks |
| Loss Function | `training.loss_fns` | 4 components, weighted | Compute training loss |

---

## Data Flow Summary

```
Video Dataset
    ↓
Frame Sampling (1 query + 7 consecutive frames → 8 frame sequence)
    ↓
Image Loading & Augmentation
    ↓
Batch Collation
    ↓
Image Encoder (Hiera-B+ + FPN) - All 8 frames
    ↓
┌─────────────────────────────────────────┐
│ Step 1: Process Query Frame (Frame 0)  │
│   ├─ Memory Attention (no memory)      │
│   ├─ SAM Mask Decoder                  │
│   └─ Memory Encoder → Store in memory  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Step 2: Process Video Frames (1-7)     │
│   ├─ Memory Attention (query memory)   │
│   ├─ SAM Mask Decoder                  │
│   └─ Skip memory encoding              │
│   (All 7 frames processed in parallel) │
└─────────────────────────────────────────┘
    ↓
Loss Computation (mask + dice + iou + class)
    ↓
Backpropagation & Optimization
```

---

## Notes

1. **Frame Sampling**: The current implementation uses 1 query frame (frame 0) + 7 consecutive video frames (frames 1-7). The query frame is encoded into memory, and all 7 video frames use this same query memory for matching.

2. **Memory Strategy**: Only the query frame (frame 0) is stored in memory. Video frames (1-7) do not contribute to memory, ensuring all video frames match against the same query representation.

3. **Parallel Processing**: Video frames (1-7) are processed in parallel, all using the same query memory. They are independent of each other, enabling efficient batch processing.

4. **Training Efficiency**: Uses mixed precision (bfloat16), gradient clipping, and memory optimization (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) for stable training.

5. **Flexibility**: Supports both point and box prompts, with iterative refinement during training.

6. **Scalability**: Currently using 4 GPUs (0, 1, 2, 3) with batch size 1 per GPU. Supports multi-node training via SLURM.

7. **Code Modifications**:
   - `training/dataset/vos_sampler.py`: Modified to select 1 query + 7 consecutive frames
   - `training/model/sam2.py`: 
     - Modified `forward_tracking()` to process query first, then all 7 video frames
     - Overrode `_prepare_memory_conditioned_features()` to ensure frames 1-7 only use query memory
     - Modified `track_step()` to skip memory encoding for frames 1-7

---

## References

- SAM2.1 Base Model: `sam2.modeling.sam2_base.SAM2Base`
- Training Model: `training.model.sam2.SAM2Train`
- Dataset: `training.dataset.vos_dataset.VOSDataset`
- Sampler: `training.dataset.vos_sampler.RandomUniformSampler`
- Loss: `training.loss_fns.MultiStepMultiMasksAndIous`
- Trainer: `training.trainer.Trainer`



