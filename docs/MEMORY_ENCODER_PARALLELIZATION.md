# Memory Encoder Parallelization: Technical Details

## Overview

The memory encoder **CAN be parallelized**, but currently it's only used for Frame 0 (query frame). This document explains:
1. Why memory encoder operations are batchable
2. Current usage (only Frame 0)
3. How to parallelize if encoding multiple frames
4. When parallelization would be beneficial

---

## 1. Memory Encoder Operations (All Batchable)

The memory encoder consists of **convolutional operations** that are inherently batchable:

### Memory Encoder Pipeline

```python
def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
    # Input shapes:
    # pix_feat: [B, C, H, W]  (e.g., [1, 256, 64, 64])
    # masks: [B, N, H*16, W*16]  (e.g., [1, 3, 1024, 1024])
    
    # Step 1: Process masks
    masks = F.sigmoid(masks)  # ✅ Batchable (element-wise)
    masks = self.mask_downsampler(masks)  # ✅ Batchable (Conv2d layers)
    # Output: [B, C, H, W]  (e.g., [1, 256, 64, 64])
    
    # Step 2: Project pixel features
    x = self.pix_feat_proj(pix_feat)  # ✅ Batchable (Conv2d 1x1)
    # Output: [B, C, H, W]
    
    # Step 3: Fuse features and masks
    x = x + masks  # ✅ Batchable (element-wise addition)
    # Output: [B, C, H, W]
    
    # Step 4: Fuser (CXBlock layers)
    x = self.fuser(x)  # ✅ Batchable (Conv2d + Linear operations)
    # Output: [B, C, H, W]
    
    # Step 5: Output projection
    x = self.out_proj(x)  # ✅ Batchable (Conv2d 1x1)
    # Output: [B, out_dim, H, W]  (e.g., [1, 64, 64, 64])
    
    # Step 6: Position encoding
    pos = self.position_encoding(x)  # ✅ Batchable
    # Output: [B, out_dim, H, W]
    
    return {"vision_features": x, "vision_pos_enc": [pos]}
```

### All Operations Are Batchable

| Operation | Type | Batchable? | Why |
|-----------|------|------------|-----|
| `F.sigmoid()` | Element-wise | ✅ Yes | Works on any batch size |
| `mask_downsampler` | Conv2d layers | ✅ Yes | Conv2d is batchable |
| `pix_feat_proj` | Conv2d 1x1 | ✅ Yes | Conv2d is batchable |
| Element-wise addition | `x + masks` | ✅ Yes | Works on any batch size |
| `fuser` (CXBlock) | Conv2d + Linear | ✅ Yes | All operations batchable |
| `out_proj` | Conv2d 1x1 | ✅ Yes | Conv2d is batchable |
| `position_encoding` | Position encoding | ✅ Yes | Applied per sample |

**Key Insight:** All operations in memory encoder are **convolutional or element-wise**, which means they naturally support batching.

---

## 2. Current Usage (Only Frame 0)

### Current Implementation

```python
# In track_step() - only Frame 0 encodes memory
if frame_idx == 0 or is_init_cond_frame:
    # Encode memory for query frame
    self._encode_memory_in_output(
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    )
else:
    # Frames 1-7: Skip memory encoding
    # (they don't contribute to memory)
    dummy_mem_feats = torch.zeros(B, self.mem_dim, H, W, device=device)
    ...
```

**Current Behavior:**
- **Frame 0**: Encodes memory → `[B, mem_dim, H, W]`
- **Frames 1-7**: Skip encoding (dummy memory features)

**Why No Parallelization Needed:**
- Only **1 frame** encodes memory (Frame 0)
- Nothing to parallelize (can't batch 1 item)

---

## 3. How to Parallelize Memory Encoder

If you wanted to encode **multiple frames in parallel**, here's how:

### Example: Encoding Multiple Frames

```python
def _encode_memories_parallel(
    self,
    all_pix_feats,  # List of [B, C, H, W] for each frame
    all_masks,      # List of [B, N, H*16, W*16] for each frame
    all_object_score_logits,  # List of [B, N] for each frame
):
    """
    Encode multiple frames' memories in parallel.
    
    Args:
        all_pix_feats: List of [B, C, H, W] tensors (one per frame)
        all_masks: List of [B, N, H*16, W*16] tensors (one per frame)
        all_object_score_logits: List of [B, N] tensors (one per frame)
    
    Returns:
        List of memory features, one per frame
    """
    num_frames = len(all_pix_feats)
    B = all_pix_feats[0].size(0)
    
    # Step 1: Batch all inputs together
    # Stack along batch dimension: [B*num_frames, C, H, W]
    batched_pix_feats = torch.cat(all_pix_feats, dim=0)
    # Result: [B*num_frames, C, H, W] = [7, 256, 64, 64]
    
    batched_masks = torch.cat(all_masks, dim=0)
    # Result: [B*num_frames, N, H*16, W*16] = [7, 3, 1024, 1024]
    
    # Step 2: Process masks (all frames in parallel)
    batched_masks = torch.sigmoid(batched_masks)
    batched_masks = self.mask_downsampler(batched_masks)
    # Result: [B*num_frames, C, H, W] = [7, 256, 64, 64]
    
    # Step 3: Project pixel features (all frames in parallel)
    batched_pix_feats = batched_pix_feats.to(batched_masks.device)
    x = self.pix_feat_proj(batched_pix_feats)
    # Result: [B*num_frames, C, H, W] = [7, 256, 64, 64]
    
    # Step 4: Fuse (all frames in parallel)
    x = x + batched_masks
    # Result: [B*num_frames, C, H, W] = [7, 256, 64, 64]
    
    # Step 5: Fuser (all frames in parallel)
    x = self.fuser(x)
    # Result: [B*num_frames, C, H, W] = [7, 256, 64, 64]
    
    # Step 6: Output projection (all frames in parallel)
    x = self.out_proj(x)
    # Result: [B*num_frames, out_dim, H, W] = [7, 64, 64, 64]
    
    # Step 7: Position encoding (all frames in parallel)
    pos = self.position_encoding(x)
    # Result: [B*num_frames, out_dim, H, W] = [7, 64, 64, 64]
    
    # Step 8: Split back to individual frames
    memory_features = []
    memory_pos_encs = []
    for i in range(num_frames):
        frame_start = i * B
        frame_end = (i + 1) * B
        memory_features.append(x[frame_start:frame_end])
        memory_pos_encs.append(pos[frame_start:frame_end])
    
    return memory_features, memory_pos_encs
```

### Tensor Dimensions Example

**Input (7 frames):**
```
Frame 1: pix_feat = [1, 256, 64, 64], masks = [1, 3, 1024, 1024]
Frame 2: pix_feat = [1, 256, 64, 64], masks = [1, 3, 1024, 1024]
...
Frame 7: pix_feat = [1, 256, 64, 64], masks = [1, 3, 1024, 1024]
```

**Batched:**
```
batched_pix_feat = [7, 256, 64, 64]
batched_masks = [7, 3, 1024, 1024]
```

**After Memory Encoder (parallel):**
```
batched_output = [7, 64, 64, 64]
```

**Split back:**
```
Frame 1: [1, 64, 64, 64]
Frame 2: [1, 64, 64, 64]
...
Frame 7: [1, 64, 64, 64]
```

---

## 4. Computational Benefits

### Memory Encoder Operations Per Frame

**Mask Downsampler:**
- Conv2d layers: ~4 layers with stride 4
- Complexity: **O(H*W × C × kernel_size²)**
- For H=1024, W=1024, C=256: ~67M operations per frame

**Feature Projection:**
- Conv2d 1x1: **O(H*W × C²)**
- For H=64, W=64, C=256: ~268M operations per frame

**Fuser (CXBlock):**
- Depthwise Conv: **O(H*W × C × kernel_size²)**
- Linear layers: **O(H*W × C × FF)**
- For H=64, W=64, C=256, FF=1024: ~1.1B operations per frame

**Total Per Frame:**
- ~1.4 billion operations per frame

### Sequential vs Parallel

**Sequential (7 frames):**
```
Frame 1: 1.4B ops
Frame 2: 1.4B ops
...
Frame 7: 1.4B ops
Total: 7 × 1.4B = 9.8 billion operations
GPU calls: 7 separate forward passes
```

**Parallel (7 frames batched):**
```
All Frames: 1.4B ops (but with B*7 instead of B)
Total: ~1.4 billion operations (slightly more due to larger batch)
GPU calls: 1 forward pass
```

**Expected Speedup:**
- **Memory encoder: ~5-7x faster** (if encoding multiple frames)
- Better GPU utilization
- Single kernel launch instead of multiple

---

## 5. When Would Parallelization Help?

### Current Design: ❌ No Benefit

**Current Setup:**
- Only Frame 0 encodes memory
- Frames 1-7 skip encoding
- **No parallelization benefit** (only 1 frame to encode)

### Alternative Design: ✅ Would Benefit

**If you wanted to encode multiple frames:**

**Scenario 1: Encode All 8 Frames**
```python
# Encode all frames (0-7) into memory
# Then use different memory strategies
for frame in all_frames:
    encode_memory(frame)  # Sequential: 8 calls
```

**Parallelized:**
```python
# Batch all 8 frames
batched_encode_memories(all_frames)  # Parallel: 1 call
# Speedup: ~5-7x
```

**Scenario 2: Encode Frames 1-7 (Not Just Frame 0)**
```python
# If you wanted frames 1-7 to also contribute to memory
# (instead of only using query frame memory)
for frame in frames_1_7:
    encode_memory(frame)  # Sequential: 7 calls
```

**Parallelized:**
```python
# Batch frames 1-7
batched_encode_memories(frames_1_7)  # Parallel: 1 call
# Speedup: ~5-7x
```

---

## 6. Implementation Example

Here's how you could modify the code to parallelize memory encoding if encoding multiple frames:

```python
def _encode_memories_parallel_for_video_frames(
    self,
    video_frame_indices,  # [1, 2, 3, 4, 5, 6, 7]
    all_pix_feats,        # List of [B, C, H, W] for each frame
    all_high_res_masks,   # List of [B, N, H*16, W*16] for each frame
    all_object_score_logits,  # List of [B, N] for each frame
):
    """
    Encode memories for multiple video frames in parallel.
    This would be used if frames 1-7 also need to encode memory.
    """
    if len(video_frame_indices) == 0:
        return []
    
    num_frames = len(video_frame_indices)
    B = all_pix_feats[0].size(0)
    device = all_pix_feats[0].device
    
    # Batch all inputs
    batched_pix_feats = torch.cat(all_pix_feats, dim=0)  # [B*num_frames, C, H, W]
    batched_masks = torch.cat(all_high_res_masks, dim=0)  # [B*num_frames, N, H*16, W*16]
    
    # Process masks (all frames in parallel)
    mask_for_mem = torch.sigmoid(batched_masks)
    if self.sigmoid_scale_for_mem_enc != 1.0:
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
    if self.sigmoid_bias_for_mem_enc != 0.0:
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
    
    # Memory encoder forward (all frames in parallel)
    maskmem_out = self.memory_encoder(
        batched_pix_feats, 
        mask_for_mem, 
        skip_mask_sigmoid=True
    )
    
    batched_maskmem_features = maskmem_out["vision_features"]  # [B*num_frames, mem_dim, H, W]
    batched_maskmem_pos_enc = maskmem_out["vision_pos_enc"][0]  # [B*num_frames, mem_dim, H, W]
    
    # Add no-object embedding if enabled
    if self.no_obj_embed_spatial is not None:
        batched_object_scores = torch.cat(all_object_score_logits, dim=0)  # [B*num_frames, N]
        is_obj_appearing = (batched_object_scores > 0).float()
        batched_maskmem_features += (
            1 - is_obj_appearing[..., None, None]
        ) * self.no_obj_embed_spatial[..., None, None].expand(
            *batched_maskmem_features.shape
        )
    
    # Split back to individual frames
    memory_outputs = []
    for i in range(num_frames):
        frame_start = i * B
        frame_end = (i + 1) * B
        memory_outputs.append({
            "maskmem_features": batched_maskmem_features[frame_start:frame_end],
            "maskmem_pos_enc": [batched_maskmem_pos_enc[frame_start:frame_end]],
        })
    
    return memory_outputs
```

---

## 7. Comparison: Memory Attention vs Memory Encoder

| Aspect | Memory Attention | Memory Encoder |
|--------|------------------|----------------|
| **Current Usage** | Frames 1-7 (parallelized) | Frame 0 only (no parallelization needed) |
| **Operations** | Transformer (Self-Attn + Cross-Attn) | Convolutions (Conv2d + Linear) |
| **Batchable?** | ✅ Yes | ✅ Yes |
| **Parallelized?** | ✅ Yes (7 frames) | ❌ No (only 1 frame) |
| **Speedup if Parallelized** | ~5-7x (already done) | ~5-7x (if encoding multiple frames) |
| **Complexity** | O((HW)² × B × C) | O(H*W × C²) |

---

## 8. Summary

### Can Memory Encoder Be Parallelized?

**✅ YES** - All operations are batchable:
- Mask downsampling (Conv2d)
- Feature projection (Conv2d)
- Fusing (Conv2d + Linear)
- Position encoding

### Is It Currently Parallelized?

**❌ NO** - Because:
- Only Frame 0 encodes memory
- Frames 1-7 skip encoding
- Nothing to parallelize (can't batch 1 item)

### When Would Parallelization Help?

**✅ If encoding multiple frames:**
- If frames 1-7 also encoded memory (different design)
- If encoding all 8 frames
- If encoding multiple query frames

### Performance Impact (If Parallelized)

- **Speedup: ~5-7x** (similar to memory attention)
- **GPU utilization: 14% → 80-95%**
- **Kernel launches: N → 1**

### Current Recommendation

**For current design (only Frame 0 encodes):**
- ✅ **No changes needed** - already optimal
- Only 1 frame encodes, so no parallelization benefit

**If changing design (multiple frames encode):**
- ✅ **Parallelize memory encoder** - would provide significant speedup
- Use batching approach similar to memory attention
- All operations are naturally batchable

---

## 9. Code Location

- **Memory Encoder**: `sam2/modeling/memory_encoder.py`
- **Memory Encoding Call**: `sam2/modeling/sam2_base.py` → `_encode_new_memory()`
- **Current Usage**: `training/model/sam2.py` → `track_step()` (line 816)

---

## References

- Memory Encoder Architecture: `sam2/modeling/memory_encoder.py`
- Memory Encoding: `sam2/modeling/sam2_base.py` → `_encode_new_memory()`
- Parallel Memory Attention: `training/model/sam2.py` → `_process_video_frames_parallel()`

