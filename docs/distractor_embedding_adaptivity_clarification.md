# Clarification: Distractor Embedding vs Attention Adaptivity

## The Question

You asked: "The distractor embedding is learnable, meaning it will be fixed in inference as stable weights. It does not change for any frame and any video, is this right?"

**Answer: YES, you are absolutely correct!**

## What is Fixed vs What is Adaptive

### Fixed (After Training):
1. **Distractor Embedding**: `distractor_embed_spatial` `[64]`
   - Same value for all DRM slots
   - Same value for all frames
   - Same value for all videos
   - Fixed after training (learned once, then constant)

2. **Attention Parameters**: Learned Q, K, V projection matrices
   - Fixed after training
   - Same for all frames/videos

### Adaptive (Computed Dynamically):
1. **Attention Weights**: Computed for each frame/video
   - **Different** for each frame
   - **Different** for each video
   - **Different** for different DRM features
   - Computed based on: `attention_weights = softmax(Q @ K^T / sqrt(d))`

## How Adaptivity Works

### Example: Same Distractor Embedding, Different Attention Weights

**Scenario 1: DRM from Frame 2 (high divergence distractor)**
```python
# Fixed distractor embedding (same for all)
drm_feats_frame2 = memory_encoder(...) + distractor_embed_spatial  # [1, 64, 64, 64]

# Attention weights computed dynamically
Q = current_frame_features @ W_q  # Current frame (Frame 5)
K = memory_features @ W_k  # Includes drm_feats_frame2
attention_weights = softmax(Q @ K^T / sqrt(d))
# Result: Low attention to DRM (e.g., 0.02) because features are very different
```

**Scenario 2: DRM from Frame 4 (moderate divergence distractor)**
```python
# Same fixed distractor embedding
drm_feats_frame4 = memory_encoder(...) + distractor_embed_spatial  # [1, 64, 64, 64]

# Attention weights computed dynamically (different!)
Q = current_frame_features @ W_q  # Current frame (Frame 5)
K = memory_features @ W_k  # Includes drm_feats_frame4
attention_weights = softmax(Q @ K^T / sqrt(d))
# Result: Slightly higher attention to DRM (e.g., 0.05) because features are somewhat similar
```

**Key Point**: The distractor embedding is the **same** in both cases, but the **attention weights are different** because:
- The DRM features themselves are different (different distractor masks)
- The query features are different (different current frames)
- The similarity between Q and K is different

## What "Adaptive" Means in This Context

When I said "Adaptive: handles different scenarios (high/low confidence RAM, different DRM types)", I meant:

1. **Different DRM Features → Different Attention Weights**
   - DRM from a very different-looking distractor → very low attention
   - DRM from a similar-looking distractor → slightly higher attention (but still low)
   - The distractor embedding helps ensure DRM generally gets low attention, but the exact weight depends on feature similarity

2. **Different Query Frames → Different Attention Weights**
   - Current frame looks very different from DRM → very low attention
   - Current frame looks somewhat similar to DRM → slightly higher attention
   - The attention mechanism adapts to the current context

3. **Different RAM Features → Different Attention Weights**
   - High confidence RAM → high attention
   - Low confidence RAM → lower attention
   - The attention mechanism adapts to RAM quality

## The Mechanism

```python
# Fixed after training
distractor_embed_spatial = [0.1, -0.2, 0.3, ...]  # [64] - same for all DRM slots

# For each frame/video, dynamically computed:
for frame in video_frames:
    # DRM features (different for each DRM slot, but all have same distractor embedding)
    drm_feats_1 = encode_distractor(mask_1) + distractor_embed_spatial
    drm_feats_2 = encode_distractor(mask_2) + distractor_embed_spatial
    
    # Attention weights (computed dynamically)
    Q = current_frame @ W_q  # Different for each frame
    K = [query, ram_1, ram_2, drm_1, drm_2] @ W_k  # Different for each memory bank
    attention_weights = softmax(Q @ K^T / sqrt(d))  # Different for each frame!
    
    # Result: Different attention weights even though distractor embedding is the same
    # attention_weights = [0.3, 0.25, 0.2, 0.02, 0.03]  # Frame 1
    # attention_weights = [0.35, 0.3, 0.25, 0.01, 0.04]  # Frame 2 (different!)
```

## Summary

**You are correct**: The distractor embedding is fixed after training (same for all frames/videos).

**But the adaptivity comes from**:
- **Dynamically computed attention weights** that vary based on:
  - Feature content (what the DRM actually looks like)
  - Context (what the current frame looks like)
  - Similarity between query and memory features

**The distractor embedding's role**:
- Acts as a fixed "tag" that helps the attention mechanism learn to distinguish DRM from RAM
- Ensures DRM generally gets low attention weights
- But the exact attention weight still depends on the feature similarity (adaptive)

So the "adaptivity" is in the **attention mechanism's response** to different features, not in the distractor embedding itself changing.
