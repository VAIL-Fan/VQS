# Position Embedding for maskmem_features: Shape Transformations

This document explains how `PositionEmbeddingSine` is applied to `maskmem_features` and how feature shapes change throughout the process.

## Overview

`PositionEmbeddingSine` is applied to `maskmem_features` **inside the `MemoryEncoder`** module. The position embedding is computed **after** the memory features are generated and is returned alongside the features for use in the memory attention mechanism.

## Process Flow

### Step 1: Memory Encoder Forward Pass

**Location**: `sam2/modeling/memory_encoder.py` → `MemoryEncoder.forward()`

```python
def forward(self, pix_feat: torch.Tensor, masks: torch.Tensor, skip_mask_sigmoid: bool = False):
    # Input shapes:
    # pix_feat: [B, C_in, H, W]  where C_in = 256 (hidden_dim from backbone)
    # masks: [B, N, H_img, W_img]  where H_img = W_img = 768 (input image size)
    
    # Step 1: Process masks
    if not skip_mask_sigmoid:
        masks = F.sigmoid(masks)  # [B, N, H_img, W_img]
    masks = self.mask_downsampler(masks)  # [B, C_in, H, W] = [B, 256, 64, 64]
    
    # Step 2: Project pixel features
    x = self.pix_feat_proj(pix_feat)  # [B, 256, 64, 64]
    
    # Step 3: Fuse features and masks
    x = x + masks  # [B, 256, 64, 64]  (element-wise addition)
    
    # Step 4: Fuser (CXBlock layers)
    x = self.fuser(x)  # [B, 256, 64, 64]
    
    # Step 5: Output projection (if out_dim != in_dim)
    x = self.out_proj(x)  # [B, out_dim, H, W] = [B, 64, 64, 64]
    
    # Step 6: Position encoding ← HERE IS WHERE PositionEmbeddingSine IS APPLIED
    pos = self.position_encoding(x).to(x.dtype)  # [B, out_dim, H, W] = [B, 64, 64, 64]
    
    return {"vision_features": x, "vision_pos_enc": [pos]}
```

### Step 2: PositionEmbeddingSine Forward Pass

**Location**: `sam2/modeling/position_encoding.py` → `PositionEmbeddingSine.forward()`

```python
def forward(self, x: torch.Tensor):
    B = x.shape[0]
    cache_key = (x.shape[-2], x.shape[-1])  # (H, W) = (64, 64)
    return self._pe(B, x.device, *cache_key)

def _pe(self, B, device, *cache_key):
    H, W = cache_key  # H = 64, W = 64
    
    # Create coordinate grids
    y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device)
    y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)  # [B, H, W]
    
    x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device)
    x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)  # [B, H, W]
    
    # Normalize coordinates
    if self.normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # [B, H, W]
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # [B, H, W]
    
    # Compute sinusoidal encodings
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
    dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
    
    pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
    pos_y = y_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
    
    # Apply sin/cos
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)  # [B, H, W, num_pos_feats]
    
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)  # [B, H, W, num_pos_feats]
    
    # Concatenate y and x encodings
    pos = torch.cat((pos_y, pos_x), dim=3)  # [B, H, W, num_pos_feats*2]
    
    # Permute to [B, C, H, W] format
    pos = pos.permute(0, 3, 1, 2)  # [B, num_pos_feats*2, H, W]
    
    return pos  # [B, out_dim, H, W] = [B, 64, 64, 64]
```

**Key Parameters**:
- `num_pos_feats`: Half of the output dimension (e.g., if `out_dim=64`, then `num_pos_feats=32`)
- `temperature`: Default 10000 (controls frequency of sinusoidal waves)
- `normalize`: Default True (normalizes coordinates to [0, scale])
- `scale`: Default `2 * math.pi` (normalization scale)

**Shape Details**:
- Input `x`: `[B, out_dim, H, W]` = `[B, 64, 64, 64]`
- Output `pos`: `[B, out_dim, H, W]` = `[B, 64, 64, 64]`
- The position embedding has the **same shape** as the input features
- The embedding dimension (`out_dim`) must be **even** (since it's split into y and x components)

### Step 3: Shape Summary

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| **Input to MemoryEncoder** | `pix_feat` | `[B, 256, 64, 64]` | Backbone features |
| | `masks` | `[B, N, 768, 768]` | High-res mask logits |
| **After mask_downsampler** | `masks` | `[B, 256, 64, 64]` | Downsampled masks |
| **After fusion** | `x` | `[B, 256, 64, 64]` | Fused features |
| **After fuser** | `x` | `[B, 256, 64, 64]` | Processed features |
| **After out_proj** | `x` | `[B, 64, 64, 64]` | Projected to memory dim |
| **Position Embedding** | `pos` | `[B, 64, 64, 64]` | Spatial position encoding |
| **Output** | `maskmem_features` | `[B, 64, 64, 64]` | Memory features |
| | `maskmem_pos_enc` | `[B, 64, 64, 64]` | Position encodings |

**Important**: The position embedding has the **same shape** as `maskmem_features` (`[B, 64, 64, 64]`). This allows them to be added together or used separately in the attention mechanism.

## Usage in Memory Attention

**Location**: `sam2/modeling/memory_attention.py` → `MemoryAttentionLayer._forward_ca()`

The position embeddings are used in the cross-attention mechanism:

```python
def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
    # Cross-Attention
    tgt2 = self.norm2(tgt)
    tgt2 = self.cross_attn_image(
        q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
        k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,  # ← pos added to keys
        v=memory,
    )
    tgt = tgt + self.dropout2(tgt2)
    return tgt
```

**Shape Flow in Memory Attention**:

1. **Input to MemoryAttention**:
   - `curr`: `[HW, B, C_img]` = `[4096, B, 256]` (image features, flattened)
   - `memory`: `[M*HW, B, C_mem]` = `[M*4096, B, 64]` (memory features, flattened)
   - `curr_pos`: `[HW, B, C_img]` = `[4096, B, 256]` (image position encodings)
   - `memory_pos`: `[M*HW, B, C_mem]` = `[M*4096, B, 64]` (memory position encodings)

2. **Before Cross-Attention**:
   - Memory features are reshaped: `[M, B, 64, 64, 64]` → `[M*4096, B, 64]`
   - Position encodings are reshaped: `[M, B, 64, 64, 64]` → `[M*4096, B, 64]`

3. **In Cross-Attention**:
   - Keys: `k = memory + pos` (element-wise addition)
   - Shape: `[M*4096, B, 64]` + `[M*4096, B, 64]` = `[M*4096, B, 64]`
   - The position encoding is **added** to the memory features before computing attention

## Key Points

1. **Position Embedding Shape**: The position embedding has the **same shape** as `maskmem_features`: `[B, 64, 64, 64]`

2. **Computation**: Position embedding is computed **after** memory features are generated, based on the spatial dimensions (H, W) of the feature map

3. **Sinusoidal Encoding**: Uses sinusoidal functions (sin/cos) to encode 2D spatial positions, similar to Transformer positional encodings

4. **Normalization**: Coordinates are normalized to [0, 2π] by default, ensuring consistent encoding across different feature map sizes

5. **Usage**: Position embeddings are **added** to memory features in the cross-attention mechanism (`k = memory + pos`), providing spatial context to the attention computation

6. **Dimension Requirement**: The embedding dimension (`out_dim`) must be **even** because it's split into y and x components (each gets `out_dim // 2` dimensions)

## Example for 768×768 Input Image

For a 768×768 input image with backbone stride 16:

- **Feature map size**: H = W = 768 / 16 = **48** (for some backbones) or **64** (for Hiera-B+)
- **Memory dimension**: C_mem = **64**
- **Batch size**: B = **1** (typically)

**Shapes**:
- `maskmem_features`: `[1, 64, 64, 64]`
- `maskmem_pos_enc`: `[1, 64, 64, 64]`

**After flattening for attention**:
- `maskmem_features`: `[1, 4096, 64]` (64×64 = 4096 spatial locations)
- `maskmem_pos_enc`: `[1, 4096, 64]`

**In memory bank** (with M memory slots):
- `memory`: `[M, 4096, 64]` (M slots × 4096 tokens × 64 dims)
- `memory_pos`: `[M, 4096, 64]` (same shape)

## Configuration

The position encoding is configured when creating the `MemoryEncoder`:

```python
from sam2.modeling.position_encoding import PositionEmbeddingSine

position_encoding = PositionEmbeddingSine(
    num_pos_feats=out_dim // 2,  # e.g., 32 for out_dim=64
    temperature=10000,
    normalize=True,
    scale=2 * math.pi,
)

memory_encoder = MemoryEncoder(
    out_dim=64,
    mask_downsampler=mask_downsampler,
    fuser=fuser,
    position_encoding=position_encoding,  # ← PositionEmbeddingSine instance
    in_dim=256,
)
```

## Summary

- **PositionEmbeddingSine** is applied **inside MemoryEncoder** after memory features are generated
- **Input shape**: `[B, out_dim, H, W]` = `[B, 64, 64, 64]`
- **Output shape**: `[B, out_dim, H, W]` = `[B, 64, 64, 64]` (same as input)
- **Computation**: Generates sinusoidal encodings for 2D spatial positions
- **Usage**: Added to memory features in cross-attention (`k = memory + pos`)
- **Purpose**: Provides spatial context to help the attention mechanism understand where features are located in the 2D space

