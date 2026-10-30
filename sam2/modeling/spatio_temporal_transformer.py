# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from torch import nn, Tensor
from typing import Optional

from sam2.modeling.sam2_utils import get_activation_fn


class SpatioTemporalTransformerLayer(nn.Module):
    """
    A transformer layer that performs self-attention across spatio-temporal tokens.
    Each token represents a (temporal, spatial_h, spatial_w) location.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=False
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = get_activation_fn(activation)
    
    def forward(
        self,
        x: Tensor,  # [T*H*W, B, C] where T=temporal, H=height, W=width
        pos_embed: Optional[Tensor] = None,  # [T*H*W, B, C] positional encoding
        **kwargs,  # Additional kwargs like chunk_size for memory efficiency
    ) -> Tensor:
        """
        Args:
            x: Input tokens of shape [T*H*W, B, C]
            pos_embed: Optional positional encoding of shape [T*H*W, B, C]
        
        Returns:
            Output tokens of shape [T*H*W, B, C]
        """
        # Add positional encoding if provided
        if pos_embed is not None:
            x = x + pos_embed
        
        # Self-attention with optional chunking for memory efficiency
        x2 = self.norm1(x)
        
        # Use chunked attention if chunk_size is provided to save memory
        # This reduces memory from O(N²) to O(N*chunk_size) where N = T*H*W
        chunk_size = kwargs.get('chunk_size', None)
        if chunk_size is not None and x2.size(0) > chunk_size:
            # Process in chunks: each chunk attends to all keys/values
            chunk_outputs = []
            for i in range(0, x2.size(0), chunk_size):
                chunk_end = min(i + chunk_size, x2.size(0))
                chunk_q = x2[i:chunk_end]  # [chunk_size, B, C]
                chunk_k = x2  # [T*H*W, B, C] - attend to all
                chunk_v = x2  # [T*H*W, B, C]
                chunk_out, _ = self.self_attn(chunk_q, chunk_k, chunk_v)
                chunk_outputs.append(chunk_out)
            x2 = torch.cat(chunk_outputs, dim=0)  # [T*H*W, B, C]
        else:
            x2, _ = self.self_attn(x2, x2, x2)  # [T*H*W, B, C]
        
        x = x + self.dropout1(x2)
        
        # Feedforward
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Spatio-temporal transformer that processes features across temporal and spatial dimensions.
    
    Input: [B*T, C, H, W] where T is number of frames, B is batch size per frame
    Output: [B*T, C, H, W] with enhanced spatio-temporal features
    
    Uses downsampling (1/4 scale) to reduce memory usage, with skip connections to preserve
    original features. Features are downsampled before transformer processing and upsampled
    back, then added to the original features via skip connection.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        use_temporal_pos_embed: bool = True,
        use_spatial_pos_embed: bool = True,
        max_temporal_pos_frames: int = 32,
        downsample_scale: float = 0.25,  # 1/4 scale to reduce memory
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_temporal_pos_embed = use_temporal_pos_embed
        self.use_spatial_pos_embed = use_spatial_pos_embed
        self.max_temporal_pos_frames = max_temporal_pos_frames
        self.downsample_scale = downsample_scale
        
        # Downsampling layer: 1/4 scale (64×64 -> 16×16)
        # Use average pooling for downsampling to preserve information
        self.downsample = nn.AvgPool2d(
            kernel_size=4, 
            stride=4, 
            padding=0
        )
        
        # Upsampling layer: 4x scale (16×16 -> 64×64)
        # Use bilinear interpolation for smooth upsampling
        self.upsample = nn.Upsample(
            scale_factor=4, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            SpatioTemporalTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Temporal positional encoding (learnable)
        if use_temporal_pos_embed:
            # Learnable temporal positional embedding.
            # IMPORTANT: must be >= the maximum T used by the spatio-temporal module.
            # In our training/inference, the query frame is handled separately, so T can be (num_frames - 1).
            # We set a safe default (32) to support ablations like 1 query + 15 video frames (T=15).
            if max_temporal_pos_frames < 1:
                raise ValueError(f"max_temporal_pos_frames must be >= 1, got {max_temporal_pos_frames}")
            self.temporal_pos_embed = nn.Parameter(torch.zeros(max_temporal_pos_frames, 1, d_model))
            nn.init.normal_(self.temporal_pos_embed, std=0.02)
        
        # Spatial positional encoding (sine)
        if use_spatial_pos_embed:
            from sam2.modeling.position_encoding import PositionEmbeddingSine
            self.spatial_pos_encoding = PositionEmbeddingSine(
                num_pos_feats=d_model,  # Use full d_model to match temporal pos embed dimension
                normalize=True,
                scale=None,
                temperature=10000,
            )
    
    def _create_spatio_temporal_pos_embed(
        self,
        T: int,  # number of temporal frames
        H: int,  # spatial height
        W: int,  # spatial width
        device: torch.device,
    ) -> Tensor:
        """
        Create combined spatio-temporal positional encoding.
        
        Returns:
            pos_embed: [T*H*W, 1, C] positional encoding
        """
        # Create spatial positional encoding
        if self.use_spatial_pos_embed:
            # Create dummy feature map to get spatial pos encoding
            dummy_feat = torch.zeros(1, self.d_model, H, W, device=device)
            spatial_pos = self.spatial_pos_encoding(dummy_feat)  # [1, C, H, W]
            spatial_pos = spatial_pos.flatten(2).permute(2, 0, 1)  # [H*W, 1, C]
            # Repeat for each temporal frame
            spatial_pos = spatial_pos.repeat(T, 1, 1)  # [T*H*W, 1, C]
        else:
            spatial_pos = torch.zeros(T * H * W, 1, self.d_model, device=device)
        
        # Add temporal positional encoding
        if self.use_temporal_pos_embed:
            if T > self.temporal_pos_embed.shape[0]:
                raise ValueError(
                    f"T={T} exceeds max_temporal_pos_frames={self.temporal_pos_embed.shape[0]}. "
                    f"Increase spatio_temporal_transformer.max_temporal_pos_frames in your YAML."
                )
            temporal_pos = self.temporal_pos_embed[:T]  # [T, 1, C]
            # Expand to [T*H*W, 1, C]
            temporal_pos = temporal_pos.repeat_interleave(H * W, dim=0)
            pos_embed = spatial_pos + temporal_pos
        else:
            pos_embed = spatial_pos
        
        return pos_embed
    
    def forward(
        self,
        x: Tensor,  # [B*T, C, H, W]
        T: Optional[int] = None,  # Optional: explicitly specify number of frames
    ) -> Tensor:
        """
        Args:
            x: Input features of shape [B*T, C, H, W]
               where B is batch size per frame, T is number of frames
            T: Optional number of temporal frames. If not provided, uses self._last_T
        
        Returns:
            Output features of shape [B*T, C, H, W]
        """
        B_T, C, H, W = x.shape
        
        # Store original features for skip connection
        x_skip = x  # [B*T, C, H, W]
        
        # Get T from parameter, stored attribute, or infer
        if T is not None:
            pass  # Use provided T
        elif hasattr(self, '_last_T'):
            T = self._last_T
        else:
            # Fallback: assume B=1 per frame, so B_T = T
            # This is a heuristic - should be avoided in practice
            T = B_T
            logging.warning(f"Inferring T={T} from input shape. Consider passing T explicitly.")
        
        B = B_T // T  # Batch size per frame
        assert B * T == B_T, f"B*T ({B}*{T}={B*T}) must equal input batch dimension ({B_T})"
        
        # Downsample features to 1/4 scale to reduce memory
        # [B*T, C, H, W] -> [B*T, C, H//4, W//4]
        x_down = self.downsample(x)  # [B*T, C, H//4, W//4]
        H_down, W_down = x_down.shape[2], x_down.shape[3]
        
        # Reshape to [T*H_down*W_down, B, C] for transformer
        # Flatten spatial and temporal dimensions
        x_flat = x_down.view(B, T, C, H_down, W_down)  # [B, T, C, H_down, W_down]
        x_flat = x_flat.permute(1, 3, 4, 0, 2)  # [T, H_down, W_down, B, C]
        x_flat = x_flat.reshape(T * H_down * W_down, B, C)  # [T*H_down*W_down, B, C]
        
        # Create positional encoding for downsampled features
        pos_embed = self._create_spatio_temporal_pos_embed(
            T=T, H=H_down, W=W_down, device=x.device
        )  # [T*H_down*W_down, 1, C]
        # Expand to batch dimension
        pos_embed = pos_embed.expand(-1, B, -1)  # [T*H_down*W_down, B, C]
        
        # Apply transformer layers with chunked attention for memory efficiency
        # Use chunk_size to process attention in smaller chunks (reduces memory from O(N²) to O(N*chunk_size))
        # For 7 frames × 16×16 = 1,792 tokens (downsampled from 28,672), use chunk_size=256 (one frame worth of tokens)
        chunk_size = H_down * W_down  # Process one frame at a time in attention
        
        for layer in self.layers:
            x_flat = layer(x_flat, pos_embed=pos_embed, chunk_size=chunk_size)
        
        # Final norm
        x_flat = self.norm(x_flat)  # [T*H_down*W_down, B, C]
        
        # Reshape back to [B*T, C, H_down, W_down]
        x_flat = x_flat.reshape(T, H_down, W_down, B, C)  # [T, H_down, W_down, B, C]
        x_flat = x_flat.permute(3, 0, 4, 1, 2)  # [B, T, C, H_down, W_down]
        x_down_out = x_flat.reshape(B * T, C, H_down, W_down)  # [B*T, C, H_down, W_down]
        
        # Upsample back to original resolution
        # [B*T, C, H_down, W_down] -> [B*T, C, H, W]
        x_up = self.upsample(x_down_out)  # [B*T, C, H, W]
        
        # Add skip connection: combine upsampled features with original features
        # This preserves the original information while adding spatio-temporal enhancements
        x_out = x_up + x_skip  # [B*T, C, H, W]
        
        return x_out

