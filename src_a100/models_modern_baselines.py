#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern 2023-2025 Multimodal Map-Free Baselines on Unified 256x256 BEV (Experiment 5):
  1. BEVFusionPlanner256 (ICRA 2023 BEVFusion-style):
     - ResNet50d + FPN + Standard Single-View Lift-Splat-Shoot (LSS) -> 256x256 Camera BEV
     - Multi-Scale 2D LiDAR BEV ConvEncoder -> 256x256 LiDAR BEV
     - BEVFusion ConvFuser (Concat 512->256 + 2 Residual Conv Blocks; NO FFT/SSL/FiLM/4D-Flow/ASPP/CrossAttn)
     - Standard Conv Seg / CenterNet Det / Global-Pool Multi-Label heads
     - Spatial Pool + Ego-State GRU Autoregressive Waypoint Planner (NO CTRA/Rule-LoRA/CBF)
  2. VADLiteQueryPlanner256 (ICCV 2023 VAD / CVPR 2023 UniAD Map-Free Query-Based Paradigm):
     - Same dual-stream BEVFusion ConvFuser BEV encoder (256x256)
     - 32x32 (1024) BEV Tokens with 2D Sinusoidal Positional Encoding
     - 6 Learnable Ego-Trajectory Queries conditioned on Ego Kinematics
     - 3-Layer Transformer Decoder (Self-Attn + Cross-Attn to 1024 BEV Tokens + FFN) -> 6-Step Waypoint Decoder
     - Standard Seg / Det / ML heads (NO HD-Map, NO Diff-DP/Rule-LoRA/CBF, NO Skip U-Net/PeakSharpener/20D Refiner)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class StandardSingleViewLSS256(nn.Module):
    """Standard single-front-camera Lift-Splat-Shoot depth frustum to 256x256 BEV (no 6-sector expansion)."""

    def __init__(self, dim: int = 256, bev_size: int = 256, num_depth_bins: int = 32):
        super().__init__()
        self.dim = dim
        self.bev_size = bev_size
        self.depth_net = nn.Conv2d(dim, num_depth_bins, kernel_size=1)
        self.ctx_net = nn.Conv2d(dim, dim, kernel_size=1)

        ys = torch.linspace(0.0, 1.0, bev_size).view(bev_size, 1).expand(bev_size, bev_size)
        xs = torch.linspace(-1.0, 1.0, bev_size).view(1, bev_size).expand(bev_size, bev_size)
        dy = ys - (16.0 / 256.0)
        dx = xs
        radius = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-4, max=1.0)
        angle = torch.atan2(dx, dy)
        # Front camera FOV ~ +/- 65 deg (1.15 rad)
        in_fov = ((angle.abs() <= 1.15) & (dy >= 0.0)).float()
        a_norm = (angle / 1.15).clamp(-1.0, 1.0)
        r_norm = (radius * 2.0 - 1.0).clamp(-1.0, 1.0)
        sample_grid = torch.stack([a_norm, r_norm], dim=-1).unsqueeze(0)  # (1, 256, 256, 2)

        self.register_buffer("sample_grid", sample_grid)
        self.register_buffer("in_fov_mask", in_fov.view(1, 1, bev_size, bev_size))
        self.bev_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, fpn_front: torch.Tensor) -> torch.Tensor:
        B = fpn_front.size(0)
        depth_probs = F.softmax(self.depth_net(fpn_front).mean(dim=2), dim=1)  # (B, D, W_f)
        col_ctx = self.ctx_net(fpn_front).mean(dim=2)                          # (B, C, W_f)
        frustum = col_ctx.unsqueeze(2) * depth_probs.unsqueeze(1)              # (B, C, D, W_f)
        grid = self.sample_grid.expand(B, -1, -1, -1).to(frustum.dtype)
        bev = F.grid_sample(frustum, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        bev = bev * self.in_fov_mask.to(bev.dtype)
        return self.bev_conv(bev)


class StandardLiDARBEVEncoder256(nn.Module):
    """Multi-scale 2D Conv encoder for 3-channel 256x256 LiDAR BEV."""

    def __init__(self, in_ch: int = 3, dim: int = 256):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResBlock2D(128),
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(128, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            ResBlock2D(dim),
        )
        self.up2 = nn.Conv2d(dim + 128, 128, 1, bias=False)
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, bev_lidar: torch.Tensor) -> torch.Tensor:
        f1 = self.s1(bev_lidar)                                                   # (B, 64, 256, 256)
        f2 = self.s2(f1)                                                          # (B, 128, 128, 128)
        f3 = self.s3(f2)                                                          # (B, 256, 64, 64)
        u2 = self.up2(torch.cat([F.interpolate(f3, size=(128, 128), mode="bilinear", align_corners=False), f2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=(256, 256), mode="bilinear", align_corners=False), f1], dim=1))
        return u1


class BEVFusionConvBackbone256(nn.Module):
    """Shared BEVFusion (ICRA 2023) dual-stream ResNet50d-LSS + LiDAR ConvFuser backbone."""

    def __init__(self, pretrained_backbone: bool = True, dim: int = 256):
        super().__init__()
        self.backbone = timm.create_model("resnet50d", pretrained=pretrained_backbone, features_only=True)
        ch = self.backbone.feature_info.channels()
        self.lat_c2 = nn.Conv2d(ch[-4], dim, 1)
        self.lat_c3 = nn.Conv2d(ch[-3], dim, 1)
        self.lat_c4 = nn.Conv2d(ch[-2], dim, 1)
        self.lat_c5 = nn.Conv2d(ch[-1], dim, 1)
        self.fpn_smooth = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.lss = StandardSingleViewLSS256(dim=dim, bev_size=256, num_depth_bins=32)
        self.lidar_enc = StandardLiDARBEVEncoder256(in_ch=3, dim=dim)

        # Signature BEVFusion ConvFuser: channel concat + 3x3 Conv + 2 ResBlocks
        self.conv_fuser = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            ResBlock2D(dim),
            ResBlock2D(dim),
        )

        # Standard 256x256 Perception Heads (NO Skip U-Net, NO PeakSharpener, NO 20D Refiner)
        self.seg_head = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, 1),
        )
        self.det_hm_head = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        self.det_reg_head = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 4, 1),
        )
        self.ml_head = nn.Sequential(
            nn.Linear(dim * 2 + 14, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 10),
        )

    def encode_and_perceive(self, img_front: torch.Tensor, bev_lidar_256: torch.Tensor, geo_14d: torch.Tensor):
        if img_front.dim() == 5:
            img_front = img_front[:, 0]
        feats = self.backbone(img_front)
        c2, c3, c4, c5 = feats[-4], feats[-3], feats[-2], feats[-1]
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        fpn = self.fpn_smooth(F.interpolate(p2, size=(64, 112), mode="bilinear", align_corners=False))
        vis_global = F.adaptive_avg_pool2d(p5, 1).flatten(1)

        cam_bev = self.lss(fpn)
        lidar_bev = self.lidar_enc(bev_lidar_256)
        fused_bev = self.conv_fuser(torch.cat([cam_bev, lidar_bev], dim=1))

        bev_global = F.adaptive_avg_pool2d(fused_bev, 1).flatten(1)
        seg_logits = self.seg_head(fused_bev)
        small_hm = self.det_hm_head(fused_bev)
        small_reg = self.det_reg_head(fused_bev)
        ml_logits = self.ml_head(torch.cat([vis_global, bev_global, geo_14d], dim=-1))
        return fused_bev, vis_global, bev_global, seg_logits, small_hm, small_reg, ml_logits


class BEVFusionPlanner256(nn.Module):
    """
    Modern Baseline #1: BEVFusion-Planner (ICRA 2023 BEVFusion ConvFuser + GRU Waypoint Decoder).
    """

    def __init__(self, pretrained_backbone: bool = True, dim: int = 256):
        super().__init__()
        self.bev_net = BEVFusionConvBackbone256(pretrained_backbone=pretrained_backbone, dim=dim)
        self.plan_init = nn.Sequential(
            nn.Linear(dim * 3 + 14, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.gru_cell = nn.GRUCell(input_size=2 + 14, hidden_size=256)
        self.step_head = nn.Linear(256, 2)

    def forward(
        self,
        corrupted_imgs: torch.Tensor,
        bev_lidar_256: torch.Tensor,
        geo_14d: torch.Tensor,
        temporal_8d: torch.Tensor = None,
        bev_prev_256: torch.Tensor = None,
    ):
        fused_bev, vis_g, bev_g, seg_logits, small_hm, small_reg, ml_logits = self.bev_net.encode_and_perceive(
            corrupted_imgs, bev_lidar_256, geo_14d
        )
        B = fused_bev.size(0)
        bev_max = F.adaptive_max_pool2d(fused_bev, 1).flatten(1)
        h = self.plan_init(torch.cat([vis_g, bev_g, bev_max, geo_14d], dim=-1))

        # Autoregressive 6-step GRU waypoint decoding from ego origin (64.0, 8.0)
        cur_pt = torch.tensor([64.0, 8.0], device=fused_bev.device, dtype=fused_bev.dtype).view(1, 2).expand(B, 2)
        steps = []
        for _ in range(6):
            cur_norm = (cur_pt - 64.0) / 64.0
            h = self.gru_cell(torch.cat([cur_norm, geo_14d.to(h.dtype)], dim=-1), h)
            delta = self.step_head(h)
            cur_pt = cur_pt + delta
            steps.append(cur_pt)
        pred_traj = torch.stack(steps, dim=1)

        return {
            "multi_label_logits": ml_logits,
            "seg_logits_256": seg_logits,
            "small_hm_256": small_hm,
            "small_reg_256": small_reg,
            "pred_traj": pred_traj,
        }


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    y = y.flatten().float() / float(h)
    x = x.flatten().float() / float(w)
    half = dim // 4
    omega = torch.exp(torch.arange(half, device=device).float() * (-math.log(10000.0) / max(half, 1)))
    out_y = y[:, None] * omega[None, :]
    out_x = x[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
    return emb.unsqueeze(0)  # (1, H*W, dim)


class VADLiteQueryPlanner256(nn.Module):
    """
    Modern Baseline #2: VAD-Lite (Map-Free) / QueryPlan-DETR (ICCV 2023 Query-Based E2E Planning Paradigm).
    Uses 6 learnable Ego-Trajectory Queries attending to 32x32 BEV tokens via a 3-Layer Transformer Decoder.
    """

    def __init__(self, pretrained_backbone: bool = True, dim: int = 256):
        super().__init__()
        self.bev_net = BEVFusionConvBackbone256(pretrained_backbone=pretrained_backbone, dim=dim)
        self.bev_pool32 = nn.AdaptiveAvgPool2d((32, 32))
        self.bev_tok_proj = nn.Linear(dim, dim)

        # Learnable Ego-Trajectory Queries (6 horizon steps) + Ego-Motion Conditioning
        self.ego_queries = nn.Parameter(torch.randn(1, 6, dim) * 0.02)
        self.ego_state_proj = nn.Sequential(
            nn.Linear(14 + 8, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=8, dim_feedforward=dim * 2, dropout=0.1, activation="gelu", batch_first=True
        )
        self.traj_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.traj_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.traj_head[-1].weight)
        nn.init.zeros_(self.traj_head[-1].bias)

    def forward(
        self,
        corrupted_imgs: torch.Tensor,
        bev_lidar_256: torch.Tensor,
        geo_14d: torch.Tensor,
        temporal_8d: torch.Tensor = None,
        bev_prev_256: torch.Tensor = None,
    ):
        fused_bev, _, _, seg_logits, small_hm, small_reg, ml_logits = self.bev_net.encode_and_perceive(
            corrupted_imgs, bev_lidar_256, geo_14d
        )
        B = fused_bev.size(0)
        if temporal_8d is None:
            v_fwd = (geo_14d[:, 0:1] * 15.0).clamp(0.0, 25.0)
            v_lft = (geo_14d[:, 1:2] * 5.0).clamp(-6.0, 6.0)
            zeros = torch.zeros_like(v_fwd)
            temporal_8d = torch.cat([v_fwd, v_lft, zeros, zeros, zeros, zeros, v_fwd, zeros], dim=-1)

        # 1024 BEV tokens with 2D sinusoidal positional embeddings
        bev_32 = self.bev_pool32(fused_bev).flatten(2).transpose(1, 2)  # (B, 1024, 256)
        pos_2d = build_2d_sincos_pos_embed(32, 32, bev_32.size(-1), bev_32.device).to(bev_32.dtype)
        memory = self.bev_tok_proj(bev_32) + pos_2d

        # Ego-motion conditioned trajectory queries -> 3-layer Transformer Decoder
        ego_emb = self.ego_state_proj(torch.cat([geo_14d, temporal_8d.to(geo_14d.dtype)], dim=-1)).unsqueeze(1)
        q = self.ego_queries.expand(B, -1, -1) + ego_emb
        dec_out = self.traj_decoder(tgt=q, memory=memory)  # (B, 6, 256)
        delta_128 = self.traj_head(dec_out)                # (B, 6, 2)

        # Linear constant-velocity ego anchor (1st-order kinematics, no 2nd-order CTRA or CBF)
        v_fwd = temporal_8d[:, 0:1].clamp(0.0, 25.0)
        ts = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=fused_bev.device, dtype=torch.float32).unsqueeze(0)
        y_lin = (8.0 + (v_fwd * ts) / 0.5).clamp(10.5, 124.0)
        x_lin = torch.full_like(y_lin, 64.0)
        anchor_128 = torch.stack([x_lin, y_lin], dim=-1).to(delta_128.dtype)
        pred_traj = anchor_128 + delta_128

        return {
            "multi_label_logits": ml_logits,
            "seg_logits_256": seg_logits,
            "small_hm_256": small_hm,
            "small_reg_256": small_reg,
            "pred_traj": pred_traj,
        }
