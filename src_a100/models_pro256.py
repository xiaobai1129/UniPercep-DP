import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models_full import AsymmetricLoss, VLMRagLoRAController

# Enable A100 TF32 hardware acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ModelEMA:
    """
    Upgrade 6: Exponential Moving Average (EMA) of model weights for stable SOTA evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            if k in msd:
                model_v = msd[k].detach()
                if ema_v.dtype.is_floating_point:
                    ema_v.mul_(self.decay).add_(model_v.to(ema_v.dtype), alpha=1.0 - self.decay)
                else:
                    ema_v.copy_(model_v)


class GatedSSLReconstructionHead(nn.Module):
    """
    Upgrade 5: Self-Supervised Reconstruction Head with Gated Feature Refinement.
    Not only reconstructs the clean front image from degraded weak-network inputs,
    but also injects denoised high-frequency residual features back into the FPN backbone.
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.denoise_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.denoise_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.rgb_decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, vis_fpn: torch.Tensor, target_hw: tuple, use_ssl: bool = True):
        if not use_ssl:
            dummy_rgb = F.interpolate(
                vis_fpn[:, :3].sigmoid(), size=target_hw, mode="bilinear", align_corners=False
            )
            return vis_fpn, dummy_rgb

        clean_feat = self.denoise_stem(vis_fpn)
        gate = self.denoise_gate(torch.cat([vis_fpn, clean_feat], dim=1))
        refined_fpn = vis_fpn + gate * clean_feat

        pred_clean_img = self.rgb_decoder(refined_fpn)
        if pred_clean_img.shape[-2:] != target_hw:
            pred_clean_img = F.interpolate(pred_clean_img, size=target_hw, mode="bilinear", align_corners=False)
        return refined_fpn, pred_clean_img


class PolarLiftSplatBEV256(nn.Module):
    """
    Upgrade 1 & 2: 6-Camera Surround-View Perspective-to-BEV Polar Sector Projection (256x256).
    Maps 6 surround camera FPN features (B, 6, C, H_f, W_f) into a unified 256x256 BEV feature map
    using explicit camera azimuth sector weights and learned radial depth distribution.
    """

    def __init__(self, dim: int = 256, bev_size: int = 256, num_depth_bins: int = 32):
        super().__init__()
        self.dim = dim
        self.bev_size = bev_size
        self.num_depth_bins = num_depth_bins

        # 6 camera sector embeddings (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)
        self.cam_sector_emb = nn.Parameter(torch.randn(1, 6, dim, 1, 1) * 0.02)

        # Depth distribution & context projection along columns
        self.depth_head = nn.Conv2d(dim, num_depth_bins, kernel_size=1)
        self.context_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Precompute 256x256 BEV polar coordinates (ego vehicle at (128, 16) looking forward along +Y)
        ys = torch.linspace(0.0, 1.0, bev_size).view(bev_size, 1).expand(bev_size, bev_size)
        xs = torch.linspace(-1.0, 1.0, bev_size).view(1, bev_size).expand(bev_size, bev_size)
        dy = ys - (16.0 / 256.0)
        dx = xs
        radius = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-4, max=1.0)
        angle = torch.atan2(dx, dy)  # 0 rad = straight ahead (+Y), +pi/2 = right, -pi/2 = left

        # 6 camera center azimuth angles in radians:
        # [Front: 0, Front-Left: -55 deg, Front-Right: +55 deg, Back: 180 deg, Back-Left: -115 deg, Back-Right: +115 deg]
        cam_angles = torch.tensor([0.0, -0.96, 0.96, 3.14159, -2.01, 2.01], dtype=torch.float32).view(6, 1, 1)
        angle_diff = torch.atan2(torch.sin(angle.unsqueeze(0) - cam_angles), torch.cos(angle.unsqueeze(0) - cam_angles))
        sector_weights = F.softmax(-2.2 * (angle_diff ** 2), dim=0)  # (6, 256, 256)

        self.register_buffer("sector_weights", sector_weights.unsqueeze(0).unsqueeze(2))  # (1, 6, 1, 256, 256)
        self.register_buffer("bev_radius", radius)  # (256, 256)
        self.register_buffer("bev_angle_norm", (angle_diff / 1.2).clamp(-1.0, 1.0))  # (6, 256, 256)

        self.bev_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, cam_feats: torch.Tensor):
        """
        cam_feats: (B, 6, C, H_f, W_f)
        returns: vis_bev_256 (B, C, 256, 256)
        """
        B, N, C, H_f, W_f = cam_feats.shape
        cam_feats = cam_feats + self.cam_sector_emb

        flat_feats = cam_feats.reshape(B * N, C, H_f, W_f)
        # Pool vertical rows into depth probability distribution (B*N, D, W_f)
        depth_logits = self.depth_head(flat_feats).mean(dim=2)  # (B*N, D, W_f)
        depth_probs = F.softmax(depth_logits, dim=1)            # (B*N, D, W_f)
        col_context = self.context_proj(flat_feats).mean(dim=2) # (B*N, C, W_f)

        # Outer product -> Polar frustum (B*N, C, D, W_f)
        polar_frustum = col_context.unsqueeze(2) * depth_probs.unsqueeze(1)

        # Sample polar frustum onto 256x256 Cartesian BEV grid for each camera
        r_grid = (self.bev_radius * 2.0 - 1.0).unsqueeze(0).expand(N, -1, -1)  # (6, 256, 256)
        a_grid = self.bev_angle_norm                                           # (6, 256, 256)
        sample_grid = torch.stack([a_grid, r_grid], dim=-1)                    # (6, 256, 256, 2)
        sample_grid = sample_grid.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * N, self.bev_size, self.bev_size, 2)

        cam_bev = F.grid_sample(
            polar_frustum, sample_grid.to(polar_frustum.dtype), mode="bilinear", padding_mode="zeros", align_corners=False
        )
        cam_bev = cam_bev.view(B, N, C, self.bev_size, self.bev_size)

        # Weighted sum across 6 surround sectors + front high-res residual
        vis_bev_256 = (cam_bev * self.sector_weights.to(cam_bev.dtype)).sum(dim=1)
        front_up = F.interpolate(cam_feats[:, 0], size=(self.bev_size, self.bev_size), mode="bilinear", align_corners=False)
        vis_bev_256 = self.bev_refine(vis_bev_256 + 0.5 * front_up)
        return vis_bev_256


class ASPPModule(nn.Module):
    """
    Upgrade 2: Multi-Scale Atrous Spatial Pyramid Pooling (dilations 1, 3, 6, 12)
    Preserves both 1-pixel distant small objects and large drivable area boundaries on 256x256 BEV.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        mid = dim // 4
        self.b0 = nn.Sequential(nn.Conv2d(dim, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.GELU())
        self.b1 = nn.Sequential(nn.Conv2d(dim, mid, 3, padding=3, dilation=3, bias=False), nn.BatchNorm2d(mid), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv2d(dim, mid, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(mid), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv2d(dim, mid, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(mid), nn.GELU())
        self.proj = nn.Sequential(
            nn.Conv2d(mid * 4, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return x + self.proj(out)


class MultiSensorCrossAttnFusion256(nn.Module):
    """
    Upgrade 1 & 2: High-Resolution 256x256 Multi-Sensor Fusion with 32x32 Token Cross-Attention & ASPP.
    """

    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(14, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # Upgraded 32x32 Cross-Attention (1024 spatial tokens, 4x higher resolution than 16x16)
        self.pool_32 = nn.AdaptiveAvgPool2d((32, 32))
        self.pos_emb_32 = nn.Parameter(torch.randn(1, 32 * 32, dim) * 0.02)
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            ASPPModule(dim),
        )

    def forward(
        self,
        vis_bev_256: torch.Tensor,
        bev_lidar_256: torch.Tensor,
        geo_14d: torch.Tensor,
        use_multisensor: bool = True,
    ):
        B = vis_bev_256.size(0)
        if not use_multisensor:
            zero_bev = torch.zeros_like(vis_bev_256)
            zero_geo = torch.zeros((B, vis_bev_256.size(1)), device=vis_bev_256.device, dtype=vis_bev_256.dtype)
            fused_256 = self.out_conv(torch.cat([vis_bev_256, zero_bev, zero_bev], dim=1))
            return fused_256, zero_geo, zero_bev

        lidar_feat_256 = self.bev_encoder(bev_lidar_256)  # (B, 256, 256, 256)
        geo_emb = self.geo_mlp(geo_14d)                  # (B, 256)

        vis_tok = self.pool_32(vis_bev_256).flatten(2).transpose(1, 2) + self.pos_emb_32
        lid_tok = self.pool_32(lidar_feat_256).flatten(2).transpose(1, 2) + self.pos_emb_32
        geo_tok = geo_emb.unsqueeze(1)

        kv = torch.cat([vis_tok, lid_tok, geo_tok], dim=1)  # (B, 2049, 256)
        q = lid_tok + vis_tok + geo_tok

        attn_out1, _ = self.attn1(query=q, key=kv, value=kv)
        x = self.norm1(q + attn_out1)
        attn_out2, _ = self.attn2(query=x, key=kv, value=kv)
        x = self.norm2(x + attn_out2)
        x = self.norm3(x + self.ffn(x))

        attn_2d_32 = x.transpose(1, 2).reshape(B, -1, 32, 32)
        attn_2d_256 = F.interpolate(attn_2d_32, size=(256, 256), mode="bilinear", align_corners=False)

        fused_256 = self.out_conv(torch.cat([vis_bev_256, lidar_feat_256, attn_2d_256], dim=1))
        return fused_256, geo_emb, lidar_feat_256


class MultiModalQueryMultiLabelHead(nn.Module):
    """
    Upgrade 4: Class-Specific Query Attention Multi-Label Head.
    Fuses 6-camera visual context, LiDAR point cloud density/height statistics,
    and 14D ego motion state using 10 learnable class queries.
    """

    def __init__(self, dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.class_queries = nn.Parameter(torch.randn(1, num_classes, dim) * 0.02)
        self.lidar_stats_proj = nn.Sequential(
            nn.Linear(12, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.cls_proj = nn.Linear(dim, 1)
        self.global_fallback = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        cam_globals: torch.Tensor,     # (B, 6, 256)
        fused_256: torch.Tensor,       # (B, 256, 256, 256)
        bev_lidar_256: torch.Tensor,   # (B, 3, 256, 256)
        geo_emb: torch.Tensor,         # (B, 256)
        global_context: torch.Tensor,  # (B, 256)
    ):
        B = cam_globals.size(0)
        # Compute explicit LiDAR BEV physical statistics (mean, max, std, occupancy ratio per channel)
        l_mean = bev_lidar_256.mean(dim=(2, 3))
        l_max = bev_lidar_256.amax(dim=(2, 3))
        l_std = bev_lidar_256.std(dim=(2, 3))
        l_occ = (bev_lidar_256 > 0.05).float().mean(dim=(2, 3))
        lidar_stats = torch.cat([l_mean, l_max, l_std, l_occ], dim=-1)  # (B, 12)
        lidar_stat_tok = self.lidar_stats_proj(lidar_stats).unsqueeze(1)  # (B, 1, 256)

        spatial_tokens = F.adaptive_avg_pool2d(fused_256, (4, 4)).flatten(2).transpose(1, 2)  # (B, 16, 256)
        memory = torch.cat(
            [cam_globals, spatial_tokens, lidar_stat_tok, geo_emb.unsqueeze(1), global_context.unsqueeze(1)], dim=1
        )  # (B, 25, 256)

        q = self.class_queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=memory, value=memory)
        q_feat = self.norm(q + attn_out)  # (B, 10, 256)

        query_logits = self.cls_proj(q_feat).squeeze(-1)  # (B, 10)
        base_logits = self.global_fallback(
            torch.cat([cam_globals.mean(dim=1), global_context, lidar_stat_tok.squeeze(1)], dim=-1)
        )
        return query_logits + base_logits


class ObstacleSegHead256(nn.Module):
    """
    256x256 High-Resolution BEV Semantic Segmentation Head.
    Outputs both 256x256 high-res logits and 128x128 compatible logits.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, fused_256: torch.Tensor):
        return self.net(fused_256)  # (B, 3, 256, 256)


class SmallObjectDetHead256(nn.Module):
    """
    256x256 High-Resolution Small Object Center Heatmap & Velocity/Size Regression Head.
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.hm_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.hm_conv[0].bias, -2.19)
        self.reg_conv = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, fused_256: torch.Tensor):
        feat = self.shared(fused_256)
        return self.hm_conv(feat), self.reg_conv(feat)


class KinematicGridSampleDPPlanner256(nn.Module):
    """
    Upgrade 1 & 3: 256x256 High-Precision Differentiable DP Planner with:
      1) Velocity-Adaptive Kinematic Prior Horizon Anchors
      2) 256x256 Differentiable Softmin DP Energy Field
      3) Local Trajectory ROI `F.grid_sample` Feature & Risk Gradient Closed-Loop Refinement
    """

    def __init__(self, in_channels: int = 256, temp: float = 0.22):
        super().__init__()
        self.temp = temp
        self.risk_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Project 256-dim BEV feature to 64-dim for fast per-waypoint local ROI grid_sample
        self.local_roi_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Kinematic prior Calibration MLP (maps ego [vx, vy, yaw, cos, sin, H] -> 6-step kinematic prior)
        self.kin_prior_mlp = nn.Sequential(
            nn.Linear(14 + 256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 12),
        )

        # Closed-loop GRU cell taking:
        # [dp_anchor(2) + kin_anchor(2) + local_roi_feat(64) + local_risk_val(1) + geo_14d(14) + dp_weights(3)] = 86
        self.gru_cell = nn.GRUCell(input_size=2 + 2 + 64 + 1 + 14 + 3, hidden_size=256)
        self.step_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        x_coords_256 = torch.arange(256, dtype=torch.float32).view(1, 1, 256)
        self.register_buffer("x_coords_256", x_coords_256)
        base_rows_256 = torch.tensor([30, 44, 58, 72, 86, 100], dtype=torch.long)
        self.register_buffer("base_rows_256", base_rows_256)

    def forward(
        self,
        fused_256: torch.Tensor,
        global_context: torch.Tensor,
        geo_14d: torch.Tensor,
        dp_weights: torch.Tensor,
        use_dp_lora: bool = True,
    ):
        B = fused_256.size(0)
        pred_risk_256 = self.risk_head(fused_256).squeeze(1)  # (B, 256, 256)
        roi_feat_map = self.local_roi_proj(fused_256)         # (B, 64, 256, 256)

        # 1. Velocity-Adaptive Kinematic Prior in [0, 128] metric coordinates (ego at (64, 8))
        vx = geo_14d[:, 0:1]  # m/s
        vy = geo_14d[:, 1:2]  # m/s
        speed = torch.sqrt(vx * vx + vy * vy + 1e-6).clamp(min=0.0, max=20.0)  # (B, 1)
        time_steps = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=fused_256.device, dtype=fused_256.dtype).unsqueeze(0)

        # Base physical longitudinal & lateral displacement in 0.5m grid units
        kin_y_base = 8.0 + (speed * time_steps) / 0.5  # (B, 6)
        kin_x_base = 64.0 + (vx * time_steps * 0.25)   # (B, 6)
        kin_base = torch.stack([kin_x_base, kin_y_base], dim=-1)  # (B, 6, 2)
        kin_residual = self.kin_prior_mlp(torch.cat([global_context, geo_14d], dim=-1)).view(B, 6, 2)
        kin_anchors_128 = kin_base + kin_residual  # (B, 6, 2)

        if not use_dp_lora:
            return pred_risk_256, pred_risk_256, kin_anchors_128

        # 2. 256x256 Differentiable Dynamic Programming (strided 4 along Y -> 64x256 exact Softmin transition)
        risk_fp32 = pred_risk_256.float()
        w_obs = dp_weights[:, 0:1].float()
        w_conflict = dp_weights[:, 1:2].float()
        w_smooth = dp_weights[:, 2:3].float()

        temp = float(self.temp)
        cost_256 = (0.6 * w_obs + 0.4 * w_conflict).unsqueeze(-1) * risk_fp32  # (B, 256, 256)
        cost_64x256 = F.avg_pool2d(cost_256.unsqueeze(1), kernel_size=(4, 1), stride=(4, 1)).squeeze(1)  # (B, 64, 256)

        x_grid = self.x_coords_256.squeeze(1).to(risk_fp32.device)  # (1, 256)
        start_prior = ((x_grid - 128.0) / 128.0) ** 2 * 0.4
        v_prev = cost_64x256[:, 0, :] + start_prior
        dp_rows = [v_prev]

        p1 = 0.025 * w_smooth * 1.0
        p2 = 0.025 * w_smooth * 4.0

        for y in range(1, 64):
            padded = F.pad(v_prev.unsqueeze(1), (2, 2), mode="replicate").squeeze(1)
            neighbors = torch.stack(
                [
                    padded[:, 0:256] + p2,
                    padded[:, 1:257] + p1,
                    padded[:, 2:258],
                    padded[:, 3:259] + p1,
                    padded[:, 4:260] + p2,
                ],
                dim=-1,
            )
            min_prev = -temp * torch.logsumexp(-neighbors / temp, dim=-1)
            v_curr = cost_64x256[:, y, :] + min_prev
            dp_rows.append(v_curr)
            v_prev = v_curr

        dp_val_64 = torch.stack(dp_rows, dim=1).unsqueeze(1)  # (B, 1, 64, 256)
        dp_value_256 = F.interpolate(dp_val_64, size=(256, 256), mode="bilinear", align_corners=False).squeeze(1)

        # Sample DP optimal lateral coordinates at velocity-adaptive rows
        dyn_rows_256 = (kin_anchors_128[:, :, 1].detach() * 2.0).long().clamp(min=8, max=248)  # (B, 6)
        batch_idx = torch.arange(B, device=fused_256.device).unsqueeze(1).expand(B, 6)
        horizon_vals = dp_value_256[batch_idx, dyn_rows_256, :]  # (B, 6, 256)
        prob_x = F.softmax(-horizon_vals / temp, dim=-1)
        exp_x_128 = (prob_x * self.x_coords_256.to(prob_x.device)).sum(dim=-1) * 0.5  # convert 256->128 scale
        exp_y_128 = kin_anchors_128[:, :, 1]
        dp_anchors_128 = torch.stack([exp_x_128, exp_y_128], dim=-1)  # (B, 6, 2)

        # 3. Local Trajectory ROI `F.grid_sample` Closed-Loop GRU Refinement
        h_state = global_context
        refined_steps = []
        for t in range(6):
            cur_pt_128 = 0.7 * kin_anchors_128[:, t, :] + 0.3 * dp_anchors_128[:, t, :].to(kin_anchors_128.dtype)
            # Normalized [-1, 1] grid coordinates for F.grid_sample on 256x256 map
            gx = (cur_pt_128[:, 0] / 64.0) - 1.0
            gy = (cur_pt_128[:, 1] / 64.0) - 1.0
            sample_pt = torch.stack([gx, gy], dim=-1).view(B, 1, 1, 2).to(roi_feat_map.dtype)

            local_feat = F.grid_sample(
                roi_feat_map, sample_pt, mode="bilinear", padding_mode="border", align_corners=False
            ).view(B, 64)
            local_risk = F.grid_sample(
                pred_risk_256.unsqueeze(1).to(roi_feat_map.dtype),
                sample_pt,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).view(B, 1)

            dp_norm = (dp_anchors_128[:, t, :].to(global_context.dtype) - 64.0) / 64.0
            kin_norm = (kin_anchors_128[:, t, :].to(global_context.dtype) - 64.0) / 64.0
            gru_in = torch.cat(
                [
                    dp_norm,
                    kin_norm,
                    local_feat.to(global_context.dtype),
                    local_risk.to(global_context.dtype),
                    geo_14d.to(global_context.dtype),
                    dp_weights.to(global_context.dtype),
                ],
                dim=-1,
            )
            h_state = self.gru_cell(gru_in, h_state)
            step_delta = self.step_out(h_state)
            refined_steps.append(cur_pt_128 + step_delta)

        pred_traj_128 = torch.stack(refined_steps, dim=1)  # (B, 6, 2)
        return pred_risk_256, dp_value_256, pred_traj_128


class UniPercepDPNet_Pro256(nn.Module):
    """
    6-Direction Upgraded Architecture (`UniPercep-DP-Surround256-Pro`):
      1. 6-Camera Surround View + 256x256 High-Resolution BEV Grid (0.25m/cell)
      2. Explicit Polar Lift-Splat Camera-to-BEV Projection + 32x32 Cross-Attention + ASPP
      3. Velocity-Adaptive Kinematic Anchors + 256x256 Differentiable DP + Local ROI `grid_sample`
      4. Multi-Modal Class-Query Attention MultiLabelHead (6-Cam + LiDAR Stats + 14D Pose)
      5. Gated Self-Supervised Denoising Feature Refinement (`GatedSSLReconstructionHead`)
      6. Modern ConvNeXt-Small / Multi-Scale Backbone + ModelEMA Support
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small.fb_in22k",
        pretrained_backbone: bool = True,
        use_ssl: bool = True,
        use_multisensor: bool = True,
        use_dp_lora: bool = True,
    ):
        super().__init__()
        self.use_ssl = use_ssl
        self.use_multisensor = use_multisensor
        self.use_dp_lora = use_dp_lora

        try:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained_backbone, features_only=True
            )
        except Exception:
            try:
                self.backbone = timm.create_model(
                    "resnet50d", pretrained=pretrained_backbone, features_only=True
                )
            except Exception:
                self.backbone = timm.create_model("resnet50d", pretrained=False, features_only=True)

        feat_channels = self.backbone.feature_info.channels()
        c2_ch, c3_ch, c4_ch, c5_ch = feat_channels[-4], feat_channels[-3], feat_channels[-2], feat_channels[-1]

        self.lat_c2 = nn.Conv2d(c2_ch, 256, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3_ch, 256, kernel_size=1)
        self.lat_c4 = nn.Conv2d(c4_ch, 256, kernel_size=1)
        self.lat_c5 = nn.Conv2d(c5_ch, 256, kernel_size=1)

        self.fpn_smooth = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Upgrade 5: Gated SSL Reconstruction & Feature Refinement
        self.ssl_head = GatedSSLReconstructionHead(in_channels=256)

        # Upgrade 1 & 2: 6-Camera Polar Lift-Splat to 256x256 BEV + 32x32 Cross-Attn + ASPP
        self.polar_bev_proj = PolarLiftSplatBEV256(dim=256, bev_size=256, num_depth_bins=32)
        self.fusion_256 = MultiSensorCrossAttnFusion256(dim=256, num_heads=8)

        self.global_context_proj = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Upgrade 4: Multi-Modal Class-Query Attention MultiLabelHead
        self.multi_label_head = MultiModalQueryMultiLabelHead(dim=256, num_classes=10)

        # Upgrade 1: 256x256 High-Resolution Perception Heads
        self.seg_head_256 = ObstacleSegHead256(in_channels=256, num_classes=3)
        self.small_det_head_256 = SmallObjectDetHead256(in_channels=256)

        # RAG + LoRA Dynamic Weight Controller (Rank=32, Alpha=64)
        self.lora_controller = VLMRagLoRAController(feat_dim=256, num_classes=10, rank=32, alpha=64.0)

        # Upgrade 3: 256x256 Kinematic + Local ROI Grid-Sample Differentiable DP Planner
        self.dp_planner_256 = KinematicGridSampleDPPlanner256(in_channels=256, temp=0.22)

    def forward(
        self,
        corrupted_imgs: torch.Tensor,
        bev_lidar: torch.Tensor,
        geo_14d: torch.Tensor,
    ):
        """
        corrupted_imgs:
          - Either (B, 6, 3, H, W) 6-camera surround images
          - Or (B, 3, H, W) front image (automatically expanded to 6 sector views)
        bev_lidar:
          - Either (B, 3, 256, 256) or (B, 3, 128, 128) (automatically upsampled to 256x256)
        """
        if corrupted_imgs.dim() == 4:
            B, C_im, H_img, W_img = corrupted_imgs.shape
            front = corrupted_imgs
            # Construct 6 surround views if single-cam tensor is passed
            fl = torch.flip(front, dims=[3])
            fr = torch.roll(front, shifts=W_img // 4, dims=3)
            back = torch.flip(front, dims=[2, 3])
            bl = torch.roll(fl, shifts=W_img // 3, dims=3)
            br = torch.roll(fr, shifts=-W_img // 3, dims=3)
            corrupted_imgs = torch.stack([front, fl, fr, back, bl, br], dim=1)
        else:
            B, _, C_im, H_img, W_img = corrupted_imgs.shape

        if bev_lidar.shape[-2:] != (256, 256):
            bev_lidar_256 = F.interpolate(bev_lidar, size=(256, 256), mode="bilinear", align_corners=False)
        else:
            bev_lidar_256 = bev_lidar

        # Extract multi-scale features across all 6 cameras
        flat_imgs = corrupted_imgs.reshape(B * 6, C_im, H_img, W_img)
        feats = self.backbone(flat_imgs)
        c2, c3, c4, c5 = feats[-4], feats[-3], feats[-2], feats[-1]

        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        fpn_feat = self.fpn_smooth(F.interpolate(p2, size=(64, 112), mode="bilinear", align_corners=False))
        cam_globals = F.adaptive_avg_pool2d(p5, 1).flatten(1).view(B, 6, 256)  # (B, 6, 256)

        # Upgrade 5: Apply Gated SSL Feature Refinement on all 6 views & reconstruct front clean image
        fpn_refined, pred_clean_all = self.ssl_head(fpn_feat, (H_img, W_img), use_ssl=self.use_ssl)
        pred_clean_img = pred_clean_all.view(B, 6, 3, H_img, W_img)[:, 0]  # Front camera SSL target

        # Upgrade 1 & 2: Polar Lift-Splat 6-Cam to 256x256 BEV + 32x32 Cross-Attention + ASPP
        cam_fpn_6 = fpn_refined.view(B, 6, 256, 64, 112)
        vis_bev_256 = self.polar_bev_proj(cam_fpn_6)  # (B, 256, 256, 256)

        fused_256, geo_emb, _ = self.fusion_256(
            vis_bev_256, bev_lidar_256, geo_14d, use_multisensor=self.use_multisensor
        )

        vis_global = cam_globals.mean(dim=1)
        fused_global = F.adaptive_avg_pool2d(fused_256, 1).flatten(1)
        global_context = self.global_context_proj(torch.cat([vis_global, fused_global, geo_emb], dim=-1))

        # Upgrade 4: Multi-Modal Class-Query MultiLabelHead
        multi_label_logits = self.multi_label_head(
            cam_globals, fused_256, bev_lidar_256, geo_emb, global_context
        )

        # Upgrade 1: 256x256 High-Resolution Segmentation & Small Object Heads
        seg_logits_256 = self.seg_head_256(fused_256)
        small_hm_256, small_reg_256 = self.small_det_head_256(fused_256)

        # Also provide 128x128 downsampled views for seamless compatibility with 128x128 evaluators
        seg_logits_128 = F.interpolate(seg_logits_256, size=(128, 128), mode="bilinear", align_corners=False)
        small_hm_128 = F.max_pool2d(small_hm_256, kernel_size=2, stride=2)
        small_reg_128 = F.avg_pool2d(small_reg_256, kernel_size=2, stride=2)

        # RAG + LoRA Dynamic Weight Controller
        dp_weights, rag_rule_idx, risk_emb = self.lora_controller(
            global_context, multi_label_logits, use_dp_lora=self.use_dp_lora
        )

        # Upgrade 3: 256x256 Kinematic + Local ROI Grid-Sample Differentiable DP Planner
        pred_risk_256, dp_value_256, pred_traj = self.dp_planner_256(
            fused_256, risk_emb, geo_14d, dp_weights, use_dp_lora=self.use_dp_lora
        )
        pred_risk_128 = F.avg_pool2d(pred_risk_256.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        dp_value_128 = F.avg_pool2d(dp_value_256.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)

        return {
            "pred_clean_img": pred_clean_img,
            "multi_label_logits": multi_label_logits,
            "seg_logits_256": seg_logits_256,
            "small_hm_256": small_hm_256,
            "small_reg_256": small_reg_256,
            "pred_risk_256": pred_risk_256,
            "dp_value_256": dp_value_256,
            "seg_logits": seg_logits_128,
            "small_hm_pred": small_hm_128,
            "small_reg_pred": small_reg_128,
            "pred_risk_map": pred_risk_128,
            "dp_value_map": dp_value_128,
            "pred_traj": pred_traj,
            "dp_weights": dp_weights,
            "rag_rule_idx": rag_rule_idx,
        }
