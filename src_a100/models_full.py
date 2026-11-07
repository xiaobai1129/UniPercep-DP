import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Enable A100 TF32 hardware acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification under class imbalance.
    """

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 0.5, clip: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        x_sigmoid = torch.sigmoid(logits.float())
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets.float() * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets.float()) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt0 = xs_pos * targets.float()
                pt1 = xs_neg * (1.0 - targets.float())
                pt = (pt0 + pt1).clamp(min=self.eps, max=1.0 - 1e-4)
                one_sided_gamma = self.gamma_pos * targets.float() + self.gamma_neg * (1.0 - targets.float())
                one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()


class SSLReconstructionHead(nn.Module):
    """
    Spatial & Frequency-aware Self-Supervised Reconstruction Decoder.
    Restores degraded weak-network images back to clean images.
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, fpn_feat_128: torch.Tensor, target_hw: tuple):
        out = self.decoder(fpn_feat_128)
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class MultiLabelHead(nn.Module):
    """
    10-class global scene attribute multi-label classification head.
    """

    def __init__(self, in_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, global_vis_feat: torch.Tensor):
        return self.net(global_vis_feat)


class ObstacleSegHead(nn.Module):
    """
    128x128 3-class BEV obstacle & drivable area semantic segmentation head.
    0: Drivable Safe Zone, 1: Static Obstacle/Boundary, 2: Dynamic High-Risk Participant.
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

    def forward(self, spatial_feat_128: torch.Tensor):
        return self.net(spatial_feat_128)


class SmallObjectDetHead(nn.Module):
    """
    128x128 Small Object Center Heatmap & Box Size/Velocity Regression Head.
    Outputs:
      - small_hm: (B, 1, 128, 128) in [0, 1]
      - small_reg: (B, 4, 128, 128) -> [w, h, vx, vy]
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

    def forward(self, spatial_feat_128: torch.Tensor):
        feat = self.shared(spatial_feat_128)
        return self.hm_conv(feat), self.reg_conv(feat)


class MultiSensorCrossAttnFusion(nn.Module):
    """
    Task 2: Multi-Sensor Spatial Alignment & Cross-Attention Fusion
    Fuses:
      - Visual FPN feature map (B, 256, 128, 128)
      - Point cloud BEV feature map (B, 3, 128, 128) [max_height, mean_intensity, density]
      - 14D geometric state vector [5D ego motion state + 9D flattened Homography H]
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

        # Two-layer Multihead Cross-Attention at 16x16 token resolution
        self.pool_16 = nn.AdaptiveAvgPool2d((16, 16))
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
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(
        self,
        vis_fpn_128: torch.Tensor,
        bev_lidar_3ch: torch.Tensor,
        geo_14d: torch.Tensor,
        use_multisensor: bool = True,
    ):
        B = vis_fpn_128.size(0)
        if not use_multisensor:
            zero_bev = torch.zeros_like(vis_fpn_128)
            zero_geo = torch.zeros((B, vis_fpn_128.size(1)), device=vis_fpn_128.device, dtype=vis_fpn_128.dtype)
            fused_128 = self.out_conv(torch.cat([vis_fpn_128, zero_bev, zero_bev], dim=1))
            return fused_128, zero_geo

        bev_128 = self.bev_encoder(bev_lidar_3ch)  # (B, 256, 128, 128)
        geo_emb = self.geo_mlp(geo_14d)            # (B, 256)

        vis_tok = self.pool_16(vis_fpn_128).flatten(2).transpose(1, 2)  # (B, 256, 256)
        bev_tok = self.pool_16(bev_128).flatten(2).transpose(1, 2)      # (B, 256, 256)
        geo_tok = geo_emb.unsqueeze(1)                                   # (B, 1, 256)

        kv = torch.cat([vis_tok, bev_tok, geo_tok], dim=1)  # (B, 513, 256)
        q = bev_tok + geo_tok

        attn_out1, _ = self.attn1(query=q, key=kv, value=kv)
        x = self.norm1(q + attn_out1)
        attn_out2, _ = self.attn2(query=x, key=kv, value=kv)
        x = self.norm2(x + attn_out2)
        x = self.norm3(x + self.ffn(x))

        attn_2d_16 = x.transpose(1, 2).reshape(B, -1, 16, 16)
        attn_2d_128 = F.interpolate(attn_2d_16, size=(128, 128), mode="bilinear", align_corners=False)

        fused_128 = self.out_conv(torch.cat([vis_fpn_128, bev_128, attn_2d_128], dim=1))
        return fused_128, geo_emb


def get_15_rag_rules_tensors():
    """
    Returns 15 fine-grained traffic safety rules:
      trigger_vecs: (15, 10) matching 10 scene attributes:
        [night, rain_fog, intersection, construction, congestion,
         ped_crossing, far_small_obj, fast_approaching, occlusion, curve]
      expert_weights: (15, 3) -> [w_obs, w_conflict, w_smooth]
    """
    rules = [
        # 0: Night clear cruising
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.8, 1.5, 1.8]),
        # 1: Rain/Fog wet road
        ([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [2.1, 1.7, 2.2]),
        # 2: Night + Rain/Fog severe weather
        ([1, 1, 0, 0, 0, 0, 1, 0, 0, 0], [2.6, 2.2, 2.4]),
        # 3: Urban intersection normal
        ([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [2.0, 2.1, 1.6]),
        # 4: Construction zone bottleneck
        ([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], [2.7, 1.9, 1.9]),
        # 5: High-density congestion
        ([0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [2.5, 2.6, 1.7]),
        # 6: Pedestrian crossing emergency
        ([0, 0, 1, 0, 0, 1, 1, 0, 0, 0], [3.1, 2.9, 1.3]),
        # 7: Distant small obstacles ahead
        ([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [2.3, 1.8, 1.7]),
        # 8: Fast approaching dynamic vehicle
        ([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [2.6, 2.8, 1.4]),
        # 9: Severe visual occlusion / packet loss
        ([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [2.4, 2.2, 2.1]),
        # 10: Sharp curve / turning maneuver
        ([0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [2.1, 1.9, 2.3]),
        # 11: Night + Pedestrian crossing + Occlusion
        ([1, 0, 1, 0, 0, 1, 1, 0, 1, 0], [3.3, 3.1, 1.5]),
        # 12: Rain/Fog + Fast approaching + Congestion
        ([0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [2.9, 3.0, 1.8]),
        # 13: Construction + Pedestrian + Small cones
        ([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [3.0, 2.7, 1.6]),
        # 14: Standard daytime open road cruising
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.4, 1.2, 2.0]),
    ]
    triggers = torch.tensor([r[0] for r in rules], dtype=torch.float32)
    weights = torch.tensor([r[1] for r in rules], dtype=torch.float32)
    return triggers, weights


class VLMRagLoRAController(nn.Module):
    """
    Task 4: RAG + PEFT/LoRA Dynamic Weight Controller (Rank=32, Alpha=64)
    Combines frozen Transformer layer + trainable LoRA matrices (lora_A, lora_B)
    with 15-rule FAISS/L2 RAG retrieval to predict [w_obs, w_conflict, w_smooth].
    """

    def __init__(self, feat_dim: int = 256, num_classes: int = 10, rank: int = 32, alpha: float = 64.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)

        in_dim = feat_dim + num_classes + 3
        self.in_proj = nn.Linear(in_dim, feat_dim)

        # Frozen Transformer block simulating VLM semantic reasoning layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=8, dim_feedforward=feat_dim * 2, batch_first=True
        )
        self.frozen_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        for p in self.frozen_transformer.parameters():
            p.requires_grad = False

        # Frozen base linear projection + Trainable LoRA (Rank=32, Alpha=64)
        self.base_proj = nn.Linear(feat_dim, 3)
        self.base_proj.weight.requires_grad = False
        self.base_proj.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(feat_dim, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, 3))

        triggers, weights = get_15_rag_rules_tensors()
        self.register_buffer("rule_triggers", triggers)  # (15, 10)
        self.register_buffer("rule_weights", weights)    # (15, 3)

    def forward(self, global_feat: torch.Tensor, multi_label_logits: torch.Tensor, use_dp_lora: bool = True):
        probs = torch.sigmoid(multi_label_logits)  # (B, 10)
        dists = torch.cdist(probs.float(), self.rule_triggers.float(), p=2.0)  # (B, 15)
        attn = F.softmax(-dists * 2.5, dim=-1)  # (B, 15)
        rag_prior = torch.matmul(attn, self.rule_weights.float()).to(global_feat.dtype)  # (B, 3)
        top_rule_idx = torch.argmin(dists, dim=-1)  # (B,)

        if not use_dp_lora:
            default_w = torch.tensor([1.8, 1.5, 1.8], device=global_feat.device, dtype=global_feat.dtype)
            return default_w.unsqueeze(0).expand(global_feat.size(0), -1), top_rule_idx, global_feat

        x = torch.cat([global_feat, probs, rag_prior], dim=-1)
        h = self.in_proj(x).unsqueeze(1)  # (B, 1, 256)
        h_trans = self.frozen_transformer(h).squeeze(1)  # (B, 256)

        base_out = self.base_proj(h_trans)
        lora_out = (h_trans @ self.lora_A @ self.lora_B) * self.scaling
        delta = torch.tanh(base_out + lora_out) * 0.8

        dp_weights = torch.clamp(rag_prior + delta, min=0.5, max=4.0)
        return dp_weights, top_rule_idx, h_trans


class DifferentiableDPPlanner128(nn.Module):
    """
    Task 3: 128x128 High-Precision Differentiable Dynamic Programming Planner
    - Predicts 128x128 spatial risk cost map `pred_risk_map`;
    - Solves Softmin state transition over dx in [-2, -1, 0, 1, 2]:
        V(y, x) = C(y, x) + softmin_{dx in [-2,2]} (V(y-1, x+dx) + w_smooth * ||dx||^2)
    - Soft-samples optimal guide waypoints and decodes 6-step trajectory via GRU + MLP.
    """

    def __init__(self, in_channels: int = 256, temp: float = 0.25):
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

        # GRU + MLP Trajectory Decoder
        self.gru_cell = nn.GRUCell(input_size=2 + 14 + 3, hidden_size=256)
        self.step_out = nn.Linear(256, 2)

        # Direct MLP fallback for Ablation without DP
        self.direct_mlp = nn.Sequential(
            nn.Linear(256 + 14, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 12),
        )

        x_coords = torch.arange(128, dtype=torch.float32).view(1, 1, 128)
        self.register_buffer("x_coords", x_coords)
        # 6 forward horizon row indices on 128x128 grid (ego at (64, 8), 0.5m/cell -> 7m..35m)
        horizon_rows = torch.tensor([15, 22, 29, 36, 43, 50], dtype=torch.long)
        self.register_buffer("horizon_rows", horizon_rows)

    def forward(
        self,
        fused_128: torch.Tensor,
        global_context: torch.Tensor,
        geo_14d: torch.Tensor,
        dp_weights: torch.Tensor,
        use_dp_lora: bool = True,
    ):
        B = fused_128.size(0)
        pred_risk_map = self.risk_head(fused_128).squeeze(1)  # (B, 128, 128)

        if not use_dp_lora:
            # Ablation mode without DP recurrence: direct MLP trajectory regression from center anchors
            exp_x = torch.full((B, 6), 64.0, device=fused_128.device, dtype=fused_128.dtype)
            exp_y = self.horizon_rows.to(fused_128.dtype).unsqueeze(0).expand(B, -1)
            base_anchors = torch.stack([exp_x, exp_y], dim=-1)
            delta = self.direct_mlp(torch.cat([global_context, geo_14d], dim=-1)).view(B, 6, 2)
            return pred_risk_map, pred_risk_map, base_anchors + delta

        # Run differentiable DP on 64x128 (strided 2 along Y for fast 5-neighbor transition, then upsample to 128x128)
        risk_fp32 = pred_risk_map.float()
        w_obs = dp_weights[:, 0:1].float()       # (B, 1)
        w_conflict = dp_weights[:, 1:2].float()  # (B, 1)
        w_smooth = dp_weights[:, 2:3].float()    # (B, 1)

        temp = float(self.temp)
        cost_grid = (0.6 * w_obs + 0.4 * w_conflict).unsqueeze(-1) * risk_fp32  # (B, 128, 128)

        # Downsample Y to 64 steps for fast vectorized 5-neighbor DP, keeping X=128 exact
        cost_64x128 = F.avg_pool2d(cost_grid.unsqueeze(1), kernel_size=(2, 1), stride=(2, 1)).squeeze(1)  # (B, 64, 128)

        x_grid = self.x_coords.squeeze(1).to(risk_fp32.device)  # (1, 128)
        start_prior = ((x_grid - 64.0) / 64.0) ** 2 * 0.4

        v_prev = cost_64x128[:, 0, :] + start_prior  # (B, 128)
        dp_rows = [v_prev]

        # 5-neighbor lateral transition costs: dx in [-2, -1, 0, +1, +2] -> ||dx||^2 in [4, 1, 0, 1, 4]
        p1 = 0.03 * w_smooth * 1.0  # (B, 1)
        p2 = 0.03 * w_smooth * 4.0  # (B, 1)

        for y in range(1, 64):
            padded = F.pad(v_prev.unsqueeze(1), (2, 2), mode="replicate").squeeze(1)  # (B, 132)
            n_m2 = padded[:, 0:128] + p2
            n_m1 = padded[:, 1:129] + p1
            n_0 = padded[:, 2:130]
            n_p1 = padded[:, 3:131] + p1
            n_p2 = padded[:, 4:132] + p2

            neighbors = torch.stack([n_m2, n_m1, n_0, n_p1, n_p2], dim=-1)  # (B, 128, 5)
            min_prev = -temp * torch.logsumexp(-neighbors / temp, dim=-1)
            v_curr = cost_64x128[:, y, :] + min_prev
            dp_rows.append(v_curr)
            v_prev = v_curr

        dp_val_64 = torch.stack(dp_rows, dim=1).unsqueeze(1)  # (B, 1, 64, 128)
        dp_value_map = F.interpolate(dp_val_64, size=(128, 128), mode="bilinear", align_corners=False).squeeze(1)

        # Softmin expected guide waypoints at 6 horizon rows
        horizon_vals = dp_value_map[:, self.horizon_rows, :]  # (B, 6, 128)
        prob_x = F.softmax(-horizon_vals / temp, dim=-1)
        exp_x = (prob_x * self.x_coords.to(prob_x.device)).sum(dim=-1)  # (B, 6)
        exp_y = self.horizon_rows.float().unsqueeze(0).expand(B, -1)    # (B, 6)
        dp_anchors = torch.stack([exp_x, exp_y], dim=-1)                # (B, 6, 2)

        # Decode 6-step smooth trajectory via GRU initialized with global_context
        h_state = global_context
        deltas = []
        for t in range(6):
            anchor_norm = (dp_anchors[:, t, :].to(global_context.dtype) - 64.0) / 64.0
            gru_in = torch.cat([anchor_norm, geo_14d.to(global_context.dtype), dp_weights.to(global_context.dtype)], dim=-1)
            h_state = self.gru_cell(gru_in, h_state)
            deltas.append(self.step_out(h_state))

        delta_traj = torch.stack(deltas, dim=1)  # (B, 6, 2)
        pred_traj = dp_anchors.to(delta_traj.dtype) + delta_traj
        return pred_risk_map, dp_value_map, pred_traj


class UniPercepDPNet_Full(nn.Module):
    """
    A100 Industrial-Grade Unified End-to-End Model (`UniPercepDPNet_Full`)
    Supports full pipeline and all 3 ablation configurations via flags:
      - use_ssl: enable/disable SSL reconstruction branch
      - use_multisensor: enable/disable 3x128x128 LiDAR BEV & 14D Homography/Odometry Cross-Attention
      - use_dp_lora: enable/disable 128x128 Differentiable DP & RAG-LoRA controller
    """

    def __init__(
        self,
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
                "resnet50d", pretrained=pretrained_backbone, features_only=True
            )
        except Exception:
            self.backbone = timm.create_model(
                "resnet50d", pretrained=False, features_only=True
            )

        # resnet50d feature channels: [64, 256, 512, 1024, 2048]
        # Multi-scale FPN lateral projections -> 128x128 unified feature map
        self.lat_c2 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat_c3 = nn.Conv2d(512, 256, kernel_size=1)
        self.lat_c4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lat_c5 = nn.Conv2d(2048, 256, kernel_size=1)

        self.fpn_smooth = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 1. SSL Reconstruction Head
        self.ssl_head = SSLReconstructionHead(in_channels=256)

        # 2. Multi-Task Dense Visual Perception Three Heads
        self.multi_label_head = MultiLabelHead(in_dim=256, num_classes=10)
        self.seg_head = ObstacleSegHead(in_channels=256, num_classes=3)
        self.small_det_head = SmallObjectDetHead(in_channels=256)

        # 3. Multi-Sensor Cross-Attention Fusion
        self.fusion_module = MultiSensorCrossAttnFusion(dim=256, num_heads=8)

        self.global_context_proj = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # 4. RAG + PEFT/LoRA Dynamic Weight Controller (Rank=32, Alpha=64)
        self.lora_controller = VLMRagLoRAController(feat_dim=256, num_classes=10, rank=32, alpha=64.0)

        # 5. 128x128 Differentiable DP Planner
        self.dp_planner = DifferentiableDPPlanner128(in_channels=256, temp=0.25)

    def set_training_stage(self, stage_name: str):
        for name, param in self.named_parameters():
            if "frozen_transformer" in name or "lora_controller.base_proj" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        if stage_name == "Stage1A_SSL_LP":
            for name, param in self.backbone.named_parameters():
                if any(k in name for k in ["conv1", "bn1", "layer1", "layer2", "layer3"]):
                    param.requires_grad = False
        elif stage_name == "Stage1B_LP_FT":
            pass
        elif stage_name == "Stage2_Fusion_DP":
            pass
        elif stage_name == "Stage3_RAG_LoRA":
            for name, param in self.backbone.named_parameters():
                if any(k in name for k in ["conv1", "bn1", "layer1", "layer2"]):
                    param.requires_grad = False
            for p in self.ssl_head.parameters():
                p.requires_grad = False
            self.lora_controller.lora_A.requires_grad = True
            self.lora_controller.lora_B.requires_grad = True

    def forward(self, corrupted_img: torch.Tensor, bev_lidar_3ch: torch.Tensor, geo_14d: torch.Tensor):
        _, _, H_img, W_img = corrupted_img.shape
        feats = self.backbone(corrupted_img)
        c2, c3, c4, c5 = feats[1], feats[2], feats[3], feats[4]

        # Top-down FPN -> 128x128
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        vis_fpn_128 = self.fpn_smooth(
            F.interpolate(p2, size=(128, 128), mode="bilinear", align_corners=False)
        )
        vis_global = F.adaptive_avg_pool2d(p5, 1).flatten(1)  # (B, 256)

        # SSL Reconstruction
        if self.use_ssl:
            pred_clean_img = self.ssl_head(vis_fpn_128, (H_img, W_img))
        else:
            pred_clean_img = corrupted_img

        # Multi-Sensor Cross-Attention Fusion
        fused_128, geo_emb = self.fusion_module(
            vis_fpn_128, bev_lidar_3ch, geo_14d, use_multisensor=self.use_multisensor
        )

        # Multi-Task Perception Heads
        multi_label_logits = self.multi_label_head(vis_global)
        seg_logits = self.seg_head(fused_128)
        small_hm_pred, small_reg_pred = self.small_det_head(fused_128)

        fused_global = F.adaptive_avg_pool2d(fused_128, 1).flatten(1)
        global_context = self.global_context_proj(torch.cat([vis_global, fused_global, geo_emb], dim=-1))

        # RAG + LoRA Dynamic Weight Controller
        dp_weights, rag_rule_idx, risk_emb = self.lora_controller(
            global_context, multi_label_logits, use_dp_lora=self.use_dp_lora
        )

        # 128x128 Differentiable DP Planner
        pred_risk_map, dp_value_map, pred_traj = self.dp_planner(
            fused_128, risk_emb, geo_14d, dp_weights, use_dp_lora=self.use_dp_lora
        )

        return {
            "pred_clean_img": pred_clean_img,
            "multi_label_logits": multi_label_logits,
            "seg_logits": seg_logits,
            "small_hm_pred": small_hm_pred,
            "small_reg_pred": small_reg_pred,
            "pred_risk_map": pred_risk_map,
            "dp_value_map": dp_value_map,
            "pred_traj": pred_traj,
            "dp_weights": dp_weights,
            "rag_rule_idx": rag_rule_idx,
        }
