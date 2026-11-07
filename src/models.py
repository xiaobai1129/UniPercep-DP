import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LoRARagController(nn.Module):
    """
    Task 4: RAG + LoRA Dynamic Weight Controller
    Freezes base linear projection layer, declares trainable low-rank matrices lora_A (rank=16)
    and lora_B, combines predicted 8D multi-label scene attributes + RAG rule bank retrieval
    to dynamically output DP cost weights [w_obs, w_smooth].
    """

    def __init__(self, in_dim: int = 256 + 8 + 2, out_dim: int = 2, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)

        # Frozen base linear projection layer
        self.base_proj = nn.Linear(in_dim, out_dim)
        self.base_proj.weight.requires_grad = False
        self.base_proj.bias.requires_grad = False

        # Trainable LoRA low-rank matrices (rank=16)
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

        # Built-in RAG rule knowledge bank (3 canonical traffic rules)
        # Rule 0: Night / Rain-Fog
        # Rule 1: Dense Pedestrian / Small Far Objects / Intersection
        # Rule 2: Normal Road Cruising
        rule_triggers = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        rule_weights = torch.tensor(
            [
                [2.2, 1.5],
                [2.8, 1.2],
                [1.2, 1.8],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("rule_triggers", rule_triggers)
        self.register_buffer("rule_weights", rule_weights)

    def retrieve_rag_prior(self, multi_label_probs: torch.Tensor):
        """
        Differentiable L2-distance RAG retrieval over the 3 rule trigger vectors.
        multi_label_probs: (B, 8)
        Returns:
          rag_weights: (B, 2) recommended [w_obs, w_smooth]
          top_rule_idx: (B,) integer rule index with smallest L2 distance
        """
        # Pairwise squared L2 distance: (B, 1, 8) vs (1, 3, 8) -> (B, 3)
        dists = torch.cdist(multi_label_probs.float(), self.rule_triggers.float(), p=2.0)
        attn = F.softmax(-dists * 2.5, dim=-1)  # (B, 3)
        rag_weights = torch.matmul(attn, self.rule_weights.float()).to(multi_label_probs.dtype)
        top_rule_idx = torch.argmin(dists, dim=-1)
        return rag_weights, top_rule_idx

    def forward(self, global_feat: torch.Tensor, multi_label_logits: torch.Tensor):
        probs = torch.sigmoid(multi_label_logits)
        rag_prior, top_rule_idx = self.retrieve_rag_prior(probs)

        x = torch.cat([global_feat, probs, rag_prior], dim=-1)
        base_out = self.base_proj(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        delta = torch.tanh(base_out + lora_out) * 0.6

        # Combine RAG prior weights with LoRA residual modulation, keeping weights strictly positive
        dp_weights = torch.clamp(rag_prior + delta, min=0.5, max=3.5)
        return dp_weights, top_rule_idx


class DifferentiableDPPlanner(nn.Module):
    """
    Task 3: Spatial Risk Field Modeling & Differentiable Dynamic Programming (DP) Trajectory Planner
    - Regresses 64x64 spatial risk cost map `pred_risk` from fused multi-sensor BEV features;
    - Performs Softmin (-temp * logsumexp(-neighbors / temp)) differentiable DP recurrence along
      the forward direction (Y axis) to compute cumulative minimum energy field `dp_value_map`;
    - Computes expected optimal target waypoints and combines with global features to regress
      future 6-step collision-free trajectory `pred_traj` (Bx6x2).
    """

    def __init__(self, in_channels: int = 128, global_dim: int = 256, temp: float = 0.25):
        super().__init__()
        self.temp = temp
        self.risk_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Trajectory regression head combining global fused context + DP softmin waypoint anchors + LoRA weights
        self.traj_mlp = nn.Sequential(
            nn.Linear(global_dim + 12 + 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 12),  # 6 steps x 2 coordinates (gx, gy)
        )

        # Register coordinate grids
        x_coords = torch.arange(64, dtype=torch.float32).view(1, 1, 64)
        self.register_buffer("x_coords", x_coords)
        # 6 target horizon row indices along forward Y axis matching physical 0.5s~3.0s horizon
        horizon_rows = torch.tensor([7, 11, 14, 17, 21, 24], dtype=torch.long)
        self.register_buffer("horizon_rows", horizon_rows)

    def forward(self, fused_bev: torch.Tensor, global_context: torch.Tensor, dp_weights: torch.Tensor):
        """
        fused_bev: (B, C, 64, 64)
        global_context: (B, global_dim)
        dp_weights: (B, 2) -> [w_obs, w_smooth]
        """
        pred_risk = self.risk_head(fused_bev).squeeze(1)  # (B, 64, 64)

        # Run differentiable DP state transition in float32 for numerical stability under FP16/BF16
        risk_fp32 = pred_risk.float()
        w_obs = dp_weights[:, 0:1].float()     # (B, 1)
        w_smooth = dp_weights[:, 1:2].float()  # (B, 1)

        B, H, W = risk_fp32.shape
        temp = float(self.temp)

        # Slight quadratic prior keeping ego near center lane (col 32) at start row y=0
        x_grid = self.x_coords.squeeze(1).to(risk_fp32.device)  # (1, 64)
        start_prior = ((x_grid - 32.0) / 32.0) ** 2 * 0.4

        dp_rows = []
        v_prev = w_obs * risk_fp32[:, 0, :] + start_prior  # (B, 64)
        dp_rows.append(v_prev)

        # Lateral shift transition penalty scaled by w_smooth
        lat_penalty = 0.08 * w_smooth  # (B, 1)

        # Differentiable DP recurrence along Y axis (rows 1 .. 63)
        for y in range(1, H):
            padded = F.pad(v_prev.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)  # (B, 66)
            left = padded[:, :-2] + lat_penalty
            center = padded[:, 1:-1]
            right = padded[:, 2:] + lat_penalty

            neighbors = torch.stack([left, center, right], dim=-1)  # (B, 64, 3)
            # Softmin: -temp * logsumexp(-neighbors / temp)
            min_prev = -temp * torch.logsumexp(-neighbors / temp, dim=-1)
            step_cost = w_obs * risk_fp32[:, y, :]
            v_curr = step_cost + min_prev
            dp_rows.append(v_curr)
            v_prev = v_curr

        dp_value_map = torch.stack(dp_rows, dim=1)  # (B, 64, 64)

        # Extract softmin optimal expected lateral positions at 6 forward horizon slices
        horizon_vals = dp_value_map[:, self.horizon_rows, :]  # (B, 6, 64)
        prob_x = F.softmax(-horizon_vals / temp, dim=-1)      # (B, 6, 64)
        exp_x = (prob_x * self.x_coords.to(prob_x.device)).sum(dim=-1)  # (B, 6)
        exp_y = self.horizon_rows.float().unsqueeze(0).expand(B, -1)    # (B, 6)

        dp_anchors = torch.stack([exp_x, exp_y], dim=-1)  # (B, 6, 2)
        dp_anchors_flat = (dp_anchors.reshape(B, 12) - 32.0) / 32.0

        mlp_in = torch.cat(
            [
                global_context,
                dp_anchors_flat.to(global_context.dtype),
                dp_weights.to(global_context.dtype),
            ],
            dim=-1,
        )
        delta_traj = self.traj_mlp(mlp_in).view(B, 6, 2)

        # Combine DP softmin anchor trajectory with residual refinement
        pred_traj = dp_anchors.to(delta_traj.dtype) + delta_traj
        return pred_risk, dp_value_map, pred_traj


class UniPercepDPNet(nn.Module):
    """
    Unified End-to-End Model:
      1. Visual Backbone (timm resnet34) + SSL Reconstruction Decoder (L1 loss_ssl)
      2. Multi-task Dense Perception Heads:
         - multi_label_head (8 classes)
         - seg_head (3 classes, 64x64)
         - small_det_head (1x64x64 Gaussian heatmap)
      3. Multi-Sensor Cross-Attention Fusion (Visual + LiDAR BEV + 14D Geometric State)
      4. DifferentiableDPPlanner (pred_risk, dp_value_map, pred_traj Bx6x2)
      5. LoRARagController (dynamic DP cost weights [w_obs, w_smooth])
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                "resnet34", pretrained=pretrained_backbone, features_only=True
            )
        except Exception:
            self.backbone = timm.create_model(
                "resnet34", pretrained=False, features_only=True
            )

        # resnet34 feature channels: [64, 64, 128, 256, 512]
        # Spatial sizes for 512x288 input:
        # c1: 256x144, c2: 128x72, c3: 64x36, c4: 32x18, c5: 16x9
        self.vis_proj = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 1. Self-Supervised Image Reconstruction Branch (SSL Decoder -> 288x512x3)
        self.ssl_decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32x18
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64x36
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(288, 512), mode="bilinear", align_corners=False),  # 288x512
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # LiDAR BEV Encoder (input: 2x64x64 -> 64x64 and 16x16 tokens)
        self.bev_stem = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.bev_token_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(128, 256, kernel_size=1),
        )

        # 14D Geometric State Encoder (5D sensor_state + 9D flattened homography H)
        self.geo_encoder = nn.Sequential(
            nn.Linear(14, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

        # 3. Multi-Sensor Cross-Attention Fusion Layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(256)

        # Fuse attended BEV tokens (upsampled to 64x64) + high-res BEV stem + visual BEV projection
        self.vis_to_bev = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 + 128 + 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 2. Multi-Task Dense Visual Perception Three Heads
        # Head 2a: Global 8-class multi-label classification head
        self.multi_label_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 8),
        )

        # Head 2b: 64x64 3-class obstacle semantic segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        # Head 2c: 64x64 small object center heatmap head
        self.small_det_head = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Global fused context projection for planner & controller
        self.global_fusion_proj = nn.Sequential(
            nn.Linear(256 + 256 + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        # 5. RAG + LoRA Dynamic Weight Controller
        self.lora_controller = LoRARagController(in_dim=256 + 8 + 2, out_dim=2, rank=16)

        # 4. Differentiable Dynamic Programming Planner
        self.dp_planner = DifferentiableDPPlanner(in_channels=128, global_dim=256, temp=0.25)

    def set_training_stage(self, stage_name: str):
        """
        Configure parameter trainable states for the 4 training stages:
          - Stage1_SSL_LP: Freeze shallow backbone (conv1, bn1, layer1, layer2), train deep backbone + SSL & perception heads
          - Stage1_LP_FT: Unfreeze full backbone + SSL & perception heads
          - Stage2_Fusion_DP: Jointly train multi-sensor fusion, risk map, and differentiable DP trajectory
          - Stage3_RAG_LoRA: Fine-tune LoRA controller (lora_A, lora_B) & planner with smoothness regularization
        """
        # First enable all parameters except frozen base_proj in LoRARagController
        for name, param in self.named_parameters():
            if "lora_controller.base_proj" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        if stage_name == "Stage1_SSL_LP":
            for name, param in self.backbone.named_parameters():
                if any(k in name for k in ["conv1", "bn1", "layer1", "layer2"]):
                    param.requires_grad = False
        elif stage_name == "Stage1_LP_FT":
            # Full backbone + perception & SSL heads unfrozen
            pass
        elif stage_name == "Stage2_Fusion_DP":
            # Jointly train fusion, risk map, DP planner, and keep perception active
            pass
        elif stage_name == "Stage3_RAG_LoRA":
            # Freeze visual backbone & SSL decoder to focus on LoRA controller and DP trajectory smoothness
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.ssl_decoder.parameters():
                param.requires_grad = False
            self.lora_controller.base_proj.weight.requires_grad = False
            self.lora_controller.base_proj.bias.requires_grad = False
            self.lora_controller.lora_A.requires_grad = True
            self.lora_controller.lora_B.requires_grad = True

    def forward(self, corrupted_img: torch.Tensor, bev_lidar: torch.Tensor, geo_state_14d: torch.Tensor):
        """
        corrupted_img: (B, 3, 288, 512)
        bev_lidar: (B, 2, 64, 64)
        geo_state_14d: (B, 14) [5D sensor_state + 9D flattened homography]
        """
        B = corrupted_img.size(0)

        # 1. Visual Backbone + SSL Reconstruction
        feats = self.backbone(corrupted_img)
        c5 = self.vis_proj(feats[-1])  # (B, 256, 9, 16)
        pred_clean_img = self.ssl_decoder(c5)  # (B, 3, 288, 512)

        vis_global = F.adaptive_avg_pool2d(c5, 1).flatten(1)  # (B, 256)
        vis_bev_64 = self.vis_to_bev(c5)  # (B, 64, 64, 64)

        # 2. Multi-Label Classification Head
        multi_label_logits = self.multi_label_head(vis_global)  # (B, 8)

        # 3. LiDAR BEV & 14D Geometric State Encoding
        bev_feat_64 = self.bev_stem(bev_lidar)  # (B, 128, 64, 64)
        Percep_feat_64 = torch.cat([bev_feat_64, vis_bev_64], dim=1)  # (B, 192, 64, 64)

        seg_logits = self.seg_head(Percep_feat_64)       # (B, 3, 64, 64)
        small_hm_pred = self.small_det_head(Percep_feat_64)  # (B, 1, 64, 64)

        # 4. Multi-Sensor Cross-Attention Fusion
        vis_tokens = c5.flatten(2).transpose(1, 2)  # (B, 144, 256)
        bev_tokens_2d = self.bev_token_proj(bev_feat_64)  # (B, 256, 16, 16)
        bev_tokens = bev_tokens_2d.flatten(2).transpose(1, 2)  # (B, 256, 256)
        geo_emb = self.geo_encoder(geo_state_14d)  # (B, 256)
        geo_token = geo_emb.unsqueeze(1)  # (B, 1, 256)

        kv_tokens = torch.cat([vis_tokens, bev_tokens, geo_token], dim=1)  # (B, 401, 256)
        q_tokens = bev_tokens + geo_token  # (B, 256, 256)
        attn_out, _ = self.cross_attn(query=q_tokens, key=kv_tokens, value=kv_tokens)
        fused_tokens = self.attn_norm(q_tokens + attn_out)  # (B, 256, 256)

        fused_2d_16 = fused_tokens.transpose(1, 2).reshape(B, 256, 16, 16)
        fused_2d_64 = F.interpolate(fused_2d_16, size=(64, 64), mode="bilinear", align_corners=False)

        fused_bev = self.fusion_conv(torch.cat([fused_2d_64, bev_feat_64, vis_bev_64], dim=1))  # (B, 128, 64, 64)
        bev_global = fused_tokens.mean(dim=1)  # (B, 256)
        global_context = self.global_fusion_proj(torch.cat([vis_global, bev_global, geo_emb], dim=-1))  # (B, 256)

        # 5. LoRA + RAG Dynamic Weight Controller
        dp_weights, rag_rule_idx = self.lora_controller(global_context, multi_label_logits)

        # 6. Differentiable DP Planner
        pred_risk, dp_value_map, pred_traj = self.dp_planner(fused_bev, global_context, dp_weights)

        return {
            "pred_clean_img": pred_clean_img,
            "multi_label_logits": multi_label_logits,
            "seg_logits": seg_logits,
            "small_hm_pred": small_hm_pred,
            "pred_risk": pred_risk,
            "dp_value_map": dp_value_map,
            "pred_traj": pred_traj,
            "dp_weights": dp_weights,
            "rag_rule_idx": rag_rule_idx,
        }
