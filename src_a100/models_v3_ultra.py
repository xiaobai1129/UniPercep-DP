import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_pro256 import (
    UniPercepDPNet_Pro256,
    PolarLiftSplatBEV256,
    MultiSensorCrossAttnFusion256,
    MultiModalQueryMultiLabelHead,
    ObstacleSegHead256,
    SmallObjectDetHead256,
    ModelEMA,
)
from models_full import VLMRagLoRAController

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class DegradationFiLMModulator(nn.Module):
    """
    Innovation #12: Degradation-Aware Feature-wise Linear Modulation (FiLM).
    Estimates image degradation severity (packet loss / fog / blur) from spatial & high-frequency statistics
    and dynamically modulates visual vs LiDAR fusion trust weights: gamma(d) * F + beta(d).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.deg_probe = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(channels * 16, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, channels * 2),
        )
        # Zero-init last layer so it starts as identity modulation (gamma=0 -> 1+0=1, beta=0)
        nn.init.zeros_(self.deg_probe[-1].weight)
        nn.init.zeros_(self.deg_probe[-1].bias)

    def forward(self, x: torch.Tensor):
        params = self.deg_probe(x)  # (B, 2C)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1) * 0.5
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1) * 0.5
        return x * (1.0 + gamma) + beta


class HomographyMotionSalienceBlock(nn.Module):
    """
    Innovation #14: Explicit 3x3 Homography Spatial Warping & Motion Salience Extractor.
    Uses the 3x3 inter-frame Homography matrix H (from `geo_14d[:, 5:14]`) to warp the front camera
    feature map and compute ego-motion compensated feature difference |F_t - Warp(F_t, H)|,
    explicitly highlighting moving distant small objects and cutting-in vehicles.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.motion_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.15))

    def forward(self, front_fpn: torch.Tensor, geo_14d: torch.Tensor):
        B, C, H, W = front_fpn.shape
        H_flat = geo_14d[:, 5:14].float()  # (B, 9)
        H_mat = H_flat.view(B, 3, 3)

        # Normalize pixel homography into [-1, 1] affine approximation for fast stable grid_sample
        eye_2x3 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=front_fpn.device, dtype=torch.float32).unsqueeze(0)
        tx = (H_mat[:, 0, 2] / 768.0).clamp(-0.15, 0.15)
        ty = (H_mat[:, 1, 2] / 448.0).clamp(-0.15, 0.15)
        rot_s = (H_mat[:, 0, 1]).clamp(-0.08, 0.08)

        theta = eye_2x3.repeat(B, 1, 1)
        theta[:, 0, 1] = rot_s
        theta[:, 1, 0] = -rot_s
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, size=front_fpn.size(), align_corners=False).to(front_fpn.dtype)
        warped = F.grid_sample(front_fpn, grid, mode="bilinear", padding_mode="border", align_corners=False)
        motion_diff = torch.abs(front_fpn - warped)
        return front_fpn + torch.tanh(self.alpha) * self.motion_conv(motion_diff)


class SpectralDenoiseBlock(nn.Module):
    """
    Innovation #7: 2D-FFT Frequency-Domain Spectral Gating Block.
    Transforms visual FPN features into the 2D Fourier frequency domain via `torch.fft.rfft2`,
    suppresses low-frequency rain/fog veil and high-frequency weak-network macroblock artifacts,
    and transforms back via `torch.fft.irfft2`.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x_fp32 = x.float()
        H, W = x_fp32.shape[-2], x_fp32.shape[-1]

        fft_x = torch.fft.rfft2(x_fp32, norm="ortho")
        cat_ri = torch.cat([fft_x.real, fft_x.imag], dim=1)
        gate_ri = self.freq_gate(cat_ri)
        g_real, g_imag = torch.chunk(gate_ri, 2, dim=1)

        filtered_fft = torch.complex(fft_x.real * g_real, fft_x.imag * g_imag)
        x_denoised = torch.fft.irfft2(filtered_fft, s=(H, W), norm="ortho")
        out = x_fp32 + torch.tanh(self.res_scale) * x_denoised
        return self.spatial_mix(out.to(orig_dtype))


class MetricEgoPositionalEncoding256(nn.Module):
    """
    Innovation #16 + #13: 2D Metric Ego-Centric Positional & Distance-Prior Encoding + LiDAR-Guided Calibration.
    Injects exact physical distance r(u,v) [m], azimuth sin/cos(theta), and >25m small-object prior mask
    into the 256x256 BEV space, plus LiDAR occupancy gating.
    """

    def __init__(self, channels: int = 256, bev_size: int = 256):
        super().__init__()
        ys = torch.linspace(0.0, 64.0, bev_size).view(bev_size, 1).expand(bev_size, bev_size)
        xs = torch.linspace(-32.0, 32.0, bev_size).view(1, bev_size).expand(bev_size, bev_size)
        dy = ys - 4.0  # ego vehicle at y = 4.0m (row 16 on 256 grid)
        dx = xs
        dist_m = torch.sqrt(dx * dx + dy * dy).clamp(min=0.1, max=64.0)
        angle = torch.atan2(dx, dy)

        # 6-channel physical coordinate map: [dx/32, dy/64, dist_m/64, sin(angle), cos(angle), is_far_25m]
        far_mask = torch.sigmoid((dist_m - 22.0) / 3.0)  # smooth step > 22m-25m for small object prior
        coord_map = torch.stack(
            [dx / 32.0, dy / 64.0, dist_m / 64.0, torch.sin(angle), torch.cos(angle), far_mask], dim=0
        )  # (6, 256, 256)
        self.register_buffer("coord_map", coord_map.unsqueeze(0))  # (1, 6, 256, 256)
        self.register_buffer("far_prior_mask", far_mask.view(1, 1, bev_size, bev_size))

        self.coord_proj = nn.Sequential(
            nn.Conv2d(6 + 3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, channels, kernel_size=1),
        )
        self.lidar_gate = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.20))

    def forward(self, bev_feat_256: torch.Tensor, bev_lidar_256: torch.Tensor):
        B = bev_feat_256.size(0)
        coords = self.coord_map.expand(B, -1, -1, -1).to(bev_feat_256.dtype)
        geo_prior = self.coord_proj(torch.cat([coords, bev_lidar_256], dim=1))
        l_gate = self.lidar_gate(bev_lidar_256)
        return bev_feat_256 * (0.85 + 0.30 * l_gate) + torch.tanh(self.scale) * geo_prior


class StripPolarAnisotropicBlock(nn.Module):
    """
    Innovation #8A: Orthogonal 1x9 and 9x1 Strip Convolutions on 256x256 BEV.
    Captures long-range longitudinal driving corridors and lateral intersection boundaries.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        mid = channels // 2
        self.down = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.strip_v = nn.Conv2d(mid, mid, kernel_size=(9, 1), padding=(4, 0), groups=mid, bias=False)
        self.strip_h = nn.Conv2d(mid, mid, kernel_size=(1, 9), padding=(0, 4), groups=mid, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        h = self.down(x)
        v_feat = self.strip_v(h)
        h_feat = self.strip_h(h)
        return x + self.fuse(torch.cat([v_feat, h_feat], dim=1))


class SceneGraphMultiLabelEnhancer(nn.Module):
    """
    Innovation #17: 10-Class Scene Attribute Co-occurrence Graph Attention Layer.
    Models physical label co-occurrence (e.g., Intersection <-> Pedestrian Crossing, Rain/Fog <-> Occlusion)
    via self-attention over class embeddings + explicit ego kinematics.
    """

    def __init__(self, num_classes: int = 10, dim: int = 64):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(1, num_classes, dim) * 0.02)
        self.logit_to_node = nn.Linear(1, dim)
        self.ctx_to_node = nn.Linear(256 + 14, dim)
        self.graph_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.node_out = nn.Linear(dim, 1)
        nn.init.zeros_(self.node_out.weight)
        nn.init.zeros_(self.node_out.bias)

    def forward(self, base_logits: torch.Tensor, global_context: torch.Tensor, geo_14d: torch.Tensor):
        B = base_logits.size(0)
        nodes = (
            self.node_emb.expand(B, -1, -1)
            + self.logit_to_node(base_logits.unsqueeze(-1))
            + self.ctx_to_node(torch.cat([global_context, geo_14d], dim=-1)).unsqueeze(1)
        )
        attn_nodes, _ = self.graph_attn(nodes, nodes, nodes)
        nodes = self.norm(nodes + attn_nodes)
        delta_logits = self.node_out(nodes).squeeze(-1)
        return base_logits + delta_logits


class CausalPotentialFieldDPPlanner256(nn.Module):
    """
    Innovation #8B, #9, & #20:
      1. Causal Perception-to-Planning Bridge (`seg_probs_256 + small_hm_256 + small_reg_256 -> risk_map`)
      2. K=3 Multi-Modal Trajectory Proposals & Risk-Integral Soft Selection
      3. Artificial Potential Field Negative Risk Gradient (-grad R) Repulsive Force Guidance in GRU
    """

    def __init__(self, in_channels: int = 256, num_modes: int = 3, temp: float = 0.20):
        super().__init__()
        self.num_modes = num_modes
        self.temp = temp

        self.causal_bridge = nn.Sequential(
            nn.Conv2d(in_channels + 8, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        self.risk_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.local_roi_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.kin_prior_mlp = nn.Sequential(
            nn.Linear(256 + 14, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 6 * 2),
        )

        self.mode_proposal_mlp = nn.Sequential(
            nn.Linear(256 + 14 + 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_modes * 12 + num_modes),
        )

        # GRU input includes 2D Negative Risk Gradient repulsive force vector (-dR/dx, -dR/dy):
        # [dp_anchor(2) + kin_anchor(2) + local_roi(64) + local_risk(1) + neg_risk_grad(2) + geo_14d(14) + dp_weights(3)] = 88
        self.gru_cell = nn.GRUCell(input_size=2 + 2 + 64 + 1 + 2 + 14 + 3, hidden_size=256)
        self.step_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        # Precompute 3x3 Sobel filters for spatial risk gradient (dR/dx, dR/dy)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_xy", torch.cat([sobel_x, sobel_y], dim=0))  # (2, 1, 3, 3)

        x_coords_256 = torch.arange(256, dtype=torch.float32).view(1, 1, 256)
        self.register_buffer("x_coords_256", x_coords_256)

    def forward(
        self,
        fused_256: torch.Tensor,
        seg_logits_256: torch.Tensor,
        small_hm_256: torch.Tensor,
        small_reg_256: torch.Tensor,
        global_context: torch.Tensor,
        geo_14d: torch.Tensor,
        dp_weights: torch.Tensor,
        use_dp_lora: bool = True,
    ):
        B = fused_256.size(0)

        # 1. Causal Perception-to-Planning Prior Bridge
        seg_probs_256 = F.softmax(seg_logits_256.detach(), dim=1)
        perc_priors = torch.cat([seg_probs_256, small_hm_256.detach(), small_reg_256.detach()], dim=1)
        causal_bev = self.causal_bridge(torch.cat([fused_256, perc_priors.to(fused_256.dtype)], dim=1))

        raw_risk = self.risk_head(causal_bev).squeeze(1)
        dyn_obs_prior = seg_probs_256[:, 2, :, :] * 0.35 + small_hm_256[:, 0, :, :] * 0.25
        pred_risk_256 = (0.75 * raw_risk + 0.25 * dyn_obs_prior.to(raw_risk.dtype)).clamp(0.0, 1.0)

        # Compute 2D Negative Risk Gradient Field (-dR/dx, -dR/dy) -> Repulsive Force Field (B, 2, 256, 256)
        risk_4d = pred_risk_256.unsqueeze(1).float()
        neg_risk_grad_256 = -F.conv2d(risk_4d, self.sobel_xy.float(), padding=1).to(fused_256.dtype)

        roi_feat_map = self.local_roi_proj(causal_bev)  # (B, 64, 256, 256)

        # 2. Velocity-Adaptive Kinematic Base + K=3 Multi-Modal Trajectory Proposals
        vx = geo_14d[:, 0:1]
        vy = geo_14d[:, 1:2]
        speed = torch.sqrt(vx * vx + vy * vy + 1e-6).clamp(min=0.0, max=20.0)
        time_steps = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=fused_256.device, dtype=fused_256.dtype).unsqueeze(0)
        kin_y_base = 8.0 + (speed * time_steps) / 0.5
        kin_x_base = 64.0 + (vx * time_steps * 0.25)
        kin_residual = self.kin_prior_mlp(torch.cat([global_context, geo_14d], dim=-1)).view(B, 6, 2)
        kin_base = torch.stack([kin_x_base, kin_y_base], dim=-1) + kin_residual

        raw_proposals = self.mode_proposal_mlp(torch.cat([global_context, geo_14d, dp_weights], dim=-1))
        mode_deltas = raw_proposals[:, : self.num_modes * 12].view(B, self.num_modes, 6, 2)
        mode_logits = raw_proposals[:, self.num_modes * 12 :]

        lateral_offsets = torch.tensor([-1.2, 0.0, 1.2], device=fused_256.device, dtype=fused_256.dtype).view(1, 3, 1)
        mode_trajs = kin_base.unsqueeze(1) + 0.20 * mode_deltas
        mode_trajs[:, :, :, 0] = mode_trajs[:, :, :, 0] + lateral_offsets * (time_steps.unsqueeze(1) / 3.0)

        gx_k = (mode_trajs[:, :, :, 0] / 64.0) - 1.0
        gy_k = (mode_trajs[:, :, :, 1] / 64.0) - 1.0
        sample_grid_k = torch.stack([gx_k, gy_k], dim=-1).to(pred_risk_256.dtype)
        sampled_risks_k = F.grid_sample(
            pred_risk_256.unsqueeze(1), sample_grid_k, mode="bilinear", padding_mode="border", align_corners=False
        ).squeeze(1)
        traj_risk_cost = sampled_risks_k.mean(dim=-1) * dp_weights[:, 0:1].to(pred_risk_256.dtype)

        mode_probs = F.softmax(mode_logits.float() - 4.5 * traj_risk_cost.float(), dim=-1).to(mode_trajs.dtype)
        kin_anchors_128 = (mode_trajs * mode_probs.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        if not use_dp_lora:
            return pred_risk_256, pred_risk_256, kin_anchors_128

        # 3. 256x256 Differentiable Dynamic Programming
        risk_fp32 = pred_risk_256.float()
        w_obs = dp_weights[:, 0:1].float()
        w_conflict = dp_weights[:, 1:2].float()
        w_smooth = dp_weights[:, 2:3].float()
        temp = float(self.temp)

        cost_256 = (0.6 * w_obs + 0.4 * w_conflict).unsqueeze(-1) * risk_fp32
        cost_64x256 = F.avg_pool2d(cost_256.unsqueeze(1), kernel_size=(4, 1), stride=(4, 1)).squeeze(1)

        x_grid = self.x_coords_256.squeeze(1).to(risk_fp32.device)
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

        dp_val_64 = torch.stack(dp_rows, dim=1).unsqueeze(1)
        dp_value_256 = F.interpolate(dp_val_64, size=(256, 256), mode="bilinear", align_corners=False).squeeze(1)

        dyn_rows_256 = (kin_anchors_128[:, :, 1].detach() * 2.0).long().clamp(min=8, max=248)
        batch_idx = torch.arange(B, device=fused_256.device).unsqueeze(1).expand(B, 6)
        horizon_vals = dp_value_256[batch_idx, dyn_rows_256, :]
        prob_x = F.softmax(-horizon_vals / temp, dim=-1)
        exp_x_128 = (prob_x * self.x_coords_256.to(prob_x.device)).sum(dim=-1) * 0.5
        dp_anchors_128 = torch.stack([exp_x_128, kin_anchors_128[:, :, 1]], dim=-1)

        # 4. Closed-Loop GRU Refinement with Negative Risk Gradient Repulsive Force
        h_state = global_context
        refined_steps = []
        for t in range(6):
            cur_pt_128 = 0.75 * kin_anchors_128[:, t, :] + 0.25 * dp_anchors_128[:, t, :].to(kin_anchors_128.dtype)
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
            repulsive_vec = F.grid_sample(
                neg_risk_grad_256.to(roi_feat_map.dtype),
                sample_pt,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).view(B, 2)

            dp_norm = (dp_anchors_128[:, t, :].to(global_context.dtype) - 64.0) / 64.0
            kin_norm = (kin_anchors_128[:, t, :].to(global_context.dtype) - 64.0) / 64.0
            gru_in = torch.cat(
                [
                    dp_norm,
                    kin_norm,
                    local_feat.to(global_context.dtype),
                    local_risk.to(global_context.dtype),
                    repulsive_vec.to(global_context.dtype),
                    geo_14d.to(global_context.dtype),
                    dp_weights.to(global_context.dtype),
                ],
                dim=-1,
            )
            h_state = self.gru_cell(gru_in, h_state)
            # Nudge trajectory along repulsive force vector + learned GRU delta
            refined_steps.append(cur_pt_128 + self.step_out(h_state) + 0.08 * repulsive_vec.to(cur_pt_128.dtype))

        pred_traj_128 = torch.stack(refined_steps, dim=1)
        return pred_risk_256, dp_value_256, pred_traj_128


class HomoscedasticUncertaintyWeighter(nn.Module):
    """
    Innovation #10: Learnable Homoscedastic Task Uncertainty Weighting.
    """

    def __init__(self, num_tasks: int = 7):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(self, losses: list):
        total = 0.0
        for i, l in enumerate(losses):
            log_v = self.log_vars[i].clamp(min=-2.5, max=2.5)
            precision = torch.exp(-log_v)
            total = total + 0.5 * precision * l + 0.5 * log_v
        return total


class UniPercepDPNet_v3_Ultra(UniPercepDPNet_Pro256):
    """
    21-Innovation Grand Master Architecture (`UniPercep-DP-v3-Ultra` / `Causal256-Ultra`):
      Integrates all 6 upgrades of v2.0-Pro256 + 15 deep algorithmic innovations:
        - SpectralDenoiseBlock (2D-FFT frequency-domain gating)
        - DegradationFiLMModulator (adaptive feature-wise linear modulation)
        - HomographyMotionSalienceBlock (3x3 H-warped motion difference stream)
        - MetricEgoPositionalEncoding256 (metric distance prior + LiDAR occupancy calibration)
        - StripPolarAnisotropicBlock (orthogonal 9x1 and 1x9 strip convolutions)
        - SceneGraphMultiLabelEnhancer (10-class co-occurrence graph attention)
        - Distance-Prior Small-Object Peak Sharpener
        - CausalPotentialFieldDPPlanner256 (perception-to-planning bridge + K=3 multi-modal scorer + -grad R repulsive force)
        - HomoscedasticUncertaintyWeighter
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small.fb_in22k",
        pretrained_backbone: bool = True,
        use_ssl: bool = True,
        use_multisensor: bool = True,
        use_dp_lora: bool = True,
    ):
        super().__init__(
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            use_ssl=use_ssl,
            use_multisensor=use_multisensor,
            use_dp_lora=use_dp_lora,
        )
        self.spectral_denoise = SpectralDenoiseBlock(channels=256)
        self.deg_film = DegradationFiLMModulator(channels=256)
        self.homo_motion = HomographyMotionSalienceBlock(channels=256)
        self.metric_pos_enc = MetricEgoPositionalEncoding256(channels=256, bev_size=256)
        self.strip_polar_bev = StripPolarAnisotropicBlock(channels=256)
        self.label_graph = SceneGraphMultiLabelEnhancer(num_classes=10, dim=64)
        self.dp_planner_256 = CausalPotentialFieldDPPlanner256(in_channels=256, num_modes=3, temp=0.20)
        self.uncertainty_weighter = HomoscedasticUncertaintyWeighter(num_tasks=7)
        self.far_hm_boost = nn.Parameter(torch.tensor(0.15))

    def forward(
        self,
        corrupted_imgs: torch.Tensor,
        bev_lidar: torch.Tensor,
        geo_14d: torch.Tensor,
    ):
        if corrupted_imgs.dim() == 4:
            B, C_im, H_img, W_img = corrupted_imgs.shape
            front = corrupted_imgs
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

        flat_imgs = corrupted_imgs.reshape(B * 6, C_im, H_img, W_img)
        feats = self.backbone(flat_imgs)
        c2, c3, c4, c5 = feats[-4], feats[-3], feats[-2], feats[-1]

        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        fpn_feat = self.fpn_smooth(F.interpolate(p2, size=(64, 112), mode="bilinear", align_corners=False))
        cam_globals = F.adaptive_avg_pool2d(p5, 1).flatten(1).view(B, 6, 256)

        # 1. Spatial Gated SSL + 2D-FFT Spectral Denoising + Degradation-Aware FiLM Modulation
        fpn_refined, pred_clean_all = self.ssl_head(fpn_feat, (H_img, W_img), use_ssl=self.use_ssl)
        fpn_spectral = self.deg_film(self.spectral_denoise(fpn_refined))
        pred_clean_img = pred_clean_all.view(B, 6, 3, H_img, W_img)[:, 0]

        # 2. Explicit Homography Motion Salience on Front Camera + 6-Camera Polar Lift-Splat
        cam_fpn_6 = fpn_spectral.view(B, 6, 256, 64, 112)
        front_motion_enhanced = self.homo_motion(cam_fpn_6[:, 0], geo_14d)
        cam_fpn_6 = torch.cat([front_motion_enhanced.unsqueeze(1), cam_fpn_6[:, 1:]], dim=1)

        vis_bev_256 = self.polar_bev_proj(cam_fpn_6)
        fused_256, geo_emb, _ = self.fusion_256(
            vis_bev_256, bev_lidar_256, geo_14d, use_multisensor=self.use_multisensor
        )

        # 3. 2D Metric Distance Positional Encoding + LiDAR Gating + Strip-Polar BEV Enhancement
        fused_256 = self.metric_pos_enc(fused_256, bev_lidar_256)
        fused_256 = self.strip_polar_bev(fused_256)

        vis_global = cam_globals.mean(dim=1)
        fused_global = F.adaptive_avg_pool2d(fused_256, 1).flatten(1)
        global_context = self.global_context_proj(torch.cat([vis_global, fused_global, geo_emb], dim=-1))

        # 4. Multi-Modal Class-Query + 10-Class Co-occurrence Graph Attention
        base_ml_logits = self.multi_label_head(
            cam_globals, fused_256, bev_lidar_256, geo_emb, global_context
        )
        multi_label_logits = self.label_graph(base_ml_logits, global_context, geo_14d)

        # 5. 256x256 Segmentation & Distance-Prior Enhanced Small Object Detection
        seg_logits_256 = self.seg_head_256(fused_256)
        raw_hm_256, small_reg_256 = self.small_det_head_256(fused_256)
        far_mask = self.metric_pos_enc.far_prior_mask.to(raw_hm_256.dtype)
        small_hm_256 = (raw_hm_256 * (1.0 + torch.tanh(self.far_hm_boost) * far_mask)).clamp(0.0, 1.0)

        dp_weights, rag_rule_idx, risk_emb = self.lora_controller(
            global_context, multi_label_logits, use_dp_lora=self.use_dp_lora
        )

        # 6. Causal Perception-to-Planning + K=3 Multi-Modal + Negative Risk Gradient DP Planner
        pred_risk_256, dp_value_256, pred_traj = self.dp_planner_256(
            fused_256,
            seg_logits_256,
            small_hm_256,
            small_reg_256,
            risk_emb,
            geo_14d,
            dp_weights,
            use_dp_lora=self.use_dp_lora,
        )

        return {
            "pred_clean_img": pred_clean_img,
            "multi_label_logits": multi_label_logits,
            "seg_logits_256": seg_logits_256,
            "small_hm_256": small_hm_256,
            "small_reg_256": small_reg_256,
            "pred_risk_256": pred_risk_256,
            "dp_value_256": dp_value_256,
            "pred_traj": pred_traj,
            "dp_weights": dp_weights,
            "rag_rule_idx": rag_rule_idx,
        }
