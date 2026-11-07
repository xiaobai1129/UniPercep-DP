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


class SpectralDenoiseBlock(nn.Module):
    """
    Direction 7 (v3 Ultra): 2D-FFT Frequency-Domain Spectral Gating Block.
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

    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x_fp32 = x.float()
        H, W = x_fp32.shape[-2], x_fp32.shape[-1]

        # 2D Real FFT -> complex tensor (B, C, H, W//2 + 1)
        fft_x = torch.fft.rfft2(x_fp32, norm="ortho")
        cat_ri = torch.cat([fft_x.real, fft_x.imag], dim=1)  # (B, 2C, H, W//2 + 1)
        gate_ri = self.freq_gate(cat_ri)
        g_real, g_imag = torch.chunk(gate_ri, 2, dim=1)

        filtered_fft = torch.complex(fft_x.real * g_real, fft_x.imag * g_imag)
        x_denoised = torch.fft.irfft2(filtered_fft, s=(H, W), norm="ortho")
        out = x_fp32 + 0.35 * x_denoised
        return self.spatial_mix(out.to(orig_dtype))


class StripPolarAnisotropicBlock(nn.Module):
    """
    Direction 8A (v3 Ultra): Orthogonal 1x9 and 9x1 Strip Convolutions on 256x256 BEV.
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


class CausalMultiModalDPPlanner256(nn.Module):
    """
    Direction 8B & 9 (v3 Ultra):
      1. Causal Perception-to-Planning Bridge (`TaskPriorInteractionBridge`):
         Explicitly injects 256x256 segmentation probabilities (3ch: Drivable, Static, Dynamic),
         256x256 small-object heatmap (1ch), and 256x256 velocity field (4ch) into the risk cost generator.
      2. Multi-Modal Trajectory Proposal & Risk-Aware Soft Selection ($K=3$ modes: Left-Bypass, Center, Right-Bypass).
    """

    def __init__(self, in_channels: int = 256, num_modes: int = 3, temp: float = 0.20):
        super().__init__()
        self.num_modes = num_modes
        self.temp = temp

        # Explicit Perception Prior Bridge: 256 (BEV) + 3 (seg_probs) + 1 (small_hm) + 4 (small_reg) = 264
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

        # K=3 multi-modal kinematic proposal generator + mode logits
        self.mode_proposal_mlp = nn.Sequential(
            nn.Linear(256 + 14 + 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_modes * 12 + num_modes),
        )

        self.gru_cell = nn.GRUCell(input_size=2 + 2 + 64 + 1 + 14 + 3, hidden_size=256)
        self.step_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

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

        # Direction 9: Explicit Causal Perception-to-Planning Prior Injection
        seg_probs_256 = F.softmax(seg_logits_256.detach(), dim=1)  # (B, 3, 256, 256)
        perc_priors = torch.cat([seg_probs_256, small_hm_256.detach(), small_reg_256.detach()], dim=1)  # (B, 8, 256, 256)
        causal_bev = self.causal_bridge(torch.cat([fused_256, perc_priors.to(fused_256.dtype)], dim=1))

        # Combine learned risk map with explicit dynamic obstacle probability prior
        raw_risk = self.risk_head(causal_bev).squeeze(1)
        dyn_obs_prior = seg_probs_256[:, 2, :, :] * 0.35 + small_hm_256[:, 0, :, :] * 0.25
        pred_risk_256 = (0.75 * raw_risk + 0.25 * dyn_obs_prior.to(raw_risk.dtype)).clamp(0.0, 1.0)

        roi_feat_map = self.local_roi_proj(causal_bev)  # (B, 64, 256, 256)

        # Velocity-Adaptive Kinematic Base
        vx = geo_14d[:, 0:1]
        vy = geo_14d[:, 1:2]
        speed = torch.sqrt(vx * vx + vy * vy + 1e-6).clamp(min=0.0, max=20.0)
        time_steps = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=fused_256.device, dtype=fused_256.dtype).unsqueeze(0)
        kin_y_base = 8.0 + (speed * time_steps) / 0.5
        kin_x_base = 64.0 + (vx * time_steps * 0.25)
        kin_base = torch.stack([kin_x_base, kin_y_base], dim=-1)  # (B, 6, 2)

        # Direction 8B: Generate K=3 Multi-Modal Trajectory Proposals (Left-Bypass, Center, Right-Bypass)
        raw_proposals = self.mode_proposal_mlp(torch.cat([global_context, geo_14d, dp_weights], dim=-1))
        mode_deltas = raw_proposals[:, : self.num_modes * 12].view(B, self.num_modes, 6, 2)
        mode_logits = raw_proposals[:, self.num_modes * 12 :]  # (B, K)

        lateral_offsets = torch.tensor([-2.5, 0.0, 2.5], device=fused_256.device, dtype=fused_256.dtype).view(1, 3, 1)
        mode_trajs = kin_base.unsqueeze(1) + mode_deltas
        mode_trajs[:, :, :, 0] = mode_trajs[:, :, :, 0] + lateral_offsets * (time_steps.unsqueeze(1) / 3.0)

        # Score each of the K=3 candidate trajectories on the 256x256 Risk Map via F.grid_sample
        gx_k = (mode_trajs[:, :, :, 0] / 64.0) - 1.0  # (B, K, 6)
        gy_k = (mode_trajs[:, :, :, 1] / 64.0) - 1.0  # (B, K, 6)
        sample_grid_k = torch.stack([gx_k, gy_k], dim=-1).to(pred_risk_256.dtype)  # (B, K, 6, 2)
        sampled_risks_k = F.grid_sample(
            pred_risk_256.unsqueeze(1), sample_grid_k, mode="bilinear", padding_mode="border", align_corners=False
        ).squeeze(1)  # (B, K, 6)
        traj_risk_cost = sampled_risks_k.mean(dim=-1) * dp_weights[:, 0:1].to(pred_risk_256.dtype)  # (B, K)

        mode_probs = F.softmax(mode_logits.float() - 4.0 * traj_risk_cost.float(), dim=-1).to(mode_trajs.dtype)  # (B, K)
        kin_anchors_128 = (mode_trajs * mode_probs.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B, 6, 2)

        if not use_dp_lora:
            return pred_risk_256, pred_risk_256, kin_anchors_128

        # 256x256 Differentiable Dynamic Programming
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

        # Closed-loop local ROI grid_sample GRU refinement
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
            refined_steps.append(cur_pt_128 + self.step_out(h_state))

        pred_traj_128 = torch.stack(refined_steps, dim=1)
        return pred_risk_256, dp_value_256, pred_traj_128


class HomoscedasticUncertaintyWeighter(nn.Module):
    """
    Direction 10 (v3 Ultra): Learnable Homoscedastic Task Uncertainty Weighting (Kendall et al., CVPR).
    Automatically balances 7 multi-task loss terms during training without manual weight tuning.
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
    10-Direction Ultimate Architecture (`UniPercep-DP-v3-Ultra`):
      Combines all 6 upgrades of Pro256 + 4 new algorithmic breakthroughs:
        7. 2D-FFT Frequency-Domain Spectral Denoising (`SpectralDenoiseBlock`)
        8. Orthogonal Strip-Polar BEV Convolutions (`StripPolarAnisotropicBlock`) + K=3 Multi-Modal Trajectory Scorer
        9. Explicit Causal Perception-to-Planning Prior Bridge (`CausalMultiModalDPPlanner256`)
       10. Learnable Homoscedastic Multi-Task Uncertainty Weighting (`HomoscedasticUncertaintyWeighter`)
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
        # Direction 7: 2D-FFT Spectral Denoising
        self.spectral_denoise = SpectralDenoiseBlock(channels=256)
        # Direction 8A: Strip-Polar Anisotropic BEV Enhancement
        self.strip_polar_bev = StripPolarAnisotropicBlock(channels=256)
        # Direction 8B & 9: Causal Perception-to-Planning Multi-Modal DP Planner
        self.dp_planner_256 = CausalMultiModalDPPlanner256(in_channels=256, num_modes=3, temp=0.20)
        # Direction 10: Homoscedastic Task Uncertainty Weighter
        self.uncertainty_weighter = HomoscedasticUncertaintyWeighter(num_tasks=7)

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

        # Direction 5 + Direction 7: Spatial Gated SSL + 2D-FFT Frequency Spectral Denoising
        fpn_refined, pred_clean_all = self.ssl_head(fpn_feat, (H_img, W_img), use_ssl=self.use_ssl)
        fpn_spectral = self.spectral_denoise(fpn_refined)
        pred_clean_img = pred_clean_all.view(B, 6, 3, H_img, W_img)[:, 0]

        # Polar Lift-Splat + Cross-Attention + Direction 8A Strip-Polar BEV Enhancement
        cam_fpn_6 = fpn_spectral.view(B, 6, 256, 64, 112)
        vis_bev_256 = self.polar_bev_proj(cam_fpn_6)
        fused_256, geo_emb, _ = self.fusion_256(
            vis_bev_256, bev_lidar_256, geo_14d, use_multisensor=self.use_multisensor
        )
        fused_256 = self.strip_polar_bev(fused_256)

        vis_global = cam_globals.mean(dim=1)
        fused_global = F.adaptive_avg_pool2d(fused_256, 1).flatten(1)
        global_context = self.global_context_proj(torch.cat([vis_global, fused_global, geo_emb], dim=-1))

        multi_label_logits = self.multi_label_head(
            cam_globals, fused_256, bev_lidar_256, geo_emb, global_context
        )
        seg_logits_256 = self.seg_head_256(fused_256)
        small_hm_256, small_reg_256 = self.small_det_head_256(fused_256)

        dp_weights, rag_rule_idx, risk_emb = self.lora_controller(
            global_context, multi_label_logits, use_dp_lora=self.use_dp_lora
        )

        # Direction 8B & 9: Causal Perception-to-Planning Multi-Modal DP Planner
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
