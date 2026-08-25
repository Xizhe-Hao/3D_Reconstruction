from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_model.arch4_model import MultiViewFusion, ParamRegressionHead, VideoViTEncoder


class PhysicsBottleneck(nn.Module):
    """
    在跨视角融合后显式拆分三类物理潜变量：
    - contact_latent
    - deformation_latent
    - stress_latent
    """

    def __init__(self, in_dim: int, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
        )
        self.shared_norm = nn.LayerNorm(in_dim)
        self.contact_head = nn.Linear(in_dim, latent_dim)
        self.deformation_head = nn.Linear(in_dim, latent_dim)
        self.stress_head = nn.Linear(in_dim, latent_dim)
        self.contact_norm = nn.LayerNorm(latent_dim)
        self.deformation_norm = nn.LayerNorm(latent_dim)
        self.stress_norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(self.input_norm(z))
        h = self.shared_norm(h)
        return {
            "contact_latent": self.contact_norm(self.contact_head(h)),
            "deformation_latent": self.deformation_norm(self.deformation_head(h)),
            "stress_latent": self.stress_norm(self.stress_head(h)),
        }


class ActionClassificationHead(nn.Module):
    """从 bottleneck 拼接潜变量预测动作类别 logits。"""

    def __init__(self, in_dim: int, num_actions: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, max(128, in_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(128, in_dim // 2), int(num_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalQueryFieldHead(nn.Module):
    """
    时间查询式场解码头：
    - 输入视角特征 [B,V,D]
    - 为每个 t 注入可学习 time embedding
    - 逐帧解码为 [B,V,C,H,W]，再拼成 [B,V,C,T,H,W]
    """

    def __init__(
        self,
        feat_dim: int,
        num_frames: int,
        dec_h: int,
        dec_w: int,
        out_channels: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.dec_h = int(dec_h)
        self.dec_w = int(dec_w)
        self.out_channels = int(out_channels)
        self.time_embed = nn.Embedding(self.num_frames, int(feat_dim))
        self.decoder = nn.Sequential(
            nn.Linear(int(feat_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.out_channels * self.dec_h * self.dec_w),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, v, d = feat.shape
        t_ids = torch.arange(self.num_frames, device=feat.device, dtype=torch.long)
        t_emb = self.time_embed(t_ids).view(1, 1, self.num_frames, d)  # [1,1,T,D]
        feat_btvd = feat.unsqueeze(2) + t_emb  # [B,V,T,D]
        y = self.decoder(feat_btvd.reshape(b * v * self.num_frames, d))
        y = y.view(b, v, self.num_frames, self.out_channels, self.dec_h, self.dec_w)
        return y.permute(0, 1, 3, 2, 4, 5).contiguous()


def _group_norm(num_channels: int) -> nn.GroupNorm:
    for g in (8, 4, 2, 1):
        if int(num_channels) % g == 0:
            return nn.GroupNorm(g, int(num_channels))
    return nn.GroupNorm(1, int(num_channels))


class ConditionalConvFieldHead(nn.Module):
    """
    轻量条件式 dense field 解码头：
    - 全局特征 global_feat [B,Dg]：整体形变流程/跨视角摘要
    - 帧特征 frame_feat [B,V,T,Df]：当前时刻视觉参考
    - 物理潜变量 phys_latent [B,Dp]：材料/接触/应力先验
    先融合为时刻条件 token，再投影成低分辨率 patch grid，经轻量卷积上采样输出场图。
    """

    def __init__(
        self,
        *,
        global_dim: int,
        frame_dim: int,
        phys_dim: int,
        num_frames: int,
        dec_h: int,
        dec_w: int,
        out_channels: int = 3,
        token_dim: int = 512,
        base_channels: int = 128,
        dropout: float = 0.1,
        output_activation: str = "sigmoid",
        output_scale_init: float = 0.1,
        output_bias_init: float = -2.0,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.dec_h = int(dec_h)
        self.dec_w = int(dec_w)
        self.out_channels = int(out_channels)
        self.base_h = max(1, (self.dec_h + 7) // 8)
        self.base_w = max(1, (self.dec_w + 7) // 8)
        self.base_channels = int(base_channels)
        self.token_dim = int(token_dim)

        self.global_norm = nn.LayerNorm(int(global_dim))
        self.frame_norm = nn.LayerNorm(int(frame_dim))
        self.phys_norm = nn.LayerNorm(int(phys_dim))
        self.global_proj = nn.Linear(int(global_dim), self.token_dim)
        self.frame_proj = nn.Linear(int(frame_dim), self.token_dim)
        self.phys_proj = nn.Linear(int(phys_dim), self.token_dim)
        self.time_embed = nn.Embedding(self.num_frames, self.token_dim)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.seed_proj = nn.Linear(
            self.token_dim,
            self.base_channels * self.base_h * self.base_w,
        )
        self.seed_norm = _group_norm(self.base_channels)

        c1 = self.base_channels
        c2 = max(self.out_channels * 8, c1 // 2)
        c3 = max(self.out_channels * 4, c2 // 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            _group_norm(c1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            _group_norm(c2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            _group_norm(c3),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, self.out_channels, kernel_size=3, padding=1),
        )
        # 输出用于合成 RGB 视频，显式约束到 [0,1]。
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))
        self.output_bias = nn.Parameter(torch.tensor(float(output_bias_init)))
        self.output_activation = str(output_activation).strip().lower()

    def _apply_output_activation(self, y: torch.Tensor) -> torch.Tensor:
        act = self.output_activation
        if act == "sigmoid":
            return torch.sigmoid(self.output_scale.abs() * y + self.output_bias)
        if act == "identity":
            return y
        if act == "tanh":
            return torch.tanh(self.output_scale.abs() * y + self.output_bias)
        raise ValueError(f"unknown output_activation: {self.output_activation}")

    def forward(
        self,
        global_feat: torch.Tensor,
        frame_feat: torch.Tensor,
        phys_latent: torch.Tensor,
    ) -> torch.Tensor:
        if global_feat.dim() != 2:
            raise ValueError(f"global_feat expect [B,D], got {tuple(global_feat.shape)}")
        if frame_feat.dim() != 4:
            raise ValueError(f"frame_feat expect [B,V,T,D], got {tuple(frame_feat.shape)}")
        if phys_latent.dim() != 2:
            raise ValueError(f"phys_latent expect [B,D], got {tuple(phys_latent.shape)}")

        b, v, t, _ = frame_feat.shape
        if int(t) > self.num_frames:
            raise ValueError(f"T exceeds configured num_frames: T={t} num_frames={self.num_frames}")

        t_ids = torch.arange(t, device=frame_feat.device, dtype=torch.long)
        global_tok = self.global_proj(self.global_norm(global_feat)).view(b, 1, 1, self.token_dim)
        frame_tok = self.frame_proj(self.frame_norm(frame_feat))
        phys_tok = self.phys_proj(self.phys_norm(phys_latent)).view(b, 1, 1, self.token_dim)
        time_tok = self.time_embed(t_ids).view(1, 1, t, self.token_dim)

        cond = self.token_norm(global_tok + frame_tok + phys_tok + time_tok)
        cond = cond + self.token_mlp(cond)
        seed = self.seed_proj(cond.reshape(b * v * t, self.token_dim))
        seed = seed.view(b * v * t, self.base_channels, self.base_h, self.base_w)
        seed = self.seed_norm(seed)

        y = self.decoder(seed)
        if int(y.shape[-2]) != self.dec_h or int(y.shape[-1]) != self.dec_w:
            y = F.interpolate(y, size=(self.dec_h, self.dec_w), mode="bilinear", align_corners=False)
        y = self._apply_output_activation(y)
        y = y.view(b, v, t, self.out_channels, self.dec_h, self.dec_w)
        return y.permute(0, 1, 3, 2, 4, 5).contiguous()


class MultiScaleSpatialConditionStem(nn.Module):
    """从原始 RGB 帧提取低分辨率空间条件，用于增强场头局部几何泛化。"""

    def __init__(self, in_channels: int = 3, out_channels: int = 48, out_h: int = 14, out_w: int = 14):
        super().__init__()
        c0 = max(16, int(out_channels) // 2)
        c1 = max(24, int(out_channels))
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), c0, kernel_size=3, stride=2, padding=1),
            _group_norm(c0),
            nn.GELU(),
            nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1),
            _group_norm(c1),
            nn.GELU(),
            nn.Conv2d(c1, int(out_channels), kernel_size=3, stride=2, padding=1),
            _group_norm(int(out_channels)),
            nn.GELU(),
            nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1),
            _group_norm(int(out_channels)),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.out_h, self.out_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expect [N,C,H,W], got {tuple(x.shape)}")
        return self.pool(self.net(x))


class TemporalSeedMixBlock(nn.Module):
    """低分辨率时序混合块：先做 depthwise temporal conv，再做 pointwise-MLP。"""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        c = int(channels)
        self.temporal = nn.Conv3d(c, c, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=c)
        self.temporal_pw = nn.Conv3d(c, c, kernel_size=1)
        self.temporal_norm = _group_norm(c)
        self.mlp = nn.Sequential(
            nn.Conv3d(c, c * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv3d(c * 2, c, kernel_size=1),
        )
        self.mlp_norm = _group_norm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expect [N,C,T,H,W], got {tuple(x.shape)}")
        y = x + self.temporal_pw(self.temporal(x))
        y = self.temporal_norm(y)
        y = y + self.mlp(y)
        return self.mlp_norm(y)


class SharedConditionalFieldDecoder(nn.Module):
    """
    共享场解码 trunk：
    - 使用 global/frame/shared_phys/time 融合生成低分辨率 seed
    - 可选注入多尺度空间条件
    - 在低分辨率 seed 上加入轻量时序偏置
    - 输出共享 dense field feature map，供三个任务头复用
    """

    def __init__(
        self,
        *,
        global_dim: int,
        frame_dim: int,
        shared_phys_dim: int,
        num_frames: int,
        dec_h: int,
        dec_w: int,
        token_dim: int = 512,
        base_channels: int = 128,
        shared_channels: int = 64,
        spatial_cond_dim: int = 0,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        geometry_dim: int = 0,
        patch_cond_dim: int = 0,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.dec_h = int(dec_h)
        self.dec_w = int(dec_w)
        self.base_h = max(1, (self.dec_h + 7) // 8)
        self.base_w = max(1, (self.dec_w + 7) // 8)
        self.base_channels = int(base_channels)
        self.shared_channels = int(shared_channels)
        self.token_dim = int(token_dim)
        self.spatial_cond_dim = int(max(0, spatial_cond_dim))
        self.geometry_dim = int(max(0, geometry_dim))
        self.patch_cond_dim = int(max(0, patch_cond_dim))

        self.global_norm = nn.LayerNorm(int(global_dim))
        self.frame_norm = nn.LayerNorm(int(frame_dim))
        self.shared_phys_norm = nn.LayerNorm(int(shared_phys_dim))
        self.global_proj = nn.Linear(int(global_dim), self.token_dim)
        self.frame_proj = nn.Linear(int(frame_dim), self.token_dim)
        self.shared_phys_proj = nn.Linear(int(shared_phys_dim), self.token_dim)
        self.geometry_norm = (
            nn.LayerNorm(int(self.geometry_dim)) if self.geometry_dim > 0 else None
        )
        self.geometry_proj = (
            nn.Linear(int(self.geometry_dim), self.token_dim) if self.geometry_dim > 0 else None
        )
        self.time_embed = nn.Embedding(self.num_frames, self.token_dim)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.seed_proj = nn.Linear(
            self.token_dim,
            self.base_channels * self.base_h * self.base_w,
        )
        self.seed_norm = _group_norm(self.base_channels)
        self.spatial_proj = (
            nn.Conv2d(self.spatial_cond_dim, self.base_channels, kernel_size=1)
            if self.spatial_cond_dim > 0
            else None
        )
        self.patch_seed_proj = (
            nn.Conv2d(self.patch_cond_dim, self.base_channels, kernel_size=1)
            if self.patch_cond_dim > 0
            else None
        )
        self.patch_out_proj = (
            nn.Conv2d(self.patch_cond_dim, self.shared_channels, kernel_size=1)
            if self.patch_cond_dim > 0
            else None
        )

        self.temporal_blocks = nn.Sequential(
            *[TemporalSeedMixBlock(self.base_channels, dropout=float(dropout)) for _ in range(max(0, int(temporal_layers)))]
        )

        c1 = self.base_channels
        c2 = max(self.shared_channels * 2, c1 // 2)
        c3 = max(self.shared_channels, c2 // 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            _group_norm(c1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            _group_norm(c2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            _group_norm(c3),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, self.shared_channels, kernel_size=3, padding=1),
            _group_norm(self.shared_channels),
            nn.GELU(),
        )

    def forward(
        self,
        global_feat: torch.Tensor,
        frame_feat: torch.Tensor,
        shared_phys_latent: torch.Tensor,
        spatial_cond: torch.Tensor | None = None,
        geometry_latent: Optional[torch.Tensor] = None,
        patch_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if global_feat.dim() != 2:
            raise ValueError(f"global_feat expect [B,D], got {tuple(global_feat.shape)}")
        if frame_feat.dim() != 4:
            raise ValueError(f"frame_feat expect [B,V,T,D], got {tuple(frame_feat.shape)}")
        if shared_phys_latent.dim() != 2:
            raise ValueError(f"shared_phys_latent expect [B,D], got {tuple(shared_phys_latent.shape)}")

        b, v, t, _ = frame_feat.shape
        if int(t) > self.num_frames:
            raise ValueError(f"T exceeds configured num_frames: T={t} num_frames={self.num_frames}")
        if spatial_cond is not None and spatial_cond.dim() != 6:
            raise ValueError(f"spatial_cond expect [B,V,T,C,H,W], got {tuple(spatial_cond.shape)}")
        if patch_cond is not None:
            if patch_cond.dim() != 6:
                raise ValueError(f"patch_cond expect [B,V,T,C,H,W], got {tuple(patch_cond.shape)}")
            if int(patch_cond.shape[3]) != self.patch_cond_dim or self.patch_seed_proj is None or self.patch_out_proj is None:
                raise ValueError(
                    f"patch_cond C mismatch or decoder has patch_cond_dim=0: got C={patch_cond.shape[3]}, "
                    f"expect patch_cond_dim={self.patch_cond_dim}"
                )
        if geometry_latent is not None:
            if self.geometry_dim <= 0 or self.geometry_norm is None or self.geometry_proj is None:
                raise ValueError("geometry_latent provided but decoder.geometry_dim is 0")
            if geometry_latent.dim() != 2 or int(geometry_latent.shape[1]) != self.geometry_dim:
                raise ValueError(
                    f"geometry_latent expect [B,{self.geometry_dim}], got {tuple(geometry_latent.shape)}"
                )

        t_ids = torch.arange(t, device=frame_feat.device, dtype=torch.long)
        global_tok = self.global_proj(self.global_norm(global_feat)).view(b, 1, 1, self.token_dim)
        frame_tok = self.frame_proj(self.frame_norm(frame_feat))
        shared_phys_tok = self.shared_phys_proj(self.shared_phys_norm(shared_phys_latent)).view(b, 1, 1, self.token_dim)
        time_tok = self.time_embed(t_ids).view(1, 1, t, self.token_dim)
        geom_tok = 0.0
        if geometry_latent is not None and self.geometry_norm is not None and self.geometry_proj is not None:
            geom_tok = self.geometry_proj(self.geometry_norm(geometry_latent)).view(b, 1, 1, self.token_dim)

        cond = self.token_norm(global_tok + frame_tok + shared_phys_tok + time_tok + geom_tok)
        cond = cond + self.token_mlp(cond)
        seed = self.seed_proj(cond.reshape(b * v * t, self.token_dim))
        seed = seed.view(b * v * t, self.base_channels, self.base_h, self.base_w)

        if spatial_cond is not None and self.spatial_proj is not None:
            sc = spatial_cond.reshape(b * v * t, spatial_cond.shape[3], spatial_cond.shape[4], spatial_cond.shape[5])
            if int(sc.shape[-2]) != self.base_h or int(sc.shape[-1]) != self.base_w:
                sc = F.interpolate(sc, size=(self.base_h, self.base_w), mode="bilinear", align_corners=False)
            seed = seed + self.spatial_proj(sc)
        if patch_cond is not None and self.patch_seed_proj is not None:
            pc = patch_cond.reshape(b * v * t, patch_cond.shape[3], patch_cond.shape[4], patch_cond.shape[5])
            if int(pc.shape[-2]) != self.base_h or int(pc.shape[-1]) != self.base_w:
                pc = F.interpolate(pc, size=(self.base_h, self.base_w), mode="bilinear", align_corners=False)
            seed = seed + self.patch_seed_proj(pc)
        seed = self.seed_norm(seed)

        seed_3d = seed.view(b * v, t, self.base_channels, self.base_h, self.base_w).permute(0, 2, 1, 3, 4).contiguous()
        if len(self.temporal_blocks) > 0:
            seed_3d = self.temporal_blocks(seed_3d)
        y = seed_3d.permute(0, 2, 1, 3, 4).contiguous().view(b * v * t, self.base_channels, self.base_h, self.base_w)

        y = self.decoder(y)
        if int(y.shape[-2]) != self.dec_h or int(y.shape[-1]) != self.dec_w:
            y = F.interpolate(y, size=(self.dec_h, self.dec_w), mode="bilinear", align_corners=False)
        if patch_cond is not None and self.patch_out_proj is not None:
            pc2 = patch_cond.reshape(b * v * t, patch_cond.shape[3], patch_cond.shape[4], patch_cond.shape[5])
            if int(pc2.shape[-2]) != self.dec_h or int(pc2.shape[-1]) != self.dec_w:
                pc2 = F.interpolate(pc2, size=(self.dec_h, self.dec_w), mode="bilinear", align_corners=False)
            y = y + self.patch_out_proj(pc2)
        y = y.view(b, v, t, self.shared_channels, self.dec_h, self.dec_w)
        return y.permute(0, 1, 3, 2, 4, 5).contiguous()


class TaskAdaptiveFieldHead(nn.Module):
    """轻量任务头：在共享场特征上做任务特定调制并输出最终场图。"""

    def __init__(
        self,
        *,
        shared_dim: int,
        task_phys_dim: int,
        out_channels: int = 3,
        dropout: float = 0.1,
        output_activation: str = "sigmoid",
        output_scale_init: float = 0.1,
        output_bias_init: float = -2.0,
    ):
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.out_channels = int(out_channels)
        self.task_phys_norm = nn.LayerNorm(int(task_phys_dim))
        self.task_mod = nn.Sequential(
            nn.Linear(int(task_phys_dim), self.shared_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(self.shared_dim, self.shared_dim, kernel_size=3, padding=1),
            _group_norm(self.shared_dim),
            nn.GELU(),
            nn.Conv2d(self.shared_dim, self.out_channels, kernel_size=3, padding=1),
        )
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))
        self.output_bias = nn.Parameter(torch.tensor(float(output_bias_init)))
        self.output_activation = str(output_activation).strip().lower()

    def _apply_output_activation(self, y: torch.Tensor) -> torch.Tensor:
        act = self.output_activation
        if act == "sigmoid":
            return torch.sigmoid(self.output_scale.abs() * y + self.output_bias)
        if act == "identity":
            return y
        if act == "tanh":
            return torch.tanh(self.output_scale.abs() * y + self.output_bias)
        raise ValueError(f"unknown output_activation: {self.output_activation}")

    def forward(self, shared_feat: torch.Tensor, task_phys_latent: torch.Tensor) -> torch.Tensor:
        if shared_feat.dim() != 6:
            raise ValueError(f"shared_feat expect [B,V,C,T,H,W], got {tuple(shared_feat.shape)}")
        if task_phys_latent.dim() != 2:
            raise ValueError(f"task_phys_latent expect [B,D], got {tuple(task_phys_latent.shape)}")

        b, v, c, t, h, w = shared_feat.shape
        if int(c) != self.shared_dim:
            raise ValueError(f"shared_feat C mismatch: expect {self.shared_dim}, got {c}")

        gamma_beta = self.task_mod(self.task_phys_norm(task_phys_latent)).view(b, 2, self.shared_dim, 1, 1, 1)
        # [B,1,C,1,1,1] 以便与 [B,V,C,T,H,W] 在 batch 与 view 维上正确广播（避免 5D gamma 与 V 轴误对齐）
        gamma = (0.25 * torch.tanh(gamma_beta[:, 0])).unsqueeze(1)
        beta = (0.25 * torch.tanh(gamma_beta[:, 1])).unsqueeze(1)
        y = shared_feat * (1.0 + gamma) + beta
        y = y.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, self.shared_dim, h, w)
        y = self.refine(y)
        y = self._apply_output_activation(y)
        y = y.view(b, v, t, self.out_channels, h, w)
        return y.permute(0, 1, 3, 2, 4, 5).contiguous()


class StressFlowForceFuser(nn.Module):
    """将共享场特征与已预测的 flow/force 通道拼接后融合，供 stress 头在因果顺序下使用。"""

    def __init__(
        self,
        *,
        shared_dim: int,
        flow_channels: int = 3,
        force_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.flow_channels = int(flow_channels)
        self.force_channels = int(force_channels)
        in_ch = self.shared_dim + self.flow_channels + self.force_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, self.shared_dim, kernel_size=1),
            _group_norm(self.shared_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(self.shared_dim, self.shared_dim, kernel_size=3, padding=1),
            _group_norm(self.shared_dim),
            nn.GELU(),
        )

    def forward(
        self,
        shared_feat: torch.Tensor,
        flow_pred: torch.Tensor,
        force_pred: torch.Tensor,
    ) -> torch.Tensor:
        if shared_feat.dim() != 6:
            raise ValueError(f"shared_feat expect [B,V,C,T,H,W], got {tuple(shared_feat.shape)}")
        b, v, c, t, h, w = shared_feat.shape
        if int(c) != self.shared_dim:
            raise ValueError(f"shared_feat C mismatch: expect {self.shared_dim}, got {c}")
        x = torch.cat([shared_feat, flow_pred, force_pred], dim=2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, x.shape[2], h, w)
        y = self.net(x)
        y = y.view(b, v, t, self.shared_dim, h, w).permute(0, 1, 3, 2, 4, 5).contiguous()
        return y


class StressSpatialEnhancer(nn.Module):
    """
    在 stress 专用阶段对融合后的场特征做空间残差细化（不改动共享 trunk 权重时仍可训练本模块）。
    输入/输出: [B,V,C,T,H,W]，C 为 shared_field 通道维（与 StressFlowForceFuser 输出一致）。
    """

    def __init__(self, *, channels: int, dropout: float = 0.1):
        super().__init__()
        c = int(channels)
        self.channels = c
        self.body = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            _group_norm(c),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            _group_norm(c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(x.shape)}")
        b, v, c, t, h, w = x.shape
        if int(c) != self.channels:
            raise ValueError(f"C mismatch: expect {self.channels}, got {c}")
        y = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, c, h, w)
        y = self.body(y)
        y = y.view(b, v, t, c, h, w).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x + y


class FlowSpatialEnhancer(nn.Module):
    """
    与 StressSpatialEnhancer 对称：在 flow 头前对共享 trunk 特征做空间残差细化，
    缓解「仅 stress 有 fuser+enhancer、flow 直接接 head」时的细节劣势。
    输入/输出: [B,V,C,T,H,W]，C 为 shared_field 通道维。
    """

    def __init__(self, *, channels: int, dropout: float = 0.1):
        super().__init__()
        c = int(channels)
        self.channels = c
        self.body = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            _group_norm(c),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            _group_norm(c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(x.shape)}")
        b, v, c, t, h, w = x.shape
        if int(c) != self.channels:
            raise ValueError(f"C mismatch: expect {self.channels}, got {c}")
        y = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, c, h, w)
        y = self.body(y)
        y = y.view(b, v, t, c, h, w).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x + y


class SharedTaskPhysRefiner(nn.Module):
    """将拼接后的物理 latent 重构为 shared + task-specific 两级条件。"""

    def __init__(self, in_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(in_dim))
        self.shared = nn.Sequential(
            nn.Linear(int(in_dim), int(in_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(in_dim), int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
        )
        self.contact = nn.Sequential(
            nn.Linear(int(in_dim), int(latent_dim)),
            nn.GELU(),
            nn.LayerNorm(int(latent_dim)),
        )
        self.deformation = nn.Sequential(
            nn.Linear(int(in_dim), int(latent_dim)),
            nn.GELU(),
            nn.LayerNorm(int(latent_dim)),
        )
        self.stress = nn.Sequential(
            nn.Linear(int(in_dim), int(latent_dim)),
            nn.GELU(),
            nn.LayerNorm(int(latent_dim)),
        )
        self.contact_out = nn.LayerNorm(int(latent_dim))
        self.deformation_out = nn.LayerNorm(int(latent_dim))
        self.stress_out = nn.LayerNorm(int(latent_dim))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError(f"expect [B,D], got {tuple(z.shape)}")
        h = self.input_norm(z)
        shared = self.shared(h)
        contact = self.contact_out(shared + self.contact(h))
        deformation = self.deformation_out(shared + self.deformation(h))
        stress = self.stress_out(shared + self.stress(h))
        return {
            "shared_phys_latent": shared,
            "contact_latent": contact,
            "deformation_latent": deformation,
            "stress_latent": stress,
        }


class DenseFieldFeatureEncoder(nn.Module):
    """从 dense field [B,V,C,T,H,W] 提取样本级特征 [B,D]。"""

    def __init__(self, in_channels: int = 3, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        c0 = 32
        c1 = 64
        self.body = nn.Sequential(
            nn.Conv3d(int(in_channels), c0, kernel_size=3, padding=1),
            _group_norm(c0),
            nn.GELU(),
            nn.Conv3d(c0, c1, kernel_size=3, padding=1, stride=(1, 2, 2)),
            _group_norm(c1),
            nn.GELU(),
            nn.Conv3d(c1, c1, kernel_size=3, padding=1, stride=(1, 2, 2)),
            _group_norm(c1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        self.proj = nn.Sequential(
            nn.Linear(c1 * 2 * 2 * 2, 256),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )
        self.view_fuse = nn.Sequential(
            nn.Linear(int(out_dim) * 2, int(out_dim)),
            nn.GELU(),
            nn.LayerNorm(int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(x.shape)}")
        b, v, c, t, h, w = x.shape
        y = x.reshape(b * v, c, t, h, w).contiguous()
        try:
            y = self.body(y)
        except RuntimeError as e:
            msg = str(e).lower()
            if y.is_cuda and ("cudnn" in msg or "unable to find a valid cudnn algorithm" in msg):
                # 某些 3D 卷积 shape 在长程训练切到 joint/param 阶段后会触发 cuDNN 选算法失败；
                # 这里局部回退到原生实现，避免整轮训练在参数场编码器处中断。
                with torch.backends.cudnn.flags(enabled=False):
                    y = self.body(y)
            else:
                raise
        y = y.flatten(1)
        y = self.proj(y).view(b, v, -1)
        y_mean = y.mean(dim=1)
        y_max = y.amax(dim=1)
        return self.view_fuse(torch.cat([y_mean, y_max], dim=1))


class LogicPhysModel(nn.Module):
    """
    最小可行版本：
    输入 `[B,V,C,T,H,W]`
    输出保持现有键，并新增 bottleneck/action 输出。
    """

    def __init__(
        self,
        *,
        num_views: int = 4,
        in_channels: int = 3,
        num_frames: int = 30,
        img_size: int = 224,
        num_targets: int = 4,
        num_actions: int = 4,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 6,
        encoder_num_heads: int = 6,
        tubelet_size: int = 1,
        patch_size: int = 32,
        fusion_dim: int = 512,
        fusion_heads: int = 8,
        use_attention_pool: bool = True,
        use_uncertainty: bool = False,
        encoder_dropout: float = 0.05,
        dec_h: int = 56,
        dec_w: int = 56,
        use_aux_field_heads: bool = True,
        head_dropout: float = 0.1,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.in_channels = int(in_channels)
        self.num_targets = int(num_targets)
        self.num_frames = int(num_frames)
        self.dec_h = int(dec_h)
        self.dec_w = int(dec_w)
        self.use_aux_field_heads = bool(use_aux_field_heads)
        self.use_uncertainty = bool(use_uncertainty)

        self.encoder = VideoViTEncoder(
            self.in_channels,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            tubelet_size,
            patch_size,
            encoder_dropout,
            img_size,
            self.num_frames,
        )
        self.fusion = MultiViewFusion(
            self.num_views,
            encoder_embed_dim,
            fusion_dim,
            fusion_heads,
            use_attention_pool,
            head_dropout,
        )

        self.physics_bottleneck = PhysicsBottleneck(
            in_dim=encoder_embed_dim,
            latent_dim=int(bottleneck_dim),
            dropout=head_dropout,
        )
        bottleneck_out_dim = int(bottleneck_dim) * 3

        self.param_head = ParamRegressionHead(
            bottleneck_out_dim,
            self.num_targets,
            head_dropout,
            self.use_uncertainty,
        )
        self.action_head = ActionClassificationHead(
            bottleneck_out_dim,
            int(num_actions),
            dropout=head_dropout,
        )

        # 将 bottleneck 全局表征注入每个视角，再走场解码头。
        self.view_condition_proj = nn.Linear(bottleneck_out_dim, encoder_embed_dim)
        # 显式视角身份编码：在场头解码前为每个 view 注入可学习区分信号。
        self.view_id_emb = nn.Embedding(self.num_views, encoder_embed_dim)
        self.view_id_scale = nn.Parameter(torch.tensor(1.0))
        # 场输出改为 RGB 三通道: [B,V,3,T,H,W]。
        # 使用时间查询式解码，避免单向量一次性硬回归整段时间序列。
        self.stress_head = TemporalQueryFieldHead(
            encoder_embed_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=head_dropout,
        )
        self.flow_head = TemporalQueryFieldHead(
            encoder_embed_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=head_dropout,
        )
        self.force_head = TemporalQueryFieldHead(
            encoder_embed_dim,
            self.num_frames,
            self.dec_h,
            self.dec_w,
            out_channels=3,
            hidden_dim=512,
            dropout=head_dropout,
        )

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(rgb.shape)}")
        b, v, c, t, h, w = rgb.shape
        if v != self.num_views:
            raise ValueError(f"V mismatch: expect {self.num_views}, got {v}")
        if c != self.in_channels:
            raise ValueError(f"C mismatch: expect {self.in_channels}, got {c}")

        feat_v = self.encoder(rgb.reshape(b * v, c, t, h, w)).view(b, v, -1)  # [B,V,D]
        z_fused = self.fusion(feat_v)  # [B,D]

        pb = self.physics_bottleneck(z_fused)
        z_phys = torch.cat(
            [pb["contact_latent"], pb["deformation_latent"], pb["stress_latent"]],
            dim=1,
        )  # [B,3*Db]

        param_pred, logvar_raw = self.param_head(z_phys)
        action_logits = self.action_head(z_phys)

        # [B,D] -> [B,V,D]
        view_bias = self.view_condition_proj(z_phys).unsqueeze(1).expand(-1, v, -1)
        view_ids = torch.arange(v, device=feat_v.device, dtype=torch.long)
        view_bias_id = self.view_id_emb(view_ids).unsqueeze(0).expand(b, -1, -1)
        feat_v_cond = feat_v + view_bias + self.view_id_scale * view_bias_id

        if self.use_aux_field_heads:
            stress_field_pred = self.stress_head(feat_v_cond)
            flow_field_pred = self.flow_head(feat_v_cond)
            force_pred = self.force_head(feat_v_cond)
        else:
            z0 = feat_v.new_zeros(b, v, 3, self.num_frames, self.dec_h, self.dec_w)
            stress_field_pred = z0
            flow_field_pred = z0
            force_pred = z0

        param_pred_raw = torch.stack(
            [
                torch.expm1(torch.clamp(param_pred[:, 0], min=0.0)),
                torch.clamp(param_pred[:, 1], min=0.0, max=0.5),
                torch.expm1(torch.clamp(param_pred[:, 2], min=0.0)),
                torch.expm1(torch.clamp(param_pred[:, 3], min=0.0)),
            ],
            dim=1,
        )
        logvar = logvar_raw if (self.use_uncertainty and logvar_raw is not None) else param_pred.new_zeros(b, self.num_targets)

        return {
            # 兼容旧输出
            "param_pred": param_pred,
            "param_pred_raw": param_pred_raw,
            "stress_field_pred": stress_field_pred,
            "flow_field_pred": flow_field_pred,
            "force_pred": force_pred,
            "logvar": logvar,
            # 新增输出
            "contact_latent": pb["contact_latent"],
            "deformation_latent": pb["deformation_latent"],
            "stress_latent": pb["stress_latent"],
            "action_logits": action_logits,
        }

