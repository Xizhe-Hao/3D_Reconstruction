# -*- coding: utf-8 -*-
"""
Arch4 物理视频回归（skills/模型搭建）：

- 输入：``rgb [B, V, C, T, H, W]``（当前流程默认 ``C=1``，仅 images 单通道时序）。
- 输出：**固定键名** ``dict[str, Tensor]``（单卡/DDP 键集合一致）：

  - ``param_pred``：``[B, num_targets]``，参数头直接输出训练目标空间（``[logE, nu, logDensity, logYield]``）
  - ``param_pred_raw``：``[B, num_targets]``，由 ``param_pred`` 反变换得到的原始物理量（便于日志/可视化）
  - ``stress_field_pred`` / ``flow_field_pred`` / ``force_pred``：``[B, V, 1, T, dec_h, dec_w]``（时序输出，不做时间压缩）
  - ``logvar``：``[B, num_targets]``；非异方差时为全零

参考 backbone：VideoMAE 风格 tubelet + ViT encoder。
"""

from __future__ import annotations

import inspect
from typing import Dict

import torch
import torch.nn as nn


def _te_kwargs() -> dict:
    sig = inspect.signature(nn.TransformerEncoder.__init__)
    return {"enable_nested_tensor": False} if "enable_nested_tensor" in sig.parameters else {}


def trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    nn.init.normal_(t, std=std)
    return t


class VideoTubeletEmbed(nn.Module):
    """``[B, C, T, H, W]`` → token 序列 ``[B, N, D]``。"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        tubelet_size: int,
        patch_size: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, d, tp, hp, wp = x.shape
        return x.flatten(2).transpose(1, 2).contiguous()


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        _sig = inspect.signature(nn.MultiheadAttention.__init__).parameters
        bias = "bias" in _sig
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True, **({"bias": True} if bias else {})
        )
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class VideoViTEncoder(nn.Module):
    """仅编码器：tubelet + pos + ViT blocks + token mean pool → ``[B, D]``。"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        tubelet_size: int,
        patch_size: int,
        dropout: float,
        img_size: int,
        num_frames: int,
    ):
        super().__init__()
        self.tubelet = VideoTubeletEmbed(in_channels, embed_dim, tubelet_size, patch_size)
        with torch.no_grad():
            n_tok = self.tubelet(
                torch.zeros(1, in_channels, num_frames, img_size, img_size)
            ).shape[1]
        self.num_tokens = n_tok
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tok, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            ViTBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.tubelet(x) + self.pos_embed
        tok = self.pos_drop(tok)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return tok.mean(dim=1)


class AttentionViewPool(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(dim)
        _sig = inspect.signature(nn.MultiheadAttention.__init__).parameters
        bias = "bias" in _sig
        self.mha = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, **({"bias": True} if bias else {})
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, d = x.shape
        q = self.query.expand(b, -1, -1)
        xn = self.norm(x)
        o, _ = self.mha(q, xn, xn, need_weights=False)
        return o.squeeze(1)


class MultiViewFusion(nn.Module):
    def __init__(
        self,
        num_views: int,
        view_dim: int,
        fusion_dim: int,
        fusion_heads: int,
        use_attention_pool: bool,
        dropout: float,
    ):
        super().__init__()
        self.num_views = num_views
        self.use_attention_pool = use_attention_pool
        self.view_emb = nn.Parameter(torch.zeros(1, num_views, view_dim))
        trunc_normal_(self.view_emb, std=0.02)
        self.proj_up = nn.Linear(view_dim, fusion_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=fusion_heads,
            dim_feedforward=int(fusion_dim * 4),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=2, **_te_kwargs())
        self.pool = AttentionViewPool(fusion_dim, fusion_heads) if use_attention_pool else None
        self.proj_down = nn.Linear(fusion_dim, view_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.view_emb
        x = self.proj_up(x)
        x = self.enc(x)
        if self.pool is not None:
            x = self.pool(x)
        else:
            x = x.mean(dim=1)
        return self.proj_down(x)


class ParamRegressionHead(nn.Module):
    """
    物理参数头（与场头解耦）：
    - 共享 trunk 后接 4 个独立小 head，降低不同参数分布差异的相互干扰；
    - 直接输出训练目标空间 [logE, nu, logDensity, logYield]。
    """

    def __init__(self, in_dim: int, num_targets: int, dropout: float, use_uncertainty: bool):
        super().__init__()
        if int(num_targets) != 4:
            raise ValueError(f"ParamRegressionHead currently expects num_targets=4, got {num_targets}")
        self.use_uncertainty = use_uncertainty
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        def _small_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        # 四个参数分开预测
        self.e_head = _small_head()
        self.nu_head = _small_head()
        self.density_head = _small_head()
        self.yield_head = _small_head()

        self.logvar_head = nn.Linear(256, num_targets) if use_uncertainty else None

    def forward(self, z: torch.Tensor) -> tuple:
        h = self.trunk(z)
        e_raw = self.e_head(h).squeeze(-1)
        nu_raw = self.nu_head(h).squeeze(-1)
        density_raw = self.density_head(h).squeeze(-1)
        yield_raw = self.yield_head(h).squeeze(-1)

        # 约束：
        # - logE/logDensity/logYield >= 0（对应 raw >= 0）
        # - nu in [0, 0.5]
        e_log = torch.nn.functional.softplus(e_raw)
        density_log = torch.nn.functional.softplus(density_raw)
        yield_log = torch.nn.functional.softplus(yield_raw)
        nu = 0.5 * torch.sigmoid(nu_raw)

        mean = torch.stack([e_log, nu, density_log, yield_log], dim=1)
        if self.logvar_head is not None:
            return mean, self.logvar_head(h)
        return mean, None


class FieldHead(nn.Module):
    """单场头：视角特征 → ``[B, V, 1, T, dec_h, dec_w]``（与 param 头解耦）。"""

    def __init__(self, feat_dim: int, num_frames: int, dec_h: int, dec_w: int, out_channels: int = 1):
        super().__init__()
        out = int(out_channels) * int(num_frames) * int(dec_h) * int(dec_w)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, out),
        )
        self.out_channels = int(out_channels)
        self.num_frames = int(num_frames)
        self.dec_h = dec_h
        self.dec_w = dec_w

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, v, d = feat.shape
        y = self.net(feat.reshape(b * v, d))
        return y.view(b, v, self.out_channels, self.num_frames, self.dec_h, self.dec_w)


class Arch4VideoMAEPhysModel(nn.Module):
    """
    多任务：参数回归 + stress/flow/force_mask 场预测（三场均自视角特征解码头）。

    ``forward`` 始终返回相同键集合（skills 质量门禁）。
    """

    ARCH_NAME = "arch4_videomae_phys"

    def __init__(
        self,
        num_views: int = 4,
        in_channels: int = 6,
        num_frames: int = 30,
        img_size: int = 224,
        num_targets: int = 4,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 12,
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
    ):
        super().__init__()
        self.num_views = num_views
        self.in_channels = in_channels
        self.num_targets = num_targets
        self.dec_h = dec_h
        self.dec_w = dec_w
        self.use_aux_field_heads = use_aux_field_heads
        self.use_uncertainty = use_uncertainty

        self.encoder = VideoViTEncoder(
            in_channels,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            tubelet_size,
            patch_size,
            encoder_dropout,
            img_size,
            num_frames,
        )
        self.fusion = MultiViewFusion(
            num_views,
            encoder_embed_dim,
            fusion_dim,
            fusion_heads,
            use_attention_pool,
            head_dropout,
        )
        self.param_head = ParamRegressionHead(
            encoder_embed_dim, num_targets, head_dropout, use_uncertainty
        )
        self.num_frames = int(num_frames)
        self.stress_head = FieldHead(encoder_embed_dim, num_frames, dec_h, dec_w, out_channels=1) if use_aux_field_heads else None
        self.flow_head = FieldHead(encoder_embed_dim, num_frames, dec_h, dec_w, out_channels=1) if use_aux_field_heads else None
        self.force_head = FieldHead(encoder_embed_dim, num_frames, dec_h, dec_w, out_channels=1) if use_aux_field_heads else None

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb.dim() != 6:
            raise ValueError(f"rgb 期望 6 维 [B,V,C,T,H,W]，得到 {tuple(rgb.shape)}")
        b, v, c, t, h, w = rgb.shape
        assert v == self.num_views, f"V 应为 {self.num_views}，得到 {v}"
        assert c == self.in_channels, f"C 应为 {self.in_channels}，得到 {c}"

        x = rgb.reshape(b * v, c, t, h, w)
        feat_v = self.encoder(x).view(b, v, -1)

        if (
            self.use_aux_field_heads
            and self.stress_head is not None
            and self.flow_head is not None
            and self.force_head is not None
        ):
            stress_field_pred = self.stress_head(feat_v)
            flow_field_pred = self.flow_head(feat_v)
            force_pred = self.force_head(feat_v)
        else:
            z0 = feat_v.new_zeros(b, v, 1, self.num_frames, self.dec_h, self.dec_w)
            stress_field_pred = z0
            flow_field_pred = z0
            force_pred = z0

        z = self.fusion(feat_v)
        param_pred, logvar_raw = self.param_head(z)
        # 反变换到 raw 物理量，供日志/可视化使用
        param_pred_raw = torch.stack(
            [
                torch.expm1(torch.clamp(param_pred[:, 0], min=0.0)),
                torch.clamp(param_pred[:, 1], min=0.0, max=0.5),
                torch.expm1(torch.clamp(param_pred[:, 2], min=0.0)),
                torch.expm1(torch.clamp(param_pred[:, 3], min=0.0)),
            ],
            dim=1,
        )

        if self.use_uncertainty and logvar_raw is not None:
            logvar = logvar_raw
        else:
            logvar = param_pred.new_zeros(b, self.num_targets)

        return {
            "param_pred": param_pred,
            "param_pred_raw": param_pred_raw,
            "stress_field_pred": stress_field_pred,
            "flow_field_pred": flow_field_pred,
            "force_pred": force_pred,
            "logvar": logvar,
        }


def build_arch4_model(**kwargs) -> Arch4VideoMAEPhysModel:
    return Arch4VideoMAEPhysModel(**kwargs)
