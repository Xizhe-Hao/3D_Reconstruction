from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from my_model.arch4_model import MultiViewFusion, ParamRegressionHead
from logic_model.losses import mask_field_with_object_mask
from logic_model.model import (
    ActionClassificationHead,
    ConditionalConvFieldHead,
    DenseFieldFeatureEncoder,
    FlowSpatialEnhancer,
    MultiScaleSpatialConditionStem,
    PhysicsBottleneck,
    SharedConditionalFieldDecoder,
    SharedTaskPhysRefiner,
    StressFlowForceFuser,
    StressSpatialEnhancer,
    TaskAdaptiveFieldHead,
)


class DinoFrameEncoder(nn.Module):
    """
    2D 预训练骨干包装：
    - 输入: [N,C,H,W]
    - 输出: [N,D]
    优先使用 torch.hub 的 DINOv2；若失败可回退到 torchvision ViT。
    """

    def __init__(
        self,
        *,
        backbone_name: str = "dinov2_vits14",
        pretrained: bool = True,
        source: str = "torchhub",
        image_size: int = 224,
        out_dim: int = 384,
        torchhub_dir: str | None = None,
        torchhub_repo: str = "facebookresearch/dinov2:main",
        force_reload: bool = False,
        trust_repo: bool = True,
        skip_validation: bool = True,
        hub_verbose: bool = False,
        log_torchhub_dir: bool = False,
        field_patch_dim: Optional[int] = None,
    ):
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.pretrained = bool(pretrained)
        self.source = str(source).lower().strip()
        self.image_size = int(image_size)
        self.out_dim = int(out_dim)
        self.torchhub_dir = (str(torchhub_dir).strip() if torchhub_dir is not None else "")
        self.torchhub_repo = str(torchhub_repo).strip() or "facebookresearch/dinov2:main"
        self.force_reload = bool(force_reload)
        self.trust_repo = bool(trust_repo)
        self.skip_validation = bool(skip_validation)
        self.hub_verbose = bool(hub_verbose)
        self.log_torchhub_dir = bool(log_torchhub_dir)

        self.backbone: nn.Module
        self.backbone_dim: int
        self._build_backbone()
        self.proj = (
            nn.Identity()
            if int(self.backbone_dim) == int(self.out_dim)
            else nn.Linear(int(self.backbone_dim), int(self.out_dim))
        )
        fpd = field_patch_dim
        self.field_patch_dim = None if fpd is None else int(fpd)
        self.patch_proj = (
            None
            if self.field_patch_dim is None
            else (
                nn.Identity()
                if int(self.backbone_dim) == int(self.field_patch_dim)
                else nn.Linear(int(self.backbone_dim), int(self.field_patch_dim))
            )
        )

        # DINOv2 采用 ImageNet 风格归一化
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _build_backbone(self) -> None:
        name = self.backbone_name

        if self.source in ("torchhub", "hub"):
            hub_dir = self._resolve_torchhub_dir()
            hub_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(hub_dir.parent)
            torch.hub.set_dir(str(hub_dir))
            if self.log_torchhub_dir:
                print(f"[LogicPhysModel2] torchhub_dir={torch.hub.get_dir()}", flush=True)
                print(f"[LogicPhysModel2] torchhub_repo={self.torchhub_repo} backbone={name}", flush=True)
            bb = self._load_dinov2_from_hub(name)
            dim = int(getattr(bb, "embed_dim", getattr(bb, "num_features", self.out_dim)))
            self.backbone = bb
            self.backbone_dim = dim
            return

        if self.source in ("torchvision", "tv"):
            from torchvision.models import vit_b_16, vit_l_16

            if name in ("vit_b_16", "vitb16"):
                bb = vit_b_16(weights="DEFAULT" if self.pretrained else None)
            elif name in ("vit_l_16", "vitl16"):
                bb = vit_l_16(weights="DEFAULT" if self.pretrained else None)
            else:
                raise ValueError(f"unsupported torchvision backbone_name: {name}")
            dim = int(bb.hidden_dim)
            self.backbone = bb
            self.backbone_dim = dim
            return

        raise ValueError(f"unsupported backbone_source: {self.source}")

    def _resolve_torchhub_dir(self) -> Path:
        if self.torchhub_dir:
            p = Path(self.torchhub_dir).expanduser()
            if p.is_absolute():
                return p.resolve()
            return (Path.cwd() / p).resolve()
        env_home = str(os.environ.get("TORCH_HOME", "") or "").strip()
        if env_home:
            return (Path(env_home).expanduser().resolve() / "hub").resolve()
        # Phys/logic_model/model2.py -> Phys/.torch/hub
        return (Path(__file__).resolve().parent.parent / ".torch" / "hub").resolve()

    def _load_dinov2_from_hub(self, name: str) -> nn.Module:
        kwargs: Dict[str, object] = {
            "pretrained": bool(self.pretrained),
            "force_reload": bool(self.force_reload),
            "trust_repo": bool(self.trust_repo),
            "skip_validation": bool(self.skip_validation),
            "verbose": bool(self.hub_verbose),
        }
        drop_order = ["trust_repo", "skip_validation", "verbose"]
        for i in range(len(drop_order) + 1):
            try:
                return torch.hub.load(self.torchhub_repo, name, **kwargs)
            except TypeError:
                if i >= len(drop_order):
                    raise
                kwargs.pop(drop_order[i], None)
        raise RuntimeError("unreachable")

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone

        # DINOv2 / timm 常见接口
        if hasattr(bb, "forward_features"):
            feats = bb.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    return feats["x_norm_clstoken"]
                if "x_prenorm" in feats and feats["x_prenorm"].dim() == 3:
                    return feats["x_prenorm"][:, 0]
            if torch.is_tensor(feats):
                if feats.dim() == 2:
                    return feats
                if feats.dim() == 3:
                    return feats[:, 0]
            raise RuntimeError("unsupported forward_features output format")

        # torchvision ViT 接口
        if hasattr(bb, "_process_input") and hasattr(bb, "encoder"):
            y = bb._process_input(x)
            n = y.shape[0]
            cls = bb.class_token.expand(n, -1, -1)
            y = torch.cat([cls, y], dim=1)
            y = bb.encoder(y)
            return y[:, 0]

        y = bb(x)
        if y.dim() == 2:
            return y
        if y.dim() == 3:
            return y[:, 0]
        raise RuntimeError("unsupported backbone output shape")

    def _backbone_forward_tokens(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """返回 cls token 与 patch tokens（原始 backbone 维）；无法取得 patch 时 patch=None。"""
        bb = self.backbone
        cls_t: Optional[torch.Tensor] = None
        patch_t: Optional[torch.Tensor] = None

        if hasattr(bb, "forward_features"):
            feats = bb.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    cls_t = feats["x_norm_clstoken"]
                if "x_norm_patchtokens" in feats:
                    patch_t = feats["x_norm_patchtokens"]
                if cls_t is None and "x_prenorm" in feats and feats["x_prenorm"].dim() == 3:
                    pr = feats["x_prenorm"]
                    cls_t = pr[:, 0]
                    patch_t = pr[:, 1:] if patch_t is None else patch_t
                elif (
                    patch_t is None
                    and "x_prenorm" in feats
                    and feats["x_prenorm"].dim() == 3
                ):
                    patch_t = feats["x_prenorm"][:, 1:]
            elif torch.is_tensor(feats):
                if feats.dim() == 2:
                    cls_t = feats
                elif feats.dim() == 3:
                    cls_t = feats[:, 0]
                    patch_t = feats[:, 1:]
            if cls_t is None:
                raise RuntimeError("unsupported forward_features output: missing cls token")
            return {"cls": cls_t, "patch": patch_t}

        if hasattr(bb, "_process_input") and hasattr(bb, "encoder"):
            y = bb._process_input(x)
            n = y.shape[0]
            cls_exp = bb.class_token.expand(n, -1, -1)
            y = torch.cat([cls_exp, y], dim=1)
            y = bb.encoder(y)
            cls_t = y[:, 0]
            patch_t = y[:, 1:]
            return {"cls": cls_t, "patch": patch_t}

        y = bb(x)
        if y.dim() == 2:
            return {"cls": y, "patch": None}
        if y.dim() == 3:
            return {"cls": y[:, 0], "patch": y[:, 1:]}
        raise RuntimeError("unsupported backbone output shape")

    def forward(
        self, x: torch.Tensor, return_patch_tokens: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if x.dim() != 4:
            raise ValueError(f"expect [N,C,H,W], got {tuple(x.shape)}")
        if int(x.shape[1]) != 3:
            raise ValueError(f"DINO encoder expects C=3, got C={int(x.shape[1])}")
        x = (x - self.mean.to(x)) / self.std.to(x)
        if not return_patch_tokens:
            f = self._backbone_forward(x)
            return self.proj(f)
        tok = self._backbone_forward_tokens(x)
        cls_raw = tok["cls"]
        patch_raw = tok["patch"]
        cls_out = self.proj(cls_raw)
        if patch_raw is None:
            return cls_out, None
        if self.patch_proj is None:
            raise RuntimeError(
                "return_patch_tokens=True but DinoFrameEncoder was built without field_patch_dim / patch_proj"
            )
        patch_out = self.patch_proj(patch_raw)
        return cls_out, patch_out


class TemporalAdapter(nn.Module):
    """将逐帧特征 [B,V,T,D] 聚合到 [B,V,D]。"""

    def __init__(
        self,
        *,
        dim: int,
        num_frames: int,
        adapter_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 6,
        dropout: float = 0.1,
        frame_pool: str = "mean",
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_frames = int(num_frames)
        self.adapter_type = str(adapter_type).lower().strip()
        self.frame_pool = str(frame_pool).lower().strip()

        self.frame_pos = nn.Parameter(torch.zeros(1, self.num_frames, self.dim))
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        if self.frame_pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        if self.adapter_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=int(num_heads),
                dim_feedforward=int(4 * self.dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
            self.norm = nn.LayerNorm(self.dim)
        elif self.adapter_type == "mean":
            self.encoder = None
            self.norm = nn.LayerNorm(self.dim)
        else:
            raise ValueError(f"unknown temporal_adapter_type: {self.adapter_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expect [B,V,T,D], got {tuple(x.shape)}")
        b, v, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"D mismatch: expect {self.dim}, got {d}")
        n = b * v
        y = x.view(n, t, d)
        pos = self.frame_pos[:, :t, :]
        y = y + pos

        if self.adapter_type == "transformer":
            if self.frame_pool == "cls":
                if self.cls_token is None:
                    raise RuntimeError("frame_pool='cls' but cls_token is not initialized")
                cls = self.cls_token.expand(n, -1, -1)
                y = torch.cat([cls, y], dim=1)
                y = self.encoder(y)
                z = y[:, 0, :]
            else:
                y = self.encoder(y)
                z = y.mean(dim=1)
        else:
            z = y.mean(dim=1)

        z = self.norm(z)
        return z.view(b, v, d).contiguous()


class StrongParamTransformer(nn.Module):
    """少量跨模态 token 的轻量融合器，输出参数头上下文向量。"""

    def __init__(
        self,
        *,
        token_dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_token_types: int,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.out_dim = int(out_dim)
        self.num_token_types = int(max(1, int(num_token_types)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(4 * self.token_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.token_type_emb = nn.Parameter(torch.zeros(1, self.num_token_types, self.token_dim))
        nn.init.trunc_normal_(self.token_type_emb, std=0.02)
        self.input_norm = nn.LayerNorm(self.token_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.out_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"expect [B,N,D], got {tuple(tokens.shape)}")
        _, num_tokens, d = tokens.shape
        if int(d) != self.token_dim:
            raise ValueError(f"token dim mismatch: expect {self.token_dim}, got {d}")
        y = self.input_norm(tokens)
        if int(num_tokens) > int(self.num_token_types):
            raise ValueError(
                f"num_tokens exceeds configured token types: tokens={num_tokens} token_types={self.num_token_types}"
            )
        y = y + self.token_type_emb[:, :num_tokens, :]
        y = self.encoder(y)
        return self.output_proj(y.mean(dim=1))


class TemporalMultiScaleFieldParamEncoder(nn.Module):
    """Encode dense full-sequence stress/flow fields into a few parameter tokens."""

    def __init__(
        self,
        *,
        in_channels: int,
        token_dim: int,
        num_views: int,
        num_frames: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.num_views = int(num_views)
        self.num_frames = int(num_frames)
        c0 = 32
        c1 = 64
        self.spatial = nn.Sequential(
            nn.Conv2d(self.in_channels, c0, kernel_size=3, padding=1),
            nn.GroupNorm(8, c0),
            nn.GELU(),
            nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c1),
            nn.GELU(),
        )
        multiscale_dim = c1 * (1 + 2 * 2 + 4 * 4)
        self.frame_proj = nn.Sequential(
            nn.Linear(multiscale_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.token_dim),
        )
        max_tokens = max(1, self.num_views * self.num_frames)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, self.token_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(4 * self.token_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.global_proj = nn.LayerNorm(self.token_dim)
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.LayerNorm(self.token_dim),
        )
        self.salient_proj = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.LayerNorm(self.token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(x.shape)}")
        b, v, c, t, h, w = x.shape
        if int(c) != self.in_channels:
            raise ValueError(f"field channel mismatch: expect {self.in_channels}, got {c}")
        y = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, c, h, w)
        try:
            y = self.spatial(y)
        except RuntimeError as e:
            msg = str(e).lower()
            if y.is_cuda and ("cudnn" in msg or "unable to find a valid cudnn algorithm" in msg):
                with torch.backends.cudnn.flags(enabled=False):
                    y = self.spatial(y)
            else:
                raise
        pooled = [
            nn.functional.adaptive_avg_pool2d(y, 1).flatten(1),
            nn.functional.adaptive_avg_pool2d(y, 2).flatten(1),
            nn.functional.adaptive_avg_pool2d(y, 4).flatten(1),
        ]
        frame_tokens = self.frame_proj(torch.cat(pooled, dim=1)).view(b, v * t, self.token_dim)
        seq_len = int(frame_tokens.shape[1])
        if seq_len > int(self.pos.shape[1]):
            raise ValueError(f"field token length {seq_len} exceeds configured max {int(self.pos.shape[1])}")
        seq = self.temporal(frame_tokens + self.pos[:, :seq_len, :])
        global_token = self.global_proj(seq.mean(dim=1))
        seq_bvtd = seq.view(b, v, t, self.token_dim)
        split = max(1, t // 4)
        early = seq_bvtd[:, :, :split, :].mean(dim=(1, 2))
        late = seq_bvtd[:, :, -split:, :].mean(dim=(1, 2))
        delta_token = self.delta_proj(late - early)
        salient_token = self.salient_proj(seq.amax(dim=1))
        return torch.stack([global_token, delta_token, salient_token], dim=1)


class LogicPhysModel2(nn.Module):
    """
    DINOv2 版本逻辑模型，保持与 LogicPhysModel 相同核心接口:
    - 输入: [B,V,C,T,H,W]
    - 返回键: param_pred/param_pred_raw/stress_field_pred/flow_field_pred/force_pred/logvar/action_logits
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
        fusion_dim: int = 512,
        fusion_heads: int = 8,
        use_attention_pool: bool = True,
        use_uncertainty: bool = False,
        dec_h: int = 56,
        dec_w: int = 56,
        use_aux_field_heads: bool = True,
        head_dropout: float = 0.1,
        bottleneck_dim: int = 128,
        dino_backbone_name: str = "dinov2_vits14",
        dino_backbone_pretrained: bool = True,
        dino_backbone_source: str = "torchhub",
        dino_out_dim: int = 384,
        temporal_adapter_type: str = "transformer",
        temporal_adapter_layers: int = 2,
        temporal_adapter_heads: int = 6,
        temporal_adapter_dropout: float = 0.1,
        frame_pool: str = "mean",
        freeze_backbone: bool = True,
        torchhub_dir: str | None = None,
        dino_torchhub_repo: str = "facebookresearch/dinov2:main",
        dino_force_reload: bool = False,
        dino_trust_repo: bool = True,
        dino_skip_validation: bool = True,
        dino_hub_verbose: bool = False,
        dino_log_torchhub_dir: bool = False,
        field_head_mode: str = "independent",
        field_token_dim: int = 512,
        field_base_channels: int = 128,
        field_shared_channels: int = 64,
        field_temporal_layers: int = 0,
        field_spatial_channels: int = 0,
        field_use_multiscale_spatial: bool = False,
        field_use_shared_task_phys: bool = False,
        field_use_geometry_residual: bool = False,
        field_sequential_stress: bool = False,
        force_out_channels: int = 3,
        use_stress_spatial_enhancer: bool = False,
        use_flow_spatial_enhancer: Optional[bool] = None,
        flow_output_scale_init: float = 0.5,
        flow_output_bias_init: float = -0.5,
        field_use_patch_tokens: bool = False,
        field_patch_dim: int = 256,
        param_chain_mode: str = "baseline",
        param_token_dim: int = 256,
        param_mixer_layers: int = 2,
        param_mixer_heads: int = 8,
        param_mixer_dropout: Optional[float] = None,
        param_use_rgb_static: bool = True,
        param_use_rgb_residual: bool = True,
        param_use_masked_field_tokens: bool = True,
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.in_channels = int(in_channels)
        self.num_targets = int(num_targets)
        self.num_actions = int(num_actions)
        self.num_frames = int(num_frames)
        self.dec_h = int(dec_h)
        self.dec_w = int(dec_w)
        self.use_aux_field_heads = bool(use_aux_field_heads)
        self.use_uncertainty = bool(use_uncertainty)
        self.freeze_backbone = bool(freeze_backbone)
        self.field_head_mode = str(field_head_mode).strip().lower()
        if self.field_head_mode not in ("independent", "shared_temporal"):
            raise ValueError(f"unknown field_head_mode: {field_head_mode}")
        self.field_use_multiscale_spatial = bool(field_use_multiscale_spatial)
        self.field_use_shared_task_phys = bool(field_use_shared_task_phys)
        self.field_use_geometry_residual = bool(field_use_geometry_residual)
        self.field_sequential_stress = bool(field_sequential_stress)
        self.force_out_channels = int(max(1, int(force_out_channels)))
        self.use_stress_spatial_enhancer = bool(use_stress_spatial_enhancer)
        self.use_flow_spatial_enhancer = bool(
            use_stress_spatial_enhancer if use_flow_spatial_enhancer is None else use_flow_spatial_enhancer
        )
        self.flow_output_scale_init = float(flow_output_scale_init)
        self.flow_output_bias_init = float(flow_output_bias_init)
        self.field_use_patch_tokens = bool(field_use_patch_tokens)
        self.field_patch_dim = int(max(1, int(field_patch_dim)))
        self.param_chain_mode = str(param_chain_mode).strip().lower()
        if self.param_chain_mode not in ("baseline", "add_shared_field", "strong_param_v1", "strong_param_v2"):
            raise ValueError(f"unknown param_chain_mode: {param_chain_mode}")
        self.param_token_dim = int(max(32, int(param_token_dim)))
        self.param_mixer_layers = int(max(1, int(param_mixer_layers)))
        self.param_mixer_heads = int(max(1, int(param_mixer_heads)))
        self.param_mixer_dropout = float(
            head_dropout if param_mixer_dropout is None else param_mixer_dropout
        )
        self.param_use_rgb_static = bool(param_use_rgb_static)
        self.param_use_rgb_residual = bool(param_use_rgb_residual)
        self.param_use_masked_field_tokens = bool(param_use_masked_field_tokens)
        self.img_size = int(img_size)

        enc_dim = int(dino_out_dim)
        self.frame_encoder = DinoFrameEncoder(
            backbone_name=dino_backbone_name,
            pretrained=bool(dino_backbone_pretrained),
            source=dino_backbone_source,
            image_size=int(img_size),
            out_dim=enc_dim,
            torchhub_dir=torchhub_dir,
            torchhub_repo=dino_torchhub_repo,
            force_reload=bool(dino_force_reload),
            trust_repo=bool(dino_trust_repo),
            skip_validation=bool(dino_skip_validation),
            hub_verbose=bool(dino_hub_verbose),
            log_torchhub_dir=bool(dino_log_torchhub_dir),
            field_patch_dim=(self.field_patch_dim if self.field_use_patch_tokens else None),
        )
        self.field_patch_norm = (
            nn.LayerNorm(self.field_patch_dim) if self.field_use_patch_tokens else None
        )
        self.temporal_adapter = TemporalAdapter(
            dim=enc_dim,
            num_frames=self.num_frames,
            adapter_type=temporal_adapter_type,
            num_layers=int(temporal_adapter_layers),
            num_heads=int(temporal_adapter_heads),
            dropout=float(temporal_adapter_dropout),
            frame_pool=frame_pool,
        )
        self.frame_feat_norm = nn.LayerNorm(enc_dim)
        self.view_feat_norm = nn.LayerNorm(enc_dim)
        self.fused_feat_norm = nn.LayerNorm(enc_dim)
        self.field_frame_norm = nn.LayerNorm(enc_dim)

        if self.freeze_backbone:
            for p in self.frame_encoder.parameters():
                p.requires_grad = False
            self.frame_encoder.eval()

        self.fusion = MultiViewFusion(
            self.num_views,
            enc_dim,
            int(fusion_dim),
            int(fusion_heads),
            bool(use_attention_pool),
            float(head_dropout),
        )

        self.physics_bottleneck = PhysicsBottleneck(
            in_dim=enc_dim,
            latent_dim=int(bottleneck_dim),
            dropout=float(head_dropout),
        )
        bottleneck_out_dim = int(bottleneck_dim) * 3

        self.param_head = ParamRegressionHead(
            bottleneck_out_dim,
            self.num_targets,
            float(head_dropout),
            self.use_uncertainty,
        )
        self.stress_field_encoder = DenseFieldFeatureEncoder(
            in_channels=3,
            out_dim=int(bottleneck_dim),
            dropout=float(head_dropout),
        )
        self.flow_field_encoder = DenseFieldFeatureEncoder(
            in_channels=3,
            out_dim=int(bottleneck_dim),
            dropout=float(head_dropout),
        )
        self.force_field_encoder = DenseFieldFeatureEncoder(
            in_channels=int(self.force_out_channels),
            out_dim=int(bottleneck_dim),
            dropout=float(head_dropout),
        )
        self.shared_field_param_encoder: Optional[DenseFieldFeatureEncoder] = None
        param_field_feat_dim = int(bottleneck_dim) * 3
        if self.param_chain_mode == "add_shared_field" and self.field_head_mode == "shared_temporal":
            self.shared_field_param_encoder = DenseFieldFeatureEncoder(
                in_channels=int(field_shared_channels),
                out_dim=int(bottleneck_dim),
                dropout=float(head_dropout),
            )
            param_field_feat_dim += int(bottleneck_dim)
        self.param_field_fusion = nn.Sequential(
            nn.Linear(bottleneck_out_dim + param_field_feat_dim, bottleneck_out_dim),
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
            nn.LayerNorm(bottleneck_out_dim),
        )
        self.phys_token_proj: Optional[nn.Module] = None
        self.rgb_static_proj: Optional[nn.Module] = None
        self.rgb_residual_phys_proj: Optional[nn.Module] = None
        self.rgb_residual_proj: Optional[nn.Module] = None
        self.field_token_proj: Optional[nn.Module] = None
        self.stress_temporal_param_encoder: Optional[TemporalMultiScaleFieldParamEncoder] = None
        self.flow_temporal_param_encoder: Optional[TemporalMultiScaleFieldParamEncoder] = None
        self.strong_param_transformer: Optional[StrongParamTransformer] = None
        if self.param_chain_mode in ("strong_param_v1", "strong_param_v2"):
            num_param_tokens = 1
            if self.param_use_rgb_static:
                num_param_tokens += 1
            if self.param_use_rgb_residual:
                num_param_tokens += 1
            if self.param_use_masked_field_tokens:
                num_param_tokens += 7 if self.param_chain_mode == "strong_param_v2" else 3
            self.phys_token_proj = nn.Sequential(
                nn.LayerNorm(bottleneck_out_dim),
                nn.Linear(bottleneck_out_dim, self.param_token_dim),
                nn.GELU(),
                nn.LayerNorm(self.param_token_dim),
            )
            self.rgb_static_proj = nn.Sequential(
                nn.LayerNorm(enc_dim),
                nn.Linear(enc_dim, self.param_token_dim),
                nn.GELU(),
                nn.LayerNorm(self.param_token_dim),
            )
            self.rgb_residual_phys_proj = nn.Linear(bottleneck_out_dim, enc_dim)
            self.rgb_residual_proj = nn.Sequential(
                nn.LayerNorm(enc_dim),
                nn.Linear(enc_dim, self.param_token_dim),
                nn.GELU(),
                nn.LayerNorm(self.param_token_dim),
            )
            self.field_token_proj = nn.Sequential(
                nn.LayerNorm(int(bottleneck_dim)),
                nn.Linear(int(bottleneck_dim), self.param_token_dim),
                nn.GELU(),
                nn.LayerNorm(self.param_token_dim),
            )
            if self.param_chain_mode == "strong_param_v2":
                self.stress_temporal_param_encoder = TemporalMultiScaleFieldParamEncoder(
                    in_channels=3,
                    token_dim=self.param_token_dim,
                    num_views=self.num_views,
                    num_frames=self.num_frames,
                    num_heads=self.param_mixer_heads,
                    dropout=self.param_mixer_dropout,
                )
                self.flow_temporal_param_encoder = TemporalMultiScaleFieldParamEncoder(
                    in_channels=3,
                    token_dim=self.param_token_dim,
                    num_views=self.num_views,
                    num_frames=self.num_frames,
                    num_heads=self.param_mixer_heads,
                    dropout=self.param_mixer_dropout,
                )
            self.strong_param_transformer = StrongParamTransformer(
                token_dim=self.param_token_dim,
                out_dim=bottleneck_out_dim,
                num_layers=self.param_mixer_layers,
                num_heads=self.param_mixer_heads,
                dropout=self.param_mixer_dropout,
                num_token_types=num_param_tokens,
            )
        self.action_head = ActionClassificationHead(
            bottleneck_out_dim,
            int(num_actions),
            dropout=float(head_dropout),
        )

        self.view_id_emb = nn.Embedding(self.num_views, enc_dim)
        self.view_id_scale = nn.Parameter(torch.tensor(1.0))
        self.field_token_dim = int(max(32, int(field_token_dim)))
        self.field_base_channels = int(field_base_channels)
        self.field_shared_channels = int(field_shared_channels)
        self.field_temporal_layers = int(field_temporal_layers)
        self.field_spatial_channels = int(field_spatial_channels)

        if self.field_head_mode == "shared_temporal":
            shared_phys_dim = int(bottleneck_dim)
            dec_geometry_dim = int(enc_dim) if self.field_use_geometry_residual else 0
            self.geom_latent_mlp = (
                nn.Sequential(
                    nn.LayerNorm(enc_dim),
                    nn.Linear(enc_dim, enc_dim),
                    nn.GELU(),
                    nn.LayerNorm(enc_dim),
                )
                if self.field_use_geometry_residual
                else None
            )
            self.field_shared_phys_proj = nn.Sequential(
                nn.Linear(bottleneck_out_dim, shared_phys_dim),
                nn.GELU(),
                nn.LayerNorm(shared_phys_dim),
            )
            self.field_task_phys_refiner = (
                SharedTaskPhysRefiner(
                    in_dim=bottleneck_out_dim,
                    latent_dim=shared_phys_dim,
                    dropout=float(head_dropout),
                )
                if self.field_use_shared_task_phys
                else None
            )
            self.field_spatial_stem = (
                MultiScaleSpatialConditionStem(
                    in_channels=3,
                    out_channels=max(16, self.field_spatial_channels),
                    out_h=max(1, (self.dec_h + 7) // 8),
                    out_w=max(1, (self.dec_w + 7) // 8),
                )
                if self.field_use_multiscale_spatial and self.field_spatial_channels > 0
                else None
            )
            self.shared_field_decoder = SharedConditionalFieldDecoder(
                global_dim=enc_dim,
                frame_dim=enc_dim,
                shared_phys_dim=shared_phys_dim,
                num_frames=self.num_frames,
                dec_h=self.dec_h,
                dec_w=self.dec_w,
                token_dim=min(int(self.field_token_dim), enc_dim),
                base_channels=self.field_base_channels,
                shared_channels=self.field_shared_channels,
                spatial_cond_dim=(max(16, self.field_spatial_channels) if self.field_spatial_stem is not None else 0),
                temporal_layers=self.field_temporal_layers,
                dropout=float(head_dropout),
                geometry_dim=dec_geometry_dim,
                patch_cond_dim=(self.field_patch_dim if self.field_use_patch_tokens else 0),
            )
            self.stress_ff_fuser = (
                StressFlowForceFuser(
                    shared_dim=self.field_shared_channels,
                    flow_channels=3,
                    force_channels=int(self.force_out_channels),
                    dropout=float(head_dropout),
                )
                if self.field_sequential_stress
                else None
            )
            self.stress_spatial_enhancer = (
                StressSpatialEnhancer(channels=self.field_shared_channels, dropout=float(head_dropout))
                if self.use_stress_spatial_enhancer
                else None
            )
            self.flow_spatial_enhancer = (
                FlowSpatialEnhancer(channels=self.field_shared_channels, dropout=float(head_dropout))
                if self.use_flow_spatial_enhancer
                else None
            )
            self.stress_head = TaskAdaptiveFieldHead(
                shared_dim=self.field_shared_channels,
                task_phys_dim=shared_phys_dim,
                out_channels=3,
                dropout=float(head_dropout),
                output_activation="sigmoid",
            )
            self.flow_head = TaskAdaptiveFieldHead(
                shared_dim=self.field_shared_channels,
                task_phys_dim=shared_phys_dim,
                out_channels=3,
                dropout=float(head_dropout),
                output_activation="sigmoid",
                output_scale_init=self.flow_output_scale_init,
                output_bias_init=self.flow_output_bias_init,
            )
            self.force_head = TaskAdaptiveFieldHead(
                shared_dim=self.field_shared_channels,
                task_phys_dim=shared_phys_dim,
                out_channels=int(self.force_out_channels),
                dropout=float(head_dropout),
                output_activation="identity",
            )
        else:
            self.geom_latent_mlp = None
            self.field_shared_phys_proj = None
            self.field_task_phys_refiner = None
            self.field_spatial_stem = None
            self.shared_field_decoder = None
            self.stress_ff_fuser = None
            self.stress_spatial_enhancer = None
            self.flow_spatial_enhancer = None
            self.stress_head = ConditionalConvFieldHead(
                global_dim=enc_dim,
                frame_dim=enc_dim,
                phys_dim=int(bottleneck_dim),
                num_frames=self.num_frames,
                dec_h=self.dec_h,
                dec_w=self.dec_w,
                out_channels=3,
                token_dim=min(512, enc_dim),
                base_channels=128,
                dropout=float(head_dropout),
                output_activation="sigmoid",
            )
            self.flow_head = ConditionalConvFieldHead(
                global_dim=enc_dim,
                frame_dim=enc_dim,
                phys_dim=int(bottleneck_dim),
                num_frames=self.num_frames,
                dec_h=self.dec_h,
                dec_w=self.dec_w,
                out_channels=3,
                token_dim=min(512, enc_dim),
                base_channels=128,
                dropout=float(head_dropout),
                output_activation="sigmoid",
                output_scale_init=self.flow_output_scale_init,
                output_bias_init=self.flow_output_bias_init,
            )
            self.force_head = ConditionalConvFieldHead(
                global_dim=enc_dim,
                frame_dim=enc_dim,
                phys_dim=int(bottleneck_dim),
                num_frames=self.num_frames,
                dec_h=self.dec_h,
                dec_w=self.dec_w,
                out_channels=int(self.force_out_channels),
                token_dim=min(512, enc_dim),
                base_channels=128,
                dropout=float(head_dropout),
                output_activation="identity",
            )
        self.training_stage: str = "joint"

    def train(self, mode: bool = True) -> "LogicPhysModel2":
        super().train(mode)
        if self.freeze_backbone:
            # 冻结 backbone 时始终保持 eval，避免被外部 model.train() 重新切回训练态。
            self.frame_encoder.eval()
        if mode and self.training_stage == "param":
            # 第二阶段仅训练参数预测相关模块，前端保持 eval 降低抖动。
            self.frame_encoder.eval()
            self.temporal_adapter.eval()
            self.fusion.eval()
            self.physics_bottleneck.eval()
            if self.field_shared_phys_proj is not None:
                self.field_shared_phys_proj.eval()
            if self.field_task_phys_refiner is not None:
                self.field_task_phys_refiner.eval()
            if self.field_spatial_stem is not None:
                self.field_spatial_stem.eval()
            if self.shared_field_decoder is not None:
                self.shared_field_decoder.eval()
            if self.geom_latent_mlp is not None:
                self.geom_latent_mlp.eval()
            if self.stress_ff_fuser is not None:
                self.stress_ff_fuser.eval()
            if self.stress_spatial_enhancer is not None:
                self.stress_spatial_enhancer.eval()
            if self.flow_spatial_enhancer is not None:
                self.flow_spatial_enhancer.eval()
            self.stress_head.eval()
            self.flow_head.eval()
            self.force_head.eval()
        if mode and self.training_stage == "stress":
            # stress 阶段：冻结主干与 flow/force，但允许 stress 相关共享场解码链一起适配。
            self.frame_encoder.eval()
            self.temporal_adapter.eval()
            self.fusion.eval()
            self.physics_bottleneck.eval()
            if self.field_shared_phys_proj is not None:
                self.field_shared_phys_proj.train()
            if self.field_task_phys_refiner is not None:
                self.field_task_phys_refiner.train()
            if self.field_spatial_stem is not None:
                self.field_spatial_stem.eval()
            if self.shared_field_decoder is not None:
                self.shared_field_decoder.train()
            if self.geom_latent_mlp is not None:
                self.geom_latent_mlp.eval()
            self.flow_head.eval()
            self.force_head.eval()
            if self.flow_spatial_enhancer is not None:
                self.flow_spatial_enhancer.eval()
            if self.stress_ff_fuser is not None:
                self.stress_ff_fuser.train()
            if self.stress_spatial_enhancer is not None:
                self.stress_spatial_enhancer.train()
            self.stress_head.train()
        return self

    @staticmethod
    def _set_module_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = bool(flag)

    def _set_optional_module_requires_grad(self, module: Optional[nn.Module], flag: bool) -> None:
        if module is not None:
            self._set_module_requires_grad(module, flag)

    def _set_param_branch_requires_grad(self, flag: bool) -> None:
        self._set_module_requires_grad(self.param_head, flag)
        self._set_module_requires_grad(self.param_field_fusion, flag)
        self._set_module_requires_grad(self.stress_field_encoder, flag)
        self._set_module_requires_grad(self.flow_field_encoder, flag)
        self._set_module_requires_grad(self.force_field_encoder, flag)
        self._set_optional_module_requires_grad(self.shared_field_param_encoder, flag)
        self._set_optional_module_requires_grad(self.phys_token_proj, flag)
        self._set_optional_module_requires_grad(self.rgb_static_proj, flag)
        self._set_optional_module_requires_grad(self.rgb_residual_phys_proj, flag)
        self._set_optional_module_requires_grad(self.rgb_residual_proj, flag)
        self._set_optional_module_requires_grad(self.field_token_proj, flag)
        self._set_optional_module_requires_grad(self.stress_temporal_param_encoder, flag)
        self._set_optional_module_requires_grad(self.flow_temporal_param_encoder, flag)
        self._set_optional_module_requires_grad(self.strong_param_transformer, flag)

    def set_training_stage(self, stage: str) -> None:
        s = str(stage).strip().lower()
        if s not in ("joint", "field", "param", "flow_force", "stress"):
            raise ValueError(f"unknown training stage: {stage}")
        self.training_stage = s
        if s == "joint":
            for p in self.parameters():
                p.requires_grad = True
            if self.freeze_backbone:
                for p in self.frame_encoder.parameters():
                    p.requires_grad = False
            return
        if s == "field":
            # 第一阶段：优先训练场预测链路。
            for p in self.parameters():
                p.requires_grad = True
            self._set_param_branch_requires_grad(False)
            self._set_module_requires_grad(self.action_head, False)
            if self.field_shared_phys_proj is not None:
                self._set_module_requires_grad(self.field_shared_phys_proj, True)
            if self.field_task_phys_refiner is not None:
                self._set_module_requires_grad(self.field_task_phys_refiner, True)
            if self.field_spatial_stem is not None:
                self._set_module_requires_grad(self.field_spatial_stem, True)
            if self.shared_field_decoder is not None:
                self._set_module_requires_grad(self.shared_field_decoder, True)
            if self.geom_latent_mlp is not None:
                self._set_module_requires_grad(self.geom_latent_mlp, True)
            if self.stress_ff_fuser is not None:
                self._set_module_requires_grad(self.stress_ff_fuser, True)
            if self.stress_spatial_enhancer is not None:
                self._set_module_requires_grad(self.stress_spatial_enhancer, True)
            if self.flow_spatial_enhancer is not None:
                self._set_module_requires_grad(self.flow_spatial_enhancer, True)
            if self.freeze_backbone:
                self._set_module_requires_grad(self.frame_encoder, False)
            return
        if s == "flow_force":
            # 三阶段之 1：只训练 flow + force（可选关闭 stress 分支参数更新）。
            for p in self.parameters():
                p.requires_grad = True
            self._set_param_branch_requires_grad(False)
            self._set_module_requires_grad(self.action_head, False)
            self._set_module_requires_grad(self.stress_head, False)
            if self.stress_ff_fuser is not None:
                self._set_module_requires_grad(self.stress_ff_fuser, False)
            if self.stress_spatial_enhancer is not None:
                self._set_module_requires_grad(self.stress_spatial_enhancer, False)
            if self.flow_spatial_enhancer is not None:
                self._set_module_requires_grad(self.flow_spatial_enhancer, True)
            if self.field_shared_phys_proj is not None:
                self._set_module_requires_grad(self.field_shared_phys_proj, True)
            if self.field_task_phys_refiner is not None:
                self._set_module_requires_grad(self.field_task_phys_refiner, True)
            if self.field_spatial_stem is not None:
                self._set_module_requires_grad(self.field_spatial_stem, True)
            if self.shared_field_decoder is not None:
                self._set_module_requires_grad(self.shared_field_decoder, True)
            if self.geom_latent_mlp is not None:
                self._set_module_requires_grad(self.geom_latent_mlp, True)
            if self.freeze_backbone:
                self._set_module_requires_grad(self.frame_encoder, False)
            return
        if s == "stress":
            # 三阶段之 2：冻结主干与 flow/force 头；允许 stress 相关共享场解码链一起适配。
            for p in self.parameters():
                p.requires_grad = False
            self._set_module_requires_grad(self.frame_encoder, False)
            self._set_module_requires_grad(self.temporal_adapter, False)
            self._set_module_requires_grad(self.fusion, False)
            self._set_module_requires_grad(self.physics_bottleneck, False)
            for p in self.view_id_emb.parameters():
                p.requires_grad = False
            self.view_id_scale.requires_grad = False
            self._set_module_requires_grad(self.frame_feat_norm, False)
            self._set_module_requires_grad(self.view_feat_norm, False)
            self._set_module_requires_grad(self.fused_feat_norm, False)
            self._set_module_requires_grad(self.field_frame_norm, False)
            if self.field_shared_phys_proj is not None:
                self._set_module_requires_grad(self.field_shared_phys_proj, True)
            if self.field_task_phys_refiner is not None:
                self._set_module_requires_grad(self.field_task_phys_refiner, True)
            if self.field_spatial_stem is not None:
                self._set_module_requires_grad(self.field_spatial_stem, False)
            if self.shared_field_decoder is not None:
                self._set_module_requires_grad(self.shared_field_decoder, True)
            if self.geom_latent_mlp is not None:
                self._set_module_requires_grad(self.geom_latent_mlp, False)
            if self.stress_ff_fuser is not None:
                self._set_module_requires_grad(self.stress_ff_fuser, True)
            if self.stress_spatial_enhancer is not None:
                self._set_module_requires_grad(self.stress_spatial_enhancer, True)
            if self.flow_spatial_enhancer is not None:
                self._set_module_requires_grad(self.flow_spatial_enhancer, False)
            self._set_module_requires_grad(self.stress_head, True)
            self._set_module_requires_grad(self.flow_head, False)
            self._set_module_requires_grad(self.force_head, False)
            self._set_param_branch_requires_grad(False)
            self._set_module_requires_grad(self.action_head, False)
            return
        # s == "param": 冻结场链路，仅训练参数预测相关模块。
        self._set_module_requires_grad(self.frame_encoder, False)
        self._set_module_requires_grad(self.temporal_adapter, False)
        self._set_module_requires_grad(self.fusion, False)
        self._set_module_requires_grad(self.physics_bottleneck, False)
        self._set_module_requires_grad(self.frame_feat_norm, False)
        self._set_module_requires_grad(self.view_feat_norm, False)
        self._set_module_requires_grad(self.fused_feat_norm, False)
        self._set_module_requires_grad(self.field_frame_norm, False)
        if self.field_patch_norm is not None:
            self._set_module_requires_grad(self.field_patch_norm, False)
        if self.field_shared_phys_proj is not None:
            self._set_module_requires_grad(self.field_shared_phys_proj, False)
        if self.field_task_phys_refiner is not None:
            self._set_module_requires_grad(self.field_task_phys_refiner, False)
        if self.field_spatial_stem is not None:
            self._set_module_requires_grad(self.field_spatial_stem, False)
        if self.shared_field_decoder is not None:
            self._set_module_requires_grad(self.shared_field_decoder, False)
        if self.geom_latent_mlp is not None:
            self._set_module_requires_grad(self.geom_latent_mlp, False)
        if self.stress_ff_fuser is not None:
            self._set_module_requires_grad(self.stress_ff_fuser, False)
        if self.stress_spatial_enhancer is not None:
            self._set_module_requires_grad(self.stress_spatial_enhancer, False)
        if self.flow_spatial_enhancer is not None:
            self._set_module_requires_grad(self.flow_spatial_enhancer, False)
        for p in self.view_id_emb.parameters():
            p.requires_grad = False
        self.view_id_scale.requires_grad = False
        self._set_module_requires_grad(self.stress_head, False)
        self._set_module_requires_grad(self.flow_head, False)
        self._set_module_requires_grad(self.force_head, False)
        self._set_module_requires_grad(self.action_head, False)
        self._set_param_branch_requires_grad(True)

    def _to_three_channels(self, x: torch.Tensor) -> torch.Tensor:
        # DINO 系列骨干默认输入 3 通道
        c = int(x.shape[1])
        if c == 3:
            return x
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        if c > 3:
            return x[:, :3, :, :]
        raise ValueError(f"unsupported input channels for DINO backbone: C={c}")

    @staticmethod
    def _infer_patch_grid(num_patches: int, ih: int, iw: int, backbone: nn.Module) -> Tuple[int, int]:
        ps = getattr(getattr(backbone, "patch_embed", None), "patch_size", None)
        if ps is not None:
            if isinstance(ps, (tuple, list)) and len(ps) >= 2:
                ph, pw = int(ps[0]), int(ps[1])
            else:
                ph = pw = int(ps)
            hp, wp = int(ih) // ph, int(iw) // pw
            if hp * wp == int(num_patches):
                return hp, wp
        s = int(math.isqrt(int(num_patches)))
        if s * s == int(num_patches):
            return s, s
        raise ValueError(
            f"无法推断 DINO patch 空间网格: num_patches={num_patches}, img={ih}x{iw}, patch_embed.patch_size={ps}"
        )

    @staticmethod
    def _patch_tokens_to_map(
        patch_nv_pd: torch.Tensor,
        ih: int,
        iw: int,
        backbone: nn.Module,
    ) -> torch.Tensor:
        """[N,P,D] -> [N,D,Hp,Wp]"""
        n, p, d = patch_nv_pd.shape
        hp, wp = LogicPhysModel2._infer_patch_grid(int(p), int(ih), int(iw), backbone)
        if hp * wp != int(p):
            raise ValueError(f"patch token 数量与网格不一致: P={p}, grid={hp}x{wp}")
        return patch_nv_pd.transpose(1, 2).contiguous().view(n, int(d), hp, wp)

    @staticmethod
    def _mix_param_field_with_gt(
        pred: torch.Tensor,
        gt: Optional[torch.Tensor],
        use_gt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if gt is None or use_gt is None:
            return pred
        if gt.dim() != pred.dim() or gt.shape[:3] != pred.shape[:3]:
            raise ValueError(f"GT field shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
        if gt.shape[3:] != pred.shape[3:]:
            b, v, c, _, _, _ = gt.shape
            gt_bv = gt.to(device=pred.device, dtype=pred.dtype).reshape(b * v, c, gt.shape[3], gt.shape[4], gt.shape[5])
            gt_bv = nn.functional.interpolate(
                gt_bv,
                size=(int(pred.shape[3]), int(pred.shape[4]), int(pred.shape[5])),
                mode="trilinear",
                align_corners=False,
            )
            gt = gt_bv.view(b, v, c, int(pred.shape[3]), int(pred.shape[4]), int(pred.shape[5]))
        mask = use_gt.to(device=pred.device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.view(-1, 1, 1, 1, 1, 1)
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
        return torch.where(mask, gt.to(device=pred.device, dtype=pred.dtype), pred)

    def forward(
        self,
        rgb: torch.Tensor,
        stage: str | None = None,
        object_mask: torch.Tensor | None = None,
        param_stress_gt: torch.Tensor | None = None,
        param_flow_gt: torch.Tensor | None = None,
        param_stress_use_gt: torch.Tensor | None = None,
        param_flow_use_gt: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if rgb.dim() != 6:
            raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(rgb.shape)}")
        b, v, c, t, h, w = rgb.shape
        if v != self.num_views:
            raise ValueError(f"V mismatch: expect {self.num_views}, got {v}")
        if c != self.in_channels:
            raise ValueError(f"C mismatch: expect {self.in_channels}, got {c}")

        # [B,V,C,T,H,W] -> [B*V*T,C,H,W]
        x = rgb.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * v * t, c, h, w)
        x = self._to_three_channels(x)
        spatial_rgb = x.float()

        patch_cond: Optional[torch.Tensor] = None
        if self.field_use_patch_tokens:
            if self.freeze_backbone:
                with torch.no_grad():
                    enc_pair = self.frame_encoder(x, return_patch_tokens=True)
            else:
                enc_pair = self.frame_encoder(x, return_patch_tokens=True)
            frame_vec, patch_tok = enc_pair[0], enc_pair[1]
            frame_feat = frame_vec
            if patch_tok is not None and self.field_patch_norm is not None:
                # pm: [B*V*T, D, Hp, Wp] -> [B,V,T,D,Hp,Wp]；LayerNorm 在通道 D 上，须保持 (T,D) 顺序，不可与时间维混淆。
                pm = self._patch_tokens_to_map(patch_tok, int(h), int(w), self.frame_encoder.backbone)
                pm = pm.view(b, v, t, self.field_patch_dim, pm.shape[-2], pm.shape[-1])
                pm_hw_d = pm.permute(0, 1, 2, 4, 5, 3)  # [B,V,T,Hp,Wp,D]
                pm_ln = self.field_patch_norm(pm_hw_d)
                patch_cond = pm_ln.permute(0, 1, 2, 5, 3, 4).contiguous()  # [B,V,T,D,Hp,Wp]
        else:
            if self.freeze_backbone:
                with torch.no_grad():
                    frame_feat = self.frame_encoder(x)
            else:
                frame_feat = self.frame_encoder(x)

        active_stage = str(stage or self.training_stage).strip().lower()
        if active_stage not in ("joint", "field", "param", "flow_force", "stress"):
            active_stage = "joint"
        run_field_heads = bool(self.use_aux_field_heads)
        run_param_chain = active_stage in ("joint", "param")
        run_action_head = active_stage == "joint"
        amp_off = (
            torch.amp.autocast(device_type=rgb.device.type, enabled=False)
            if rgb.device.type in ("cuda", "cpu")
            else nullcontext()
        )
        with amp_off:
            shared_field_param_feat: Optional[torch.Tensor] = None
            # vit-g/14 的高维后处理链路更容易在 fp16 中间激活溢出，
            # 因此将 backbone 之后的 field/main branch 固定为 float32。
            frame_feat = self.frame_feat_norm(frame_feat.float().view(b, v, t, -1))
            feat_v = self.view_feat_norm(self.temporal_adapter(frame_feat))  # [B,V,D]

            z_fused = self.fused_feat_norm(self.fusion(feat_v))  # [B,D]
            pb = self.physics_bottleneck(z_fused)
            z_phys = torch.cat(
                [pb["contact_latent"], pb["deformation_latent"], pb["stress_latent"]],
                dim=1,
            )

            view_ids = torch.arange(v, device=feat_v.device, dtype=torch.long)
            view_bias_id = self.view_id_emb(view_ids).view(1, v, 1, -1).expand(b, -1, t, -1)
            frame_feat_cond = self.field_frame_norm(frame_feat + self.view_id_scale.float() * view_bias_id.float())

            if run_field_heads:
                if self.field_head_mode == "shared_temporal":
                    if self.field_task_phys_refiner is not None:
                        phys_cond = self.field_task_phys_refiner(z_phys)
                        shared_phys_latent = phys_cond["shared_phys_latent"]
                        stress_phys = phys_cond["stress_latent"]
                        flow_phys = phys_cond["deformation_latent"]
                        force_phys = phys_cond["contact_latent"]
                    else:
                        shared_phys_latent = self.field_shared_phys_proj(z_phys)
                        stress_phys = pb["stress_latent"]
                        flow_phys = pb["deformation_latent"]
                        force_phys = pb["contact_latent"]
                    spatial_cond = None
                    if self.field_spatial_stem is not None:
                        spatial_cond = self.field_spatial_stem(spatial_rgb)
                        spatial_cond = spatial_cond.view(
                            b,
                            v,
                            t,
                            spatial_cond.shape[1],
                            spatial_cond.shape[2],
                            spatial_cond.shape[3],
                        ).contiguous()
                    geom_latent_t = None
                    if self.geom_latent_mlp is not None:
                        geom_latent_t = self.geom_latent_mlp(frame_feat[:, :, 0, :].float().mean(dim=1))
                        frame_for_decoder = frame_feat_cond - frame_feat_cond[:, :, 0:1, :].expand_as(
                            frame_feat_cond
                        )
                    else:
                        frame_for_decoder = frame_feat_cond
                    shared_field_feat = self.shared_field_decoder(
                        z_fused,
                        frame_for_decoder,
                        shared_phys_latent,
                        spatial_cond=spatial_cond,
                        geometry_latent=geom_latent_t,
                        patch_cond=patch_cond,
                    )
                    if self.shared_field_param_encoder is not None:
                        if object_mask is not None:
                            shared_field_for_param = mask_field_with_object_mask(shared_field_feat, object_mask)
                        else:
                            shared_field_for_param = shared_field_feat
                        shared_field_param_feat = self.shared_field_param_encoder(shared_field_for_param)
                    flow_feat_in = shared_field_feat
                    if self.flow_spatial_enhancer is not None:
                        flow_feat_in = self.flow_spatial_enhancer(shared_field_feat)
                    if self.stress_ff_fuser is not None:
                        flow_field_pred = self.flow_head(flow_feat_in, flow_phys)
                        force_logits = self.force_head(shared_field_feat, force_phys)
                        force_pred = torch.sigmoid(force_logits)
                        if object_mask is not None:
                            flow_to_stress = mask_field_with_object_mask(flow_field_pred, object_mask)
                            force_to_stress = mask_field_with_object_mask(force_pred, object_mask)
                        else:
                            flow_to_stress = flow_field_pred
                            force_to_stress = force_pred
                        stress_in = self.stress_ff_fuser(shared_field_feat, flow_to_stress, force_to_stress)
                        if self.stress_spatial_enhancer is not None:
                            stress_in = self.stress_spatial_enhancer(stress_in)
                        stress_field_pred = self.stress_head(stress_in, stress_phys)
                    else:
                        stress_feat_in = shared_field_feat
                        if self.stress_spatial_enhancer is not None:
                            stress_feat_in = self.stress_spatial_enhancer(shared_field_feat)
                        stress_field_pred = self.stress_head(stress_feat_in, stress_phys)
                        flow_field_pred = self.flow_head(flow_feat_in, flow_phys)
                        force_logits = self.force_head(shared_field_feat, force_phys)
                        force_pred = torch.sigmoid(force_logits)
                else:
                    stress_field_pred = self.stress_head(
                        z_fused,
                        frame_feat_cond,
                        pb["stress_latent"],
                    )
                    flow_field_pred = self.flow_head(
                        z_fused,
                        frame_feat_cond,
                        pb["deformation_latent"],
                    )
                    force_logits = self.force_head(
                        z_fused,
                        frame_feat_cond,
                        pb["contact_latent"],
                    )
                    force_pred = torch.sigmoid(force_logits)
            else:
                z3 = feat_v.new_zeros(b, v, 3, self.num_frames, self.dec_h, self.dec_w)
                stress_field_pred = z3
                flow_field_pred = z3
                zf = feat_v.new_zeros(b, v, int(self.force_out_channels), self.num_frames, self.dec_h, self.dec_w)
                force_logits = zf.new_full(zf.shape, -25.0)
                force_pred = torch.sigmoid(force_logits)

            if run_param_chain:
                use_param_gt_fields = bool(self.training and active_stage == "param")
                if use_param_gt_fields:
                    stress_param_source = self._mix_param_field_with_gt(
                        stress_field_pred,
                        param_stress_gt,
                        param_stress_use_gt,
                    )
                    flow_param_source = self._mix_param_field_with_gt(
                        flow_field_pred,
                        param_flow_gt,
                        param_flow_use_gt,
                    )
                else:
                    stress_param_source = stress_field_pred
                    flow_param_source = flow_field_pred
                if object_mask is not None:
                    stress_for_param = mask_field_with_object_mask(stress_param_source, object_mask)
                    flow_for_param = mask_field_with_object_mask(flow_param_source, object_mask)
                    force_for_param = mask_field_with_object_mask(force_pred, object_mask)
                else:
                    stress_for_param = stress_param_source
                    flow_for_param = flow_param_source
                    force_for_param = force_pred
                stress_feat = self.stress_field_encoder(stress_for_param)
                flow_feat = self.flow_field_encoder(flow_for_param)
                force_feat = self.force_field_encoder(force_for_param)
                if self.param_chain_mode in ("strong_param_v1", "strong_param_v2"):
                    if self.phys_token_proj is None or self.strong_param_transformer is None:
                        raise RuntimeError(f"{self.param_chain_mode} is enabled but parameter token modules are not initialized")
                    param_tokens = [self.phys_token_proj(z_phys)]
                    if self.param_use_rgb_static:
                        if self.rgb_static_proj is None:
                            raise RuntimeError("rgb_static_proj is not initialized")
                        param_tokens.append(self.rgb_static_proj(z_fused))
                    if self.param_use_rgb_residual:
                        if self.rgb_residual_phys_proj is None or self.rgb_residual_proj is None:
                            raise RuntimeError("rgb residual projection modules are not initialized")
                        rgb_residual = z_fused - self.rgb_residual_phys_proj(z_phys)
                        param_tokens.append(self.rgb_residual_proj(rgb_residual))
                    if self.param_use_masked_field_tokens:
                        if self.field_token_proj is None:
                            raise RuntimeError("field_token_proj is not initialized")
                        if self.param_chain_mode == "strong_param_v2":
                            if self.stress_temporal_param_encoder is None or self.flow_temporal_param_encoder is None:
                                raise RuntimeError("strong_param_v2 temporal field encoders are not initialized")
                            stress_tokens = self.stress_temporal_param_encoder(stress_for_param)
                            flow_tokens = self.flow_temporal_param_encoder(flow_for_param)
                            param_tokens.extend([stress_tokens[:, i, :] for i in range(stress_tokens.shape[1])])
                            param_tokens.extend([flow_tokens[:, i, :] for i in range(flow_tokens.shape[1])])
                            param_tokens.append(self.field_token_proj(force_feat))
                        else:
                            param_tokens.extend(
                                [
                                    self.field_token_proj(stress_feat),
                                    self.field_token_proj(flow_feat),
                                    self.field_token_proj(force_feat),
                                ]
                            )
                    param_ctx = self.strong_param_transformer(torch.stack(param_tokens, dim=1))
                else:
                    param_parts = [z_phys, stress_feat, flow_feat, force_feat]
                    if shared_field_param_feat is not None:
                        param_parts.append(shared_field_param_feat)
                    param_ctx = self.param_field_fusion(torch.cat(param_parts, dim=1))
                param_pred, logvar_raw = self.param_head(param_ctx)
            else:
                stress_feat = z_fused.new_zeros(b, self.physics_bottleneck.contact_head.out_features)
                flow_feat = z_fused.new_zeros(b, self.physics_bottleneck.contact_head.out_features)
                force_feat = z_fused.new_zeros(b, self.physics_bottleneck.contact_head.out_features)
                param_pred = z_fused.new_zeros(b, self.num_targets)
                logvar_raw = param_pred.new_zeros(b, self.num_targets)

            if run_action_head:
                action_logits = self.action_head(z_phys)
            else:
                action_logits = z_fused.new_zeros(b, self.num_actions)

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
            "param_pred": param_pred,
            "param_pred_raw": param_pred_raw,
            "stress_field_pred": stress_field_pred,
            "flow_field_pred": flow_field_pred,
            "force_logits": force_logits,
            "force_pred": force_pred,
            "logvar": logvar,
            "contact_latent": pb["contact_latent"],
            "deformation_latent": pb["deformation_latent"],
            "stress_latent": pb["stress_latent"],
            "action_logits": action_logits,
            "field_feat_stress": stress_feat,
            "field_feat_flow": flow_feat,
            "field_feat_force": force_feat,
            "field_feat_shared": (
                shared_field_param_feat
                if shared_field_param_feat is not None
                else z_fused.new_zeros(b, self.physics_bottleneck.contact_head.out_features)
            ),
        }

