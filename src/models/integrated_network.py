from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.models.attention_smoothing_cbam1d import CBAM1D


class Projection(nn.Module):
    """
    1D conv stack (+ optional CBAM) that projects transformer frames
    -> fixed-length vector via adaptive pooling.
    Input expected shape after backbone: (B, T, C).
    """
    def __init__(
        self,
        in_planes: int = 49,
        out_planes: int = 1024,
        n_layers: int = 3,
        use_cbam: bool = True,
        alpha: float = 0.1,
        cbam_kernel_size: int = 7,
        use_smoothing: bool = True,
        res_block: bool = False,
    ) -> None:
        super().__init__()
        self.out_planes = out_planes
        self.use_cbam = use_cbam

        layers = []
        for i in range(n_layers):
            in_ch = in_planes if i == 0 else in_planes * (2 ** i)
            out_ch = in_planes * (2 ** (i + 1))
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding="same"),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
        if use_cbam:
            layers += [CBAM1D(out_ch, reduction=2, sam_smoothing_alpha=alpha,
                              sam_kernel_size=cbam_kernel_size,
                              cam_smoothing_alpha=alpha,
                              use_smoothing=use_smoothing, residual=res_block)]

        self.projector = nn.Sequential(*layers)

    def forward(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        x = self.projector(x_bt_c) 
        # flatten channels x time and pool to fixed 'out_planes'
        x = F.adaptive_avg_pool1d(x.view(x.shape[0], -1), self.out_planes)
        return x


class Discriminator(nn.Module):
    """Simple MLP discriminator returning a single logit per item."""
    def __init__(self, in_features: int, n_layers: int = 3, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = in_features if hidden is None else hidden

        body = []
        for i in range(max(1, n_layers - 1)):
            in_dim = in_features if i == 0 else hidden_dim
            hidden_dim = int(hidden_dim // 1.5) if hidden is None else hidden
            body.append(
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.body = nn.Sequential(*body)
        self.tail = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.tail(x)


@dataclass
class IntegratedNetworkConfig:
    num_classes: int = 12
    noise_std: float = 0.04
    channel: int = 49         # projector input planes (start)
    width: int = 1024         # projector output width & MLP width
    proj_layers: int = 3
    use_cbam: bool = True
    use_noise: bool = True
    cbam_alpha: float = 0.1
    use_smoothing: bool = True
    cbam_kernel_size: int = 7
    cbam_resblock: bool = False
    disc_layers: int = 3
    device: str = "cuda"


class IntegratedNetwork(nn.Module):
    """
    Dual-head network:
    - Backbone (wav2vec2/wavlm) -> Projection -> Classifier (only on normal samples)
    - Sub-projection (+ optional Gaussian noise for training) -> Discriminator (anomaly)
    """
    def __init__(self, backbone: PreTrainedModel, cfg: IntegratedNetworkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        self.projector = Projection(
            in_planes=cfg.channel,
            out_planes=cfg.width,
            n_layers=cfg.proj_layers,
            use_cbam=cfg.use_cbam,
            alpha=cfg.cbam_alpha,
            res_block=cfg.cbam_resblock,
            use_smoothing=cfg.use_smoothing,
        )

        # shallow sub-projection on top of the main projection vector (as 1xW 1D signal)
        self.sub_projector = nn.Sequential(
            nn.Conv1d(1, 1, 3, padding="same"),
            nn.Conv1d(1, 1, 3, padding="same"),
            nn.Conv1d(1, 1, 3, padding="same"),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(cfg.width),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.LazyLinear(cfg.width // 2),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.LazyLinear(cfg.width // 4),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.LazyLinear(cfg.num_classes),
        )

        self.discriminator = Discriminator(cfg.width, n_layers=cfg.disc_layers)

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        # Returns (B, T, C)
        return self.backbone(x).last_hidden_state

    def _sub_project(self, proj_vec: torch.Tensor) -> torch.Tensor:
        # proj_vec: (B, W) -> (B, 1, W) -> sub-proj -> (B, W)
        x = proj_vec.unsqueeze(1)
        x = self.sub_projector(x)
        return x.squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        anomaly_label: Optional[torch.Tensor] = None,
        train_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            class_logits: (B_normal, num_classes) — only for normal samples if labels provided, else (B, num_classes)
            anomaly_logits: (N_mix, 1) — discriminator logits
        """
        feats_bt_c = self._backbone_features(x)
        proj_vec = self.projector(feats_bt_c)             # (B, W)

        if anomaly_label is not None:
            normal_mask = (anomaly_label == 1)
            cls_input = proj_vec[normal_mask]
        else:
            cls_input = proj_vec

        class_logits = self.classifier(cls_input)         # (B_normal or B, C)

        sub_proj = self._sub_project(proj_vec)            # (B, W)

        if train_mode and anomaly_label is not None:
            # Build discriminator input: [all sub_proj; noisy copies of normal sub_proj]
            normal_sub = sub_proj[normal_mask]
            if self.cfg.use_noise and normal_sub.numel() > 0:
                noise = torch.normal(
                    mean=0.0,
                    std=self.cfg.noise_std,
                    size=normal_sub.shape,
                    device=normal_sub.device,
                )
                noisy = normal_sub + noise
                disc_in = torch.cat([sub_proj, noisy], dim=0)
            else:
                disc_in = sub_proj
        else:
            disc_in = sub_proj

        anomaly_logits = self.discriminator(disc_in)      # (N_mix, 1)

        return class_logits, anomaly_logits
