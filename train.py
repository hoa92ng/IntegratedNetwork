from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Model,
    WavLMModel,
    get_scheduler,
)

from torchmetrics.classification import Accuracy

from src.models.integrated_network import IntegratedNetwork, IntegratedNetworkConfig
from src.data.custom_audio_dataset import CustomAudioDataset


# ----------------------- Utils -----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_backbone(name: str, device: str):
    """
    Accepts: wav2vec2-base, wav2vec2-base-960h, wav2vec2-large, wavlm-base, wavlm-large
    """
    if name.startswith("wav2vec2"):
        return Wav2Vec2Model.from_pretrained(f"facebook/{name}").to(device), AutoFeatureExtractor.from_pretrained(f"facebook/{name}")
    if name.startswith("wavlm"):
        return WavLMModel.from_pretrained(f"microsoft/{name}").to(device), AutoFeatureExtractor.from_pretrained(f"microsoft/{name}")
    raise ValueError(f"Unknown backbone: {name}")


# ------------------- Training Loop -------------------

def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    run_dir = os.path.join("models", "runs", f"{args.backbone_model}_{args.exp_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # model, feature extractor (match backbone family)
    backbone, feat_extractor = make_backbone(args.backbone_model, device)

    # data
    train_ds = CustomAudioDataset(args.train_data_path, feature_extractor=feat_extractor, use_data_from_disk=args.use_data_from_disk, split="train", data_version=args.data_version)
    valid_ds = CustomAudioDataset(args.valid_data_path, feature_extractor=feat_extractor, use_data_from_disk=args.use_data_from_disk, split="validation", data_version=args.data_version)

    if args.show_qty:
        train_ds.show_counts()
        valid_ds.show_counts()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    
    cfg = IntegratedNetworkConfig(
        num_classes=args.num_classes,
        noise_std=args.noise_std,
        channel=args.proj_in_planes,
        width=args.width,
        proj_layers=args.proj_layers,
        use_cbam=not args.no_cbam,
        use_noise=not args.no_noise,
        cbam_alpha=not args.cbam_alpha,
        use_smoothing=not args.use_smoothing,
        cbam_resblock=args.cbam_resblock,
        disc_layers=args.disc_layers,
        cbam_kernel_size=args.cbam_kernel_size,
        device=device,
    )
    model = IntegratedNetwork(backbone, cfg).to(device)

    # loss & optim
    cls_criterion = nn.CrossEntropyLoss()
    anom_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # metrics (torchmetrics >=0.11 style)
    acc_cls = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
    acc_anom = Accuracy(task="binary").to(device)

    scaler = GradScaler(enabled=not args.no_amp)

    best_valid_loss = float("inf")

    # ------------------- epochs -------------------
    step_bar = tqdm(total=num_training_steps, desc="training", leave=False)
    for epoch in range(1, args.epochs + 1):
        model.train()
        acc_cls.reset(); acc_anom.reset()

        running_cls_loss = 0.0
        running_anom_loss = 0.0

        for batch in train_loader:
            x = batch["input_values"].to(device)
            y_anom = batch["nomaly_label"].to(device)  # 1 normal, 0 anomaly/unknown
            y_cls = batch["label"][batch["nomaly_label"] == 1].to(device)  # only normals have class labels

            if y_cls.numel() == 0:
                # no normal samples in this batch; skip classification loss
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=not args.no_amp):
                    # still forward to train discriminator
                    class_logits, anom_logits = model(x, anomaly_label=y_anom, train_mode=True)

                    # Build labels for discriminator:
                    # first half: original sub_proj -> "original" labels = y_anom (0/1) as float
                    # second half (if noise added): synthetic normals -> all zeros (fake)
                    orig_count = x.shape[0]
                    proj_labels = y_anom.float().unsqueeze(-1)  # (B, 1)
                    if anom_logits.shape[0] > orig_count:
                        extra = anom_logits.shape[0] - orig_count
                        proj_labels = torch.cat([proj_labels, torch.zeros(extra, 1, device=device)], dim=0)
                    anom_loss = anom_criterion(anom_logits, proj_labels)

                scaler.scale(anom_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step_bar.update(1)
                running_anom_loss += float(anom_loss.detach())
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=not args.no_amp):
                class_logits, anom_logits = model(x, anomaly_label=y_anom, train_mode=True)

                # classification loss on normals only
                cls_loss = cls_criterion(class_logits, y_cls)

                # discriminator labels:
                orig_count = x.shape[0]
                p_origin = anom_logits[:orig_count]
                p_extra  = anom_logits[orig_count:]  # noisy normals (fake)

                # True=1 for normals in original part; 0 otherwise
                true_normal_mask = y_anom == 1
                true_labels = true_normal_mask.float().unsqueeze(-1)  # (B,1)
                fake_labels = torch.zeros_like(p_extra)

                anom_loss = (
                    2.0 * anom_criterion(p_origin, true_labels) +
                    anom_criterion(p_extra, fake_labels) if p_extra.numel() > 0
                    else anom_criterion(p_origin, true_labels)
                )

                loss = args.cls_weight * cls_loss + anom_loss

            # metrics
            acc_cls.update(class_logits.detach().softmax(dim=-1), y_cls.detach())
            acc_anom.update(torch.sigmoid(anom_logits.detach()),
                            torch.cat([true_labels, fake_labels], dim=0) if p_extra.numel() > 0 else true_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step_bar.update(1)
            running_cls_loss += float(cls_loss.detach())
            running_anom_loss += float(anom_loss.detach())

        # ------------------- validation -------------------
        model.eval()
        acc_cls.reset(); acc_anom.reset()
        valid_loss = 0.0
        with torch.no_grad(), autocast(enabled=not args.no_amp):
            for batch in valid_loader:
                x = batch["input_values"].to(device)
                y_anom = batch["nomaly_label"].to(device)
                y_cls = batch["label"][batch["nomaly_label"] == 1].to(device)

                class_logits, anom_logits = model(x, anomaly_label=y_anom, train_mode=False)

                # For val, compute the same combined objective but without noise
                cls_loss = cls_criterion(class_logits, y_cls) if y_cls.numel() > 0 else torch.tensor(0.0, device=device)
                anom_targets = y_anom.float().unsqueeze(-1)  # only originals
                anom_loss = anom_criterion(anom_logits, anom_targets)
                batch_loss = args.cls_weight * cls_loss + anom_loss

                valid_loss += float(batch_loss)
                if y_cls.numel() > 0:
                    acc_cls.update(class_logits.softmax(dim=-1), y_cls)
                acc_anom.update(torch.sigmoid(anom_logits), anom_targets)

        valid_loss /= max(1, len(valid_loader))
        acc_cls_val = acc_cls.compute().item() if acc_cls._update_count > 0 else 0.0
        acc_anom_val = acc_anom.compute().item()

        # checkpoints
        torch.save(
            model.state_dict(),
            os.path.join(run_dir, f"epoch_{epoch:03d}_{valid_loss:.3f}_{acc_cls_val:.3f}_{acc_anom_val:.3f}.pth"),
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        tqdm.write(
            f"Epoch [{epoch}/{args.epochs}] "
            f"| valid_loss: {valid_loss:.3f} "
            f"| acc_cls: {acc_cls_val:.3f} "
            f"| acc_anom: {acc_anom_val:.3f}"
        )

    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))
    step_bar.close()
