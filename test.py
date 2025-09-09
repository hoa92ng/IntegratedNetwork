from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from transformers import AutoFeatureExtractor, Wav2Vec2Model, WavLMModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from src.data.custom_audio_dataset import CustomAudioDataset
from src.models.integrated_network import IntegratedNetwork, IntegratedNetworkConfig


def make_backbone(name: str, device: str):
    """
    Load backbone model and feature extractor.
    Accepts: wav2vec2-base, wav2vec2-base-960h, wav2vec2-large, wavlm-base, wavlm-large
    """
    if name.startswith("wav2vec2"):
        return (
            Wav2Vec2Model.from_pretrained(f"facebook/{name}").to(device),
            AutoFeatureExtractor.from_pretrained(f"facebook/{name}")
        )
    if name.startswith("wavlm"):
        return (
            WavLMModel.from_pretrained(f"microsoft/{name}").to(device),
            AutoFeatureExtractor.from_pretrained(f"microsoft/{name}")
        )
    raise ValueError(f"Unknown backbone: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_model", default="wavlm-base")
    ap.add_argument("--test_data_path")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ckpt", type=str, default=None, help="Optional state_dict to load")
    ap.add_argument("--num_classes", type=int, default=11)

    # Network configuration
    ap.add_argument("--noise_std", type=float, default=0.04)
    ap.add_argument("--proj_in_planes", type=int, default=49)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--proj_layers", type=int, default=3)
    ap.add_argument("--disc_layers", type=int, default=3)
    ap.add_argument("--cbam_alpha", type=float, default=0.1)
    ap.add_argument("--cbam_kernel_size", type=int, default=7)
    ap.add_argument("--cbam_resblock", action="store_true")

    # Training / inference options
    ap.add_argument("--cls_weight", type=float, default=5.0)
    ap.add_argument("--no_cbam", action="store_true")
    ap.add_argument("--no_noise", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--use_smoothing", action="store_true")  # smoothing enabled if passed

    # Dataset config
    ap.add_argument("--use_data_from_disk", action="store_true")
    ap.add_argument("--data_version", type=str, default="v1")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load backbone and feature extractor
    backbone, feat_extractor = make_backbone(args.backbone_model, device)

    # Build test dataset
    test_va = CustomAudioDataset(
        feature_extractor=feat_extractor,
        use_data_from_disk=args.use_data_from_disk,
        split="test",
        data_version=args.data_version,
        dataset_path=args.test_data_path,
    )
    dl = DataLoader(test_va, batch_size=args.batch_size, shuffle=False)

    # Build model configuration
    cfg = IntegratedNetworkConfig(
        num_classes=args.num_classes,
        noise_std=args.noise_std,
        channel=args.proj_in_planes,
        width=args.width,
        proj_layers=args.proj_layers,
        use_cbam=not args.no_cbam,
        use_noise=not args.no_noise,
        cbam_alpha=args.cbam_alpha,
        use_smoothing=not args.use_smoothing,
        cbam_resblock=args.cbam_resblock,
        disc_layers=args.disc_layers,
        cbam_kernel_size=args.cbam_kernel_size,
        device=device,
    )
    model = IntegratedNetwork(backbone, cfg).to(device)

    # Load checkpoint if provided
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")

    # Metrics
    acc_cls = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
    acc_an = Accuracy(task="binary").to(device)

    ce = CrossEntropyLoss()
    bce = BCEWithLogitsLoss()

    model.eval()
    tot_loss = 0.0

    # For computing final combined accuracy
    final_correct = 0
    total_samples = 0

    with torch.no_grad():
        for b in dl:
            x = b["input_values"].to(device)
            y_an = b["anomaly_label"].to(device).long()  # 1 = valid command, 0 = anomaly
            y_cls = b["label"].to(device).long()

            # Forward pass
            cls_logits, an_logits = model(x, anomaly_label=y_an, train_mode=False)

            # ----- Loss -----
            # Compute classification loss only on valid samples
            valid_mask = (y_an == 1)
            if valid_mask.any():
                cls_loss = ce(cls_logits[valid_mask], y_cls[valid_mask])
                acc_cls.update(cls_logits[valid_mask].softmax(-1), y_cls[valid_mask])
            else:
                cls_loss = torch.tensor(0., device=device)

            # Anomaly detection loss
            an_loss = bce(an_logits, y_an.float().unsqueeze(-1))
            tot_loss += float(args.cls_weight * cls_loss + an_loss)

            # Update anomaly detection accuracy
            acc_an.update(torch.sigmoid(an_logits), y_an.float().unsqueeze(-1))

            # ----- Final Accuracy -----
            # pred_an: 1 = predicted valid, 0 = predicted anomaly
            pred_an = (torch.sigmoid(an_logits).squeeze(-1) >= 0.5).long()
            pred_cls = cls_logits.argmax(dim=-1)

            # Case A: anomaly sample is correct if predicted anomaly
            correct_anom = (y_an == 0) & (pred_an == 0)

            # Case B: valid sample is correct if predicted valid AND correct class
            correct_valid = (y_an == 1) & (pred_an == 1) & (pred_cls == y_cls)

            correct = correct_anom | correct_valid
            final_correct += int(correct.sum().item())
            total_samples += int(y_an.numel())

    # Compute accuracies safely
    try:
        an_acc = acc_an.compute().item()
    except Exception:
        an_acc = float("nan")

    try:
        cls_acc = acc_cls.compute().item()
    except Exception:
        cls_acc = float("nan")

    final_acc = final_correct / max(1, total_samples)

    print(
        f"valid_loss: {tot_loss/len(dl):.4f} | "
        f"acc_anom: {an_acc:.4f} | "
        f"acc_cls(valid): {cls_acc:.4f} | "
        f"final_acc: {final_acc:.4f} "
        f"(N={total_samples})"
    )


if __name__ == "__main__":
    main()
