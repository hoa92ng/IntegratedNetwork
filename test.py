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
    Accepts: wav2vec2-base, wav2vec2-base-960h, wav2vec2-large, wavlm-base, wavlm-large
    """
    if name.startswith("wav2vec2"):
        return Wav2Vec2Model.from_pretrained(f"facebook/{name}").to(device), AutoFeatureExtractor.from_pretrained(f"facebook/{name}")
    if name.startswith("wavlm"):
        return WavLMModel.from_pretrained(f"microsoft/{name}").to(device), AutoFeatureExtractor.from_pretrained(f"microsoft/{name}")
    raise ValueError(f"Unknown backbone: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_model", default="wavlm-base")
    ap.add_argument("--test_data_path", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ckpt", type=str, default=None, help="Optional state_dict to load")
    ap.add_argument("--num_classes", type=int, default=11)
    ap.add_argument("--noise_std", type=float, default=0.04)
    ap.add_argument("--proj_in_planes", type=int, default=49)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--proj_layers", type=int, default=3)
    ap.add_argument("--disc_layers", type=int, default=3)
    ap.add_argument("--cbam_alpha", type=float, default=0.1)
    ap.add_argument("--cls_weight", type=float, default=5.0)
    ap.add_argument("--cbam_kernel_size", type=int, default=7)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--no_cbam", action="store_true")
    ap.add_argument("--no_noise", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--use_smoothing", action="store_true")
    ap.add_argument("--cbam_resblock", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, feat_extractor = make_backbone(args.backbone_model, device)
    
    test_va = CustomAudioDataset(feature_extractor=feat_extractor, use_data_from_disk=args.use_data_from_disk, split="test", data_version=args.data_version)

    dl = DataLoader(test_va, batch_size=args.batch_size, shuffle=False)
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

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")

    acc_cls = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)
    acc_an  = Accuracy(task="binary").to(device)
    ce = CrossEntropyLoss(); bce = BCEWithLogitsLoss()

    model.eval()
    tot = 0.0
    with torch.no_grad():
        for b in dl:
            x = b["input_values"].to(device)
            y_an = b["anomaly_label"].to(device)
            y_cls = b["label"][b["anomaly_label"] == 1].to(device)
            cls_logits, an_logits = model(x, anomaly_label=y_an, train_mode=False)

            cls_loss = ce(cls_logits, y_cls) if y_cls.numel() else torch.tensor(0., device=device)
            an_loss = bce(an_logits, y_an.float().unsqueeze(-1))
            tot += float(args.cls_weight*cls_loss + an_loss)

            if y_cls.numel():
                acc_cls.update(cls_logits.softmax(-1), y_cls)
            acc_an.update(torch.sigmoid(an_logits), y_an.float().unsqueeze(-1))

    print(f"valid_loss: {tot/len(dl):.4f} | acc_cls: {acc_cls.compute().item():.4f} | acc_anom: {acc_an.compute().item():.4f}")

if __name__ == "__main__":
    main()
