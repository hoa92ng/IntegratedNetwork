import argparse
from train import *
# ------------------- CLI -------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Integrated KWS + Anomaly Training")
    p.add_argument("--exp_name", type=str, default="exp")
    p.add_argument("--backbone_model", type=str, default="wavlm-base",
                   help="wav2vec2-base | wav2vec2-base-960h | wav2vec2-large | wavlm-base | wavlm-large")
    p.add_argument("--feature_model", type=str, default=None,
                   help="Optional HF feature extractor id (defaults to backbone_model)")

    p.add_argument("--use_data_from_disk", action="store_true")
    p.add_argument("--train_data_path", type=str)
    p.add_argument("--valid_data_path", type=str)
    p.add_argument("--data_version", type=str, default="v0.01")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # model config
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--noise_std", type=float, default=0.04)
    p.add_argument("--proj_in_planes", type=int, default=49)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--proj_layers", type=int, default=3)
    p.add_argument("--disc_layers", type=int, default=3)
    p.add_argument("--cbam_alpha", type=float, default=0.1)
    p.add_argument("--cbam_kernel_size", type=int, default=7)
    p.add_argument("--use_smoothing", action="store_true")
    p.add_argument("--cbam_resblock", action="store_true")
    p.add_argument("--no_cbam", action="store_true")
    p.add_argument("--no_noise", action="store_true")
    p.add_argument("--no_amp", action="store_true")

    # loss weights
    p.add_argument("--cls_weight", type=float, default=5.0)

    # misc
    p.add_argument("--show_qty", action="store_true")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)

