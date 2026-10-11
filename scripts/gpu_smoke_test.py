"""Check that Pointcept + Sonata run on this GPU and how much memory a step takes.

Run from inside the Pointcept checkout (after envs/setup_2080.sh):
    python /path/to/repo/scripts/gpu_smoke_test.py --config configs/heritage/semseg-sonata-heritage-dec.py \
        --weight weights/sonata/pretrain-sonata-v1m1-0-base.pth --points 80000

Does one training step (forward + backward, fp16 AMP) on a synthetic scene and
prints peak VRAM. If it fails with OOM, lower --points and use that value as
point_max in the config's SphereCrop.
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from pointcept.models import build_model  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--points", type=int, default=80000)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model).cuda().train()

    if args.weight:
        state = torch.load(args.weight, map_location="cpu", weights_only=False)["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        state = {k.replace("student.backbone", "backbone", 1): v for k, v in state.items()
                 if k.startswith("student.backbone")}
        info = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(info.unexpected_keys)
        print(f"loaded {loaded} backbone tensors from Sonata, {len(info.missing_keys)} missing (decoder/head expected)")

    n = args.points
    coord = torch.rand(n, 3, device="cuda") * torch.tensor([10.0, 10.0, 15.0], device="cuda")
    color = torch.rand(n, 3, device="cuda") * 2 - 1
    normal = torch.nn.functional.normalize(torch.randn(n, 3, device="cuda"), dim=1)
    batch = dict(
        coord=coord,
        grid_coord=torch.div(coord, 0.02, rounding_mode="floor").int(),
        feat=torch.cat([coord, color, normal], 1),
        offset=torch.tensor([n], device="cuda"),
        segment=torch.randint(0, cfg.model.num_classes, (n,), device="cuda"),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(batch)["loss"]
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)} | {n} points | loss {loss.item():.3f} | "
          f"peak {peak:.2f} / {total:.1f} GB | trainable params {sum(p.numel() for p in params)/1e6:.1f}M")
    print("OK")


if __name__ == "__main__":
    main()
