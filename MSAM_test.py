import argparse
import os
import sys
from typing import List

import torch
from PIL import Image
from torchvision.utils import save_image

# ensure local sam3 package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "sam3"))

from dataset import MVTecMetaDataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official


def build_loader(root: str, meta_path: str, mode: str, batch_size: int):
    ds = MVTecMetaDataset(root=root, meta_path=meta_path, mode=mode)

    def collate_fn(batch):
        imgs, masks, prompt_lists, is_anomaly, class_names = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        return imgs, masks, list(prompt_lists), list(class_names)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )


def load_model(args, device):
    if args.use_official:
        model = FineTuneSAM3Official(
            bpe_path=args.bpe_path,
            sam3_ckpt=args.sam3_ckpt,
            enable_lora=not args.disable_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            device=device,
        )
    else:
        model = FineTuneSAM3(
            bpe_path=args.bpe_path,
            enable_lora=not args.disable_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_vision=args.freeze_vision,
            freeze_text=args.freeze_text,
            device=device,
        )
        if args.sam3_ckpt and os.path.exists(args.sam3_ckpt):
            state = torch.load(args.sam3_ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print(f"[INFO] Loaded fine-tuned weights from {args.ckpt}")
    model.eval()
    return model.to(device)


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = build_loader(args.data_root, args.meta_path or os.path.join(args.data_root, "meta.json"), args.mode, args.batch_size)
    model = load_model(args, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # parse custom prompt if provided (comma-separated words)
    custom_prompt: List[str] = []
    if args.prompt:
        custom_prompt = [w.strip() for w in args.prompt.split(",") if w.strip()]

    idx = 0
    for images, masks, prompt_lists, class_names in loader:
        images = images.to(device)
        # override dataset prompts with a custom prompt if given
        if custom_prompt:
            prompt_lists = [custom_prompt for _ in prompt_lists]
        out = model(images, prompt_lists)
        pred_masks = out["pred_masks"]
        if pred_masks is None:
            continue
        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        if pred_masks.dim() == 4:
            pred_masks = pred_masks.max(dim=1, keepdim=True).values
        pred_masks = torch.sigmoid(pred_masks)
        # upsample to original mask size
        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        for b in range(pred_masks.shape[0]):
            save_path = os.path.join(args.output_dir, f"pred_{idx}.png")
            save_image(pred_masks[b], save_path)
            idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MSAM test")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="test", choices=["train", "train_all", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt", type=str, default=None, help="fine-tuned PEFT checkpoint (sam3_peft_best.pth)")
    parser.add_argument("--sam3_ckpt", type=str, default=None, help="base SAM3 checkpoint for official build")
    parser.add_argument("--use_official", action="store_true")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--bpe_path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt words, comma separated, to override dataset prompts.")
    args = parser.parse_args()
    run_inference(args)
