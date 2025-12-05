import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

# ensure local sam3 package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "sam3"))

from dataset import MVTecMetaDataset
from model_wrapper import FineTuneSAM3, FineTuneSAM3Official
from sam3.visualization_utils import draw_masks_to_frame


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


def get_color_map(palette, prompt: str):
    import hashlib
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(palette)
    return palette[idx]


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = build_loader(args.data_root, args.meta_path or os.path.join(args.data_root, "meta.json"), args.mode, args.batch_size)
    model = load_model(args, device)
    os.makedirs(args.output_dir, exist_ok=True)

    custom_prompt: List[str] = []
    if args.prompt:
        custom_prompt = [w.strip() for w in args.prompt.split(",") if w.strip()]

    to_pil = transforms.ToPILImage()
    palette = [
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (218, 165, 32),
        (199, 21, 133),
        (70, 130, 180),
    ]
    font = ImageFont.load_default()

    idx = 0
    total_imgs = 0
    total_time = 0.0
    dice_sum = 0.0
    dice_cnt = 0

    pbar = tqdm(loader, desc="Inference", leave=True)
    for images, masks, prompt_lists, class_names in pbar:
        images = images.to(device)
        if custom_prompt:
            prompt_lists = [custom_prompt for _ in prompt_lists]

        start = time.time()
        out = model(images, prompt_lists)
        infer_time = time.time() - start
        total_time += infer_time
        total_imgs += images.size(0)

        pred_masks = out["pred_masks"]
        if pred_masks is None:
            continue
        if pred_masks.dim() == 5:
            pred_masks = pred_masks[-1]
        if pred_masks.dim() == 4:
            pred_masks = pred_masks.max(dim=1, keepdim=True).values
        pred_masks = torch.sigmoid(pred_masks)
        if pred_masks.shape[-2:] != masks.shape[-2:]:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        gt = (masks > 0.5).float().to(device)
        valid = gt.flatten(1).sum(dim=1) > 0
        if valid.any():
            pm = pred_masks[valid]
            gm = gt[valid]
            pm_bin = (pm > 0.5).float()
            num = 2 * (pm_bin * gm).sum(dim=(1, 2, 3))
            den = pm_bin.sum(dim=(1, 2, 3)) + gm.sum(dim=(1, 2, 3)) + 1e-6
            dice_batch = (num / den).mean().item()
            dice_sum += dice_batch * valid.sum().item()
            dice_cnt += valid.sum().item()

        for b in range(pred_masks.shape[0]):
            cls_name = class_names[b]
            prompts_b = prompt_lists[b] if prompt_lists else []
            prompts_b = prompts_b if isinstance(prompts_b, list) else [prompts_b]
            prompt_text = "_".join(prompts_b) if prompts_b else "prompt"
            sample_dir = os.path.join(args.output_dir, cls_name)
            os.makedirs(sample_dir, exist_ok=True)

            img_pil = to_pil(images[b].cpu())
            mask_np = pred_masks[b].squeeze().cpu().numpy()
            if mask_np.ndim == 2:
                masks_stack = (mask_np > 0.5)[None, ...]
                prompts_for_color = prompts_b if prompts_b else ["prompt"]
            else:
                masks_stack = mask_np > 0.5
                prompts_for_color = prompts_b if prompts_b else [f"c{i}" for i in range(mask_np.shape[0])]

            colors = np.array([get_color_map(palette, p) for p in prompts_for_color], dtype=np.uint8)
            frame = np.array(img_pil.convert("RGB"), dtype=np.uint8)
            frame = draw_masks_to_frame(frame, masks_stack.astype(bool), colors)
            overlay_pil = Image.fromarray(frame)

            # legend bottom-left with prompt text
            draw = ImageDraw.Draw(overlay_pil)
            row_h = 12
            box_w = 10
            pad = 4
            legend_items = []
            max_w = 0
            for p in prompts_for_color:
                text = p
                bbox = font.getbbox(text)
                w = bbox[2] - bbox[0]
                max_w = max(max_w, w)
                legend_items.append((text, w))
            legend_h = row_h * len(prompts_for_color) + pad * 2
            legend_w = box_w + 4 + max_w + pad * 2
            legend_x = 5
            legend_y = overlay_pil.height - legend_h - 5
            draw.rectangle(
                [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
                fill=(0, 0, 0),
            )
            for i, (text, _) in enumerate(legend_items):
                color = get_color_map(palette, prompts_for_color[i])
                y = legend_y + pad + i * row_h
                draw.rectangle(
                    [legend_x + pad, y, legend_x + pad + box_w, y + box_w],
                    fill=color,
                )
                draw.text(
                    (legend_x + pad + box_w + 4, y - 2),
                    text,
                    fill=(255, 255, 255),
                    font=font,
                )

            filename = f"{cls_name}_{prompt_text}_{idx}.png"
            overlay_path = os.path.join(sample_dir, filename)
            overlay_pil.save(overlay_path)
            idx += 1

        speed = total_imgs / total_time if total_time > 0 else 0.0
        avg_dice = dice_sum / dice_cnt if dice_cnt > 0 else 0.0
        pbar.set_postfix(imgs=total_imgs, fps=speed, dice=avg_dice)

    final_speed = total_imgs / total_time if total_time > 0 else 0.0
    final_dice = dice_sum / dice_cnt if dice_cnt > 0 else 0.0
    print(f"[INFO] Inference done. images={total_imgs}, fps={final_speed:.2f}, avg_dice={final_dice:.4f}")


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
