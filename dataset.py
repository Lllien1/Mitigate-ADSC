import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from utils.defect_definitions import mvtec_short_keywords

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _default_transforms(image_size: int = 1008) -> Tuple[Callable, Callable]:
    """Default image/mask transforms aligned with SAM3 1008px input."""
    img_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )
    return img_tf, mask_tf


@dataclass
class SampleEntry:
    img_path: str
    mask_path: str
    cls_name: str
    anomaly: int
    specie_name: str


class MVTecMetaDataset(Dataset):
    """Dataset that mirrors FiLo `mvtec_supervised.py` meta.json sampling (train/train_all/test).

    - meta.json structure: meta['train'/'test'][cls] is a list of dicts with keys
      {'img_path','mask_path','cls_name','specie_name','anomaly'}.
    - train: k-shot per class_name (obj_name) from train split; train_all: k-shot per class across train split;
      test: full test split per meta.json.
    - aug_rate: probability to synthesize a 2x2 mosaic from random defects in test set of the same class.
    - Returns: (image_tensor, mask_tensor, prompt_list, is_anomaly, cls_name)
    """

    def __init__(
        self,
        root: str,
        meta_path: Optional[str] = None,
        mode: str = "test",
        k_shot: int = 0,
        obj_name: Optional[str] = None,
        include_test_defects: bool = True,
        goods_per_class: Optional[int] = 10,
        aug_rate: float = 0.0,
        prompt_dict: Optional[Dict[str, List[str]]] = None,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        save_dir: Optional[str] = None,
        # new for specie-splitting from test defects:
        train_from_test: bool = False,            # when True, build per-specie train/test from test split defects
        specie_split_ratio: float = 0.8,         # train ratio per specie (e.g. 0.8 => 80% train, 20% test)
        specie_split_seed: int = 42,             # deterministic seed for per-specie split
    ) -> None:
        """
        Dataset constructor.

        If train_from_test=True, we will build per-specie splits from meta_info['test'][cls]
        and (if save_dir provided) save them to JSON for reproducibility; test-time will
        prefer loading those JSON files if present to guarantee consistency.

        If train_from_test=False, behavior falls back to original k_shot logic (with
        include_test_defects fallback).
        """
        super().__init__()
        self.root = root
        self.aug_rate = aug_rate
        self.prompt_dict = prompt_dict or mvtec_short_keywords
        img_tf, mask_tf = _default_transforms()
        self.image_transform = image_transform or img_tf
        self.mask_transform = mask_transform or mask_tf

        # Persist new flags
        self.train_from_test = train_from_test
        self.specie_split_ratio = specie_split_ratio
        self.specie_split_seed = specie_split_seed
        self.save_dir = save_dir

        meta_file = meta_path or os.path.join(root, "meta.json")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)

        if mode == "train_all":
            split_meta = meta_info["train"]
            cls_names = list(split_meta.keys())
        elif mode == "train":
            split_meta = meta_info["train"]
            cls_names = [obj_name] if obj_name is not None else list(split_meta.keys())
        else:
            split_meta = meta_info[mode]
            cls_names = list(split_meta.keys())

        self.entries: List[SampleEntry] = []

        for cls in cls_names:
            data_list = split_meta[cls]

            # ==== Case 1: train_from_test mode (preferred) ====
            if self.train_from_test:
                chosen: List[Dict] = []

                # Try to load previously saved split file for reproducibility if provided
                if self.save_dir is not None:
                    split_file = os.path.join(self.save_dir, f"specie_splits_{cls}.json")
                    if os.path.exists(split_file):
                        try:
                            with open(split_file, "r", encoding="utf-8") as f:
                                js = json.load(f)
                            sel_list = js.get("train", []) if mode == "train" else js.get("test", [])
                            # Build lookup from available meta (test+train) to recover full dicts
                            pool = []
                            if "test" in meta_info and cls in meta_info["test"]:
                                pool.extend(meta_info["test"][cls])
                            if "train" in meta_info and cls in meta_info["train"]:
                                pool.extend(meta_info["train"][cls])
                            lookup = {}
                            for d in pool:
                                key = (d.get("img_path"), d.get("mask_path", ""))
                                lookup[key] = d
                            loaded = []
                            for item in sel_list:
                                key = (item.get("img_path"), item.get("mask_path", ""))
                                if key in lookup:
                                    loaded.append(lookup[key])
                            if len(loaded) > 0:
                                chosen = loaded
                        except Exception:
                            # if any error, fall back to deterministic split below
                            chosen = []

                # If not loaded from file, do deterministic per-specie split from test meta
                if not chosen:
                    test_meta_for_cls = meta_info.get("test", {}).get(cls, [])
                    # group only defects by specie_name and require mask_path existence
                    specie_map = {}
                    for d in test_meta_for_cls:
                        if int(d.get("anomaly", 0)) != 1:
                            continue
                        maskp = d.get("mask_path")
                        if not maskp:
                            continue
                        full_maskp = os.path.join(self.root, maskp)
                        if not os.path.exists(full_maskp):
                            # skip entries missing mask file
                            continue
                        specie = d.get("specie_name", d.get("cls_name", cls))
                        specie_map.setdefault(specie, []).append(d)

                    rng = random.Random(self.specie_split_seed)
                    class_train: List[Dict] = []
                    class_test: List[Dict] = []

                    for specie, items in specie_map.items():
                        rng.shuffle(items)
                        n = len(items)
                        if n == 0:
                            continue
                        if n == 1:
                            n_train = 1
                        else:
                            n_train = int(max(1, round(n * self.specie_split_ratio)))
                            if n_train >= n:
                                n_train = n - 1
                        train_items = items[:n_train]
                        test_items = items[n_train:]
                        class_train.extend(train_items)
                        class_test.extend(test_items)
                    
                    # Add normal (anomaly==0) samples from TRAIN split, capped by the number of defect samples in this class.
                    # Rule: if this cls has n defect samples assigned to training (after specie split),
                    # we sample at most n normal samples for training (prefer train folder), for a ~1:1 balance.
                    n_def_train = len(class_train)
                    if n_def_train > 0:
                        train_meta_for_cls = meta_info.get("train", {}).get(cls, [])
                        normal_train_images = [d for d in train_meta_for_cls if int(d.get("anomaly", 0)) == 0]
                        if normal_train_images:
                            k = min(len(normal_train_images), n_def_train)
                            # 使用与 specie_split_seed 相同的 rng 确保随机抽样可复现
                            normal_sampled = normal_train_images.copy() if len(normal_train_images) <= k else rng.sample(normal_train_images, k)
                            class_train.extend(normal_sampled)

                    goods_from_test = [d for d in test_meta_for_cls if int(d.get("anomaly", 0)) == 0]
                    if goods_from_test:
                        # avoid duplicates by (img_path, mask_path)
                        seen_keys = set((x.get("img_path"), x.get("mask_path", "")) for x in class_test)
                        for gd in goods_from_test:
                            key = (gd.get("img_path"), gd.get("mask_path", ""))
                            if key not in seen_keys:
                                class_test.append(gd)
                                seen_keys.add(key)
                    
                    chosen = class_train.copy() if mode == "train" else class_test.copy()

                    # save splits for reproducibility
                    if self.save_dir is not None:
                        os.makedirs(self.save_dir, exist_ok=True)
                        out_map_path = os.path.join(self.save_dir, f"specie_splits_{cls}.json")
                        json_map = {"train": [], "test": []}
                        for d in class_train:
                            json_map["train"].append({"img_path": d["img_path"], "mask_path": d.get("mask_path", "")})
                        for d in class_test:
                            json_map["test"].append({"img_path": d["img_path"], "mask_path": d.get("mask_path", "")})
                        try:
                            with open(out_map_path, "w", encoding="utf-8") as f:
                                json.dump(json_map, f, indent=2)
                        except Exception:
                            pass

            # ==== Case 2: legacy k_shot handling (train/test from train split, optional include_test_defects) ====
            elif mode in ("train", "train_all", "test") and k_shot > 0:
                # original k_shot logic, using train split primarily and optional inclusion of test defects/goods
                train_defects = [d for d in data_list if int(d.get("anomaly", 0)) == 1 and d.get("mask_path")]
                train_goods = [d for d in data_list if int(d.get("anomaly", 0)) == 0]

                test_defects = []
                test_goods = []
                if include_test_defects:
                    try:
                        test_meta = meta_info.get("test", {})
                        cls_test_list = test_meta.get(cls, [])
                    except Exception:
                        cls_test_list = []
                    for d in cls_test_list:
                        if int(d.get("anomaly", 0)) == 1 and d.get("mask_path"):
                            test_defects.append(d)
                        elif int(d.get("anomaly", 0)) == 0:
                            test_goods.append(d)

                chosen_defects = test_defects.copy() if include_test_defects and len(test_defects) > 0 else []
                if len(chosen_defects) < k_shot:
                    needed = k_shot - len(chosen_defects)
                    if len(train_defects) >= needed:
                        chosen_defects.extend(random.sample(train_defects, needed))
                    else:
                        chosen_defects.extend(train_defects)
                        remaining = k_shot - len(chosen_defects)
                        if remaining > 0:
                            pool = [d for d in data_list if d not in chosen_defects]
                            if len(pool) >= remaining:
                                chosen_defects.extend(random.sample(pool, remaining))
                            else:
                                chosen_defects.extend(pool)

                n_goods = goods_per_class if goods_per_class is not None else max(k_shot, 50)
                goods_pool = test_goods if include_test_defects and len(test_goods) > 0 else train_goods

                if len(goods_pool) >= n_goods:
                    chosen_goods = random.sample(goods_pool, n_goods)
                else:
                    union_pool = []
                    if "train" in meta_info and cls in meta_info["train"]:
                        union_pool.extend([d for d in meta_info["train"][cls] if int(d.get("anomaly", 0)) == 0])
                    if "test" in meta_info and cls in meta_info["test"]:
                        union_pool.extend([d for d in meta_info["test"][cls] if int(d.get("anomaly", 0)) == 0])
                    seen = set()
                    unique_union = []
                    for d in union_pool:
                        key = (d["img_path"], d.get("mask_path", ""))
                        if key not in seen:
                            seen.add(key)
                            unique_union.append(d)
                    union_pool = unique_union

                    if len(union_pool) >= n_goods:
                        chosen_goods = random.sample(union_pool, n_goods)
                    else:
                        chosen_goods = union_pool.copy()

                chosen = chosen_defects + chosen_goods
                random.shuffle(chosen)

            else:
                # default: use all entries from the provided split
                chosen = data_list

            # Append validated entries to self.entries (ensure we only add items with mask_path for anomalies)
            for d in chosen:
                # If anomaly flagged but mask missing, don't add as anomaly (skip to avoid later issues)
                if int(d.get("anomaly", 0)) == 1:
                    if not d.get("mask_path"):
                        continue
                    full_maskp = os.path.join(self.root, d.get("mask_path", ""))
                    if not os.path.exists(full_maskp):
                        continue
                self.entries.append(
                    SampleEntry(
                        img_path=d["img_path"],
                        mask_path=d.get("mask_path", ""),
                        cls_name=d["cls_name"],
                        anomaly=int(d.get("anomaly", 0)),
                        specie_name=d.get("specie_name", d["cls_name"]),
                    )
                )

        # cache class-wise test paths for mosaic augmentation
        self.test_cache = split_meta if "test" in meta_info else meta_info.get("test", {})


    def __len__(self) -> int:
        return len(self.entries)

    def _combine_img(self, cls_name: str) -> Tuple[Image.Image, Image.Image]:
        """Mimic mvtec_supervised combine_img: 2x2 mosaic from random test defects."""
        img_paths_root = os.path.join(self.root, cls_name, "test")
        img_ls, mask_ls = [], []
        defects = os.listdir(img_paths_root)
        for _ in range(4):
            defect = random.choice(defects)
            files = os.listdir(os.path.join(img_paths_root, defect))
            random_file = random.choice(files)
            img_path = os.path.join(img_paths_root, defect, random_file)
            mask_path = os.path.join(
                self.root, cls_name, "ground_truth", defect, random_file[:3] + "_mask.png"
            )
            img = Image.open(img_path).convert("RGB")
            img_ls.append(img)
            if defect == "good":
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            else:
                mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
                img_mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
            mask_ls.append(img_mask)

        w, h = img_ls[0].size
        result_image = Image.new("RGB", (2 * w, 2 * h))
        result_mask = Image.new("L", (2 * w, 2 * h))
        for i, (img, msk) in enumerate(zip(img_ls, mask_ls)):
            row, col = divmod(i, 2)
            x, y = col * w, row * h
            result_image.paste(img, (x, y))
            result_mask.paste(msk, (x, y))
        return result_image, result_mask

    def __getitem__(self, idx: int):
        data = self.entries[idx]
        img_path = os.path.join(self.root, data.img_path)
        mask_path = os.path.join(self.root, data.mask_path) if data.mask_path else None
        cls_name = data.cls_name
        is_anomaly = data.anomaly != 0

        try:
            if random.random() < self.aug_rate:
                img, img_mask = self._combine_img(cls_name)
            else:
                img = Image.open(img_path).convert("RGB")
                if not is_anomaly or mask_path is None or not os.path.exists(mask_path):
                    img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
                    is_anomaly = False
                else:
                    mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
                    img_mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")

            img = self.image_transform(img)
            img_mask = self.mask_transform(img_mask)
        except (OSError, ValueError) as e:
            # log skipped file for troubleshooting
            print(f"[WARN] Skip corrupted sample idx={idx} img={img_path} mask={mask_path} err={e}")
            # fallback to next sample to avoid worker crash on truncated images
            return self.__getitem__((idx + 1) % len(self.entries))
        specie_name = getattr(data, "specie_name", "") or ""

        # Build prompt list (dual-template style)
        # - first element is ALWAYS a short state+cls template (keeps SAM3 prompt style)
        # - remaining elements are class-agnostic defect descriptors (used for grouping / contrastive)
        def _norm(s: str) -> str:
            return (s or "").strip().lower().replace("_", " ")

        def _select_defect_keywords(cls_name: str, specie_name: str):
            """Pick a small set of defect keywords for this sample.
            Priority:
              1) specie_name words (from meta.json)
              2) prompt_dict[cls_name] filtered by specie_name words (if available)
              3) fallback to a small generic set
            """
            kws = []
            sp = _norm(specie_name)
            if sp and sp not in ("good", "normal", "ok"):
                kws.append(sp)

            # pull candidates from prompt_dict (usually class->list[str])
            candidates = None
            if self.prompt_dict is not None:
                candidates = (
                    self.prompt_dict.get(cls_name)
                    or self.prompt_dict.get(cls_name.lower())
                    or self.prompt_dict.get(cls_name.replace("_", " "))
                    or self.prompt_dict.get(cls_name.replace("_", " ").lower())
                )
            if isinstance(candidates, str):
                candidates = [w.strip() for w in candidates.split(",") if w.strip()]
            if candidates:
                # if we have specie_name, try to filter
                if sp:
                    sp_words = [w for w in sp.split(" ") if w]
                    filtered = [c for c in candidates if any(w in _norm(c) for w in sp_words)]
                    if filtered:
                        candidates = filtered
                # add a few candidates
                for c in candidates:
                    c2 = _norm(c)
                    if c2 and c2 not in kws:
                        kws.append(c2)
                    if len(kws) >= 4:
                        break

            if not kws:
                kws = ["defect", "flaw", "crack", "broken"]

            return kws[:4]

        if is_anomaly:
            # positive: damaged template + defect descriptors (specie_name / keywords)
            prompt_list = [f"anomaly {cls_name}", "damaged", "defect"]
            prompt_list.extend(_select_defect_keywords(cls_name, specie_name))
        else:
            # positive: normal template (keep short)
            prompt_list = [f"normal {cls_name}", "perfect", "good"]

        return img, img_mask, prompt_list, is_anomaly, cls_name, specie_name


# ==================== VisA Dataset ====================

class VisADataset(Dataset):
    """Dataset for VisA (Visual Anomaly) benchmark.
    
    Reads from CSV file (1cls.csv format) with columns:
    - object: class name (candle, capsules, etc.)
    - split: train/test
    - label: normal/anomaly
    - image: image path relative to root
    - mask: mask path relative to root (empty for normal samples)
    
    Returns: (image_tensor, mask_tensor, prompt_list, is_anomaly, cls_name, specie_name)
    """
    
    def __init__(
        self,
        root: str,
        csv_path: str,
        mode: str = "test",  # "train" or "test"
        obj_name: Optional[str] = None,  # filter by class name
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = root
        img_tf, mask_tf = _default_transforms()
        self.image_transform = image_transform or img_tf
        self.mask_transform = mask_transform or mask_tf
        
        # Parse CSV
        import csv
        self.entries: List[SampleEntry] = []
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = row.get("object", "").strip()
                split = row.get("split", "").strip().lower()
                label = row.get("label", "").strip().lower()
                img_path = row.get("image", "").strip()
                mask_path = row.get("mask", "").strip()
                
                # Filter by split
                if split != mode.lower():
                    continue
                    
                # Filter by class name if specified
                if obj_name is not None and obj != obj_name:
                    continue
                
                # Determine anomaly status
                is_anomaly = 1 if label == "anomaly" else 0
                
                # specie_name: for VisA, we don't have defect types, use "anomaly" or "normal"
                specie_name = "anomaly" if is_anomaly else "normal"
                
                self.entries.append(SampleEntry(
                    img_path=img_path,
                    mask_path=mask_path if mask_path else "",
                    cls_name=obj,
                    anomaly=is_anomaly,
                    specie_name=specie_name,
                ))
        
        print(f"[INFO] VisADataset: loaded {len(self.entries)} samples from {csv_path} (mode={mode})")
        
        # Count by class
        cls_counts = {}
        for e in self.entries:
            cls_counts[e.cls_name] = cls_counts.get(e.cls_name, 0) + 1
        for cls, cnt in sorted(cls_counts.items()):
            anom_cnt = sum(1 for e in self.entries if e.cls_name == cls and e.anomaly)
            norm_cnt = cnt - anom_cnt
            print(f"  {cls}: {cnt} samples (anomaly={anom_cnt}, normal={norm_cnt})")
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int):
        data = self.entries[idx]
        img_path = os.path.join(self.root, data.img_path)
        mask_path = os.path.join(self.root, data.mask_path) if data.mask_path else None
        cls_name = data.cls_name
        is_anomaly = data.anomaly != 0
        specie_name = data.specie_name
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            if not is_anomaly or mask_path is None or not os.path.exists(mask_path):
                # Normal sample: empty mask
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
                is_anomaly = False
            else:
                # Anomaly sample: load mask
                mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
                img_mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
            
            img = self.image_transform(img)
            img_mask = self.mask_transform(img_mask)
            
        except (OSError, ValueError) as e:
            print(f"[WARN] Skip corrupted sample idx={idx} img={img_path} mask={mask_path} err={e}")
            return self.__getitem__((idx + 1) % len(self.entries))
        
        # Build prompt list for VisA
        # VisA doesn't have detailed specie_name, so we use simple templates
        if is_anomaly:
            prompt_list = [f"damaged {cls_name}", "defect", "flaw", "anomaly"]
        else:
            prompt_list = [f"normal {cls_name}", "clean", "intact"]
        
        return img, img_mask, prompt_list, is_anomaly, cls_name, specie_name