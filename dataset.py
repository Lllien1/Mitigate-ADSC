"""
dataset.py

MVTec-AD 和 VisA 数据集加载器

支持两种prompt模式（通过 prompt_mode 参数控制）：
- "simple": 简化版，只有 ["{anomaly/normal} {cls_name}"]
           适用于 CoOp/CoCoOp，让可学习向量真正学习语义
- "full": 完整版，包含关键词 ["{anomaly/normal} {cls_name}", "kw1", "kw2", ...]
         适用于需要细粒度specie区分的场景
"""

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
    """Dataset that mirrors FiLo `mvtec_supervised.py` meta.json sampling.

    支持两种prompt模式:
    - prompt_mode="simple": ["{anomaly/normal} {cls_name}"]
    - prompt_mode="full": ["{anomaly/normal} {cls_name}", "kw1", "kw2", ...]
    
    Returns: (image_tensor, mask_tensor, prompt_list, is_anomaly, cls_name, specie_name)
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
        train_from_test: bool = False,
        specie_split_ratio: float = 0.8,
        specie_split_seed: int = 42,
        # ========== 新增: prompt模式 ==========
        prompt_mode: str = "simple",  # "simple" 或 "full"
    ) -> None:
        """
        Args:
            prompt_mode: 
                "simple" - 简化版prompt，只有类别描述 ["{anomaly/normal} {cls_name}"]
                           推荐用于CoOp/CoCoOp，让可学习向量学习通用语义
                "full" - 完整版prompt，包含关键词 ["{anomaly/normal} {cls_name}", "kw1", ...]
                         适用于需要细粒度specie区分的场景
        """
        super().__init__()
        self.root = root
        self.aug_rate = aug_rate
        self.prompt_dict = prompt_dict or mvtec_short_keywords
        self.prompt_mode = prompt_mode.lower()
        
        img_tf, mask_tf = _default_transforms()
        self.image_transform = image_transform or img_tf
        self.mask_transform = mask_transform or mask_tf

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

            # ==== Case 1: train_from_test mode ====
            if self.train_from_test:
                chosen: List[Dict] = []

                if self.save_dir is not None:
                    split_file = os.path.join(self.save_dir, f"specie_splits_{cls}.json")
                    if os.path.exists(split_file):
                        try:
                            with open(split_file, "r", encoding="utf-8") as f:
                                js = json.load(f)
                            sel_list = js.get("train", []) if mode == "train" else js.get("test", [])
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
                            chosen = []

                if not chosen:
                    test_data = meta_info.get("test", {}).get(cls, [])
                    train_goods_data = [d for d in data_list if int(d.get("anomaly", 0)) == 0]

                    from collections import defaultdict
                    by_specie = defaultdict(list)
                    for d in test_data:
                        sp = d.get("specie_name", d.get("cls_name", "unknown"))
                        by_specie[sp].append(d)

                    train_items = []
                    test_items = []
                    rng = random.Random(self.specie_split_seed)
                    for sp, items in by_specie.items():
                        rng.shuffle(items)
                        n_train = max(1, int(len(items) * self.specie_split_ratio))
                        train_items.extend(items[:n_train])
                        test_items.extend(items[n_train:])

                    # add goods from train split
                    train_items.extend(train_goods_data)

                    if self.save_dir is not None:
                        split_file = os.path.join(self.save_dir, f"specie_splits_{cls}.json")
                        try:
                            with open(split_file, "w", encoding="utf-8") as f:
                                json.dump({"train": train_items, "test": test_items}, f, indent=2)
                        except Exception:
                            pass

                    chosen = train_items if mode == "train" else test_items

            # ==== Case 2: k_shot mode ====
            elif k_shot > 0:
                train_goods = [d for d in data_list if int(d.get("anomaly", 0)) == 0]
                train_defects = [d for d in data_list if int(d.get("anomaly", 0)) == 1]

                test_meta = meta_info.get("test", {}).get(cls, [])
                test_goods = [d for d in test_meta if int(d.get("anomaly", 0)) == 0]
                test_defects = [d for d in test_meta if int(d.get("anomaly", 0)) == 1]

                if include_test_defects:
                    if len(test_defects) >= k_shot:
                        chosen_defects = random.sample(test_defects, k_shot)
                    else:
                        chosen_defects = test_defects.copy()
                        remaining = k_shot - len(chosen_defects)
                        if remaining > 0:
                            pool = train_defects + [d for d in test_defects if d not in chosen_defects]
                            if len(pool) >= remaining:
                                chosen_defects.extend(random.sample(pool, remaining))
                            else:
                                chosen_defects.extend(pool)
                else:
                    needed = k_shot
                    if len(train_defects) >= needed:
                        chosen_defects = random.sample(train_defects, needed)
                    else:
                        chosen_defects = train_defects.copy()
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
                chosen = data_list

            for d in chosen:
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

        self.test_cache = split_meta if "test" in meta_info else meta_info.get("test", {})
        
        print(f"[MVTecMetaDataset] Loaded {len(self.entries)} samples, prompt_mode='{self.prompt_mode}'")

    def __len__(self) -> int:
        return len(self.entries)

    def _combine_img(self, cls_name: str) -> Tuple[Image.Image, Image.Image]:
        """2x2 mosaic from random test defects."""
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
            print(f"[WARN] Skip corrupted sample idx={idx} img={img_path} mask={mask_path} err={e}")
            return self.__getitem__((idx + 1) % len(self.entries))
        
        specie_name = getattr(data, "specie_name", "") or ""

        # ========== 构建 prompt_list ==========
        if self.prompt_mode == "simple":
            # 简化版: 只有类别描述
            # 适用于 CoOp/CoCoOp，让可学习向量学习通用语义
            if is_anomaly:
                prompt_list = [f"anomaly {cls_name}"]
            else:
                prompt_list = [f"normal {cls_name}"]
        else:
            # 完整版: 包含关键词
            # 适用于需要细粒度specie区分的场景
            def _norm(s: str) -> str:
                return (s or "").strip().lower().replace("_", " ")

            def _select_defect_keywords(cls_name: str, specie_name: str):
                kws = []
                sp = _norm(specie_name)
                if sp and sp not in ("good", "normal", "ok"):
                    kws.append(sp)

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
                    if sp:
                        sp_words = [w for w in sp.split(" ") if w]
                        filtered = [c for c in candidates if any(w in _norm(c) for w in sp_words)]
                        if filtered:
                            candidates = filtered
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
                prompt_list = [f"anomaly {cls_name}", "damaged", "defect"]
                prompt_list.extend(_select_defect_keywords(cls_name, specie_name))
            else:
                prompt_list = [f"normal {cls_name}", "perfect", "good"]

        return img, img_mask, prompt_list, is_anomaly, cls_name, specie_name


# ==================== VisA Dataset ====================

class VisADataset(Dataset):
    """Dataset for VisA (Visual Anomaly) benchmark.
    
    支持两种prompt模式:
    - prompt_mode="simple": ["{anomaly/normal} {cls_name}"]
    - prompt_mode="full": ["{anomaly/normal} {cls_name}", "kw1", ...]
    """
    
    def __init__(
        self,
        root: str,
        csv_path: str,
        mode: str = "test",
        obj_name: Optional[str] = None,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        prompt_mode: str = "simple",  # 新增
        missing_mask_behavior: str = "error",
    ) -> None:
        super().__init__()
        self.root = root
        self.prompt_mode = prompt_mode.lower()
        self.missing_mask_behavior = str(missing_mask_behavior).lower()
        self._missing_mask_warned = 0
        
        img_tf, mask_tf = _default_transforms()
        self.image_transform = image_transform or img_tf
        self.mask_transform = mask_transform or mask_tf
        
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
                
                if split != mode.lower():
                    continue
                    
                if obj_name is not None and obj != obj_name:
                    continue
                
                is_anomaly = 1 if label == "anomaly" else 0
                specie_name = "anomaly" if is_anomaly else "normal"
                
                self.entries.append(SampleEntry(
                    img_path=img_path,
                    mask_path=mask_path if mask_path else "",
                    cls_name=obj,
                    anomaly=is_anomaly,
                    specie_name=specie_name,
                ))
        
        print(f"[VisADataset] Loaded {len(self.entries)} samples, prompt_mode='{self.prompt_mode}'")
        
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

            mask_missing = (mask_path is None) or (not os.path.exists(mask_path))
            if is_anomaly and mask_missing:
                beh = self.missing_mask_behavior
                if beh == "error":
                    raise FileNotFoundError(f"VisA anomaly sample missing mask file: {mask_path}")
                if beh == "skip":
                    if self._missing_mask_warned < 50:
                        print(f"[WARN] VisA missing mask (skip): idx={idx} img={img_path} mask={mask_path}")
                        self._missing_mask_warned += 1
                    return self.__getitem__((idx + 1) % len(self.entries))
                if beh == "flip_to_normal":
                    if self._missing_mask_warned < 50:
                        print(f"[WARN] VisA missing mask (flip_to_normal): idx={idx} img={img_path} mask={mask_path}")
                        self._missing_mask_warned += 1
                    is_anomaly = False

            if (not is_anomaly) or mask_missing:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
            else:
                mask_arr = np.array(Image.open(mask_path).convert("L")) > 0
                img_mask = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode="L")
            
            img = self.image_transform(img)
            img_mask = self.mask_transform(img_mask)
            
        except (OSError, ValueError) as e:
            print(f"[WARN] Skip corrupted sample idx={idx} img={img_path} mask={mask_path} err={e}")
            return self.__getitem__((idx + 1) % len(self.entries))
        
        # ========== 构建 prompt_list ==========
        if self.prompt_mode == "simple":
            if is_anomaly:
                prompt_list = [f"anomaly {cls_name}"]
            else:
                prompt_list = [f"normal {cls_name}"]
        else:
            if is_anomaly:
                prompt_list = [f"damaged {cls_name}", "defect", "flaw", "anomaly"]
            else:
                prompt_list = [f"normal {cls_name}", "clean", "intact"]
        
        return img, img_mask, prompt_list, is_anomaly, cls_name, specie_name
