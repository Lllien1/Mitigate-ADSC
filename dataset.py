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
    ) -> None:
        super().__init__()
        self.root = root
        self.aug_rate = aug_rate
        self.prompt_dict = prompt_dict or mvtec_short_keywords
        img_tf, mask_tf = _default_transforms()
        self.image_transform = image_transform or img_tf
        self.mask_transform = mask_transform or mask_tf

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
            # === REPLACE k-shot SECTION in MVTecMetaDataset.__init__ ===
            # 新增参数说明（在 __init__ signature）： include_test_defects: bool = False, goods_per_class: int = 50
            # (如果还没有，请在函数定义处添加对应默认参数)
            #
            # 替换原有 k_shot 处理逻辑为 defect-first + goods 补齐（优先取 test 中缺陷，并在无 good 时从 train 找 good）
            
            if mode in ("train", "train_all", "test") and k_shot > 0:
                # data_list: all entries in the 'train' split for this class (since current code chooses split_meta earlier)
                # But we want optionally to include defects from test -> we'll handle that in a mode-agnostic way:
                # Build three pools:
                #   - train_defects: defects from train split (if any)
                #   - test_defects: defects from test split (meta_info['test']) if available and include_test_defects True
                #   - goods_pool: goods from test if available else goods from train
                train_defects = [d for d in data_list if int(d.get("anomaly", 0)) == 1 and d.get("mask_path")]
                train_goods = [d for d in data_list if int(d.get("anomaly", 0)) == 0]
            
                test_defects = []
                test_goods = []
                # If meta_info has 'test' we can use it for defect harvesting
                if include_test_defects:
                    test_meta = None
                    # meta_info variable in outer scope holds whole meta; we can read meta_info['test'][cls] if exists
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
            
                # Build chosen list:
                chosen = []
            
                # 1) defects selection: prefer test defects (all), then train defects to reach k_shot if necessary.
                # Per your requirement: "将 test 下 cls 中的所有缺陷都划分进来", so include all test defects.
                if include_test_defects and len(test_defects) > 0:
                    chosen_defects = test_defects.copy()
                else:
                    chosen_defects = []
            
                # If we want at least k_shot defect examples per specie, supplement from train_defects
                if len(chosen_defects) < k_shot:
                    needed = k_shot - len(chosen_defects)
                    # sample from train_defects if available, otherwise sample from train overall
                    if len(train_defects) >= needed:
                        chosen_defects.extend(random.sample(train_defects, needed))
                    else:
                        chosen_defects.extend(train_defects)
                        # If still not enough, sample from data_list (possibly goods) as fallback
                        remaining = k_shot - len(chosen_defects)
                        if remaining > 0:
                            pool = [d for d in data_list if d not in chosen_defects]
                            if len(pool) >= remaining:
                                chosen_defects.extend(random.sample(pool, remaining))
                            else:
                                chosen_defects.extend(pool)
            
                # 2) goods selection: we want some goods per class to balance; prefer test goods if exist, else use train_goods
                n_goods = goods_per_class if goods_per_class is not None else max(k_shot, 50)
                goods_pool = []
                if include_test_defects and len(test_goods) > 0:
                    goods_pool = test_goods
                else:
                    goods_pool = train_goods
            
                if len(goods_pool) >= n_goods:
                    chosen_goods = random.sample(goods_pool, n_goods)
                else:
                    # If insufficient goods in the chosen pool, fallback to entire train/test union
                    union_pool = []
                    if "train" in meta_info and cls in meta_info["train"]:
                        union_pool.extend([d for d in meta_info["train"][cls] if int(d.get("anomaly",0))==0])
                    if "test" in meta_info and cls in meta_info["test"]:
                        union_pool.extend([d for d in meta_info["test"][cls] if int(d.get("anomaly",0))==0])

                    # 去重（按 img_path + mask_path）
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

                # Merge defects + goods and shuffle
                chosen = chosen_defects + chosen_goods
                random.shuffle(chosen)
                # Optionally save k_shot defect list
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, f"k_shot_{cls}.txt"), "a", encoding="utf-8") as f:
                        for d in chosen_defects:
                            f.write(d["img_path"] + "\n")
            else:
                chosen = data_list
            # === END REPLACEMENT ===
            
            for d in chosen:
                self.entries.append(
                    SampleEntry(
                        img_path=d["img_path"],
                        mask_path=d["mask_path"],
                        cls_name=d["cls_name"],
                        anomaly=int(d["anomaly"]),
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

        # Build prompt list
        if is_anomaly:
            prompt_list = [cls_name] + self.prompt_dict.get(cls_name, [])
        else:
            prompt_list = ["normal", "clean", cls_name]

        return img, img_mask, prompt_list, is_anomaly, cls_name
