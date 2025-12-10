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
        super().__init__()
        self.root = root
        self.aug_rate = aug_rate
        self.prompt_dict = prompt_dict or mvtec_short_keywords
        img_tf, mask_tf = _default_transforms()
        self.image_transform = image_transform or img_tf
        self.mask_transform = mask_transform or mask_tf

        # persist new flags for use in selection logic:
        self.train_from_test = train_from_test
        self.specie_split_ratio = specie_split_ratio
        self.specie_split_seed = specie_split_seed

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
            
            if getattr(self, "train_from_test", False):
                test_meta_for_cls = meta_info.get("test", {}).get(cls, [])
                # group only defects by specie_name
                specie_map = {}
                for d in test_meta_for_cls:
                    if int(d.get("anomaly", 0)) != 1:
                        continue
                    specie = d.get("specie_name", d.get("cls_name", cls))
                    specie_map.setdefault(specie, []).append(d)

                rng = random.Random(getattr(self, "specie_split_seed", 42))
                class_train, class_test = [], []

                for specie, items in specie_map.items():
                    rng.shuffle(items)
                    n = len(items)
                    if n == 0:
                        continue
                    # determine train count: round(n * ratio), ensure at least 1 train if n>0 and at least 1 test if n>1
                    if n == 1:
                        n_train = 1
                    else:
                        n_train = int(max(1, round(n * getattr(self, "specie_split_ratio", 0.8))))
                        if n_train >= n:
                            n_train = n - 1
                    train_items = items[:n_train]
                    test_items = items[n_train:]
                    class_train.extend(train_items)
                    class_test.extend(test_items)

                if mode == "train":
                    chosen = class_train.copy()
                else:
                    chosen = class_test.copy()

                # save splits for reproducibility
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    out_map_path = os.path.join(save_dir, f"specie_splits_{cls}.json")
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

            elif mode in ("train", "train_all", "test") and k_shot > 0:
                # === REPLACEMENT: use test-split defects only, per-specie 80/20 split ===
                # 新增 __init__ 参数（请保证函数签名包含这些参数）:
                #    train_from_test: bool = False,
                #    specie_split_ratio: float = 0.8,
                #    specie_split_seed: int = 42,
                #
                # 语义：
                #  - 当 train_from_test=True 且 mode == "train" 时:
                #       -> 使用 meta_info['test'][cls] 中的缺陷样本（anomaly==1），
                #       -> 按 specie_name 分组，对每个组按 specie_split_ratio 随机切分为 train / test，
                #       -> 将 train 子集加入 self.entries
                #  - 当 train_from_test=True 且 mode == "test" 时:
                #       -> 使用上面分割的 test 子集作为测试集合（同一 split seed 保证对应）
                #
                # 如果 train_from_test=False，保留原来的 k_shot/goods 逻辑（兼容旧行为）。

                # read new flags from local variables (ensure __init__ signature contains them)
                # train_from_test: use test split defects to build train/test per-specie
                train_from_test = getattr(self, "train_from_test", False)
                specie_split_ratio = getattr(self, "specie_split_ratio", 0.8)
                specie_split_seed = getattr(self, "specie_split_seed", 42)

                if train_from_test:
                    # Build per-class per-specie splits from meta_info['test'][cls]
                    chosen = []
                    # get test list for this class (may be empty)
                    test_meta_for_cls = meta_info.get("test", {}).get(cls, [])
                    # filter only defect entries (anomaly==1) and group by specie_name
                    specie_map = {}
                    for d in test_meta_for_cls:
                        if int(d.get("anomaly", 0)) != 1:
                            continue
                        specie = d.get("specie_name", d.get("cls_name", cls))
                        specie_map.setdefault(specie, []).append(d)

                    # deterministic RNG
                    rng = random.Random(specie_split_seed)

                    # we'll accumulate chosen train/test lists for this class
                    class_train = []
                    class_test = []

                    for specie, items in specie_map.items():
                        # shuffle deterministically
                        rng.shuffle(items)
                        n = len(items)
                        if n == 0:
                            continue
                        # number of train samples (at least 1 when n>0)
                        if n == 1:
                            n_train = 1
                        else:
                            n_train = int(max(1, round(n * specie_split_ratio)))
                            # ensure at least one left for test if n>1 and ratio yields all
                            if n_train >= n:
                                n_train = n - 1
                        train_items = items[:n_train]
                        test_items = items[n_train:]
                        class_train.extend(train_items)
                        class_test.extend(test_items)

                    if mode == "train":
                        chosen = class_train.copy()
                    else:
                        # mode == "test" -> use the test portion
                        chosen = class_test.copy()

                    # if save_dir is provided, write out the per-class per-specie split mapping (for reproducibility)
                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        out_map_path = os.path.join(save_dir, f"specie_splits_{cls}.json")
                        # prepare serializable dict
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

                else:
                    # fallback to original k_shot handling if not using test-derived defects
                    if mode in ("train", "train_all", "test") and k_shot > 0:
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
                                union_pool.extend([d for d in meta_info["train"][cls] if int(d.get("anomaly",0))==0])
                            if "test" in meta_info and cls in meta_info["test"]:
                                union_pool.extend([d for d in meta_info["test"][cls] if int(d.get("anomaly",0))==0])
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
