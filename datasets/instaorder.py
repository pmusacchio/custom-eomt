from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import pil_to_tensor
from pycocotools import mask as coco_mask

from datasets.instaorder_utils import (
    load_json,
    build_instaorder_lookup,
    merge_instaorder_with_coco_records,
)


class InstaOrderDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        coco_json_path: str | Path,
        instaorder_json_path: str | Path,
        transforms: Callable | None = None,
        class_mapping: dict[int, int] | None = None,
        include_occlusion: bool = True,
        include_depth: bool = False,
        remove_occ_bidirec: bool = False,
        remove_depth_overlap: bool = False,
        check_empty_targets: bool = True,
    ) -> None:
        super().__init__()
        self.image_root = Path(image_root)
        self.transforms = transforms
        self.class_mapping = class_mapping or {}
        self.include_occlusion = include_occlusion
        self.include_depth = include_depth
        self.check_empty_targets = check_empty_targets

        coco_records = self._load_coco_records(coco_json_path)

        instaorder_by_image_id = build_instaorder_lookup(
            instaorder_json_path,
            include_occlusion=include_occlusion,
            include_depth=include_depth,
            remove_occ_bidirec=remove_occ_bidirec,
            remove_depth_overlap=remove_depth_overlap,
        )

        records = merge_instaorder_with_coco_records(
            coco_records,
            instaorder_by_image_id,
        )

        if self.check_empty_targets:
            records = [
                record for record in records if self._has_valid_annotations(record)
            ]

        self.records = records

    @staticmethod
    def _load_coco_records(coco_json_path: str | Path) -> list[dict[str, Any]]:
        data = load_json(coco_json_path)

        images_by_id = {img["id"]: img for img in data["images"]}

        anns_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image_id[ann["image_id"]].append(ann)

        records: list[dict[str, Any]] = []
        for image_id, image_info in images_by_id.items():
            records.append(
                {
                    "image_id": image_id,
                    "file_name": image_info["file_name"],
                    "height": image_info["height"],
                    "width": image_info["width"],
                    "annotations": anns_by_image_id.get(image_id, []),
                }
            )

        return records

    def _has_valid_annotations(self, record: dict[str, Any]) -> bool:
        for ann in record["annotations"]:
            if ann["category_id"] in self.class_mapping:
                return True
        return False

    @staticmethod
    def _decode_segmentation(
        segmentation: Any,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if isinstance(segmentation, list):
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
        elif isinstance(segmentation, dict):
            if isinstance(segmentation["counts"], list):
                rle = coco_mask.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation
        else:
            raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

        mask = coco_mask.decode(rle)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)

        return torch.as_tensor(mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]

        image = Image.open(self.image_root / record["file_name"]).convert("RGB")
        image = tv_tensors.Image(pil_to_tensor(image))

        masks: list[torch.Tensor] = []
        labels: list[int] = []
        is_crowd: list[bool] = []
        kept_positions: list[int] = []
        kept_ann_ids: list[int] = []

        for pos, ann in enumerate(record["annotations"]):
            cls_id = ann["category_id"]
            if cls_id not in self.class_mapping:
                continue

            mask = self._decode_segmentation(
                ann["segmentation"],
                record["height"],
                record["width"],
            )

            if mask.sum() == 0:
                continue

            masks.append(mask)
            labels.append(self.class_mapping[cls_id])
            is_crowd.append(bool(ann.get("iscrowd", 0)))
            kept_positions.append(pos)
            kept_ann_ids.append(int(ann["id"]))

        if len(masks) == 0:
            if self.check_empty_targets:
                raise RuntimeError(
                    f"No valid annotations left for image_id={record['image_id']} after filtering."
                )

            masks_tensor = torch.zeros(
                (0, record["height"], record["width"]),
                dtype=torch.bool,
            )
        else:
            masks_tensor = torch.stack(masks, dim=0)

        target: dict[str, Any] = {
            "masks": tv_tensors.Mask(masks_tensor),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "is_crowd": torch.as_tensor(is_crowd, dtype=torch.bool),
            "image_id": torch.tensor(record["image_id"], dtype=torch.long),
            "annotation_ids": torch.as_tensor(kept_ann_ids, dtype=torch.long),
        }

        if self.include_occlusion and "occlusion" in record:
            occlusion = record["occlusion"]
            occlusion = occlusion[np.ix_(kept_positions, kept_positions)]
            target["occlusion"] = torch.as_tensor(occlusion, dtype=torch.float32)

        if self.include_depth and "depth" in record:
            depth = record["depth"][np.ix_(kept_positions, kept_positions)]
            overlap = record["overlap"][np.ix_(kept_positions, kept_positions)]
            count = record["count"][np.ix_(kept_positions, kept_positions)]

            target["depth"] = torch.as_tensor(depth, dtype=torch.long)
            target["overlap"] = torch.as_tensor(overlap, dtype=torch.long)
            target["count"] = torch.as_tensor(count, dtype=torch.long)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
