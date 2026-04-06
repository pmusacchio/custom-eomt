from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(json_path: str | Path) -> dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)


def get_occlusion_matrix(
    gt_occlusion: list[dict[str, Any]],
    nb_instances: int,
    rm_bidirec: bool = False,
) -> np.ndarray:
    """
    Build an NxN occlusion matrix.

    Convention:
    - diagonal = -1
    - 1 means row instance occludes column instance
    - 0 means no annotated occlusion relation
    - -1 is used for ignored / invalid entries
    """
    gt_occ_matrix = np.zeros((nb_instances, nb_instances), dtype=np.int64)
    np.fill_diagonal(gt_occ_matrix, -1)

    if len(gt_occlusion) == 0:
        return gt_occ_matrix

    for relation in gt_occlusion:
        order = relation["order"]

        if "&" in order and rm_bidirec:
            instance1, instance2 = map(int, order.split(" & ")[0].split("<"))
            gt_occ_matrix[instance1, instance2] = -1
            gt_occ_matrix[instance2, instance1] = -1

        elif "&" in order:
            instance1, instance2 = map(int, order.split(" & ")[0].split("<"))
            gt_occ_matrix[instance1, instance2] = 1
            gt_occ_matrix[instance2, instance1] = 1

        else:
            instance1, instance2 = map(int, order.split("<"))
            gt_occ_matrix[instance1, instance2] = 1

    return gt_occ_matrix


def get_depth_overlap_count_matrices(
    gt_depth: list[dict[str, Any]],
    nb_instances: int,
    rm_overlap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build NxN depth / overlap / count matrices.

    Depth convention:
    - diagonal = -1
    - 1 means row instance is in front of column instance
    - 0 means row instance is behind column instance
    - 2 means equal depth
    """
    gt_depth_matrix = np.zeros((nb_instances, nb_instances), dtype=np.int64)
    is_overlap_matrix = np.zeros((nb_instances, nb_instances), dtype=np.int64)
    count_matrix = np.ones((nb_instances, nb_instances), dtype=np.int64)

    np.fill_diagonal(gt_depth_matrix, -1)
    np.fill_diagonal(is_overlap_matrix, -1)
    np.fill_diagonal(count_matrix, -1)

    if len(gt_depth) == 0:
        return gt_depth_matrix, is_overlap_matrix, count_matrix

    for overlap_count in gt_depth:
        depth_order = overlap_count["order"]
        is_overlap = overlap_count["overlap"]
        count = overlap_count["count"]

        split_char = "<" if "<" in depth_order else "="
        idx1, idx2 = map(int, depth_order.split(split_char))

        if rm_overlap and is_overlap:
            is_overlap_matrix[idx1, idx2] = -1
            is_overlap_matrix[idx2, idx1] = -1
        elif is_overlap:
            is_overlap_matrix[idx1, idx2] = 1
            is_overlap_matrix[idx2, idx1] = 1
        else:
            is_overlap_matrix[idx1, idx2] = 0
            is_overlap_matrix[idx2, idx1] = 0

        if split_char == "<":
            gt_depth_matrix[idx1, idx2] = 1
            gt_depth_matrix[idx2, idx1] = 0
        elif split_char == "=":
            gt_depth_matrix[idx1, idx2] = 2
            gt_depth_matrix[idx2, idx1] = 2

        count_matrix[idx1, idx2] = count
        count_matrix[idx2, idx1] = count

    return gt_depth_matrix, is_overlap_matrix, count_matrix


def build_instaorder_lookup(
    json_path: str | Path,
    include_occlusion: bool = True,
    include_depth: bool = True,
    remove_occ_bidirec: bool = False,
    remove_depth_overlap: bool = False,
) -> dict[int, dict[str, Any]]:
    """
    Build a dict keyed by image_id.

    Each value contains:
    - instance_ids: list[int]
    - optionally occlusion: np.ndarray [N, N]
    - optionally depth / overlap / count: np.ndarray [N, N]
    """
    raw = load_json(json_path)["annotations"]

    lookup: dict[int, dict[str, Any]] = {}
    for sample in raw:
        instance_ids = sample["instance_ids"]
        nb_instances = len(instance_ids)

        item: dict[str, Any] = {
            "instance_ids": instance_ids,
        }

        if include_occlusion:
            item["occlusion"] = get_occlusion_matrix(
                sample["occlusion"],
                nb_instances,
                rm_bidirec=remove_occ_bidirec,
            )

        if include_depth:
            depth, overlap, count = get_depth_overlap_count_matrices(
                sample["depth"],
                nb_instances,
                rm_overlap=remove_depth_overlap,
            )
            item["depth"] = depth
            item["overlap"] = overlap
            item["count"] = count

        lookup[sample["image_id"]] = item

    return lookup


def filter_coco_annotations_by_instance_ids(
    annotations: list[dict[str, Any]],
    instance_ids: list[int],
) -> list[dict[str, Any]]:
    """
    Keep only COCO annotations whose annotation id is in instance_ids,
    preserving the exact order of instance_ids.
    """
    ann_by_id = {ann["id"]: ann for ann in annotations}
    return [ann_by_id[ann_id] for ann_id in instance_ids if ann_id in ann_by_id]


def merge_instaorder_with_coco_records(
    coco_records: list[dict[str, Any]],
    instaorder_by_image_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge COCO per-image records with InstaOrder relation info.

    Output records are restricted to images that exist in InstaOrder.
    The COCO annotation list is filtered and reordered to match
    InstaOrder instance_ids exactly.
    """
    merged_records: list[dict[str, Any]] = []

    for record in coco_records:
        image_id = record["image_id"]
        if image_id not in instaorder_by_image_id:
            continue

        io_info = instaorder_by_image_id[image_id]
        instance_ids = io_info["instance_ids"]

        annotations = filter_coco_annotations_by_instance_ids(
            record["annotations"],
            instance_ids,
        )

        merged = {
            **record,
            "annotations": annotations,
            "instance_ids": instance_ids,
        }

        if "occlusion" in io_info:
            merged["occlusion"] = io_info["occlusion"]
        if "depth" in io_info:
            merged["depth"] = io_info["depth"]
            merged["overlap"] = io_info["overlap"]
            merged["count"] = io_info["count"]

        merged_records.append(merged)

    return merged_records
