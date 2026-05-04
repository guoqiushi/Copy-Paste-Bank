#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict

import ijson


def load_categories(annotation_file):
    """
    读取 categories，建立:
    1. category name -> category id
    2. category id -> category name
    """
    name_to_id = {}
    id_to_name = {}

    with open(annotation_file, "rb") as f:
        for cat in ijson.items(f, "categories.item"):
            cat_id = cat["id"]
            cat_name = cat["name"]
            name_to_id[cat_name] = cat_id
            id_to_name[cat_id] = cat_name

    return name_to_id, id_to_name


def collect_image_ids_by_category(annotation_file, target_cat_ids):
    """
    遍历 annotations，收集指定类别出现过的 image_id
    """
    image_ids = set()
    image_to_cat_ids = defaultdict(set)

    with open(annotation_file, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            cat_id = ann["category_id"]
            if cat_id in target_cat_ids:
                image_id = ann["image_id"]
                image_ids.add(image_id)
                image_to_cat_ids[image_id].add(cat_id)

    return image_ids, image_to_cat_ids


def collect_image_paths(annotation_file, image_ids, image_root=None):
    """
    遍历 images，根据 image_id 提取 file_name/path
    """
    image_paths = {}

    with open(annotation_file, "rb") as f:
        for img in ijson.items(f, "images.item"):
            img_id = img["id"]

            if img_id not in image_ids:
                continue

            # Objects365 / COCO 常见字段是 file_name
            file_name = img.get("file_name", None)

            # 某些版本可能是 path / url / image_name
            if file_name is None:
                file_name = img.get("path", None)
            if file_name is None:
                file_name = img.get("image_name", None)

            if file_name is None:
                print(f"[Warning] image id {img_id} has no file_name/path/image_name")
                continue

            if image_root is not None:
                full_path = str(Path(image_root) / file_name)
            else:
                full_path = file_name

            image_paths[img_id] = full_path

    return image_paths


def save_results_txt(image_paths, output_txt):
    with open(output_txt, "w", encoding="utf-8") as f:
        for _, path in sorted(image_paths.items()):
            f.write(path + "\n")


def save_results_json(image_paths, image_to_cat_ids, id_to_name, output_json):
    results = []

    for image_id, path in sorted(image_paths.items()):
        cat_ids = sorted(list(image_to_cat_ids.get(image_id, [])))
        cat_names = [id_to_name[cid] for cid in cat_ids]

        results.append({
            "image_id": image_id,
            "image_path": path,
            "category_ids": cat_ids,
            "category_names": cat_names
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract image paths of specified categories from Objects365 annotation file."
    )

    parser.add_argument(
        "--ann",
        required=True,
        help="Objects365 annotation json file, e.g. objects365_train.json"
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Target category names, e.g. person 'cell phone'"
    )

    parser.add_argument(
        "--image_root",
        default=None,
        help="Optional image root path. If set, output full image path."
    )

    parser.add_argument(
        "--output_txt",
        default="selected_image_paths.txt",
        help="Output txt file path."
    )

    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional output json file path."
    )

    args = parser.parse_args()

    ann_file = args.ann
    target_classes = args.classes

    print("[1/4] Loading categories...")
    name_to_id, id_to_name = load_categories(ann_file)

    print(f"Total categories: {len(name_to_id)}")

    target_cat_ids = set()
    missing_classes = []

    for cls_name in target_classes:
        if cls_name in name_to_id:
            target_cat_ids.add(name_to_id[cls_name])
        else:
            missing_classes.append(cls_name)

    if missing_classes:
        print("[Warning] These classes are not found:")
        for cls_name in missing_classes:
            print(f"  - {cls_name}")

    if len(target_cat_ids) == 0:
        raise ValueError("No valid target classes found.")

    print("Target classes:")
    for cid in sorted(target_cat_ids):
        print(f"  {cid}: {id_to_name[cid]}")

    print("[2/4] Collecting image ids from annotations...")
    image_ids, image_to_cat_ids = collect_image_ids_by_category(
        ann_file,
        target_cat_ids
    )

    print(f"Matched image num: {len(image_ids)}")

    print("[3/4] Collecting image paths from images...")
    image_paths = collect_image_paths(
        ann_file,
        image_ids,
        image_root=args.image_root
    )

    print(f"Collected image paths: {len(image_paths)}")

    print("[4/4] Saving results...")
    save_results_txt(image_paths, args.output_txt)
    print(f"Saved txt to: {args.output_txt}")

    if args.output_json is not None:
        save_results_json(
            image_paths,
            image_to_cat_ids,
            id_to_name,
            args.output_json
        )
        print(f"Saved json to: {args.output_json}")


if __name__ == "__main__":
    main()
