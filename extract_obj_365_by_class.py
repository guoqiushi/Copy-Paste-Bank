#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import ijson


def load_categories(annotation_file):
    """
    Load category name <-> category id mapping.
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


def collect_image_ids_by_classes(annotation_file, target_cat_ids):
    """
    Collect image ids for each target category.

    Return:
        cat_id_to_image_ids:
            {
                cat_id: set(image_id)
            }
        image_id_to_cat_ids:
            {
                image_id: set(cat_id)
            }
    """
    cat_id_to_image_ids = defaultdict(set)
    image_id_to_cat_ids = defaultdict(set)

    with open(annotation_file, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            cat_id = ann["category_id"]

            if cat_id not in target_cat_ids:
                continue

            image_id = ann["image_id"]

            cat_id_to_image_ids[cat_id].add(image_id)
            image_id_to_cat_ids[image_id].add(cat_id)

    return cat_id_to_image_ids, image_id_to_cat_ids


def collect_image_info(annotation_file, all_target_image_ids, image_root=None):
    """
    Collect image file paths by image id.
    """
    image_id_to_info = {}

    with open(annotation_file, "rb") as f:
        for img in ijson.items(f, "images.item"):
            img_id = img["id"]

            if img_id not in all_target_image_ids:
                continue

            file_name = (
                img.get("file_name")
                or img.get("path")
                or img.get("image_name")
            )

            if file_name is None:
                print(f"[Warning] image id {img_id} has no file_name/path/image_name")
                continue

            if image_root is not None:
                image_path = str(Path(image_root) / file_name)
            else:
                image_path = file_name

            image_id_to_info[img_id] = {
                "image_id": img_id,
                "file_name": file_name,
                "image_path": image_path,
                "raw_info": img,
            }

    return image_id_to_info


def safe_class_name(class_name):
    """
    Convert class name to safe directory name.
    """
    return class_name.replace("/", "_").replace(" ", "_")


def save_class_results(
    class_name,
    cat_id,
    image_ids,
    image_id_to_info,
    output_dir,
):
    """
    Save one class result to its own subdirectory.
    """
    class_dir = Path(output_dir) / safe_class_name(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)

    txt_path = class_dir / "image_paths.txt"
    json_path = class_dir / "image_info.json"

    results = []

    with open(txt_path, "w", encoding="utf-8") as f:
        for image_id in sorted(image_ids):
            if image_id not in image_id_to_info:
                continue

            info = image_id_to_info[image_id]
            f.write(info["image_path"] + "\n")

            results.append({
                "image_id": image_id,
                "image_path": info["image_path"],
                "file_name": info["file_name"],
                "category_id": cat_id,
                "category_name": class_name,
            })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"[Saved] class={class_name}, "
        f"num={len(results)}, "
        f"dir={class_dir}"
    )


def copy_one_image(task):
    """
    Copy one image to target directory.

    task:
        {
            "src": str,
            "dst": str
        }
    """
    src = Path(task["src"])
    dst = Path(task["dst"])

    if not src.exists():
        return False, str(src), "source not found"

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(src, dst)
        return True, str(src), str(dst)
    except Exception as e:
        return False, str(src), str(e)


def build_copy_tasks(
    class_name,
    image_ids,
    image_id_to_info,
    output_dir,
    keep_original_subdir=False,
):
    """
    Build copy tasks for one class.
    """
    class_dir = Path(output_dir) / safe_class_name(class_name)
    image_save_dir = class_dir / "images"

    tasks = []

    for image_id in sorted(image_ids):
        if image_id not in image_id_to_info:
            continue

        info = image_id_to_info[image_id]
        src_path = Path(info["image_path"])

        if keep_original_subdir:
            rel_name = info["file_name"]
            dst_path = image_save_dir / rel_name
        else:
            dst_path = image_save_dir / src_path.name

        tasks.append({
            "src": str(src_path),
            "dst": str(dst_path),
        })

    return tasks


def copy_images_multiprocess(tasks, num_workers):
    """
    Copy images using multiprocessing.
    """
    if len(tasks) == 0:
        print("[Copy] No images to copy.")
        return

    print(f"[Copy] Total copy tasks: {len(tasks)}")
    print(f"[Copy] Workers: {num_workers}")

    success = 0
    failed = 0

    with Pool(processes=num_workers) as pool:
        for ok, src, msg in pool.imap_unordered(copy_one_image, tasks, chunksize=100):
            if ok:
                success += 1
            else:
                failed += 1
                print(f"[Warning] Failed to copy: {src}, reason: {msg}")

    print(f"[Copy Done] success={success}, failed={failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract image paths from Objects365 annotation file by multiple classes."
    )

    parser.add_argument(
        "--ann",
        required=True,
        help="Objects365 annotation JSON file."
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help='Target class names, e.g. --classes person cat dog "cell phone"'
    )

    parser.add_argument(
        "--image_root",
        default=None,
        help="Image root path. If set, output full image paths."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="If set, copy matched images into class subdirectories."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of workers for multiprocessing image copying."
    )

    parser.add_argument(
        "--keep_original_subdir",
        action="store_true",
        help="Keep original relative subdirectory structure when copying images."
    )

    args = parser.parse_args()

    ann_file = args.ann
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading categories...")
    name_to_id, id_to_name = load_categories(ann_file)

    target_cat_ids = set()
    class_name_to_id = {}

    missing_classes = []

    for class_name in args.classes:
        if class_name in name_to_id:
            cat_id = name_to_id[class_name]
            target_cat_ids.add(cat_id)
            class_name_to_id[class_name] = cat_id
        else:
            missing_classes.append(class_name)

    if missing_classes:
        print("[Warning] These classes are not found in annotation file:")
        for name in missing_classes:
            print(f"  - {name}")

    if len(target_cat_ids) == 0:
        raise ValueError("No valid target classes found.")

    print("Target classes:")
    for class_name, cat_id in class_name_to_id.items():
        print(f"  {class_name}: {cat_id}")

    print("[2/5] Collecting matched image ids from annotations...")
    cat_id_to_image_ids, image_id_to_cat_ids = collect_image_ids_by_classes(
        ann_file,
        target_cat_ids
    )

    all_target_image_ids = set()
    for image_ids in cat_id_to_image_ids.values():
        all_target_image_ids.update(image_ids)

    print(f"Total matched unique images: {len(all_target_image_ids)}")

    for class_name, cat_id in class_name_to_id.items():
        print(
            f"  {class_name}: "
            f"{len(cat_id_to_image_ids.get(cat_id, set()))} images"
        )

    print("[3/5] Collecting image paths from images...")
    image_id_to_info = collect_image_info(
        ann_file,
        all_target_image_ids,
        image_root=args.image_root
    )

    print(f"Collected image info: {len(image_id_to_info)}")

    print("[4/5] Saving path files into class subdirectories...")
    for class_name, cat_id in class_name_to_id.items():
        image_ids = cat_id_to_image_ids.get(cat_id, set())

        save_class_results(
            class_name=class_name,
            cat_id=cat_id,
            image_ids=image_ids,
            image_id_to_info=image_id_to_info,
            output_dir=output_dir,
        )

    if args.copy_images:
        print("[5/5] Copying images with multiprocessing...")

        all_tasks = []

        for class_name, cat_id in class_name_to_id.items():
            image_ids = cat_id_to_image_ids.get(cat_id, set())

            tasks = build_copy_tasks(
                class_name=class_name,
                image_ids=image_ids,
                image_id_to_info=image_id_to_info,
                output_dir=output_dir,
                keep_original_subdir=args.keep_original_subdir,
            )

            all_tasks.extend(tasks)

        copy_images_multiprocess(
            tasks=all_tasks,
            num_workers=args.num_workers
        )
    else:
        print("[5/5] Skip image copying. Only image paths are saved.")

    print("Done.")


if __name__ == "__main__":
    main()
