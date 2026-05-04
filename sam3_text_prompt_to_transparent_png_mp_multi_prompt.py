# -*- coding: utf-8 -*-
"""
sam3_text_prompt_to_transparent_png_mp_multi_prompt.py

功能：
1. 单张图：
   --img xxx.jpg --prompt "person, car"
   会分别生成多个透明 PNG

2. 批量模式：
   --dir /path/to/images --prompt "person, car"
   输出到：
   out_dir/person_sticker/...
   out_dir/car_sticker/...

3. 多 GPU / 多进程：
   --mp --gpus 0,1 --procs_per_gpu 1

默认：
- --ckpt = ../models/sam3/sam3.pt
- --device = cuda:0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import os
import sys
import traceback

import torch
import torch.nn.functional as F
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================
# Helper functions
# =========================
def _ensure_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _to_3d_masks(m: torch.Tensor) -> torch.Tensor:
    """Normalize masks tensor to shape (N, H, W)."""
    if m is None:
        return torch.empty((0, 0, 0))
    if m.dim() == 2:
        return m.unsqueeze(0)
    if m.dim() == 3:
        return m
    if m.dim() == 4:
        if m.shape[1] == 1:
            return m[:, 0, :, :]
        if m.shape[0] == 1:
            return m[0]
        return m.reshape(-1, m.shape[-2], m.shape[-1])
    return m.reshape(-1, m.shape[-2], m.shape[-1])


def _resize_masks_to_image(masks_nhw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if masks_nhw.numel() == 0:
        return masks_nhw
    hm, wm = int(masks_nhw.shape[-2]), int(masks_nhw.shape[-1])
    if (hm, wm) == (h, w):
        return masks_nhw
    x = masks_nhw.unsqueeze(1).float()
    x = F.interpolate(x, size=(h, w), mode="nearest")
    return x[:, 0, :, :]


def _extract_masks_and_scores(out: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    masks = None
    for k in ("masks", "pred_masks", "segmentation", "seg", "mask"):
        if k in out and out[k] is not None:
            masks = out[k]
            break

    scores = None
    for k in ("scores", "score", "iou_scores", "pred_scores"):
        if k in out and out[k] is not None:
            scores = out[k]
            break

    masks_t = _ensure_tensor(masks) if masks is not None else None
    scores_t = _ensure_tensor(scores).flatten() if scores is not None else None
    return masks_t, scores_t


def _mask_to_bbox(mask_bool_hw: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    if mask_bool_hw.numel() == 0 or (not bool(mask_bool_hw.any().item())):
        return None
    ys, xs = torch.where(mask_bool_hw)
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1
    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1
    return x1, y1, x2, y2


def _safe_prompt_name(prompt: str) -> str:
    s = prompt.strip().lower().replace(" ", "_")
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_.-")
    return s[:80] if len(s) > 80 else s


def _parse_prompts(prompt_text: str) -> List[str]:
    prompts = []
    for x in prompt_text.split(","):
        x = x.strip()
        if x:
            prompts.append(x)

    # 去重，保留顺序
    seen = set()
    uniq = []
    for x in prompts:
        key = x.lower()
        if key not in seen:
            uniq.append(x)
            seen.add(key)
    return uniq


def _prompt_folder_name(prompt: str) -> str:
    return f"{_safe_prompt_name(prompt)}_sticker"


def _save_empty_or_fail(out_path: str, prompt: str, return_empty_as_1x1: bool, reason: str) -> Dict[str, Any]:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if return_empty_as_1x1:
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(str(out_p))
    return {
        "ok": False,
        "bbox": None,
        "out_path": str(out_p),
        "prompt": prompt,
        "kept_masks": 0,
        "reason": reason,
    }


# =========================
# SAM3 session
# =========================
@dataclass
class Sam3Session:
    processor: Sam3Processor
    device: str


def load_sam3_session(checkpoint_path: str, device: str = "cuda:0") -> Sam3Session:
    ckpt = str(checkpoint_path)
    if ckpt and not Path(ckpt).exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {ckpt}")

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(":")[1]))

    model = build_sam3_image_model(checkpoint_path=ckpt)
    model.eval()
    try:
        model.to(device)
    except Exception:
        pass

    processor = Sam3Processor(model)
    return Sam3Session(processor=processor, device=device)


# =========================
# Core single prompt API
# =========================
@torch.inference_mode()
def _get_transparent_png_with_session(
    *,
    processor: Sam3Processor,
    img_path: str,
    text_prompt: str,
    out_path: str,
    score_thr: float = 0.25,
    mask_thr: float = 0.5,
    union: bool = True,
    return_empty_as_1x1: bool = True,
) -> Dict[str, Any]:
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(str(p)).convert("RGB")
    w, h = image.size

    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt=text_prompt)
    masks_t, scores_t = _extract_masks_and_scores(out)

    if masks_t is None:
        return _save_empty_or_fail(out_path, text_prompt, return_empty_as_1x1, reason="no_mask_returned")

    masks_nhw = _to_3d_masks(masks_t)
    if masks_nhw.numel() == 0:
        return _save_empty_or_fail(out_path, text_prompt, return_empty_as_1x1, reason="empty_masks_tensor")

    masks_nhw = _resize_masks_to_image(masks_nhw, h=h, w=w)

    if scores_t is not None and scores_t.numel() > 0:
        n = min(int(masks_nhw.shape[0]), int(scores_t.shape[0]))
        masks_nhw = masks_nhw[:n]
        scores_t = scores_t[:n]
        keep = scores_t >= float(score_thr)
        masks_nhw = masks_nhw[keep]
        scores_t = scores_t[keep]

    if masks_nhw.numel() == 0:
        return _save_empty_or_fail(out_path, text_prompt, return_empty_as_1x1, reason="all_filtered_by_score")

    m = (masks_nhw.float() > float(mask_thr))

    if (not union) and (scores_t is not None) and (scores_t.numel() == m.shape[0]):
        best_i = int(torch.argmax(scores_t).item())
        mask_bool = m[best_i]
        kept = 1
    else:
        mask_bool = m.any(dim=0)
        kept = int(m.shape[0])

    mask_bool = mask_bool.to(torch.bool).cpu()
    bbox = _mask_to_bbox(mask_bool)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if bbox is None:
        return _save_empty_or_fail(out_path, text_prompt, return_empty_as_1x1, reason="mask_empty_after_threshold")

    x1, y1, x2, y2 = bbox
    crop_rgb = image.crop((x1, y1, x2, y2))
    mask_crop = mask_bool[y1:y2, x1:x2]

    rgba = crop_rgb.convert("RGBA")
    alpha = Image.fromarray((mask_crop.numpy().astype("uint8") * 255), mode="L")
    rgba.putalpha(alpha)
    rgba.save(str(out_p))

    return {
        "ok": True,
        "bbox": bbox,
        "out_path": str(out_p),
        "prompt": text_prompt,
        "prompt_name": _safe_prompt_name(text_prompt),
        "kept_masks": kept,
        "image_size": (w, h),
        "score_thr": float(score_thr),
        "mask_thr": float(mask_thr),
        "union": bool(union),
    }


@torch.inference_mode()
def get_transparent_png(
    img_path: str,
    text_prompt: str,
    out_path: str,
    *,
    checkpoint_path: str = "../models/sam3/sam3.pt",
    device: str = "cuda:0",
    score_thr: float = 0.25,
    mask_thr: float = 0.5,
    union: bool = True,
    return_empty_as_1x1: bool = True,
) -> Dict[str, Any]:
    session = load_sam3_session(checkpoint_path=checkpoint_path, device=device)
    return _get_transparent_png_with_session(
        processor=session.processor,
        img_path=img_path,
        text_prompt=text_prompt,
        out_path=out_path,
        score_thr=score_thr,
        mask_thr=mask_thr,
        union=union,
        return_empty_as_1x1=return_empty_as_1x1,
    )


# =========================
# Multi prompt API
# =========================
def get_transparent_png_multi(
    img_path: str,
    prompt_text: str,
    out: Optional[str] = None,
    *,
    checkpoint_path: str = "../models/sam3/sam3.pt",
    device: str = "cuda:0",
    score_thr: float = 0.25,
    mask_thr: float = 0.5,
    union: bool = True,
    return_empty_as_1x1: bool = True,
) -> Dict[str, Any]:
    prompts = _parse_prompts(prompt_text)
    if len(prompts) == 0:
        raise ValueError("No valid prompts found in --prompt")

    session = load_sam3_session(checkpoint_path=checkpoint_path, device=device)
    img_p = Path(img_path)
    results = []

    for prompt in prompts:
        prompt_name = _safe_prompt_name(prompt)
        if out is None:
            out_path = str(Path.cwd() / f"{img_p.stem}_{prompt_name}.png")
        else:
            out_p = Path(out)
            if len(prompts) == 1:
                out_path = str(out_p)
            else:
                if out_p.suffix:
                    out_path = str(out_p.parent / f"{out_p.stem}_{prompt_name}{out_p.suffix}")
                else:
                    out_dir = out_p / _prompt_folder_name(prompt)
                    out_path = str(out_dir / f"{img_p.stem}_{prompt_name}.png")

        info = _get_transparent_png_with_session(
            processor=session.processor,
            img_path=str(img_p),
            text_prompt=prompt,
            out_path=out_path,
            score_thr=score_thr,
            mask_thr=mask_thr,
            union=union,
            return_empty_as_1x1=return_empty_as_1x1,
        )
        results.append(info)

    return {
        "img_path": str(img_p),
        "num_prompts": len(prompts),
        "prompts": prompts,
        "results": results,
    }


# =========================
# Batch helpers
# =========================
def _iter_images_in_dir(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    exts_l = tuple(e.lower().lstrip(".") for e in exts)
    imgs: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts_l:
            imgs.append(p)
    imgs.sort()
    return imgs


def run_batch(
    input_dir: str,
    prompt: str,
    *,
    out_dir: Optional[str] = None,
    checkpoint_path: str = "../models/sam3/sam3.pt",
    device: str = "cuda:0",
    score_thr: float = 0.25,
    mask_thr: float = 0.5,
    union: bool = True,
    return_empty_as_1x1: bool = True,
    exts: Tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "webp"),
    show_progress: bool = True,
) -> Dict[str, Any]:
    in_root = Path(input_dir)
    if not in_root.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if out_dir is None:
        out_root = in_root.parent / (in_root.name + "_stickers")
    else:
        out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    prompts = _parse_prompts(prompt)
    if len(prompts) == 0:
        raise ValueError("No valid prompts found in --prompt")

    session = load_sam3_session(checkpoint_path=checkpoint_path, device=device)
    processor = session.processor

    images = _iter_images_in_dir(in_root, exts)
    results: Dict[str, Any] = {
        "total_images": len(images),
        "num_prompts": len(prompts),
        "total_tasks": len(images) * len(prompts),
        "ok": 0,
        "failed": 0,
        "out_dir": str(out_root),
        "prompts": prompts,
    }

    tasks = []
    for img_p in images:
        rel = img_p.relative_to(in_root)
        for one_prompt in prompts:
            tasks.append((img_p, rel, one_prompt))

    it = tasks
    if show_progress and tqdm is not None:
        it = tqdm(tasks, desc="SAM3 batch", unit="task")

    for img_p, rel, one_prompt in it:
        prompt_name = _safe_prompt_name(one_prompt)
        class_root = out_root / _prompt_folder_name(one_prompt)
        out_subdir = class_root / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_p = out_subdir / f"{img_p.stem}_{prompt_name}.png"

        try:
            info = _get_transparent_png_with_session(
                processor=processor,
                img_path=str(img_p),
                text_prompt=one_prompt,
                out_path=str(out_p),
                score_thr=score_thr,
                mask_thr=mask_thr,
                union=union,
                return_empty_as_1x1=return_empty_as_1x1,
            )
            if info.get("ok", False):
                results["ok"] += 1
            else:
                results["failed"] += 1
        except Exception:
            results["failed"] += 1

    return results


# =========================
# Multi-process / multi-GPU
# =========================
def _split_round_robin(items: List[Any], k: int) -> List[List[Any]]:
    parts: List[List[Any]] = [[] for _ in range(k)]
    for i, p in enumerate(items):
        parts[i % k].append(p)
    return parts


def _worker_process(
    rank: int,
    device: str,
    tasks: List[Tuple[str, str, str]],
    in_root: str,
    out_root: str,
    ckpt: str,
    score_thr: float,
    mask_thr: float,
    union: bool,
    empty_1x1: bool,
    progress_q,
):
    try:
        session = load_sam3_session(checkpoint_path=ckpt, device=device)
        processor = session.processor

        out_root_p = Path(out_root)

        ok = 0
        failed = 0

        for img_s, rel_s, prompt in tasks:
            img_p = Path(img_s)
            rel = Path(rel_s)

            prompt_name = _safe_prompt_name(prompt)
            class_root = out_root_p / _prompt_folder_name(prompt)
            out_subdir = class_root / rel.parent
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_p = out_subdir / f"{img_p.stem}_{prompt_name}.png"

            try:
                info = _get_transparent_png_with_session(
                    processor=processor,
                    img_path=str(img_p),
                    text_prompt=prompt,
                    out_path=str(out_p),
                    score_thr=score_thr,
                    mask_thr=mask_thr,
                    union=union,
                    return_empty_as_1x1=empty_1x1,
                )
                if info.get("ok", False):
                    ok += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            try:
                progress_q.put(("tick", 1), block=False)
            except Exception:
                pass

        try:
            progress_q.put(("done", rank, ok, failed), block=False)
        except Exception:
            pass

    except Exception as e:
        try:
            progress_q.put(("error", rank, str(e), traceback.format_exc()), block=False)
        except Exception:
            pass


def run_batch_mp(
    input_dir: str,
    prompt: str,
    *,
    out_dir: Optional[str] = None,
    checkpoint_path: str = "../models/sam3/sam3.pt",
    gpus: str = "0",
    procs_per_gpu: int = 1,
    score_thr: float = 0.25,
    mask_thr: float = 0.5,
    union: bool = True,
    return_empty_as_1x1: bool = True,
    exts: Tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "webp"),
) -> Dict[str, Any]:
    from multiprocessing import get_context

    in_root = Path(input_dir)
    if not in_root.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if out_dir is None:
        out_root = in_root.parent / (in_root.name + "_stickers")
    else:
        out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    prompts = _parse_prompts(prompt)
    if len(prompts) == 0:
        raise ValueError("No valid prompts found in --prompt")

    images = _iter_images_in_dir(in_root, exts)
    total_images = len(images)
    total_tasks = total_images * len(prompts)

    if total_images == 0:
        return {
            "total_images": 0,
            "num_prompts": len(prompts),
            "total_tasks": 0,
            "ok": 0,
            "failed": 0,
            "out_dir": str(out_root),
            "note": "no_images_found",
        }

    gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
    if len(gpu_ids) == 0:
        gpu_ids = ["0"]

    devices: List[str] = []
    if gpus.strip().lower() == "cpu":
        devices = ["cpu"] * max(1, int(procs_per_gpu))
    else:
        for gid in gpu_ids:
            for _ in range(int(procs_per_gpu)):
                devices.append(f"cuda:{gid}")

    tasks: List[Tuple[str, str, str]] = []
    for img_p in images:
        rel = img_p.relative_to(in_root)
        for one_prompt in prompts:
            tasks.append((str(img_p), str(rel), one_prompt))

    nprocs = min(len(devices), len(tasks))
    devices = devices[:nprocs]
    parts = _split_round_robin(tasks, nprocs)

    ctx = get_context("spawn")
    q = ctx.Queue()

    procs = []
    for rank in range(nprocs):
        p = ctx.Process(
            target=_worker_process,
            args=(
                rank,
                devices[rank],
                parts[rank],
                str(in_root),
                str(out_root),
                str(checkpoint_path),
                float(score_thr),
                float(mask_thr),
                bool(union),
                bool(return_empty_as_1x1),
                q,
            ),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    ok = 0
    failed = 0
    done_workers = 0

    bar = None
    if tqdm is not None:
        bar = tqdm(total=total_tasks, desc="SAM3 mp batch", unit="task")

    while done_workers < nprocs:
        msg = q.get()
        if not msg:
            continue

        typ = msg[0]
        if typ == "tick":
            if bar is not None:
                bar.update(1)
        elif typ == "done":
            _, rank, ok_i, failed_i = msg
            ok += int(ok_i)
            failed += int(failed_i)
            done_workers += 1
        elif typ == "error":
            _, rank, err, tb = msg
            print(f"[Worker {rank}] ERROR: {err}\n{tb}", file=sys.stderr)
            done_workers += 1

    if bar is not None:
        bar.close()

    for p in procs:
        p.join()

    return {
        "total_images": total_images,
        "num_prompts": len(prompts),
        "total_tasks": total_tasks,
        "ok": ok,
        "failed": failed,
        "out_dir": str(out_root),
        "workers": nprocs,
        "devices": devices,
        "prompts": prompts,
    }


# =========================
# CLI
# =========================
def _build_argparser():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompt",
        required=True,
        help='支持多个类别，用逗号分隔，例如: "person, car"',
    )

    ap.add_argument("--img", default=None, help="单张图路径")
    ap.add_argument("--out", default=None, help="单张图输出路径；多 prompt 时也可作为输出目录")

    ap.add_argument("--dir", default=None, help="批量输入目录，递归处理")
    ap.add_argument("--out_dir", default=None, help="批量输出目录，默认 <dir>_stickers")
    ap.add_argument("--exts", default="jpg,jpeg,png,bmp,webp", help="批量图像后缀")

    ap.add_argument("--ckpt", default="../models/sam3/sam3.pt", help="SAM3 checkpoint path")
    ap.add_argument("--device", default="cuda:0", help='单进程设备，如 "cuda:0" 或 "cpu"')

    ap.add_argument("--score_thr", type=float, default=0.25, help="score 阈值")
    ap.add_argument("--mask_thr", type=float, default=0.5, help="mask 二值化阈值")
    ap.add_argument("--no_union", action="store_true", help="只保留 best mask")
    ap.add_argument("--empty_1x1", action="store_true", help="空结果时保存 1x1 透明图")

    ap.add_argument("--mp", action="store_true", help="启用多进程 batch")
    ap.add_argument("--gpus", default="0", help='GPU ids，例如 "0,1"，CPU 模式可填 "cpu"')
    ap.add_argument("--procs_per_gpu", type=int, default=1, help="每张 GPU 开几个进程")

    ap.add_argument("--no_pbar", action="store_true", help="关闭进度条")
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    union = not args.no_union
    empty_1x1 = bool(args.empty_1x1)

    if args.dir is not None:
        exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())

        if args.mp:
            res = run_batch_mp(
                input_dir=args.dir,
                prompt=args.prompt,
                out_dir=args.out_dir,
                checkpoint_path=args.ckpt,
                gpus=args.gpus,
                procs_per_gpu=args.procs_per_gpu,
                score_thr=args.score_thr,
                mask_thr=args.mask_thr,
                union=union,
                return_empty_as_1x1=empty_1x1,
                exts=exts,
            )
        else:
            res = run_batch(
                input_dir=args.dir,
                prompt=args.prompt,
                out_dir=args.out_dir,
                checkpoint_path=args.ckpt,
                device=args.device,
                score_thr=args.score_thr,
                mask_thr=args.mask_thr,
                union=union,
                return_empty_as_1x1=empty_1x1,
                exts=exts,
                show_progress=not args.no_pbar,
            )
        print(res)

    else:
        if args.img is None:
            raise ValueError("请提供 --img 或 --dir")

        info = get_transparent_png_multi(
            img_path=str(args.img),
            prompt_text=args.prompt,
            out=args.out,
            checkpoint_path=args.ckpt,
            device=args.device,
            score_thr=args.score_thr,
            mask_thr=args.mask_thr,
            union=union,
            return_empty_as_1x1=empty_1x1,
        )
        print(info)
