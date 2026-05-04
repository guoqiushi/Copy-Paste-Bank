import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


'''
python clip_gallery_index.py \
  --mode build \
  --image_dir ./copy_paste_bank \
  --output_dir ./clip_gallery_index \
  --batch_size 64
'''

class CLIPGalleryIndexer:
    def __init__(self, model_path="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image_paths, batch_size=32):
        """
        Encode object crops / mask crops into visual embeddings.
        """
        all_features = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(img)

            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)

            all_features.append(image_features.cpu())

        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def encode_text(self, texts, batch_size=64):
        """
        Encode class names / prompts / VLM queries into text embeddings.
        """
        all_features = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)

            all_features.append(text_features.cpu())

        return torch.cat(all_features, dim=0)


def collect_images(image_dir):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_paths = []

    for ext in exts:
        image_paths.extend(Path(image_dir).rglob(f"*{ext}"))

    return sorted([str(p) for p in image_paths])


def build_gallery_index(args):
    indexer = CLIPGalleryIndexer(
        model_path=args.clip_model,
        device=args.device
    )

    image_paths = collect_images(args.image_dir)

    print(f"Found {len(image_paths)} images.")

    image_features = indexer.encode_image(
        image_paths=image_paths,
        batch_size=args.batch_size
    )

    gallery_meta = []
    for idx, path in enumerate(image_paths):
        # 默认类别名取父目录名，例如:
        # copy_paste_bank/person/xxx.png -> person
        category = Path(path).parent.name

        gallery_meta.append({
            "id": idx,
            "image_path": path,
            "category": category
        })

    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(image_features, os.path.join(args.output_dir, "visual_features.pt"))

    with open(os.path.join(args.output_dir, "gallery_meta.json"), "w", encoding="utf-8") as f:
        json.dump(gallery_meta, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(os.path.join(args.output_dir, "visual_features.pt"))
    print(os.path.join(args.output_dir, "gallery_meta.json"))


def query_gallery(args):
    indexer = CLIPGalleryIndexer(
        model_path=args.clip_model,
        device=args.device
    )

    visual_features = torch.load(args.visual_features, map_location="cpu")
    visual_features = F.normalize(visual_features, dim=-1)

    with open(args.gallery_meta, "r", encoding="utf-8") as f:
        gallery_meta = json.load(f)

    queries = args.query

    # 可以增强文本 prompt，使 CLIP 更稳定
    prompts = [f"a photo of a {q}" for q in queries]

    text_features = indexer.encode_text(prompts)
    text_features = F.normalize(text_features, dim=-1)

    sim = text_features @ visual_features.T

    for qi, q in enumerate(queries):
        scores, indices = torch.topk(sim[qi], k=args.topk)

        print("=" * 80)
        print(f"Query: {q}")
        print("=" * 80)

        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            item = gallery_meta[idx.item()]
            print(
                f"[Top {rank}] "
                f"score={score.item():.4f}, "
                f"category={item['category']}, "
                f"path={item['image_path']}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["build", "query"])

    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./clip_gallery_index")

    parser.add_argument("--visual_features", type=str, default="./clip_gallery_index/visual_features.pt")
    parser.add_argument("--gallery_meta", type=str, default="./clip_gallery_index/gallery_meta.json")

    parser.add_argument("--query", nargs="+", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if args.mode == "build":
        assert args.image_dir is not None, "--image_dir is required for build mode"
        build_gallery_index(args)

    elif args.mode == "query":
        assert args.query is not None, "--query is required for query mode"
        query_gallery(args)


if __name__ == "__main__":
    main()
