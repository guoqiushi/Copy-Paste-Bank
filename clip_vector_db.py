import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPVectorDatabase:
    def __init__(
        self,
        clip_model_path="openai/clip-vit-base-patch32",
        db_dir="./vector_db",
        device=None,
    ):
        self.clip_model_path = clip_model_path
        self.db_dir = db_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.db_dir, exist_ok=True)

        self.index_path = os.path.join(self.db_dir, "faiss.index")
        self.meta_path = os.path.join(self.db_dir, "metadata.json")

        self.model = CLIPModel.from_pretrained(self.clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_path)
        self.model.eval()

        self.dim = self.model.config.projection_dim

        self.index = None
        self.metadata = []

        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            print(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)

            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            print(f"Loaded {len(self.metadata)} items.")
        else:
            print("Creating new FAISS index.")

            # IndexFlatIP 使用 inner product。
            # 因为我们会对 CLIP 特征做 L2 normalize，
            # 所以 inner product 等价于 cosine similarity。
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    @torch.no_grad()
    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        image_feature = self.model.get_image_features(**inputs)
        image_feature = F.normalize(image_feature, dim=-1)

        image_feature = image_feature.cpu().numpy().astype("float32")

        return image_feature

    def add_image(self, image_path, category=None, extra_info=None):
        """
        Add a single image into vector database.

        Args:
            image_path: path to image
            category: object category, e.g., person, car, soccer ball
            extra_info: optional dict, e.g., mask_path, source_image_path, bbox
        """
        image_path = str(image_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        feature = self.encode_image(image_path)

        item_id = len(self.metadata)

        self.index.add(feature)

        meta = {
            "id": item_id,
            "image_path": image_path,
            "category": category,
        }

        if extra_info is not None:
            meta.update(extra_info)

        self.metadata.append(meta)

        self.save()

        print(f"Added image to database:")
        print(f"  id: {item_id}")
        print(f"  image_path: {image_path}")
        print(f"  category: {category}")

    def search_by_image(self, query_image_path, topk=5):
        """
        Search similar images using a single query image.
        """
        if self.index.ntotal == 0:
            print("Vector database is empty.")
            return []

        query_feature = self.encode_image(query_image_path)

        scores, indices = self.index.search(query_feature, topk)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]

            result = {
                "score": float(score),
                "id": meta["id"],
                "image_path": meta["image_path"],
                "category": meta.get("category", None),
            }

            for k, v in meta.items():
                if k not in result:
                    result[k] = v

            results.append(result)

        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def info(self):
        print("=" * 60)
        print("Vector Database Info")
        print("=" * 60)
        print(f"DB dir: {self.db_dir}")
        print(f"Index path: {self.index_path}")
        print(f"Metadata path: {self.meta_path}")
        print(f"Feature dim: {self.dim}")
        print(f"Total items: {self.index.ntotal}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["add", "search", "info"],
        help="add: add one image; search: search by one image; info: show db info",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="image path for add or search",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="category name for added image",
    )

    parser.add_argument(
        "--db_dir",
        type=str,
        default="./vector_db",
        help="directory for vector database",
    )

    parser.add_argument(
        "--clip_model_path",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model path or local model path",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="top-k search results",
    )

    args = parser.parse_args()

    db = CLIPVectorDatabase(
        clip_model_path=args.clip_model_path,
        db_dir=args.db_dir,
    )

    if args.mode == "add":
        if args.image_path is None:
            raise ValueError("--image_path is required for add mode")

        db.add_image(
            image_path=args.image_path,
            category=args.category,
        )

    elif args.mode == "search":
        if args.image_path is None:
            raise ValueError("--image_path is required for search mode")

        results = db.search_by_image(
            query_image_path=args.image_path,
            topk=args.topk,
        )

        print("=" * 80)
        print(f"Query image: {args.image_path}")
        print("=" * 80)

        for rank, item in enumerate(results, start=1):
            print(
                f"[Top {rank}] "
                f"score={item['score']:.4f}, "
                f"id={item['id']}, "
                f"category={item['category']}, "
                f"path={item['image_path']}"
            )

    elif args.mode == "info":
        db.info()


if __name__ == "__main__":
    main()
