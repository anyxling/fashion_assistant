from fashion_clip.fashion_clip import FashionCLIP
from PIL import Image
import json
import os
from tqdm import tqdm
import pickle
import torch
torch.backends.cudnn.benchmark = True
from concurrent.futures import ThreadPoolExecutor
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def get_valid_items(all_sets):
    # Filter and collect valid (image_path, description) pairs
    valid_items = []
    for outfit in all_sets:
        set_id = outfit.get("set_id")
        for item in outfit.get("items", []):
            name = item.get("name", "").strip().lower()
            if not name or name == 'polyvore':
                continue
            index = item["index"]
            local_path = f"polyvore/images/{set_id}/{index}.jpg"
            if os.path.exists(local_path):  # only if image exists locally
                valid_items.append((local_path, item["name"]))

    return valid_items


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


def create_embeds(valid_items, batch_size=1024):
    image_embeddings = []
    text_embeddings = []

    fclip = FashionCLIP("fashion-clip")
    fclip.device = "cuda"

    item_ids = []

    for i in tqdm(range(0, len(valid_items), batch_size)):
        batch = valid_items[i:i + batch_size]
        image_paths = [img_path for img_path, _ in batch]
        descriptions = [desc for _, desc in batch]

        # Load images
        with ThreadPoolExecutor(max_workers=8) as executor:
            images = list(executor.map(load_image, image_paths))

        # Filter
        filtered_batch = [(img, desc, path) for img, desc, path in zip(images, descriptions, image_paths) if img is not None]
        if not filtered_batch:
            continue

        images, descriptions, paths = zip(*filtered_batch)

        # Create IDs from image path
        ids = [p.split("/")[-2] + "_" + p.split("/")[-1].split(".")[0] for p in paths]  # e.g., "119704139_1"
        item_ids.extend(ids)

        # Embed
        img_emb = fclip.encode_images(list(images), batch_size=len(images))
        txt_emb = fclip.encode_text(list(descriptions), batch_size=len(images))

        image_embeddings.append(torch.from_numpy(img_emb))
        text_embeddings.append(torch.from_numpy(txt_emb))

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    return item_ids, image_embeddings, text_embeddings


def map_id_embed(item_ids, embeddings):
    id_embed = {item_id: emb for item_id, emb in zip(item_ids, embeddings)}
    return id_embed


def create_compatibility_data(all_sets, item_ids):
    positive_outfits = []
    for outfit in all_sets:
        pos_lines = [1]
        set_id = outfit.get("set_id")
        for item in outfit.get("items", []):
            index = item["index"]
            item_id = f"{set_id}_{index}"
            if item_id in item_ids:
                pos_lines.append(item_id)
        positive_outfits.append(pos_lines)

    all_item_ids = []
    for outfit in all_sets:
        set_id = outfit.get("set_id")
        for item in outfit.get("items", []):
            index = item["index"]
            item_id = f"{set_id}_{index}"
            if item_id in item_ids:
                all_item_ids.append(item_id)

    negative_outfits = []
    for _ in range(len(positive_outfits)):
        neg_lines = [0]
        neg_ids = random.sample(all_item_ids, random.randint(4, 8))  # random outfit size
        for i in neg_ids:
            neg_lines.append(i)
        negative_outfits.append(neg_lines)

    all_outfits = positive_outfits + negative_outfits
    random.shuffle(all_outfits)

    return all_outfits
