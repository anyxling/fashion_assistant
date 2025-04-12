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
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import hashlib

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


# def init_category_map(category_id):
#     category_map = {}
#     for cat in category_id:
#         cat = cat.strip()
#         num, text = cat.split(" ", 1)
#         category_map[int(num)] = {"description": text}, {"train": []}, {"val": []}, {"test": []}
#     return category_map


def map_category_items(data, dataname, category_map, item_ids):
    if dataname == "train":
        idx = 1
    elif dataname == "val":
        idx = 2
    else:
        idx = 3
    for outfit in data:
        set_id = outfit.get("set_id")
        for item in outfit.get("items", []):
            name = item.get("name", "").strip().lower()
            if not name or name == 'polyvore':
                continue
            index = item["index"]
            item_id = f"{set_id}_{index}"
            categoryid = item.get("categoryid") 
            if item_id in item_ids:
                category_map[categoryid][idx][str(dataname)].append(item_id)
    return category_map


def create_fitb_data(data, dataname, category_map):
    if dataname == "train":
        idx = 1
    elif dataname == "val":
        idx = 2
    else:
        idx = 3
    fill_in_blank = []
    for outfit in data:
        all_items = []
        categories = []
        set_id = outfit.get("set_id")
        for item in outfit.get("items", []):
            index = item["index"]
            item_id = f"{set_id}_{index}"
            all_items.append(item_id)
            categories.append(item["categoryid"])
        if len(all_items) < 4:
            continue
        blank_idx = random.randint(0, len(all_items)-1)
        correct = all_items[blank_idx]
        questions = all_items[:blank_idx] + all_items[blank_idx+1:]
        correct_category = categories[blank_idx]
        candidates = category_map[correct_category][idx][str(dataname)]
        valid_cands = [cand for cand in candidates if cand != correct]
        if len(valid_cands) < 4:
            final_cands = valid_cands
        else:
            final_cands = random.sample(valid_cands, 3)
        final_cands.insert(0, correct)
        fill_in_blank.append(
            {
                "question":questions,
                "answers": final_cands,
                "blank_position":blank_idx+1
            }
        )
    return fill_in_blank

class OutfitGenerator:
    def __init__(self, text_embeddings, map_id_img_embed, item_ids, score_model, city, occasion, sex, top_k=1):
        self.text_embeddings = text_embeddings
        self.map_id_img_embed = map_id_img_embed
        self.k = top_k
        self.model = score_model
        self.item_ids = item_ids
        self.city = city
        self.occasion = occasion
        self.sex = sex

    def generate_outfit_from_llm(self):
        # Initialize OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
        # # Ask user for input
        # city = input("which city are you currently in?")
        # occasion = input("What occasion do you need this outfit for?")
        # sex = input("what's your gender")

        # Get temperature
        API_KEY = "0db13be3f5563366aed5781b81d39c7f"
        url = f'https://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={API_KEY}&units=imperial'
        response_weather = requests.get(url)
        data = response_weather.json()

        # Build dynamic prompt
        prompt = f"""You're a fashion stylist. A {self.sex} user is going to the following occasion: {self.occasion}.
        The weather now in {self.city} is {data['weather'][0]['main']} with a temperature of {data['main']['temp']}°F, 
        ranging from a low of {data['main']['temp_min']}°F to a high of {data['main']['temp_max']}°F.
        Based on the weather condition and occasion, suggest no less than 3 distinct fashion/clothing/accessory items they should wear.
        - Do not include beauty products.
        - Do not offer alternatives like “or”.
        - Return only a single line of text.
        - Format the output as: item1, item2, item3, item4
        - Do not use bullet points or line breaks.
        - Do not output anything except just a line of items separate with comma.
        """
        # Call OpenRouter model
        completion = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            extra_headers={
                "HTTP-Referer": "https://yourproject.example",  # optional
                "X-Title": "FashionRAG",  # optional
            },
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # clean response
        response_llm = completion.choices[0].message.content
        item_texts = [item.strip() for item in response_llm.split(',')]
        weather_text = f"The weather now in {self.city} is {data['weather'][0]['main']} with a temperature of {data['main']['temp']}°F, ranging from a low of {data['main']['temp_min']}°F to a high of {data['main']['temp_max']}°F."
        return item_texts, weather_text
    
    def encode_query_items(self, item_texts):
        fclip = FashionCLIP("fashion-clip")  # already on CUDA
        fclip.device = "cuda"
        return fclip.encode_text(item_texts, batch_size=len(item_texts))

    # Retrieve real catalog items (nearest neighbor search)
    def retrieve_similar_items(self, item_text_embeds):
        knn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        knn.fit(self.text_embeddings.cpu().numpy())
        indices = knn.kneighbors(item_text_embeds.cpu().numpy(), return_distance=False)
        retrieved_ids = [[self.item_ids[i] for i in row] for row in indices]
        return retrieved_ids

    # Score combinations using your compatibility model
    def score_outfit(self, retrieved_ids):
        if not all(x in self.map_id_img_embed for x in retrieved_ids):
            return -np.inf
        embeds = torch.stack([self.map_id_img_embed[x] for x in retrieved_ids]).unsqueeze(0).cuda()
        with torch.no_grad():
            logit = self.model(embeds) # shape: [1]
            score = torch.sigmoid(logit).item()  # Now in [0, 1]
        return score

    # Generate and rerank outfit from prompt
    def start_generate(self):
        item_texts, weather_text = self.generate_outfit_from_llm()
        item_text_embeds = self.encode_query_items(item_texts)
        retrieved_ids = self.retrieve_similar_items(torch.tensor(item_text_embeds).cuda())
        score = self.score_outfit([x[0] if isinstance(x, list) else x for x in retrieved_ids])
        return f"{weather_text}\n" \
            f"We recommend you these items: {item_texts}, " \
            f"the corresponding ids are: {retrieved_ids}. " \
            f"This outfit is {score * 100:.1f}% likely to be compatible."

    
# Transformer-Based Scorer
class TransformerScorer(nn.Module):
    def __init__(self, embed_dim, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # aggregate item representations
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.transformer(x)                  # (batch, num_items, embed_dim)
        x = x.mean(dim=1)                        # (batch, embed_dim)
        return self.scorer(x).squeeze()          # (batch,)


def create_caption_for_image(img_path):
    with open(img_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    },
    extra_body={},
    model="google/gemini-pro-vision",
    # model="meta-llama/llama-3.2-90b-vision-instruct",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Look at the image and generate a short, "
                    "accurate caption for this clothing item"
                    "Example: 'white animal printed ladies t-shirt'. Output only the caption, "
                    "without extra commentary. Return no more than 15 words. It has to be accurate."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
                }}
            ]
        }
        ]
    )
    return completion.choices[0].message.content


def segment_and_paste_on_white(model, image_rgb):
    # # Load image
    # image = cv2.imread(image_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.array(image_rgb)

    # Run segmentation
    results = model(image_rgb)[0]  # get the first result
    masks = results.masks.data.cpu().numpy()  # shape: (num_instances, H, W)

    if len(masks) == 0:
        print("No object detected.")
        return

    # Combine all masks
    combined_mask = np.any(masks, axis=0).astype(np.uint8)

    # Resize mask to match original size (optional)
    combined_mask = cv2.resize(combined_mask, (image_rgb.shape[1], image_rgb.shape[0]))

    # Create white background
    white_bg = np.ones_like(image_rgb, dtype=np.uint8) * 255

    # Apply mask
    for c in range(3):  # for each color channel
        white_bg[:, :, c] = np.where(combined_mask == 1, image_rgb[:, :, c], 255)

    # Return as PIL Image
    return Image.fromarray(white_bg)