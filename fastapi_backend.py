from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import json
import torch
import numpy as np
from PIL import Image
from building_blocks import FashionCLIP, TransformerScorer, OutfitGenerator
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base wardrobe folder
WARDROBE_FOLDER = "wardrobe"
os.makedirs(WARDROBE_FOLDER, exist_ok=True)

# Load models once
fclip = FashionCLIP("fashion-clip")
fclip.device = "cuda"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model = TransformerScorer(embed_dim=512)
model.load_state_dict(torch.load("checkpoints_fitb/epoch5.pt"))
model = model.cuda()
model.eval()

@app.post("/upload_wardrobe/")
async def upload_wardrobe(user_name: str = Form(...), images: List[UploadFile] = File(...)):
    user_folder = os.path.join(WARDROBE_FOLDER, user_name)
    os.makedirs(user_folder, exist_ok=True)

    embeddings = []
    for i, image in enumerate(images):
        save_path = os.path.join(user_folder, image.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        img = Image.open(save_path).convert("RGB")
        inputs = processor(img, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        img_emb = fclip.encode_images([img])[0]
        txt_emb = fclip.encode_text([caption])[0]

        embeddings.append({
            "index": i,
            "filename": image.filename,
            "path": save_path,
            "caption": caption,
            "image_embedding": img_emb.tolist(),
            "text_embedding": txt_emb.tolist(),
        })

    with open(os.path.join(user_folder, "embeddings.json"), "w") as f:
        json.dump(embeddings, f)

@app.get("/options/")
def get_available_tasks():
    return {
        "message": "Hey there! I'm your AI stylist. How can I help you today?",
        "options": [
            "1. Check if your outfit looks good together",
            "2. Suggest a new outfit for your occasion"
        ]
    }

@app.post("/predict_compatibility/")
async def predict_compatibility(user_name: str = Form(...), item_ids: List[str] = Form(...)):
    user_folder = os.path.join(WARDROBE_FOLDER, user_name)
    with open(os.path.join(user_folder, "embeddings.json"), "r") as f:
        embeddings = json.load(f)

    embed_map = {e["filename"]: torch.tensor(e["image_embedding"]).float() for e in embeddings}
    if not all(x in embed_map for x in item_ids):
        return {"score": -1, "message": "Some items are missing in the wardrobe."}

    embeds = torch.stack([embed_map[x] for x in item_ids]).unsqueeze(0).cuda()
    with torch.no_grad():
        score = torch.sigmoid(model(embeds)).item()

    return {
        "score": score,
        "message": f"This outfit is {score * 100:.1f}% likely to be compatible."
    }

@app.post("/generate_outfit/")
async def generate_outfit(user_name: str = Form(...)):
    user_folder = os.path.join(WARDROBE_FOLDER, user_name)
    with open(os.path.join(user_folder, "embeddings.json"), "r") as f:
        embeddings = json.load(f)

    item_ids = [e["filename"] for e in embeddings]
    text_embeddings = torch.tensor(np.array([e["text_embedding"] for e in embeddings])).float().cuda()
    map_id_img_embed = {
        e["filename"]: torch.tensor(e["image_embedding"]).float().cuda()
        for e in embeddings
    }

    og = OutfitGenerator(text_embeddings, map_id_img_embed, item_ids, model)
    result = og.start_generate()