from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import shutil
import os
import json
import torch
import numpy as np
from PIL import Image
from building_blocks import FashionCLIP, TransformerScorer, OutfitGenerator, segment_and_paste_on_white, create_caption_for_image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoConfig
import cv2
import numpy as np
from ultralytics import YOLO
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.ip_adapter.ip_adapter import IPAdapter
import io
import base64
from fastapi.responses import JSONResponse
import hashlib

app = FastAPI()

# Serve wardrobe images
app.mount("/wardrobe", StaticFiles(directory="wardrobe"), name="wardrobe")

# Allow React frontend
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

WARDROBE_FOLDER = "wardrobe"
os.makedirs(WARDROBE_FOLDER, exist_ok=True)

fclip = FashionCLIP("fashion-clip")
fclip.device = "cuda"

# Load YOLOv8 segmentation model
yolo = YOLO("yolov8s-seg.pt")  # You can try 'yolov8n-seg.pt' if speed is more important

model = TransformerScorer(embed_dim=512)
model.load_state_dict(torch.load("../checkpoints_fitb/epoch5.pt"))
model = model.cuda()
model.eval()


# Load/prepare for stable diffusion
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "../models/ip_adapter/clip-vit-large-patch14/"
ip_ckpt = "../models/ip_adapter/models/ip-adapter_sd15.bin"
device = "cuda"
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)


@app.get("/check_user/")
def check_user(user_name: str):
    user_folder = os.path.join(WARDROBE_FOLDER, user_name)
    if os.path.exists(user_folder):
        with open(os.path.join(user_folder, "embeddings.json"), "r") as f:
            wardrobe = json.load(f)
        return {"exists": True, "wardrobe": wardrobe}
    else:
        return {"exists": False}

@app.post("/upload_wardrobe/")
async def upload_wardrobe(user_name: str = Form(...), image: UploadFile = File(...)):
    if not user_name:
        return {"error": "User name is required."}

    user_folder = os.path.join(WARDROBE_FOLDER, user_name)
    os.makedirs(user_folder, exist_ok=True)

    embeddings_path = os.path.join(user_folder, "embeddings.json")
    hash_file_path = os.path.join(user_folder, "image_hashes.json")

    if not os.path.exists(embeddings_path):
        with open(embeddings_path, "w") as f:
            json.dump([], f)

    if not os.path.exists(hash_file_path):
        with open(hash_file_path, "w") as f:
            json.dump([], f)

    with open(hash_file_path, "r") as f:
        existing_hashes = json.load(f)

    # Compute hash of uploaded image
    image_bytes = await image.read()
    uploaded_image_hash = hashlib.sha256(image_bytes).hexdigest()
    image.file.seek(0)  # Reset pointer for saving

    # Check for duplicate
    if uploaded_image_hash in existing_hashes:
        return {"message": "This item is already in your wardrobe."}
    else:
        existing_hashes.append(uploaded_image_hash)

    image_path = os.path.join(user_folder, image.filename)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Open and preprocess image
    img = Image.open(image_path).convert("RGB")
    processed_img = segment_and_paste_on_white(yolo, img)

    # Resize to max size if needed
    max_size = (768, 768)
    if processed_img.height > max_size[0] or processed_img.width > max_size[1]:
        processed_img.thumbnail(max_size, Image.Resampling.LANCZOS)

    processed_img.save(image_path)  # Save cleaned and resized image

    # Generate caption
    caption = create_caption_for_image(image_path)

    # Generate embeddings
    img_emb = fclip.encode_images([processed_img], batch_size=1)[0]
    txt_emb = fclip.encode_text([caption], batch_size=1)[0]

    new_entry = {
        "filename": image.filename,
        "path": f"wardrobe/{user_name}/{image.filename}",
        "caption": caption,
        "image_embedding": img_emb.tolist(),
        "text_embedding": txt_emb.tolist(),
    }

    with open(embeddings_path, "r") as f:
        existing = json.load(f)
    existing.append(new_entry)

    with open(embeddings_path, "w") as f:
        json.dump(existing, f)
    
    with open(hash_file_path, "w") as f:
        json.dump(existing_hashes, f)

    return {"user": user_name, "wardrobe": [new_entry]}

@app.get("/options/")
def get_available_tasks():
    return {
        "message": "What would you like help with today?",
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
async def generate_outfit(
    user_name: str = Query(...),
    city: str = Query(...),
    occasion: str = Query(...),
    sex: str = Query(...)
):
    user_folder = os.path.join(WARDROBE_FOLDER, user_name)
    with open(os.path.join(user_folder, "embeddings.json"), "r") as f:
        embeddings = json.load(f)

    item_ids = [e["filename"] for e in embeddings]
    text_embeddings = torch.tensor(np.array([e["text_embedding"] for e in embeddings])).float().cuda()
    map_id_img_embed = {
        e["filename"]: torch.tensor(e["image_embedding"]).float().cuda()
        for e in embeddings
    }

    og = OutfitGenerator(text_embeddings, map_id_img_embed, item_ids, model, city, occasion, sex)
    result = og.start_generate()
    return result


@app.post("/generate_virtual_tryon/")
async def generate_virtual_tryon(recommend_items:str = Form(...), occasion: str = Form(...)):
    print("Received items:", recommend_items)
    print("Occasion:", occasion)
    image = Image.open("../girl.png")
    image = image.resize((256, 256))
    prompt = f"best quality, high quality, wearing {recommend_items} for the occasion {occasion}"
    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42,
        prompt=prompt, scale=1.0)
    generated_image = images[0]

    # Convert result to base64 for frontend
    buffer = io.BytesIO()
    generated_image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()

    return JSONResponse(content={"generated_image_base64": image_base64})