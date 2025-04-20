from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
from app.models.generator import ConditionalUNetGenerator
from app.models.clip_encoder import CLIPTextEncoder
from app.inference_utils import generate_design_from_upload

# إعدادات عامة
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "/content/drive/MyDrive/Roomify/checkpoints_512/generator_epoch4.pth"

# تحميل النموذج وتشفير النص مرة واحدة عند تشغيل السيرفر
generator = ConditionalUNetGenerator(in_channels=518, out_channels=3).to(DEVICE)
generator.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
generator.eval()

clip_encoder = CLIPTextEncoder(device=DEVICE).eval()

# إعداد FastAPI
app = FastAPI(
    title="Roomify AI API",
    description="API for generating interior design images based on room image and text prompt",
    version="1.0"
)

# إعداد CORS للسماح بطلبات من الويب أو الموبايل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # عدلها لو عايز السماح لدومين معين
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 API endpoint
@app.post("/generate-design")
async def generate_endpoint(prompt: str = Form(...), image: UploadFile = File(...)):
    """
    Generate a new room design based on input image and description.
    """
    output_path = await generate_design_from_upload(image, prompt, generator, clip_encoder, device=DEVICE)
    return {"status": "success", "generated_image_path": output_path}
