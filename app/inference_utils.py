import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from uuid import uuid4
from fastapi import UploadFile

# إعداد ثابت
IMAGE_SIZE = (512, 512)
OUTPUT_DIR = "/content/drive/MyDrive/Roomify/generated_api_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✨ الدالة الأساسية لتوليد التصميم من صورة ووصف
async def generate_design_from_upload(image_file: UploadFile, prompt: str, generator, clip_encoder, device="cuda"):
    # 1. قراءة الصورة من UploadFile
    image_bytes = await image_file.read()
    image = Image.open(image_file.file).convert("RGB")

    # 2. تحويل الصورة إلى Tensor
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 3. توليد mask (قناع)
    mask = torch.ones_like(image_tensor).to(device)

    # 4. تشفير النص
    with torch.no_grad():
        text_embedding = clip_encoder([prompt]).to(device)

    # 5. توليد الصورة الجديدة
    generator.eval()
    with torch.no_grad():
        generated_image = generator(image_tensor, mask, text_embedding)
        generated_image = torch.clamp(generated_image, 0, 1)

    # 6. حفظ الصورة في Google Drive
    filename = f"generated_{uuid4().hex[:8]}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    save_image(generated_image, output_path)

    return output_path
