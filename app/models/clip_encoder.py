import torch
import torch.nn as nn
import clip  # install with: !pip install git+https://github.com/openai/CLIP.git
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class CLIPTextEncoder(nn.Module):
    def __init__(self, device="cuda", model_name="ViT-B/32"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()  # CLIP is frozen

    @torch.no_grad()
    def forward(self, texts):
        # texts: List[str]
        tokenized = clip.tokenize(texts).to(self.device)  # [B, 77]
        embeddings = self.model.encode_text(tokenized)     # [B, 512]
        return embeddings / embeddings.norm(dim=-1, keepdim=True)
