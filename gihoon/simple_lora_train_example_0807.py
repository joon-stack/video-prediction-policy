import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
import imageio
import os

from diffusers import StableVideoDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from peft import LoraConfig, get_peft_model

# ── 설정 ──
VIDEO_MODEL_PATH = "yjguo/svd-robot-calvin-ft"
CLIP_MODEL_PATH  = "openai/clip-vit-base-patch32"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# 학습용 더미 데이터 파라미터
BATCH_SIZE = 2
NUM_FRAMES = 4
CHANNELS   = 3
HEIGHT     = 256
WIDTH      = 256
DATASET_LEN = 8

# LoRA/학습 파라미터
RANK    = 8
ALPHA   = 16
LR      = 1e-4
EPOCHS  = 1  # 테스트용

# ------------------------------
# 더미 데이터셋
# ------------------------------
class DummyVideoDataset(Dataset):
    def __len__(self):
        return DATASET_LEN
    def __getitem__(self, idx):
        # 랜덤 노이즈 프레임
        frames = torch.randn(NUM_FRAMES, CHANNELS, HEIGHT, WIDTH)
        # 고정 프롬프트
        prompt = "A test prompt"
        return {"frames": frames, "text": prompt}

dataset = DummyVideoDataset()
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
# 텍스트 인코딩 함수 (인용 코드 사용)
# ------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb

def encode_text(texts, tokenizer, text_encoder, max_length=20):
    with torch.no_grad():
        inputs = tokenizer(texts, padding='max_length',
                           truncation=True, max_length=max_length,
                           return_tensors="pt").to(text_encoder.device)
        outputs = text_encoder(**inputs)
        hs = outputs.last_hidden_state  # (B, seq, C)
        # position encode
        B, seq, C = hs.shape
        pos = np.arange(seq, dtype=np.float64)
        pe  = get_1d_sincos_pos_embed_from_grid(C, pos)
        pe  = torch.from_numpy(pe).to(hs.device).unsqueeze(0)  # (1, seq, C)
        hs = hs + pe
        # concat for SVD-embedding
        hs = torch.cat([hs, hs], dim=-1)  # (B, seq, 2C)
        # take only first token
        hs = hs[:, :1]
    return hs

# ------------------------------
# 1) 파이프라인 로드 & LoRA 세팅
# ------------------------------
pipe = StableVideoDiffusionPipeline.from_pretrained(
    VIDEO_MODEL_PATH,
    torch_dtype=torch.float16,
).to(DEVICE)
pipe.enable_attention_slicing()

# UNet 분리 후 freeze
unet = pipe.unet
unet.requires_grad_(False)

# PEFT LoRA 삽입
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none",
    #task_type="UNET"
)

unet = get_peft_model(unet, lora_config)
pipe.unet = unet

# 옵티마이저: LoRA 파라미터만
optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)

# CLIP 토크나이저·인코더 로드 (학습에서는 고정)
tokenizer   = AutoTokenizer.from_pretrained(CLIP_MODEL_PATH, use_fast=False, use_safetensors=True)
text_encoder = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_PATH, use_safetensors=True).to(DEVICE)

text_encoder.requires_grad_(False)

# ------------------------------
# 2) 테스트 학습 루프 (더미 데이터로 Forward만)
# ------------------------------

unet.train()
for epoch in range(EPOCHS):
    for batch in loader:
        optimizer.zero_grad()
        frames = batch["frames"].to(DEVICE, torch.float16)  # (B, T, C, H, W)
        prompts = [batch["text"][i] for i in range(len(batch["text"]))]
        text_embeds = encode_text(prompts, tokenizer, text_encoder)  # (B, 1, 1024)

        # init_frame: 첫 프레임만 사용
        init_frame = frames[:, 0]  # (B, C, H, W)

        # SVD Inference 호출 (loss 리턴)
        out = pipe(
            init_image=init_frame,
            text=text_embeds,
            width=WIDTH, height=HEIGHT,
            num_frames=NUM_FRAMES,
            return_dict=True,
            output_type="pt"
        )
        loss = out.loss
        loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch+1}] loss: {loss.item():.4f}")
        break  # 테스트용 1배치만

print("더미 학습 루프 정상 종료!")
