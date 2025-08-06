import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torchvision.transforms as T
import numpy as np
import imageio
import os
from video_models.pipeline import MaskStableVideoDiffusionPipeline

# ------------------------------
# 설정
# ------------------------------

VIDEO_MODEL_PATH = "yjguo/svd-robot-calvin-ft"#"yjguo/svd-robot"#"yjguo/svd-robot-calvin-ft"#"yjguo/svd-robot"  # SVD 모델 경로 # "yjguo/svd-robot-calvin-ft"
CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"   # CLIP 모델 경로
PROMPT = "take the red block."
INPUT_IMAGE_PATH = "/home/s2/gihoonkim/gihoon/shared_gihoon/videodiff/frames_output/frame_000.png"  # 초기 이미지
OUTPUT_PATH = "/home/s2/gihoonkim/gihoon/shared_gihoon/videodiff/video-prediction-policy/output_video.mp4"

# ------------------------------
# 텍스트 인코딩 함수
# ------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def encode_text(texts, tokenizer, text_encoder, position_encode=True, max_length=20):
    with torch.no_grad():
        inputs = tokenizer(texts, padding='max_length', return_tensors="pt",
                           truncation=True, max_length=max_length).to(text_encoder.device)
        outputs = text_encoder(**inputs)
        encoder_hidden_states = outputs.last_hidden_state  # (batch, seq_len, 512)

        if position_encode:
            embed_dim, pos_num = encoder_hidden_states.shape[-1], encoder_hidden_states.shape[1]
            pos = np.arange(pos_num, dtype=np.float64)
            position_encode = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
            position_encode = torch.tensor(position_encode, device=encoder_hidden_states.device,
                                           dtype=encoder_hidden_states.dtype, requires_grad=False)
            encoder_hidden_states += position_encode

        # 최종 shape: (batch, seq_len, 1024) = text_embed x2
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=-1)
        
        ## 추가 
        encoder_hidden_states = encoder_hidden_states[:, :1]   # (B,1,1024)
    return encoder_hidden_states

# ------------------------------
# Inference 함수
# ------------------------------
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Stable Video Diffusion 모델 로드
    pipeline = StableVideoDiffusionPipeline.from_pretrained(VIDEO_MODEL_PATH, torch_dtype=torch.float16).to(device)

    # 2) CLIP tokenizer & encoder 로드
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_PATH, use_fast=False)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_PATH).to(device)
    text_encoder.requires_grad_(False)

    # 3) 텍스트 임베딩 생성
    text_embeds = encode_text(PROMPT, tokenizer, text_encoder, position_encode=True)

    # 4) 초기 이미지 로드 및 전처리
    init_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(init_image).unsqueeze(0).to(device)

    # 5) 비디오 생성
    videos = MaskStableVideoDiffusionPipeline.__call__(
        pipeline,
        image=image_tensor,
        text=text_embeds,
        width=256,
        height=256,
        num_frames=32,
        num_inference_steps=30,
        decode_chunk_size=2,
        fps=4,
        motion_bucket_id=127,
        mask=None
    ).frames

    # 6) 결과 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    writer = imageio.get_writer(OUTPUT_PATH, fps=4)
    for frame in videos[0]:  # batch 1개 가정
        writer.append_data(np.array(frame))
    writer.close()

    print(f"Saved video to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_inference()
