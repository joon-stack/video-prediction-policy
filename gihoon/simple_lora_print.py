# ------------------------------------------------------------------------------
# Stable Video Diffusion 모델(UNetSpatioTemporalConditionModel)에
# ① 기존 파라미터는 전부 고정(freeze)한 뒤
# ② 모든 Cross-Attention 모듈(attn2: Spatial + Temporal)에
#    LoRA 어댑터(Q/K/V/Out 전체, rank=8)를 삽입하고
# ③ 학습 가능한 LoRA 파라미터 수를 출력하는 스크립트
#
# ‣ Spatial Cross-Attention과 Temporal Cross-Attention을 별도 구분하지 않고
#   “attn2”가 포함된 모든 블록에 LoRA를 달아 Identity(appearance)와
#   Motion 모두를 동시에 미세조정할 수 있도록 구성
# ‣ diffusers 버전에 따라 set_attn_processor / set_processor 둘 다 지원
# ------------------------------------------------------------------------------

# =============================================================================
# LoRAAttnProcessor 하이퍼파라미터 설명
# -----------------------------------------------------------------------------
# • hidden_size :  Q/K/V/Out 선형층(in_features) 차원.  **필수**.
# • rank        :  저랭크 차원 r (A: d×r, B: r×d). 작게 잡으면 파라미터·VRAM↓,
#                 너무 작으면 표현력↓. 4·8·16이 실무 상 흔함.
# • network_alpha(α) :
#       - 원본 LoRA 논문은 최종 weight 를  W + (α / r) · (A·B) 로 합산.
#       - 기본값 None → diffusers는 α = rank 로 자동설정(즉 α/r = 1).
#       - α를 크게 하면 LoRA 효과(scale) ↑ → 빠른 적응·과적합 위험↑.
# • dropout     :  학습 시 (A·B)의 출력에 Dropout(p) 적용.
#                 0.0(기본)~0.1 정도 넣어주면 과적합 감소, 너무 크면 수렴↓.
# • init_weights:  A·B 행렬 초기화 방법.
#       "gaussian" (N(0, 0.01))  ←  대부분 사용.
#       "zeros"     (전부 0)     ←  overfit 억제·느린 시작.
# • use_bias    :  A·B 에 bias 벡터 추가 여부. 일반적으론 False.
# -----------------------------------------------------------------------------
# Inference 시:
#   pipe.set_lora_scale(γ)  # 0~1.  1 = 학습 때 그대로, 0 = LoRA OFF
#   γ를 낮추면 LoRA 강도↓ (base 모델 특성 더 반영), 높이면 LoRA 효과↑
# =============================================================================

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor

MODEL_ID = "yjguo/svd-robot-calvin-ft"
DEVICE   = "cuda"
RANK     = 8

pipe = StableVideoDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16
).to(DEVICE)

# 1. UNet freeze
for p in pipe.unet.parameters():
    p.requires_grad_(False)

inserted = 0
for blk_name, blk in pipe.unet.named_modules():
    if hasattr(blk, "attn2"):                      # Cross-Attention
        attn = blk.attn2

        # Q-proj 차원
        q_proj = getattr(attn, "to_q", None) or getattr(attn, "q_proj", None)
        hidden = q_proj.in_features # 입력을 차원을 받아오기 위함! 학습은 k q c out 다 해당함 
        
        # temporal - spatial 구분 없이 
        
        '''
        (attn2): Attention(
              (to_q): Linear(in_features=320, out_features=320, bias=False)
              (to_k): Linear(in_features=1024, out_features=320, bias=False)
              (to_v): Linear(in_features=1024, out_features=320, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=320, out_features=320, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
        '''
    
        lora_proc = LoRAAttnProcessor(hidden_size=hidden, rank=RANK)

        # diffusers 버전에 따라 메서드 선택
        if hasattr(attn, "set_attn_processor"):
            attn.set_attn_processor(lora_proc)
        elif hasattr(attn, "set_processor"):
            attn.set_processor(lora_proc)
        else:
            raise RuntimeError("이 어텐션 모듈은 LoRA 주입 메서드를 지원하지 않습니다.")

        inserted += 1

print(f"✔  LoRA inserted into {inserted} Cross-Attention modules")

# 학습 대상 = LoRA 행렬
lora_params = [p for n, p in pipe.unet.named_parameters() if "lora" in n.lower()]
print("trainable LoRA params :", sum(p.numel() for p in lora_params))