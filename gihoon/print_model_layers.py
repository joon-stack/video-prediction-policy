from diffusers import StableVideoDiffusionPipeline
import torch

VIDEO_MODEL_PATH = "yjguo/svd-robot-calvin-ft"   # SVD 모델 경로
OUTPUT_TXT       = "unet_full_structure.txt"     # 저장 파일명

# 1) 모델 로드
pipe = StableVideoDiffusionPipeline.from_pretrained(
    VIDEO_MODEL_PATH, torch_dtype=torch.float16
).to("cuda")

# 2) UNet 전체 구조를 (모듈 객체 문자열 : 이름) 형태로 저장
with open(OUTPUT_TXT, "w") as f:
    f.write("=== UNet 모듈 전체 구조 ===\n")
    for name, module in pipe.unet.named_modules():
        f.write(f"{module} : {name}\n")

print(f"UNet 구조를 '{OUTPUT_TXT}' 파일로 저장했습니다.")
