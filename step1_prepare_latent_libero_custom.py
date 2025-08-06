import cv2
import os
import random
import math
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
import torch
import numpy as np
import json
import glob
import pandas as pd
import subprocess
import tempfile

# Libero 데이터셋 경로 설정
raw_data_path = '/shared/s2/lab01/junhachun/dataset/libero_combined_no_noops_lerobot_v21/videos'
task_name = "libero_custom"

output_dir = '/shared/s2/lab01/youngjoonjeong/vpp/libero_custom'

# 출력 디렉토리 설정
video_dir = os.path.join(output_dir, 'videos')
latent_video_dir = os.path.join(output_dir, 'latent_videos')
anno_dir = os.path.join(output_dir, 'annotation')
os.makedirs(video_dir, exist_ok=True)
os.makedirs(latent_video_dir, exist_ok=True)
os.makedirs(anno_dir, exist_ok=True)

def get_all_mp4_files():
    """모든 chunk에서 MP4 파일들을 수집합니다."""
    mp4_files = []
    
    # chunk-000과 chunk-001에서 파일 수집
    for chunk in ['chunk-000', 'chunk-001']:
        chunk_path = os.path.join(raw_data_path, chunk)
        
        # observation.images.image (메인 카메라)
        image_path = os.path.join(chunk_path, 'observation.images.image')
        if os.path.exists(image_path):
            image_files = glob.glob(os.path.join(image_path, '*.mp4'))
            mp4_files.extend(image_files)
        
        # observation.images.wrist_image (손목 카메라)
        wrist_path = os.path.join(chunk_path, 'observation.images.wrist_image')
        if os.path.exists(wrist_path):
            wrist_files = glob.glob(os.path.join(wrist_path, '*.mp4'))
            mp4_files.extend(wrist_files)
    
    return sorted(mp4_files)

def load_video(video_path):
    """MP4 비디오를 로드합니다. (ffmpeg 직접 사용)"""
    try:
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # ffmpeg으로 프레임 추출 (AV1 무시하고 강제로 처리)
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'fps=10',  # 10fps로 추출
                '-pix_fmt', 'rgb24',  # RGB 형식으로
                '-f', 'image2',
                os.path.join(temp_dir, 'frame_%04d.png'),
                '-y'  # 기존 파일 덮어쓰기
            ]
            
            # ffmpeg 실행 (에러 출력 무시)
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 추출된 프레임들 읽기
            frames = []
            frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.png')])
            
            for frame_file in frame_files:
                frame_path = os.path.join(temp_dir, frame_file)
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if frame is not None:
                    # BGR to RGB 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            if len(frames) > 0:
                print(f"Successfully loaded {len(frames)} frames with ffmpeg")
                return np.array(frames)
            else:
                print(f"No frames extracted from {video_path}")
                return None
                
    except Exception as e:
        print(f"ffmpeg failed for {video_path}: {e}")
        return None

def process_video(video, target_size=(256, 256)):
    """비디오를 전처리합니다. (cv2만 사용)"""
    if video is None:
        return None
    processed_frames = []
    for frame in video:
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        processed_frames.append(frame_resized)
    return np.array(processed_frames)

def load_task_descriptions():
    """tasks.jsonl 파일에서 작업 설명을 로드합니다."""
    tasks_file = '/shared/s2/lab01/junhachun/dataset/libero_combined_no_noops_lerobot_v21/meta/tasks.jsonl'
    task_descriptions = {}
    
    with open(tasks_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            task_descriptions[data['task_index']] = data['task']
    
    return task_descriptions

def get_task_description_from_parquet(episode_id):
    """parquet 파일에서 task_index를 읽어서 해당하는 작업 설명을 반환합니다."""
    # chunk-000과 chunk-001에서 해당 에피소드 파일 찾기
    for chunk in ['chunk-000', 'chunk-001']:
        parquet_path = f'/shared/s2/lab01/junhachun/dataset/libero_combined_no_noops_lerobot_v21/data/{chunk}/episode_{episode_id:06d}.parquet'
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                if 'task_index' in df.columns:
                    task_index = df['task_index'].iloc[0]  # 첫 번째 행의 task_index 사용
                    return task_index
            except Exception as e:
                print(f"Error reading parquet file {parquet_path}: {e}")
                continue
    
    return None

def convert_numpy_to_list(obj):
    """numpy 배열을 리스트로 변환하는 함수"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    else:
        return obj

def main():
    # 작업 설명 로드
    print("Loading task descriptions...")
    task_descriptions = load_task_descriptions()
    print(f"Loaded {len(task_descriptions)} task descriptions")
    
    # VAE 모델 로드
    print("Loading VAE model...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "/shared/s2/lab01/youngjoonjeong/stable_diffusion/stable-video-diffusion-img2vid", 
        subfolder="vae"
    ).to("cuda")
    
    # 모든 MP4 파일 수집
    print("Collecting MP4 files...")
    mp4_files = get_all_mp4_files()
    print(f"Found {len(mp4_files)} MP4 files")
    
    # 시작 인덱스 설정 (중간에 끊어진 경우 여기서 수정)
    start_index = 2818  # 0부터 시작, 중간에 끊어진 경우 이 값을 수정
    print(f"Starting from index {start_index}")
    
    failed_num = 0
    success_num = 0
    
    for file_num, file_path in enumerate(mp4_files[start_index:], start=start_index):
        try:
            # 에피소드 ID 추출
            filename = os.path.basename(file_path)
            episode_id = int(filename.replace('episode_', '').replace('.mp4', ''))
            
            # 카메라 타입에 따라 idx 설정
            if 'observation.images.image' in file_path:
                idx = 0  # 메인 카메라
            elif 'observation.images.wrist_image' in file_path:
                idx = 1  # 손목 카메라
            else:
                idx = 0  # 기본값
            
            # train/val 분할 (5%를 validation으로)
            data_type = 'val' if episode_id % 20 == 0 else 'train'
            
            print(f"Processing {filename} ({file_num + 1}/{len(mp4_files)})")
            
            # 비디오 로드 및 처리
            video = load_video(file_path)
            if video is None:
                failed_num += 1
                continue

            processed_video = process_video(video)
            if processed_video is None:
                failed_num += 1
                continue
            
            # 디버깅: processed_video 상태 확인
            print(f"processed_video shape: {processed_video.shape if processed_video is not None else 'None'}")
            print(f"processed_video dtype: {processed_video.dtype if processed_video is not None else 'None'}")
            
            # 실제 작업 설명 가져오기
            task_index = get_task_description_from_parquet(episode_id)
            if task_index is not None and task_index in task_descriptions:
                task_description = task_descriptions[task_index]
            else:
                print(f"Warning: Could not find task description for episode {episode_id}")
                raise Exception(f"Could not find task description for episode {episode_id}")
                task_description = "pick up the object and place it on the table"  # 기본 설명
            
            # 비디오를 텐서로 변환
            frames = torch.tensor(processed_video, dtype=torch.float32).permute(0, 3, 1, 2).to("cuda") / 255.0 * 2 - 1
            
            # VAE 인코딩
            with torch.no_grad():
                batch_size = 64
                latents = []
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                    latents.append(latent)
                latent_tensor = torch.cat(latents, dim=0)
            
            # 출력 디렉토리 생성
            latent_output_path = os.path.join(latent_video_dir, data_type, str(episode_id))
            os.makedirs(latent_output_path, exist_ok=True)
            
            # Latent 저장 (idx별로)
            torch.save(latent_tensor, os.path.join(latent_output_path, f"{idx}.pt"))
            
            # 실제 상태와 액션 데이터 가져오기
            states = []
            actions = []
            
            # parquet 파일에서 상태와 액션 데이터 읽기
            for chunk in ['chunk-000', 'chunk-001']:
                parquet_path = f'/shared/s2/lab01/junhachun/dataset/libero_combined_no_noops_lerobot_v21/data/{chunk}/episode_{episode_id:06d}.parquet'
                if os.path.exists(parquet_path):
                    try:
                        df = pd.read_parquet(parquet_path)
                        
                        # 상태 데이터 처리
                        if 'observation.state' in df.columns:
                            for state_str in df['observation.state']:
                                if isinstance(state_str, str):
                                    # 문자열을 리스트로 변환
                                    state_list = eval(state_str)
                                    states.append(state_list)
                                else:
                                    states.append(state_str)
                        
                        # 액션 데이터 처리
                        if 'action' in df.columns:
                            for action_str in df['action']:
                                if isinstance(action_str, str):
                                    # 문자열을 리스트로 변환
                                    action_list = eval(action_str)
                                    actions.append(action_list)
                                else:
                                    actions.append(action_str)
                        
                        break  # 파일을 찾았으면 루프 종료
                        
                    except Exception as e:
                        print(f"Error reading parquet file {parquet_path}: {e}")
                        continue
            
            # 비디오 길이 정의
            video_length = len(processed_video)
            
            # 데이터가 없으면 더미 데이터 생성
            if not states or not actions:
                print(f"Warning: Could not find state/action data for episode {episode_id}, using dummy data")
                states = np.random.randn(video_length, 38).tolist()  # 38차원 상태
                actions = np.random.randn(video_length, 7).tolist()   # 7차원 액션
            
            # JSON 어노테이션 생성
            info = {
                "task": "robot_trajectory_prediction",
                "texts": [task_description],
                "videos": [
                    {
                        "video_path": file_path  # 원본 MP4 파일 경로 사용
                    }
                ],
                "episode_id": episode_id,
                "video_length": video_length,
                "latent_videos": [
                    {
                        "latent_video_path": f"latent_videos/{data_type}/{episode_id}/{idx}.pt"
                    }
                ],
                "states": convert_numpy_to_list(states),
                "actions": convert_numpy_to_list(actions),
            }
            
            # JSON 파일 저장
            os.makedirs(f"{output_dir}/annotation/{data_type}", exist_ok=True)
            with open(f"{output_dir}/annotation/{data_type}/{episode_id}.json", "w") as f:
                json.dump(info, f, indent=2)
            
            success_num += 1
            print(f"Success: {task_description}, episode {episode_id}")
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            break
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_num += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Success: {success_num}")
    print(f"Failed: {failed_num}")
    print(f"Total: {len(mp4_files)}")

if __name__ == "__main__":
    main()

# 실행 명령어
# CUDA_VISIBLE_DEVICES=2 python step1_prepare_latent_libero_custom.py 