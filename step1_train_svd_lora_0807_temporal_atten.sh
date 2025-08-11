#!/bin/bash
#SBATCH --job-name=svd_ft_lora_t            # Job name에 job ID가 포함되도록 설정
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node (1 task per node)
#SBATCH --gres=gpu:1                   # Number of GPUs per node
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --time=0-12:00:00               # Maximum time limit for the job
#SBATCH --partition=P2                 # Use the correct partition (e.g., P2)
#SBATCH --mem=50G
#SBATCH --output=svd_ft_lora_t%j.log      # 로그 파일 이름에 job ID가 포함되도록 설정

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate vpp

#accelerate launch --main_process_port 29506 --num_processes 1 step1_train_svd_lora_0807_temporal_atten.py --config video_conf/train_libero_svd_lora_temporal.yaml
accelerate launch --main_process_port 29506 --num_processes 1 step1_train_svd_lora_0807_temporal_atten.py --config video_conf/train_libero_svd_lora_temporal.yaml -- resume_from_checkpoint=/home/s2/gihoonkim/gihoon/shared_gihoon/videodiff/robodiff/video-prediction-policy/results_temporal/train_2025-08-07T09-41-36/checkpoint-210000
