#!/bin/bash
#SBATCH --job-name=svd_ft_lora                 # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                         # Using 1 gpu
#SBATCH --time=4-00:00:00                     # 1 hour timelimit
#SBATCH --mem=50GB                         # Using 10GB CPU Memory
#SBATCH --partition=laal5                     # Using "b" partition 
#SBATCH --cpus-per-task=12                     # Using 4 maximum processor
#SBATCH --output=svd_ft_lora_%j.log      # 로그 파일 이름에 job ID가 포함되도록 설정

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate vpp

accelerate launch --main_process_port 29506 --num_processes 1 step1_train_svd_lora_0807.py --config video_conf/train_libero_svd_lora.yaml
