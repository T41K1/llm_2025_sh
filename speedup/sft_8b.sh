#!/bin/bash
#SBATCH --job-name=sft_qwen8b_deep_math
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[84]
#SBATCH --cpus-per-task=240
#SBATCH --time=30:00:00

#gpu高速化

