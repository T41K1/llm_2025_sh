#!/bin/bash
#SBATCH --job-name=grpo_full_pipeline
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00


# Command(s) goes here
nvidia-smi
# エラー時に停止
set -e

# Step 1-4: 強化学習（GRPO）の実行
echo "=== Step 1-4: GRPO Training ==="

# ディレクトリ作成（パス統一）
mkdir -p ~/training/sft_grpo_custom_reward
mkdir -p ~/training/sft_grpo_custom_reward/checkpoints
cd ~/training/sft_grpo_custom_reward

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited

# Ray クラスターの起動
echo "Starting Ray cluster..."
ray stop  # 既存のRayプロセスを停止
# Rayのヘッドノードを起動
# --num-cpusはノードのCPU数に合わせて調整
# --num-gpusは使用するGPUの数に合わせて調整
ray start --head --port=6379 --num-cpus=240 --num-gpus=8
echo "Ray cluster started"

export WANDB_PROJECT_NAME="llm_2025_grpo"
export WANDB_RUN_NAME="Qwen3_32b_SFT_GRPO_000"

echo "Starting GRPO training..."

# GRPO学習実行(omni-math)
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
 data.val_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/val.parquet \
 data.train_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/train.parquet \
 trainer.n_gpus_per_node=8 \
 data.train_batch_size=128 \
 data.max_prompt_length=256 \
 data.max_response_length=1024 \
 data.dataloader_num_workers=0 \
 data.prompt_key=question \
 ++data.response_key=r1_solution_1 \
 custom_reward_function.path=$HOME/custom_reward.py \
 custom_reward_function.name=compute_score \
 actor_rollout_ref.model.path=$HOME/model/Qwen3_SFT_MATH/checkpoints/global_step_116/huggingface \
 actor_rollout_ref.actor.optim.lr=5e-7 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
 algorithm.adv_estimator=grpo \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.val_before_train=False \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.default_local_dir=$HOME/training/sft_grpo_custom_reward/checkpoints \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$WANDB_PROJECT_NAME \
 trainer.experiment_name=$WANDB_RUN_NAME \
 trainer.total_epochs=15 2>&1 | tee verl_grpo.log

echo "GRPO training completed"

# Step 1-5: チェックポイントの変換
echo "=== Step 1-5: Converting checkpoint to HuggingFace format ==="

# 最新のチェックポイントを探す
LATEST_CHECKPOINT=$(find $HOME/training/sft_grpo_001/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Converting checkpoint: $LATEST_CHECKPOINT"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT/actor \
    --target_dir $LATEST_CHECKPOINT/actor/huggingface

echo "Checkpoint conversion completed"

# Step 1-6: モデルのアップロード（オプション）
echo "=== Step 1-6: Model upload (optional) ==="

# HF_TOKENが設定されている場合は自動アップロード
if [ -n "$HF_TOKEN" ]; then
    echo "Uploading model to HuggingFace Hub..."
    huggingface-cli upload \
        Ta1k1/Qwen3-32B-SFT-GRPO \
        $LATEST_CHECKPOINT/actor/huggingface \
        --token $HF_TOKEN
    echo "Model upload completed"
else
    echo "HF_TOKEN not set. Upload manually if needed:"
    echo "huggingface-cli upload Ta1k1/Qwen3-32B-SFT-GRPO $LATEST_CHECKPOINT/actor/huggingface --token YOUR_TOKEN"
fi

echo "=== GRPO Full Pipeline Completed ==="
echo "End time: $(date)"
echo "Checkpoint location: $LATEST_CHECKPOINT/actor/huggingface"