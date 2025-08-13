#!/bin/bash
#SBATCH --job-name=sft_qwen3_32b
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00

#srun --partition P12 --nodes=1 --nodelist osk-gpu[85] --gpus-per-node=8  --cpus-per-task=120 --time=30:00:00 --pty bash -i

source "$HOME/login.sh"

# 失敗を即検知（最初の数回だけ残せばOK）
echo "=== Conda Env Check ==="
echo "which python: $(which python)"; python -V
conda info --envs | sed -n '1,20p'
if [ "$(which python)" != "/home/Competition2025/P12/P12U025/conda_env/bin/python" ]; then
  echo "❌ envが有効化されていません。中断します。" >&2; exit 1
fi

echo "=== Conda Env Check ==="
echo "which python: $(which python)"
python -V
conda info --envs
echo "======================="


mkdir -p ~/training/sft_Qwen3-32B_deepmath_val
mkdir -p ~/training/sft_Qwen3-32B_deepmath_val/checkpoints

cd ~/training/sft_Qwen3-32B_deepmath_val



#基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
ulimit -v unlimited


export WANDB_PROJECT_NAME="competition_sft_qwen3_32b"
export WANDB_RUN_NAME="Qwen3-32B_deepmath_val"






torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m verl.trainer.fsdp_sft_trainer \
  data.val_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/val.parquet \
  data.train_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/train.parquet \
  data.prompt_key=question \
  data.response_key=r1_solution_1 \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=2 \
  model.fsdp_config.model_dtype=bf16 \
  +data.dataloader_num_workers=16 \
  data.max_length=8000 \
  +trainer.val_before_train=True \
  ++data.filter_overlong_prompts=True \
  data.truncation=right \
  trainer.test_freq=0 \
  trainer.save_freq=0 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/home/Competition2025/P12/P12U025/training/sft_Qwen3-32B_deepmath/checkpoints/global_step_1000/huggingface \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT_NAME \
  +model.override_config.attn_implementation=flash_attention_2 \
  +model.use_remove_padding=True \
  +model.use_fused_kernels=True \
  model.enable_gradient_checkpointing=True  \
  ++model.fsdp_config.forward_prefetch=True \
  trainer.experiment_name=$WANDB_RUN_NAME | tee $HOME/training/sft_Qwen3-32B_deepmath_val/verl_sft.log


torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/train.parquet \
  data.val_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/val.parquet \
  data.prompt_key=question \
  data.response_key=r1_solution_1 \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=2 \
  data.max_length=8000 \
  +data.dataloader_num_workers=16 \
  model.fsdp_config.model_dtype=bf16 \
  data.truncation=right \
  +trainer.val_before_train=True \
  ++data.filter_overlong_prompts=True \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/home/Competition2025/P12/P12U025/training/sft_Qwen3-32B_deepmath/checkpoints/global_step_1000/huggingface \
  model.partial_pretrain=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
  trainer.total_epochs=2 \
  trainer.save_freq=0 \
  trainer.test_freq=0 \
  trainer.default_local_dir=$HOME/training/sft_Qwen3-32B_deepmath_0811/checkpoints \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT_NAME \
  +model.override_config.attn_implementation=flash_attention_2 \
  +model.use_remove_padding=True \
  +model.use_fused_kernels=True \
  model.enable_gradient_checkpointing=True  \
  ++model.fsdp_config.forward_prefetch=True \
  trainer.experiment_name=$WANDB_RUN_NAME | tee ~/training/sft_Qwen3-32B_deepmath_val/verl_sft.log