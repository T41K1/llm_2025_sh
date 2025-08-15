#!/bin/bash
#SBATCH --job-name=sft_qwen3_32b
#SBATCH --partition=P12
#SBATCH --nodes=2
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[84,85]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00



export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=64

# distributed settings
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "MASTER_ADDR=${MASTER_ADDR}"
MASTER_PORT=29500
echo "MASTER_PORT=${MASTER_PORT}"
NODE_RANK=${SLURM_PROCID}
echo "Node rank: "$NODE_RANK
NNODES=${SLURM_NNODES}
echo "Node num: "$NNODES
GPUS_PER_NODE=8
echo "GPUs per node: "$GPUS_PER_NODE
#srun --partition P12 --nodes=1 --nodelist osk-gpu[84,85] --gpus-per-node=8  --cpus-per-task=240 --time=30:00:00 --pty bash -i
source "$HOME/login.sh"


export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited


#実行ディレクトリを作成
mkdir -p ~/training/multinode/sft
#sbatchのログの保存パス
mkdir -p ~/training/multinode/sft/logs
#学習済みのモデル保存する場所
mkdir -p ~/training/multinode/sft/checkpoints



export WANDB_PROJECT_NAME="competition_verl_test"
export WANDB_RUN_NAME="llama3.2_SFT_test"

torchrun --rdzv_backend c10d \
         --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${NNODES} --nproc_per_node ${GPUS_PER_NODE} \
         --node_rank ${NODE_RANK} \
         -m verl.trainer.fsdp_sft_trainer \
         data.train_files=$HOME/data/gsm8k/train.parquet \
         data.val_files=$HOME/data/gsm8k/test.parquet \
         data.prompt_key=extra_info \
         data.response_key=extra_info \
         data.prompt_dict_keys=['question'] \
         +data.response_dict_keys=['answer'] \
         data.micro_batch_size_per_gpu=8 \
         model.partial_pretrain=$HOME/model/Llama-3.2-1B-Instruct \
         trainer.project_name=gsm8k-sft \
         trainer.experiment_name=$HOME/model/Llama-3.2-1B-Instruct \
         trainer.total_epochs=2 \
         trainer.default_local_dir=$HOME/training/multinode/sft/checkpoints \
         trainer.logger=['console','wandb'] \
         trainer.project_name=$WANDB_PROJECT_NAME \
         trainer.experiment_name=$WANDB_RUN_NAME \
         +checkpoint.cpu_offload=true \
         +fsdp.load_on_cpu=true \
         +model.init_device=meta \
         +fsdp.param_init=meta \
         +fsdp.state_dict_type=sharded \