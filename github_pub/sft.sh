#!/bin/bash
#SBATCH --job-name=sft_qwen32b
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00
#SBATCH --output=/home/Competition2025/P12/P12U007/baseline/train/logs/output.out
#SBATCH --error=/home/Competition2025/P12/P12U007/baseline/train/logs/error.out

################################################################################
# スクリプト名: sft_qwen32b.sh
# 概要:
#   Qwen-3 32Bモデルに対し、教師ありファインチューニング（SFT）を行うための
#   SBATCHジョブスクリプト。
#
# 目的:
#   数学問題（MATHデータセット）を用いてモデルの推論能力を向上させる。
#   学習後、Hugging Faceフォーマットに変換し、オプションでHugging Face Hubにアップロードする。
#
# 前提条件:
#   - torchrun、verlなどの必要なライブラリが環境にインストールされていること。
#   - Qwen3-32Bモデルのベースモデルが$HOME/model/Qwen3-32Bに存在すること。
#   - 学習データが$HOME/data/math/に存在すること。
#   - Hugging Face Hubへのアップロードには、HF_TOKEN環境変数が設定されていること。
#
# 実行方法:
#   sbatch sft_qwen32b.sh
################################################################################


# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version

#step0 でインストールした conda のディレクトリ
export CONDA_PATH="~/conda_env"

source ~/.bashrc

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate

# 作成したPython仮想環境を有効化。
conda activate $CONDA_PATH

# Hugging Face 認証
export HF_TOKEN=<Huggingfaceのトークン>
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

# エラー時に停止
set -e

# Step 1: SFT（教師ありファインチューニング）の実行
echo "=== Step 1: Supervised Fine-Tuning (SFT) ==="

# ディレクトリ作成（パス統一）
# SFTのチェックポイントを保存するためのディレクトリ
mkdir -p ~/training/sft_Qwen3_math_1
mkdir -p ~/training/sft_Qwen3_math_1/checkpoints
cd ~/training/sft_Qwen3_math_1

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited
ulimit -m unlimited

# WandBのプロジェクト設定
export WANDB_PROJECT_NAME="Qwen3_32B_SFT_1"
export WANDB_RUN_NAME="Qwen3_32B_SFT_MATH"

echo "Starting SFT training..."

# SFT学習実行
# torchrunを使用して、verlのfsdp_sft_trainerを8GPUで実行
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=$HOME/data/math/train.parquet \
  data.val_files=$HOME/data/math/test.parquet \
  data.prompt_key=problem \
  data.response_key=solution \
  data.prompt_dict_keys=[] \
  +data.response_dict_keys=[] \
  data.micro_batch_size_per_gpu=4 \
  data.max_length=4096 \
  model.partial_pretrain=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
  '+model.attn_implementation=flash_attention_2' \
  '+model.torch_dtype=bfloat16' \
  model.use_liger=True \
  model.fsdp_config.model_dtype=bfloat16 \
  model.enable_gradient_checkpointing=true \
  trainer.project_name=$WANDB_PROJECT_NAME \
  trainer.experiment_name=$WANDB_RUN_NAME \
  trainer.total_epochs=4 \
  trainer.default_local_dir=$HOME/training/sft_Qwen3_math_1/checkpoints \
  trainer.logger=['console','wandb'] \
  trainer.resume_mode=disable \
  trainer.total_epochs=15 2>&1 | tee verl_sft.log

echo "SFT training completed"

# Step 2: チェックポイントの変換
echo "=== Step 2: Converting checkpoint to HuggingFace format ==="

# 最新のチェックポイントを探す
LATEST_CHECKPOINT=$(find $HOME/training/sft_Qwen3_math/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Converting checkpoint: $LATEST_CHECKPOINT"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT \
    --target_dir $LATEST_CHECKPOINT/huggingface

echo "Checkpoint conversion completed"

# Step 3: モデルのアップロード（オプション）
echo "=== Step 3: Model upload (optional) ==="

# HF_TOKENが設定されている場合は自動アップロード
if [ -n "$HF_TOKEN" ]; then
    echo "Uploading model to HuggingFace Hub..."
    huggingface-cli upload \
        Ta1k1/Qwen3-32B-SFT-MATH \
        $LATEST_CHECKPOINT/huggingface \
        --token $HF_TOKEN
    echo "Model upload completed"
else
    echo "HF_TOKEN not set. Upload manually if needed:"
    echo "huggingface-cli upload Ta1k1/Qwen3-32B-SFT-MATH $LATEST_CHECKPOINT/huggingface --token YOUR_TOKEN"
fi

# GPU監視プロセスを終了
kill $pid_nvsmi

echo "=== SFT Full Pipeline Completed ==="
echo "End time: $(date)"
echo "Checkpoint location: $LATEST_CHECKPOINT/huggingface"