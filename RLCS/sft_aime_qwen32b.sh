#!/bin/bash
#SBATCH --job-name=sft_qwen32b
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[84]
#SBATCH --cpus-per-task=240
#SBATCH --time=12:00:00

mkdir -p ~/training/sft_aime_qwen32b

mkdir -p ~/training/sft_aime_qwen32b/checkpoints

cd ~/training/sft_aime_qwen32b
#基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited


#今回、AIME2024(29問)の数学問題に対して以下を実施しました。
# ※AIME2024には、Answer以外にSolution(解法)が付属しています。
# (1)ヒント(解法ステップ)を問題文(プロンプト)にフル実装
# (2)重要度の高いヒント(解法ステップ)の半分程度を問題文(プロンプト)に実装
# (3)最重要の解法のヒントのみを問題文(プロンプト)に実装
# (1)->(2)->(3)->(元の問題)に行くほどヒントが減少し段階的に学習できることを期待。
# データセットは、/home/Competition2025/P12/shareP12/datasets/synthetic_datasets/amouri_20250808_qwen3-32B_sampleに格納しています。ファイル名は以下の通りです。
# -------------------------------------------------------------------
# (1) AIME2024_QwFullH.parquet : 29問
# (2) AIME2024_QwHalfH.parquet : 29問
# (3) AIME2024_QwOneH.parquet : 29問
# (元) AIME2024_Q.parquet : 29問


#データセットに関しては前処理をしました。そのままではカラム名の都合場合で学習できません。
#YOU_TEAM を wandb の組織名に置き換えてください。
export WANDB_PROJECT_NAME="llm_2025_rlcs"
export WANDB_RUN_NAME="Qwen32b_SFT_AIME"

export datasets="/home/Competition2025/P12/P12U025/data"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$datasets/AIME2024_QwFullH.parquet \
    data.val_files=$datasets/AIME2025_Q.parquet \
    data.max_length=4096 \
    data.prompt_key=problem \
    data.response_key=solution \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
    trainer.experiment_name=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
    trainer.total_epochs=2 \
    model.lora_rank=32 \
    model.lora_alpha=32 \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$HOME/training/sft_aime_qwen32b/checkpoints \
    trainer.logger=["console","wandb"] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_RUN_NAME | tee ~/training/sft_aime_qwen32b/verl_sft.log



#datsets 前処理 pathと、.parquetファイル名は適宜変更してください。
# python - <<'PY'
# import pandas as pd

# # 元データ読み込み
# df = pd.read_parquet("AIME2024_QwFullH.parquet")

# # user と assistant を抽出
# df["problem"] = df["messages"].apply(lambda msgs: msgs[0]["content"])
# df["solution"] = df["messages"].apply(lambda msgs: msgs[1]["content"])

# # messages カラムを削除
# df = df.drop(columns=["messages"])

# # 上書き保存
# df.to_parquet("AIME2024_QwFullH.parquet", index=False)

# print("変換完了:", df.shape, "→ AIME2024_QwFullH.parquet に上書き保存しました")
# PY


