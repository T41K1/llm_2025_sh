#!/bin/bash
#SBATCH --job-name=sft_qwen3_32b
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[84]
#SBATCH --cpus-per-task=240
#SBATCH --time=50:00:00

mkdir -p ~/training/sft_Qwen3-32B_deepmath
mkdir -p ~/training/sft_Qwen3-32B_deepmath/checkpoints

cd ~/training/sft_Qwen3-32B_deepmath

#基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited


export WANDB_PROJECT_NAME="competition_sft_qwen3_32b"
export WANDB_RUN_NAME="Qwen3-32B_deepmath"


# 以下でdeepmathをtrain valに分ける
# cd /home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data

# python - <<'EOF'
# import os
# import sys
# import pandas as pd
# import numpy as np

# # ベースパス
# base = "/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data"

# # 連番パターン（train-00000-of-00010.parquet ... -00009-of-00010.parquet）
# files = [os.path.join(base, f"train-000{i:02d}-of-00010.parquet") for i in range(10)]

# # 存在チェック
# missing = [p for p in files if not os.path.exists(p)]
# if missing:
#     print("❌ 次のファイルが見つかりませんでした:")
#     for p in missing:
#         print("   ", p)
#     sys.exit(1)
# print("✅ 全ファイル存在を確認")

# # 読み込み＆結合（メモリに乗る前提。厳しい場合はPyArrowストリーミングを検討）
# dfs = []
# for p in files:
#     print("読み込み中:", p)
#     # engine='pyarrow' を明示（環境に合わせて fastparquet でもOK）
#     dfs.append(pd.read_parquet(p, engine="pyarrow"))
# df = pd.concat(dfs, ignore_index=True)
# print(f"→ 結合完了: {len(dfs)} ファイル, 合計 {len(df)} 行")

# # 8:2 にランダム分割（固定シードで再現可能）
# rng = np.random.default_rng(42)
# perm = rng.permutation(len(df))
# n_train = int(len(df) * 0.8)
# train_idx = perm[:n_train]
# val_idx   = perm[n_train:]

# df_train = df.iloc[train_idx].reset_index(drop=True)
# df_val   = df.iloc[val_idx].reset_index(drop=True)

# # 書き出し（圧縮はデフォルトのsnappy。変更したい場合は compression= で）
# out_train = os.path.join(base, "train.parquet")
# out_val   = os.path.join(base, "val.parquet")

# df_train.to_parquet(out_train, index=False, engine="pyarrow")
# df_val.to_parquet(out_val, index=False, engine="pyarrow")

# print("✅ 作成完了")
# print("  -", out_train, f"({len(df_train)} rows)")
# print("  -", out_val,   f"({len(df_val)} rows)")
# EOF


# == シャードごとの行数 ==
# train-00000-of-00010.parquet    rows=10,303  size=206.77 MB
# train-00001-of-00010.parquet    rows=10,303  size=202.54 MB
# train-00002-of-00010.parquet    rows=10,302  size=203.72 MB
# train-00003-of-00010.parquet    rows=10,302  size=198.50 MB
# train-00004-of-00010.parquet    rows=10,302  size=197.44 MB
# train-00005-of-00010.parquet    rows=10,302  size=198.12 MB
# train-00006-of-00010.parquet    rows=10,302  size=197.46 MB
# train-00007-of-00010.parquet    rows=10,302  size=197.77 MB
# train-00008-of-00010.parquet    rows=10,302  size=260.77 MB
# train-00009-of-00010.parquet    rows=10,302  size=174.07 MB

# == シャード合計 ==
# total_rows (shards) = 103,022

# == 単一ファイルの行数 ==
# train.parquet    rows=82,417  size=1628.42 MB
# val.parquet      rows=20,605  size=408.64 MB




#勾配蓄積


torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/train.parquet \
    data.val_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/val.parquet \
    data.prompt_key=question \
    data.response_key=r1_solution_1 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8000 \
    +data.dataloader_num_workers=8 \
    model.fsdp_config.model_dtype=bf16 \
    data.truncation=right \
    ++data.filter_overlong_prompts=True \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.partial_pretrain=/home/Competition2025/P12/shareP12/models/Qwen3-32B \
    trainer.total_epochs=2 \
    trainer.save_freq=1000 \
    trainer.default_local_dir=$HOME/training/sft_Qwen3-32B_deepmath/checkpoints \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    +model.override_config.attn_implementation=flash_attention_2 \
    model.enable_gradient_checkpointing=True \
    ++model.fsdp_config.forward_prefetch=True \
    trainer.experiment_name=$WANDB_RUN_NAME | tee ~/training/sft_Qwen3-32B_deepmath/verl_sft.log


cd $HOME/model/sft_Qwen3-32B_deepmath/checkpoints
ls -la

#勾配蓄積したいverこれを引数追加
#     model.enable_gradient_checkpointing=True \
echo "=== SFT Training Completed ==="

# 最新チェックポイントを自動検出
echo "=== Converting to HuggingFace format ==="
LATEST_CHECKPOINT=$(find $HOME/training/sft_Qwen3-32B_deepmath/checkpoints -name "global_step_*" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoint found!"
    exit 1
fi

echo "Found checkpoint: $LATEST_CHECKPOINT"

# HuggingFace形式に変換
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT \
    --target_dir $LATEST_CHECKPOINT/huggingface

echo "=== Uploading to HuggingFace ==="

# 適切なリポジトリ名でアップロード
huggingface-cli upload \
    Ta1k1/sft_Qwen3-32B-DeepMath \
    $LATEST_CHECKPOINT/huggingface \
    --token $HF_TOKEN

echo "🎉 Complete! Model uploaded to: https://huggingface.co/Ta1k1/sft_Qwen3-32B-DeepMath"
echo "📁 Local path: $LATEST_CHECKPOINT/huggingface"
