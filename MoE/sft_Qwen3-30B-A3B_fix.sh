#!/bin/bash
#SBATCH --job-name=sft_qwen8b_deep_math
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[84]
#SBATCH --cpus-per-task=240
#SBATCH --time=30:00:00

#MoEモデルをsftしてみよう

#srunする場合は以下のように実行
#bash ../shareP12/scancel_hatakeyama.sh gpu85
#srun --partition P12 --nodes=1 --nodelist osk-gpu[85] --gpus-per-node=8  --cpus-per-task=240 --time=30:00:00 --pty bash -i

mkdir -p ~/training/sft_Qwen3-30B-A3B_fix
mkdir -p ~/training/sft_Qwen3-30B-A3B_fix/checkpoints

cd ~/training/sft_Qwen3-30B-A3B_fix

#基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
#CUDA_VISIBLE_DEVICESでトレーニングに使用するGPUの数を制御します。
#例えば、単一GPUの場合は以下のように設定します：
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -v unlimited

#YOU_TEAM を wandb の組織名に置き換えてください。

#data.train_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/train.parque
#これは
#DeepMath-103KデータセットをParquet形式で結合して単一のファイルにするスクリプトです。以下参考
# 
#     cd /home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data

#     python - <<'EOF'
#     import os
#     import pandas as pd

#     # パスを組み立て
#     base = "/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data"
#     files = [os.path.join(base, f"train-000{i:02d}-of-00010.parquet") for i in range(10)]

#     # 存在チェック
#     missing = [p for p in files if not os.path.exists(p)]
#     if missing:
#         print('❌ 次のファイルが見つかりませんでした:')
#         for p in missing: print('   ', p)
#         exit(1)
#     print('✅ 全ファイル存在を確認')

#     # 読み込んで結合
#     dfs = []
#     for p in files:
#         print('読み込み中:', p)
#         dfs.append(pd.read_parquet(p))
#     df = pd.concat(dfs, ignore_index=True)
#     print(f'→ 結合完了: {len(dfs)} ファイル, 合計 {len(df)} 行')

#     # 単一 Parquet に書き出し
#     out = "train.parquet"
#     df.to_parquet(out, index=False)
#     print('✅ 作成完了:', out)
#     EOF
# 


#    data.max_length=16384 \これ大きくないと <think>タグの中身を全部含めれないので注意



export WANDB_PROJECT_NAME="competition_sft_deep_math"
export WANDB_RUN_NAME="Qwen3-30B-A3B_fix"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/train.parquet \
    data.val_files=/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data/val.parquet \
    data.prompt_key=question \
    data.response_key=r1_solution_1 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8000 \
    model.fsdp_config.model_dtype=bf16 \
    data.truncation=right \
    +data.dataloader_num_workers=8 \
    ++data.filter_overlong_prompts=True \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.partial_pretrain=/home/Competition2025/P12/P12U025/model/Qwen3-30B-A3B-Base \
    trainer.total_epochs=2 \
    trainer.save_freq=1000 \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$HOME/training/sft_Qwen3-30B-A3B_fix/checkpoints \
    trainer.logger=['console','wandb'] \
    ++model.fsdp_config.forward_prefetch=True \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_RUN_NAME | tee ~/training/sft_Qwen3-30B-A3B_fix/verl_sft.log

cd $HOME/model/sft_Qwen3-30B-A3B_fix/checkpoints
ls -la

#勾配蓄積したいverこれを引数追加
#     model.enable_gradient_checkpointing=True \
echo "=== SFT Training Completed ==="

# 最新チェックポイントを自動検出
echo "=== Converting to HuggingFace format ==="
LATEST_CHECKPOINT=$(find $HOME/training/sft_Qwen3-30B-A3B_fix/checkpoints -name "global_step_*" -type d | sort -V | tail -1)

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
    Ta1k1/sft_Qwen3-30B-A3B-SFT-DeepMath \
    $LATEST_CHECKPOINT/huggingface \
    --token $HF_TOKEN

echo "🎉 Complete! Model uploaded to: https://huggingface.co/Ta1k1/sft_Qwen3-30B-A3B-SFT-DeepMath"
echo "📁 Local path: $LATEST_CHECKPOINT/huggingface"



