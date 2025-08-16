#!/bin/bash
# filepath: /home/Competition2025/P12/P12U025/train_sh/convert_to_fsdp.sh

# 環境設定
source /etc/profile.d/modules.sh
module load miniconda/24.7.1-py311
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda activate /home/Competition2025/P12/P12U025/conda_env

# 変換実行
python -m verl.utils.convert_checkpoint \
    --input_dir /home/Competition2025/P12/shareP12/models/Qwen3-32B \
    --output_dir /home/Competition2025/P12/shareP12/models/Qwen3-32B-FSDP \
    --target_format fsdp \
    --model_dtype bf16