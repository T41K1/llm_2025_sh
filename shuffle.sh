#!/bin/bash

# 実行前に場所だけご確認ください
cd /home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data

python - <<'EOF'
import os
import sys
import pandas as pd
import numpy as np

# ベースパス
base = "/home/Competition2025/P12/P12U025/data/DeepMath-103K-parquet/data"

# 連番パターン（train-00000-of-00010.parquet ... -00009-of-00010.parquet）
files = [os.path.join(base, f"train-000{i:02d}-of-00010.parquet") for i in range(10)]

# 存在チェック
missing = [p for p in files if not os.path.exists(p)]
if missing:
    print("❌ 次のファイルが見つかりませんでした:")
    for p in missing:
        print("   ", p)
    sys.exit(1)
print("✅ 全ファイル存在を確認")

# 読み込み＆結合（メモリに乗る前提。厳しい場合はPyArrowストリーミングを検討）
dfs = []
for p in files:
    print("読み込み中:", p)
    # engine='pyarrow' を明示（環境に合わせて fastparquet でもOK）
    dfs.append(pd.read_parquet(p, engine="pyarrow"))
df = pd.concat(dfs, ignore_index=True)
print(f"→ 結合完了: {len(dfs)} ファイル, 合計 {len(df)} 行")

# 8:2 にランダム分割（固定シードで再現可能）
rng = np.random.default_rng(42)
perm = rng.permutation(len(df))
n_train = int(len(df) * 0.8)
train_idx = perm[:n_train]
val_idx   = perm[n_train:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)

# 書き出し（圧縮はデフォルトのsnappy。変更したい場合は compression= で）
out_train = os.path.join(base, "train.parquet")
out_val   = os.path.join(base, "val.parquet")

df_train.to_parquet(out_train, index=False, engine="pyarrow")
df_val.to_parquet(out_val, index=False, engine="pyarrow")

print("✅ 作成完了")
print("  -", out_train, f"({len(df_train)} rows)")
print("  -", out_val,   f"({len(df_val)} rows)")
EOF