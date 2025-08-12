#!/usr/bin/env python3
"""
Custom reward function for GRPO training with Verl framework
数学問題の回答品質を評価するカスタム報酬関数
"""

import re
import ast
import math
from typing import Optional, Union


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """
    Verl用のカスタム報酬関数
    
    Args:
        data_source: データセットの名前 (例: "gsm8k", "math")
        solution_str: モデルが生成した回答
        ground_truth: 正解
        extra_info: 追加情報（オプション）
    
    Returns:
        float: 報酬スコア (0.0 - 1.0)
    """
    if not solution_str or not ground_truth:
        return 0.0
    
    # データセットに応じた評価
    if data_source.lower() in ["gsm8k", "math"]:
        return evaluate_math_solution(solution_str, ground_truth)
    else:
        # 汎用的な評価
        return evaluate_general_solution(solution_str, ground_truth)


def evaluate_math_solution(solution_str: str, ground_truth: str) -> float:
    """
    数学問題の評価関数
    
    計算過程と最終的な答えの両方を考慮して評価
    """
    # 最終答えの抽出と比較
    predicted_answer = extract_final_answer(solution_str)
    true_answer = extract_final_answer(ground_truth)
    
    # 基本スコア: 正解なら0.6、不正解なら0.0
    base_score = 0.6 if compare_answers(predicted_answer, true_answer) else 0.0
    
    # ボーナス要素の評価
    bonus_score = evaluate_solution_quality(solution_str)
    
    # 最終スコア (0.0 - 1.0)
    final_score = min(1.0, base_score + bonus_score)
    
    return final_score


def evaluate_general_solution(solution_str: str, ground_truth: str) -> float:
    """
    汎用的な解答評価
    """
    # 文字列の類似度ベースの評価
    similarity = calculate_similarity(solution_str.lower(), ground_truth.lower())
    
    # 長さに基づく補正
    length_factor = min(1.0, len(solution_str) / max(1, len(ground_truth)))
    
    return min(1.0, similarity * length_factor)


def extract_final_answer(text: str) -> str:
    """
    テキストから最終的な数値答えを抽出
    """
    # 一般的な答えのパターンを検索
    patterns = [
        r'(?:答え|Answer|answer)(?:\s*[:：]\s*)?(\d+(?:\.\d+)?)',
        r'(?:結果|Result|result)(?:\s*[:：]\s*)?(\d+(?:\.\d+)?)',
        r'(?:したがって|Therefore|therefore)(?:.*?)(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*$',  # 文末の数値
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # 数値が見つからない場合は元のテキストを返す
    return text.strip()


def compare_answers(pred: str, true: str) -> bool:
    """
    予測値と正解を比較
    """
    try:
        # 数値として比較
        pred_num = float(pred)
        true_num = float(true)
        return abs(pred_num - true_num) < 1e-6
    except ValueError:
        # 文字列として比較
        return pred.strip().lower() == true.strip().lower()


def evaluate_solution_quality(solution: str) -> float:
    """
    解答プロセスの品質を評価してボーナススコアを計算
    
    Returns:
        float: ボーナススコア (0.0 - 0.4)
    """
    bonus = 0.0
    
    # 計算過程の存在
    if re.search(r'\d+\s*[+\-*/]\s*\d+', solution):
        bonus += 0.1
    
    # 段階的な解法の存在
    steps_indicators = ['まず', 'next', 'then', '次に', 'step', 'ステップ']
    if any(indicator in solution.lower() for indicator in steps_indicators):
        bonus += 0.1
    
    # 説明の丁寧さ（適度な長さ）
    if 50 <= len(solution) <= 500:
        bonus += 0.1
    
    # 論理的な接続詞の使用
    logical_words = ['したがって', 'therefore', 'because', 'なぜなら', 'so', 'thus']
    if any(word in solution.lower() for word in logical_words):
        bonus += 0.1
    
    return min(0.4, bonus)


def calculate_similarity(text1: str, text2: str) -> float:
    """
    シンプルな文字列類似度計算
    """
    if not text1 or not text2:
        return 0.0
    
    # Jaccard係数を使用
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


# 追加的なカスタム報酬関数の例
def compute_score_strict(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """
    より厳格な評価を行う報酬関数
    正解のみに高いスコアを与え、過程は重視しない
    """
    predicted_answer = extract_final_answer(solution_str)
    true_answer = extract_final_answer(ground_truth)
    
    return 1.0 if compare_answers(predicted_answer, true_answer) else 0.0


def compute_score_process_heavy(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """
    計算過程を重視する報酬関数
    正解でなくても適切な過程があれば部分点を与える
    """
    # 基本的な正解チェック
    predicted_answer = extract_final_answer(solution_str)
    true_answer = extract_final_answer(ground_truth)
    
    if compare_answers(predicted_answer, true_answer):
        return 1.0
    
    # 計算過程の評価による部分点
    process_score = evaluate_solution_quality(solution_str) * 2.5  # 最大1.0まで
    
    return min(1.0, process_score)