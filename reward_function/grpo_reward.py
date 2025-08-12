import re
import math
from Levenshtein import ratio as levenshtein_ratio
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def extract_contents(completions: list[dict[str, str]] | list[str]) -> list[str]:
    if isinstance(completions[0], list):
        contents = [completion[0]['content'] for completion in completions]
    else:
        contents = completions
    return contents


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def check_answer(completion: str, answer: str):
    gold_parsed = parse('\\boxed{' + answer + '}')
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            print(
                f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}"
            )
            reward = 0.0
    else:
        reward = 1.0
    return reward


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"^<think>.*?</think>.*?oxed{(.*?)}.*?$"
    pattern = r"^.*?</think>.*?oxed{(.*?)}.*?$"
    matches = [re.match(pattern, content, re.DOTALL) for content in extract_contents(completions)]
    return [1.0 if match else 0.0 for match in matches]


def format_reward2(completions, **kwargs):
    pattern = r"^.*?oxed{(.*?)}.*?</think>.*?$"
    matches = [re.match(pattern, content, re.DOTALL) for content in extract_contents(completions)]
    return [1.0 if match else 0.0 for match in matches]


def accuracy_reward(completions, answer, **kwargs):
    # contents = [extract_boxed_text(content) for content in extract_contents(completions)]
    # return [1.0 if c == str(gt) else 0.0 for c, gt in zip(contents, answer)]
    return [check_answer(content, str(gt)) for content, gt in zip(extract_contents(completions), answer)]


def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    matches = [len(re.findall(pattern, content)) for content in extract_contents(completions)]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[str], answer: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    # First check correctness of answers
    contents = extract_contents(completions)
    correctness = [check_answer(content, str(gt)) for content, gt in zip(contents, answer)]

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def levenshtein_reward(completions, solution, **kwargs):
    contents = extract_contents(completions)
    res = []
    for content, sol in zip(contents, solution):
        if '</think>' in content:
            t = content.split('</think>')[-1]
            res.append(levenshtein_ratio(t, sol))
        else:
            res.append(0.0)
    return res
       

def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 8192,
    clip_len: bool = False,
):
    def cosine_scaled_reward(completions, answer, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        rewards = []

        for content, gt in zip(extract_contents(completions), answer):
            is_correct = check_answer(content, str(gt))
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            if clip_len:
                progress = min(1.0, progress)
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -1.0):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in extract_contents(completions):
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Main compute_score function for Verl framework
    
    Args:
        data_source: Dataset name (e.g., "gsm8k", "math")
        solution_str: Model generated response
        ground_truth: Correct answer
        extra_info: Optional additional information
    
    Returns:
        float: Reward score
    """
    # Use accuracy reward as the primary evaluation
    completions = [solution_str]
    answers = [ground_truth]
    
    # Get accuracy reward
    accuracy_score = accuracy_reward(completions, answers)[0]
    
    # Get cosine scaled reward for length consideration
    cosine_reward_func = get_cosine_scaled_reward()
    cosine_score = cosine_reward_func(completions, answers)[0]
    
    # Combine accuracy and length-aware scoring
    # 70% accuracy, 30% length-aware cosine scaling
    final_score = 0.7 * accuracy_score + 0.3 * cosine_score
    
    return float(final_score)


REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "format2": format_reward2,
    "reasoning_steps": reasoning_steps_reward,
    "cosine": get_cosine_scaled_reward(),
    "repetition_penalty": get_repetition_penalty_reward(),
    "length": len_reward,
    "levenshtein": levenshtein_reward,
}