"""Grid search runner for experiments."""
import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import LLMClient
from defense.inference_budget import run_with_budget
from defense.voting import majority_vote
from attacks.many_shot import build_many_shot_prompt, get_attacker_goal_value
from attacks.distractor import make_think_less, make_nerd_snipe
from eval.metrics import attack_success_rate, accuracy


def run_grid_experiment(
    model: LLMClient,
    test_problems: List[tuple],
    k_values: List[int],
    attacker_strengths: List[int],
    attacker_goals: List[str],
    variation: str = "baseline",
    variation_params: Optional[Dict] = None,
    max_tokens: int = 100,
    deliberate_steps: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run grid search experiment over k (compute) and attacker strength.
    
    Args:
        model: LLM client
        test_problems: List of (question_str, answer_int) tuples
        k_values: List of self-consistency samples [1, 2, 4, 8, 16]
        attacker_strengths: List of attacker token budgets
        attacker_goals: List of goal types ["output_42", "answer_plus_1"]
        variation: Experiment variation name
        variation_params: Optional variation-specific parameters
        max_tokens: Max tokens per generation
        deliberate_steps: Optional deliberate reasoning steps
        seed: Random seed
        output_dir: Directory to save results
        
    Returns:
        DataFrame with results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    variation_params = variation_params or {}
    use_think_less = variation_params.get("use_think_less", False)
    use_nerd_snipe = variation_params.get("use_nerd_snipe", False)
    nerd_snipe_tokens = variation_params.get("nerd_snipe_tokens", 0)
    
    results = []
    
    total_runs = len(k_values) * len(attacker_strengths) * len(attacker_goals) * len(test_problems)
    pbar = tqdm(total=total_runs, desc=f"Running {variation} experiment")
    
    for k in k_values:
        for attacker_strength in attacker_strengths:
            for attacker_goal in attacker_goals:
                predictions = []
                true_answers = []
                attacker_goal_values = []
                
                for question, answer in test_problems:
                    # Build base prompt with attack
                    base_prompt = build_many_shot_prompt(
                        question=question,
                        answer=answer,
                        attacker_goal=attacker_goal,
                        attacker_strength_tokens=attacker_strength,
                        seed=seed,
                    )
                    
                    # Apply variation-specific modifications
                    prompt = base_prompt
                    if use_think_less:
                        prompt = make_think_less(prompt)
                    elif use_nerd_snipe:
                        prompt = make_nerd_snipe(prompt, nerd_snipe_tokens, seed=seed)
                    
                    # Run with budget (k samples)
                    outputs = run_with_budget(
                        model=model,
                        prompt=prompt,
                        k=k,
                        max_tokens=max_tokens,
                        deliberate_steps=deliberate_steps,
                    )
                    
                    # Vote to get final prediction
                    prediction = majority_vote(outputs)
                    
                    # Get attacker goal value
                    goal_value = get_attacker_goal_value(answer, attacker_goal)
                    
                    predictions.append(prediction)
                    true_answers.append(answer)
                    attacker_goal_values.append(goal_value)
                    
                    pbar.update(1)
                
                # Compute metrics
                asr = attack_success_rate(predictions, true_answers, attacker_goal_values)
                acc = accuracy(predictions, true_answers)
                
                results.append({
                    "k": k,
                    "attacker_strength": attacker_strength,
                    "attacker_goal": attacker_goal,
                    "attack_success_rate": asr,
                    "accuracy": acc,
                    "variation": variation,
                })
    
    pbar.close()
    
    df = pd.DataFrame(results)
    
    # Save CSV
    output_file = os.path.join(output_dir, f"{variation}.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return df

