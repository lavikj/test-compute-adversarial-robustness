"""Demo script to generate Figure 2 with dummy data for visualization."""
import os
import pandas as pd
import numpy as np
from eval.plotting import plot_figure2_grid


def generate_dummy_data(task, goal, k_values, attacker_strengths, seed=42):
    """
    Generate realistic dummy data for demonstration.
    
    Pattern:
    - Simple task + simple goal: Low attack success (dark purple)
    - Hard task + hard goal: High attack success (yellow)
    - Attack success decreases as k increases
    - Attack success increases as attacker_strength increases
    """
    np.random.seed(seed)
    
    # Task difficulty mapping (0=easy, 1=medium, 2=hard)
    task_difficulty = {"addition": 0, "multiplication": 1, "math": 2}
    goal_difficulty = {"output_42": 0, "answer_plus_1": 1, "answer_times_7": 2}
    
    task_diff = task_difficulty[task]
    goal_diff = goal_difficulty[goal]
    
    # Base difficulty (0 to 1)
    base_difficulty = (task_diff + goal_diff) / 4.0  # Average of 0-2 scales
    
    data = []
    for k_idx, k in enumerate(k_values):
        for strength_idx, strength in enumerate(attacker_strengths):
            # Attack success rate calculation
            # - Higher with more attacker strength (bottom of heatmap)
            # - Lower with more k (right side of heatmap)
            # - Higher with harder task/goal combinations
            
            # Normalize indices to 0-1
            k_effect = k_idx / (len(k_values) - 1)  # 0 to 1
            strength_effect = strength_idx / (len(attacker_strengths) - 1)  # 0 to 1
            
            # Attack success = base_difficulty + strength_effect - k_effect
            asr = base_difficulty + 0.3 * strength_effect - 0.4 * k_effect
            asr = np.clip(asr, 0.0, 1.0)
            
            # Add some noise for realism
            asr += np.random.normal(0, 0.05)
            asr = np.clip(asr, 0.0, 1.0)
            
            # Accuracy is inverse of attack success (simplified)
            acc = 1.0 - asr + np.random.normal(0, 0.05)
            acc = np.clip(acc, 0.0, 1.0)
            
            data.append({
                'k': k,
                'attacker_strength': strength,
                'attacker_goal': goal,
                'attack_success_rate': asr,
                'accuracy': acc,
                'variation': f'demo_{task}_{goal}',
            })
    
    return pd.DataFrame(data)


def main():
    """Generate Figure 2 with dummy data."""
    print("Generating Figure 2 demo with dummy data...")
    print("=" * 80)
    
    # Parameters matching Figure 2
    tasks = ["addition", "multiplication", "math"]
    goals = ["output_42", "answer_plus_1", "answer_times_7"]
    k_values = [316, 1000, 3162, 10000]
    attacker_strengths = [316, 1000, 3162, 10000]
    
    # Generate dummy data for all 9 conditions
    results_dict = {}
    
    for task in tasks:
        for goal in goals:
            print(f"  Generating dummy data for {task} × {goal}")
            df = generate_dummy_data(task, goal, k_values, attacker_strengths)
            results_dict[(task, goal)] = df
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Generate figure
    print("\nCreating Figure 2 visualization...")
    output_file = "results/figure2_demo.png"
    plot_figure2_grid(
        results_dict=results_dict,
        output_file=output_file,
        tasks=tasks,
        goals=goals,
    )
    
    print("=" * 80)
    print(f"✅ Demo figure generated: {output_file}")
    print()
    print("This is a DEMO with dummy data showing the expected layout:")
    print("  - Top-left: Dark purple (simple task + simple goal)")
    print("  - Bottom-right: Yellow/green (hard task + hard goal)")
    print("  - Gradient from easy-to-defend to hard-to-defend")
    print()
    print("To generate real results, run:")
    print("  python run_figure2.py --n_samples 10")


if __name__ == "__main__":
    main()

