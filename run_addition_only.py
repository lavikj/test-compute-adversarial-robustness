"""Quick runner: Only 2-digit addition with o1 model for fast results."""
import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from data.gen_math import sample_add
from eval.grid_runner import run_grid_experiment
from eval.plotting import plot_figure2_grid

# Lazy import for o1
def create_o1_model(model_name: str):
    from models.openai_client import OpenAIClient
    return OpenAIClient(
        model_name=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def main():
    parser = argparse.ArgumentParser(description="Quick Figure 2 - Addition only with o1")
    parser.add_argument(
        "--model_name",
        type=str,
        default="o1",
        help="Model name (o1, o1-mini, o1-preview)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of test samples per condition",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing model: {args.model_name}")
    model = create_o1_model(args.model_name)
    
    # Parameters for quick experiments
    tasks = ["addition"]  # Only addition
    goals = ["output_42", "answer_plus_1", "answer_times_7"]
    
    # Smaller k values for speed (still shows pattern)
    k_values = [1, 4, 16, 64]
    
    # Smaller attacker strengths for speed
    attacker_strengths = [100, 500, 1000, 2000]
    
    print(f"\n⚡ QUICK MODE - Addition Only ⚡")
    print(f"Model: {args.model_name}")
    print(f"k values: {k_values}")
    print(f"Attacker strengths: {attacker_strengths}")
    print(f"Goals: {goals}")
    print(f"Samples per condition: {args.n_samples}")
    print(f"Total configs: {len(k_values) * len(attacker_strengths) * len(goals)} = {len(k_values)}k × {len(attacker_strengths)}strength × {len(goals)}goals")
    
    # Store results
    results_dict = {}
    
    # Generate addition problems
    print(f"\n{'='*80}")
    print(f"Generating 2-digit addition problems...")
    print(f"{'='*80}")
    
    problems = sample_add(args.n_samples, digits=2, seed=args.seed)
    test_problems = problems
    
    # Run experiments for each goal
    for goal_idx, goal in enumerate(goals):
        print(f"\n[Experiment {goal_idx+1}/{len(goals)}] Task: addition, Goal: {goal}")
        print("-" * 80)
        
        # Run grid experiment
        df = run_grid_experiment(
            model=model,
            test_problems=test_problems,
            k_values=k_values,
            attacker_strengths=attacker_strengths,
            attacker_goals=[goal],
            variation=f"addition_{goal}",
            variation_params={},
            max_tokens=50,  # Low tokens for quick responses
            deliberate_steps=None,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        
        # Store results
        results_dict[("addition", goal)] = df
        
        # Save individual CSV
        csv_file = os.path.join(args.output_dir, f"addition_{goal}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")
    
    # Generate figure
    print(f"\n{'='*80}")
    print("Generating heatmap...")
    print(f"{'='*80}")
    
    output_file = os.path.join(args.output_dir, "addition_heatmap.png")
    
    # Call plotting function
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Custom colormap
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde724']
    cmap = LinearSegmentedColormap.from_list('viridis', colors, N=100)
    
    # Create 1x3 grid for 3 goals
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Addition (2-digit) - Many-shot attack on {args.model_name}", fontsize=14)
    
    goal_labels = {
        "output_42": "Goal: output 42",
        "answer_plus_1": "Goal: answer + 1",
        "answer_times_7": "Goal: answer × 7"
    }
    
    for j, goal in enumerate(goals):
        ax = axes[j]
        df = results_dict[("addition", goal)]
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            index='attacker_strength',
            columns='k',
            values='attack_success_rate',
            aggfunc='mean'
        )
        
        # Plot
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=1, origin='lower')
        
        # Ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)
        
        ax.set_title(goal_labels[goal], fontsize=11)
        ax.set_xlabel("k (inference compute)", fontsize=10)
        
        if j == 0:
            ax.set_ylabel("Attack strength (tokens)", fontsize=10)
        
        # Grid
        ax.set_xticks(np.arange(len(pivot.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(pivot.index)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_label('P[attack success]', fontsize=11)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.90, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {output_file}")
    
    plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for goal in goals:
        df = results_dict[("addition", goal)]
        print(f"\n{goal}:")
        print(f"  Mean attack success: {df['attack_success_rate'].mean():.3f}")
        print(f"  Mean accuracy: {df['accuracy'].mean():.3f}")
        
        # By k
        by_k = df.groupby('k')['attack_success_rate'].mean()
        print(f"  By k: {dict(by_k)}")


if __name__ == "__main__":
    main()

