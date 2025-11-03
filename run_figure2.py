"""Generate Figure 2 from the paper: 3x3 grid of attack success rates."""
import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from data.gen_math import sample_add, sample_mul, sample_math, train_test_split
try:
    from data.math_dataset import sample_math_hendrycks
    HENDRYCKS_MATH_AVAILABLE = True
except ImportError:
    HENDRYCKS_MATH_AVAILABLE = False
from eval.grid_runner import run_grid_experiment
from eval.plotting import plot_figure2_grid


def create_model(backend: str, model_name: str, device: str = None):
    """Create model client."""
    if backend == "openai":
        # Lazy import to avoid transformers dependency when using OpenAI
        from models.openai_client import OpenAIClient
        return OpenAIClient(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif backend == "huggingface":
        # Lazy import to avoid OpenAI dependency when using HuggingFace
        from models.hf_client import HuggingFaceClient
        return HuggingFaceClient(
            model_name=model_name,
            device=device,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2: 3x3 grid of attack experiments")
    parser.add_argument(
        "--backend",
        type=str,
        default="huggingface",
        choices=["openai", "huggingface"],
        help="Model backend",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for HuggingFace models (cuda/cpu)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
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
    parser.add_argument(
        "--use_hendrycks_math",
        action="store_true",
        help="Use real Hendrycks MATH dataset instead of synthetic problems (requires: pip install datasets)",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of digits for addition/multiplication (default: 2, paper uses 4 for o1-preview)",
    )
    args = parser.parse_args()
    
    # Check if Hendrycks MATH is requested but not available
    if args.use_hendrycks_math and not HENDRYCKS_MATH_AVAILABLE:
        print("Error: --use_hendrycks_math requires the 'datasets' library.")
        print("Install with: pip install datasets")
        return
    
    # Load environment variables
    load_dotenv()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print(f"Initializing model: {args.model_name} ({args.backend})")
    model = create_model(args.backend, args.model_name, args.device)
    
    # Define experimental parameters
    tasks = ["addition", "multiplication", "math"]
    goals = ["output_42", "answer_plus_1", "answer_times_7"]
    
    # Figure 2 uses log-scale values
    # k_values: inference-time compute (2.5, 3, 3.5, 4, 4.5 in log scale)
    k_values = [316, 1000, 3162, 10000]  # 10^2.5, 10^3, 10^3.5, 10^4
    
    # attacker_strengths: attack length in tokens (2.5, 3, 3.5, 4, 4.5 in log scale)
    attacker_strengths = [316, 1000, 3162, 10000]  # Same log scale
    
    print(f"\nExperiment Configuration:")
    print(f"  Tasks: {tasks}")
    print(f"  Goals: {goals}")
    print(f"  k values (inference compute): {k_values}")
    print(f"  Attacker strengths (tokens): {attacker_strengths}")
    print(f"  Samples per condition: {args.n_samples}")
    
    # Store results for each (task, goal) combination
    results_dict = {}
    
    # Run all 9 experiments
    total_experiments = len(tasks) * len(goals)
    experiment_num = 0
    
    for task in tasks:
        # Generate data for this task
        print(f"\n{'='*80}")
        print(f"Generating {task} problems...")
        print(f"{'='*80}")
        
        if task == "addition":
            problems = sample_add(args.n_samples, digits=args.digits, seed=args.seed)
        elif task == "multiplication":
            problems = sample_mul(args.n_samples, digits=args.digits, seed=args.seed)
        elif task == "math":
            if args.use_hendrycks_math:
                print(f"  Using real Hendrycks MATH dataset...")
                problems = sample_math_hendrycks(args.n_samples, seed=args.seed)
            else:
                print(f"  Using synthetic MATH problems...")
                problems = sample_math(args.n_samples, seed=args.seed)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Use all problems as test (no train/test split needed for this figure)
        test_problems = problems
        
        for goal in goals:
            experiment_num += 1
            print(f"\n[Experiment {experiment_num}/{total_experiments}] Task: {task}, Goal: {goal}")
            print("-" * 80)
            
            # Run grid experiment
            df = run_grid_experiment(
                model=model,
                test_problems=test_problems,
                k_values=k_values,
                attacker_strengths=attacker_strengths,
                attacker_goals=[goal],
                variation=f"figure2_{task}_{goal}",
                variation_params={},
                max_tokens=100,
                deliberate_steps=None,
                seed=args.seed,
                output_dir=args.output_dir,
            )
            
            # Store results
            results_dict[(task, goal)] = df
            
            # Save individual CSV
            csv_file = os.path.join(args.output_dir, f"figure2_{task}_{goal}.csv")
            df.to_csv(csv_file, index=False)
            print(f"Results saved to {csv_file}")
    
    # Generate Figure 2
    print(f"\n{'='*80}")
    print("Generating Figure 2...")
    print(f"{'='*80}")
    
    output_file = os.path.join(args.output_dir, "figure2.png")
    plot_figure2_grid(
        results_dict=results_dict,
        output_file=output_file,
        tasks=tasks,
        goals=goals,
    )
    
    print(f"\n✓ Figure 2 generation complete!")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()

