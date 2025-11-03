"""Fast version of Figure 2 with reduced k values for quick testing."""
import os
import sys

# Get the original run_figure2 script and modify k values
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from the original script
from run_figure2 import *

# Override the main function to use smaller k values
def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2 (FAST version with smaller k)")
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "huggingface"],
        help="Model backend",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
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
        default=1,
        help="Number of test samples per condition (default: 1 for speed)",
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
        help="Use real Hendrycks MATH dataset",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of digits for addition/multiplication",
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
    
    # FAST VERSION: Use much smaller k values for speed
    # Original: [316, 1000, 3162, 10000]
    # Fast: [1, 4, 16, 64] - still shows the pattern but much faster!
    k_values = [1, 4, 16, 64]
    
    # Also use fewer attacker strengths for speed
    # Original: [316, 1000, 3162, 10000]
    # Fast: [100, 500, 1000, 2000]
    attacker_strengths = [100, 500, 1000, 2000]
    
    print(f"\n⚡ FAST MODE ENABLED ⚡")
    print(f"Using smaller k values for speed: {k_values}")
    print(f"Using smaller attacker strengths: {attacker_strengths}")
    print(f"\nExperiment Configuration:")
    print(f"  Tasks: {tasks}")
    print(f"  Goals: {goals}")
    print(f"  k values (inference compute): {k_values}")
    print(f"  Attacker strengths (tokens): {attacker_strengths}")
    print(f"  Samples per condition: {args.n_samples}")
    print(f"  Total model calls: {len(tasks) * len(goals) * len(k_values) * len(attacker_strengths) * args.n_samples}")
    
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
        
        # Use all problems as test
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
                variation=f"figure2_fast_{task}_{goal}",
                variation_params={},
                max_tokens=100,
                deliberate_steps=None,
                seed=args.seed,
                output_dir=args.output_dir,
            )
            
            # Store results
            results_dict[(task, goal)] = df
            
            # Save individual CSV
            csv_file = os.path.join(args.output_dir, f"figure2_fast_{task}_{goal}.csv")
            df.to_csv(csv_file, index=False)
            print(f"Results saved to {csv_file}")
    
    # Generate Figure 2
    print(f"\n{'='*80}")
    print("Generating Figure 2 (FAST version)...")
    print(f"{'='*80}")
    
    output_file = os.path.join(args.output_dir, "figure2_fast.png")
    plot_figure2_grid(
        results_dict=results_dict,
        output_file=output_file,
        tasks=tasks,
        goals=goals,
    )
    
    print(f"\n✓ Figure 2 (FAST) generation complete!")
    print(f"  Output: {output_file}")
    print(f"\nNote: This used smaller k values {k_values} for speed.")
    print(f"For full reproduction with k=[316, 1000, 3162, 10000], use run_figure2.py")


if __name__ == "__main__":
    main()

