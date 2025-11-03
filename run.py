"""Main entry point for experiments."""
import argparse
import yaml
import os
from dotenv import load_dotenv

from data.gen_math import sample_add, sample_mul, sample_math, train_test_split
from eval.grid_runner import run_grid_experiment
from eval.plotting import generate_plots


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: dict):
    """Create model client from config."""
    backend = config["model"]["backend"]
    
    if backend == "openai":
        # Lazy import to avoid transformers dependency when using OpenAI
        from models.openai_client import OpenAIClient
        return OpenAIClient(
            model_name=config["model"]["model_name"],
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif backend == "huggingface":
        # Lazy import to avoid OpenAI dependency when using HuggingFace
        from models.hf_client import HuggingFaceClient
        return HuggingFaceClient(
            model_name=config["model"]["model_name"],
            device=config["model"].get("device"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description="Run inference-time compute vs robustness experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load config
    config = load_config(args.config)
    variation = config.get("variation", "baseline")
    
    print(f"Running experiment: {variation}")
    print(f"Config: {args.config}")
    
    # Create model
    print("Initializing model...")
    model = create_model(config)
    print(f"Model: {config['model']['model_name']} ({config['model']['backend']})")
    
    # Generate data
    print("Generating data...")
    data_config = config["data"]
    digits = data_config.get("digits", 2)
    n_samples = data_config.get("n_samples", 1000)
    seed = config.get("seed", 42)
    
    # Generate problems
    if data_config.get("task") == "addition":
        problems = sample_add(n_samples, digits=digits, seed=seed)
    elif data_config.get("task") == "multiplication":
        problems = sample_mul(n_samples, digits=digits, seed=seed)
    elif data_config.get("task") == "math":
        problems = sample_math(n_samples, seed=seed)
    else:
        # Default: mix
        add_problems = sample_add(n_samples // 2, digits=digits, seed=seed)
        mul_problems = sample_mul(n_samples // 2, digits=digits, seed=seed + 1)
        problems = add_problems + mul_problems
    
    # Train/test split
    train_ratio = data_config.get("train_ratio", 0.7)
    _, test_problems = train_test_split(problems, train_ratio=train_ratio, seed=seed)
    
    print(f"Test problems: {len(test_problems)}")
    
    # Get experiment parameters
    exp_config = config["experiment"]
    k_values = exp_config.get("k_values", [1, 2, 4, 8, 16])
    attacker_strengths = exp_config.get("attacker_strengths", [64, 256, 512, 1024, 2048])
    attacker_goals = exp_config.get("attacker_goals", ["output_42", "answer_plus_1"])
    
    # Variation-specific parameters
    variation_params = {}
    if variation == "variation_thinkless":
        if exp_config.get("use_think_less", False):
            variation_params["use_think_less"] = True
        if exp_config.get("use_nerd_snipe", False):
            variation_params["use_nerd_snipe"] = True
            variation_params["nerd_snipe_tokens"] = exp_config.get("nerd_snipe_tokens", 512)
    
    # Run experiment
    print("Running grid experiment...")
    df = run_grid_experiment(
        model=model,
        test_problems=test_problems,
        k_values=k_values,
        attacker_strengths=attacker_strengths,
        attacker_goals=attacker_goals,
        variation=variation,
        variation_params=variation_params,
        max_tokens=exp_config.get("max_tokens", 100),
        deliberate_steps=exp_config.get("deliberate_steps"),
        seed=seed,
        output_dir=config.get("output_dir", "results"),
    )
    
    # Generate plots
    print("Generating plots...")
    generate_plots(df, variation, output_dir=config.get("output_dir", "results"))
    
    print(f"\nExperiment completed: {variation}")
    print(f"Results saved to: {config.get('output_dir', 'results')}/{variation}.csv")


if __name__ == "__main__":
    main()

