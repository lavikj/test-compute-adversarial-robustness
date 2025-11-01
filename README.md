# Inference-Time Compute vs Robustness

A reproducible Python repository that measures how *inference-time compute* (self-consistency votes, deliberate steps) affects robustness to simple attacks on unambiguous math tasks.

## Overview

This experiment investigates whether increasing inference-time compute (via self-consistency sampling) improves robustness to many-shot jailbreak attacks on simple math tasks. It includes:

- **Baseline**: 2-digit addition/multiplication with many-shot attacks
- **Variation A**: Dataset shift (3-digit problems)
- **Variation B**: "Think-less" and "nerd-sniping" distractor attacks

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** (for OpenAI models):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # Or create a .env file with: OPENAI_API_KEY=your-api-key-here
   ```

3. **Configure model** in the YAML config files (`config/*.yaml`):
   - For OpenAI: Set `backend: openai` and specify `model_name`
   - For HuggingFace: Set `backend: huggingface` and specify `model_name`

## Usage

### Run Baseline Experiment
```bash
python run.py --config config/baseline.yaml
```

### Run Dataset Shift Variation
```bash
python run.py --config config/variation_dataset_shift.yaml
```

### Run Think-Less Variation
```bash
python run.py --config config/variation_thinkless.yaml
```

## Outputs

Each experiment generates:

1. **CSV file**: `results/{variation}.csv` with per-condition results
   - Columns: `k`, `attacker_strength`, `attacker_goal`, `attack_success_rate`, `accuracy`

2. **Plots** (PNG):
   - `results/{variation}_heatmap.png`: Heatmap of attack success rate over `attacker_strength × k`
   - `results/{variation}_line_k.png`: Line plot showing Attack Success Rate vs `k` (grouped by attacker strength)
   - `results/{variation}_line_strength.png`: Line plot showing Attack Success Rate vs attacker strength (grouped by `k`)

## Project Structure

```
test-compute-adversarial-robustness/
├── config/              # YAML configuration files
├── models/              # LLM client implementations (OpenAI, HuggingFace)
├── data/                # Math problem generation
├── attacks/             # Attack strategies (many-shot, distractor)
├── defense/             # Defense mechanisms (self-consistency, voting)
├── eval/                # Evaluation utilities (metrics, grid runner, plotting)
├── results/             # Output directory (CSVs and plots)
├── run.py               # Main entry point
└── requirements.txt     # Dependencies
```

## Configuration

Edit YAML config files to customize:

- **Model**: `backend`, `model_name`, `device` (for HF)
- **Data**: `task` (addition/multiplication/mixed), `digits`, `n_samples`
- **Experiment**: `k_values`, `attacker_strengths`, `attacker_goals`, `max_tokens`
- **Variations**: `use_think_less`, `use_nerd_snipe`, `nerd_snipe_tokens`

## Expected Results

- **Baseline**: Attack success rate should decrease as `k` increases (for fixed attacker strength)
- **Variation A**: Curves shift - defender needs larger `k` to suppress attacks
- **Variation B**: "Think-less" increases Attack Success Rate at low `k`; "nerd-snipe" can blunt gains at high `k`

## Notes

- Experiments are deterministic given seeds
- Progress bars show real-time progress
- Results are saved incrementally to CSV
