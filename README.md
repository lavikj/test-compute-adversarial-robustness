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

### Generate Figure 2 (3×3 Grid)
To reproduce Figure 2 from the paper (3 tasks × 3 attacker goals):

**With o1-mini (as in paper)**:
```bash
export OPENAI_API_KEY="your-key"
python run_figure2.py --backend openai --model_name o1-mini --n_samples 100 --use_hendrycks_math
```
📖 **See detailed guide**: `REPRODUCE_FIGURE2_O1MINI.md`

**With other models**:
```bash
# HuggingFace (free, local)
python run_figure2.py --backend huggingface --model_name microsoft/Phi-3-mini-4k-instruct --n_samples 100

# GPT-4 (faster than o1-mini)
python run_figure2.py --backend openai --model_name gpt-4 --n_samples 100
```

This will:
1. Run 9 experiments (addition/multiplication/math × output_42/answer_plus_1/answer_times_7)
2. Generate individual CSVs for each condition
3. Create `results/figure2.png` with the complete 3×3 grid visualization

## Outputs

Each experiment generates:

1. **CSV file**: `results/{variation}.csv` with per-condition results
   - Columns: `k`, `attacker_strength`, `attacker_goal`, `attack_success_rate`, `accuracy`

2. **Plots** (PNG):
   - `results/{variation}_heatmap.png`: Heatmap of attack success rate over `attacker_strength × k`
   - `results/{variation}_line_k.png`: Line plot showing Attack Success Rate vs `k` (grouped by attacker strength)
   - `results/{variation}_line_strength.png`: Line plot showing Attack Success Rate vs attacker strength (grouped by `k`)

3. **Figure 2** (when running `run_figure2.py`):
   - `results/figure2.png`: 3×3 grid of heatmaps showing attack success across all task/goal combinations
   - `results/figure2_{task}_{goal}.csv`: Individual CSV files for each of the 9 conditions

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
- **Data**: `task` (addition/multiplication/math/mixed), `digits`, `n_samples`
- **Experiment**: `k_values`, `attacker_strengths`, `attacker_goals` (output_42/answer_plus_1/answer_times_7), `max_tokens`
- **Variations**: `use_think_less`, `use_nerd_snipe`, `nerd_snipe_tokens`

## Expected Results

- **Baseline**: Attack success rate should decrease as `k` increases (for fixed attacker strength)
- **Variation A**: Curves shift - defender needs larger `k` to suppress attacks
- **Variation B**: "Think-less" increases Attack Success Rate at low `k`; "nerd-snipe" can blunt gains at high `k`
- **Figure 2**: 
  - Top-left (simple task + simple goal): Dark purple (attacks fail) with high inference compute
  - Bottom-right (hard task + hard goal): More yellow/green (attacks succeed) even with high compute
  - Gradient from easy-to-defend (top-left) to hard-to-defend (bottom-right)

## Notes

- Experiments are deterministic given seeds
- Progress bars show real-time progress
- Results are saved incrementally to CSV
