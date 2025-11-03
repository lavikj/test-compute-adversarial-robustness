# Inference-Time Compute vs Adversarial Robustness

A reproducible Python repository implementing many-shot jailbreak attacks on OpenAI's o1/o3 reasoning models, measuring how *test-time compute* (reasoning effort) affects adversarial robustness.

## Overview

This repository reproduces experiments from the paper **"Trading Inference-Time Compute for Adversarial Robustness"** (Anil et al., 2024), investigating whether increasing test-time compute improves robustness to many-shot jailbreak attacks on math reasoning tasks.

### Key Features

- **Many-shot jailbreak attacks** with persuasive prompts from the paper
- **Test-time compute via reasoning effort** (`low`, `medium`, `high`) using o1/o3 models
- **Three attack goals**: Output 42, Answer+1, Answer×7
- **Attacker strength scaling**: 100-2000 tokens
- **Comprehensive visualization**: Heatmaps showing attack success vs defense strength

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

### Quick Start: Many-Shot Attack on o1 (Recommended)

Test many-shot jailbreak attacks on OpenAI's o1 model with varying reasoning effort:

```bash
export OPENAI_API_KEY="your-key"
python run_addition_verbose.py
```

**This will:**
- Test 4 addition problems across 3 reasoning levels (low/medium/high)
- Try 4 attack strengths (100, 500, 1000, 2000 tokens)
- Test 3 attack goals (output_42, answer_plus_1, answer_times_7)
- Print every model response with detailed logging
- Save results to `results/verbose_results.csv`

**Total**: 144 API calls (~5-10 minutes, ~$5-10)

**Then plot the results:**
```bash
python plot_verbose_results.py
```

Generates `results/verbose_results_heatmap.png` showing:
- **X-axis**: Reasoning effort (low → medium → high) = Test-time compute
- **Y-axis**: Attack strength (tokens) = Attacker budget
- **Colors**: Attack success rate (purple = defense works, yellow = attack succeeds)

**Expected pattern**: Attack success should **decrease** from left to right (more compute = more robust)

---

### Advanced: Generate Figure 2 (3×3 Grid)

To reproduce the full Figure 2 from the paper (3 tasks × 3 attacker goals):

**With o1 (recommended)**:
```bash
python run_figure2.py --backend openai --model_name o1 --n_samples 100 --use_hendrycks_math
```

**With o3-mini (newer, not deprecated)**:
```bash
python run_figure2.py --backend openai --model_name o3-mini --n_samples 100
```

📖 **See detailed guide**: `REPRODUCE_FIGURE2_O1MINI.md`

---

### Legacy: YAML Config-Based Experiments

```bash
python run.py --config config/baseline.yaml
python run.py --config config/variation_dataset_shift.yaml
python run.py --config config/variation_thinkless.yaml
```

## Outputs

### Quick Start Outputs (`run_addition_verbose.py`)

1. **`results/verbose_results.csv`**: Detailed per-sample results
   - Columns: `goal`, `reasoning_effort`, `strength`, `correct_answer`, `attack_target`, `prediction`, `attack_success`, `accuracy`
   - One row per problem × reasoning_effort × strength × goal combination

2. **`results/verbose_results_heatmap.png`**: 1×3 grid of heatmaps
   - One heatmap per attack goal (output_42, answer_plus_1, answer_times_7)
   - X-axis: Reasoning effort (low, medium, high)
   - Y-axis: Attack strength (100, 500, 1000, 2000 tokens)
   - Color: Attack success rate (0.0 = purple/defense, 1.0 = yellow/attack)

### Advanced Outputs (`run_figure2.py`)

1. **`results/figure2.png`**: 3×3 grid of heatmaps
   - Rows: Tasks (addition, multiplication, math)
   - Columns: Attack goals (output_42, answer_plus_1, answer_times_7)
   - Shows complete attack landscape across all conditions

2. **`results/figure2_{task}_{goal}.csv`**: Individual CSV files for each of 9 conditions

### Legacy Outputs (`run.py`)

1. **CSV file**: `results/{variation}.csv` with per-condition results
2. **Heatmaps and line plots**: Showing attack success vs k and attacker strength

## Project Structure

```
test-compute-adversarial-robustness/
├── config/                      # YAML configuration files (legacy)
├── models/                      # LLM client implementations
│   ├── openai_client.py        #   - OpenAI o1/o3 with reasoning_effort support
│   └── hf_client.py            #   - HuggingFace local models
├── data/                        # Math problem generation
│   ├── gen_math.py             #   - Addition, multiplication, Hendrycks MATH
│   └── math_dataset.py         #   - Hendrycks MATH dataset loader
├── attacks/                     # Attack strategies
│   ├── many_shot.py            #   - Many-shot jailbreak (paper prompts)
│   └── distractor.py           #   - Think-less, nerd-sniping
├── defense/                     # Defense mechanisms
│   ├── inference_budget.py     #   - Test-time compute management
│   └── voting.py               #   - Majority voting
├── eval/                        # Evaluation utilities
│   ├── grid_runner.py          #   - Run experiments over parameter grids
│   ├── metrics.py              #   - Attack success rate, accuracy
│   └── plotting.py             #   - Heatmap and line plot generation
├── results/                     # Output directory (CSVs and plots)
├── run_addition_verbose.py      # ⭐ Quick start: Test o1 with detailed logging
├── plot_verbose_results.py      # ⭐ Plot verbose results as heatmaps
├── run_figure2.py               # Advanced: Reproduce full Figure 2
├── run.py                       # Legacy: YAML config-based runner
└── requirements.txt             # Dependencies
```

## Configuration

### Verbose Runner (`run_addition_verbose.py`)

Edit the script directly to customize:

```python
model_name = "o1"                                    # o1, o3-mini, gpt-4, etc.
n_samples = 4                                        # Number of test problems
reasoning_efforts = ["low", "medium", "high"]        # Test-time compute levels
attacker_strengths = [100, 500, 1000, 2000]         # Attack token budgets
goals = ["output_42", "answer_plus_1", "answer_times_7"]  # Attack objectives
```

### YAML Configs (Legacy)

Edit `config/*.yaml` files to customize:

- **Model**: `backend`, `model_name`, `device` (for HF)
- **Data**: `task` (addition/multiplication/math/mixed), `digits`, `n_samples`
- **Experiment**: `k_values`, `attacker_strengths`, `attacker_goals`, `max_tokens`
- **Variations**: `use_think_less`, `use_nerd_snipe`, `nerd_snipe_tokens`

## Expected Results

### Verbose Runner Results

**Attack success should DECREASE with more test-time compute:**

```
                low      medium    high
strength=100   [0.X]     [0.Y]    [0.0]  ← Weak attack
strength=500   [0.X]     [0.Y]    [0.0]  
strength=1000  [0.X]     [0.Y]    [0.0]
strength=2000  [0.X]     [0.Y]    [0.0]  ← Strong attack
```

- **Left (low)**: Quick thinking → More vulnerable → Higher attack success
- **Right (high)**: Deep thinking → More robust → Lower attack success
- **Bottom (low strength)**: Few examples → Easier to defend
- **Top (high strength)**: Many examples → Harder to defend

### Figure 2 Results

- **Top-left (simple task + simple goal)**: Dark purple (attacks fail) with high inference compute
- **Bottom-right (hard task + hard goal)**: More yellow/green (attacks succeed) even with high compute
- **Gradient**: Easy-to-defend (top-left) → Hard-to-defend (bottom-right)

## Known Issues

⚠️ **Empty Responses**: When o1/o3 models use high reasoning effort, they may detect the attack pattern and refuse to respond (return empty string). This is currently handled as:
- **Current behavior**: Empty → Treated as attack success (model disrupted)
- **Alternative interpretation**: Empty → Defense worked (model detected attack)

The interpretation affects whether you see an **inverse** or **expected** relationship in results.

## Notes

- Experiments are deterministic given seeds
- Progress bars show real-time progress
- Results are saved incrementally to CSV
- Empty model responses indicate safety system activation
