# Guide to Reproducing Figure 2

This guide explains how to reproduce Figure 2 from the paper, which shows a 3×3 grid of attack success rate heatmaps across different tasks and attacker goals.

## Figure 2 Overview

**Figure 2** demonstrates how inference-time compute (self-consistency voting) affects robustness across:

### Rows (Tasks - increasing difficulty):
1. **Simple Task**: 2-digit addition (e.g., "23 + 45 =")
2. **Medium Task**: 2-digit multiplication (e.g., "23 × 45 =")
3. **Harder Task**: MATH problems (algebra, multi-step arithmetic, word problems)

### Columns (Attacker Goals - increasing difficulty):
1. **Simple Goal**: `output_42` - Always output 42
2. **Medium Goal**: `answer_plus_1` - Output correct answer + 1
3. **Harder Goal**: `answer_times_7` - Output correct answer × 7

### Axes:
- **X-axis**: Inference time compute (log-scale) - k values [316, 1000, 3162, 10000]
- **Y-axis**: Attack length in tokens (log-scale) - attacker strengths [316, 1000, 3162, 10000]

### Color Scale:
- **Dark purple (0.0)**: Attack failed, defense works
- **Yellow (1.0)**: Attack succeeded, defense failed

## Quick Start

### Option 1: Using the Script (Recommended)

```bash
# With HuggingFace model (default)
python run_figure2.py --backend huggingface \
                      --model_name microsoft/Phi-3-mini-4k-instruct \
                      --n_samples 100

# With OpenAI model
export OPENAI_API_KEY="your-api-key"
python run_figure2.py --backend openai \
                      --model_name gpt-4 \
                      --n_samples 100

# With GPU acceleration (HuggingFace)
python run_figure2.py --backend huggingface \
                      --model_name microsoft/Phi-3-mini-4k-instruct \
                      --device cuda \
                      --n_samples 100
```

### Option 2: Manual Individual Experiments

You can also run each of the 9 experiments individually:

```bash
# Create config files for each task
python run.py --config config/figure2_addition_output42.yaml
python run.py --config config/figure2_addition_plus1.yaml
# ... etc for all 9 combinations
```

## Script Arguments

```
--backend         Model backend: "openai" or "huggingface" (default: huggingface)
--model_name      Model name (default: microsoft/Phi-3-mini-4k-instruct)
--device          Device for HF models: "cuda" or "cpu" (default: None = auto)
--n_samples       Number of test samples per condition (default: 100)
--output_dir      Output directory (default: "results")
--seed            Random seed (default: 42)
```

## Outputs

After running `run_figure2.py`, you will get:

### 1. Main Figure
- `results/figure2.png` - The complete 3×3 grid visualization

### 2. Individual CSV Files (9 total)
- `results/figure2_addition_output_42.csv`
- `results/figure2_addition_answer_plus_1.csv`
- `results/figure2_addition_answer_times_7.csv`
- `results/figure2_multiplication_output_42.csv`
- `results/figure2_multiplication_answer_plus_1.csv`
- `results/figure2_multiplication_answer_times_7.csv`
- `results/figure2_math_output_42.csv`
- `results/figure2_math_answer_plus_1.csv`
- `results/figure2_math_answer_times_7.csv`

Each CSV contains columns:
- `k`: Inference compute (self-consistency samples)
- `attacker_strength`: Attack length in tokens
- `attacker_goal`: Goal type
- `attack_success_rate`: P[attack success]
- `accuracy`: Model accuracy (correct answers)

## Expected Patterns

### Top-Left (Simple + Simple)
- **Dark purple** at high k values
- Defense easily succeeds with modest inference compute
- Attack success drops quickly as k increases

### Bottom-Right (Hard + Hard)
- **More yellow/green** even at high k values
- Defense struggles even with large inference compute
- Attack success remains high

### Diagonal Gradient
- Difficulty increases from top-left to bottom-right
- Shows limits of scaling inference compute

## Computational Requirements

### Approximate Runtime
For `n_samples=100` per condition:
- **Total samples**: 100 × 9 conditions × 4 k-values × 4 attack strengths = 14,400 model calls
- **Estimated time** (depends on model):
  - Small local model (Phi-3-mini): ~2-4 hours
  - Large API model (GPT-4): ~4-8 hours (with rate limits)

### Memory Requirements
- **HuggingFace models**: 4-16GB GPU RAM (depends on model size)
- **OpenAI API**: Minimal local memory, API credits required

## Tips for Faster Experimentation

1. **Reduce samples**: Use `--n_samples 10` for quick testing
2. **Use smaller model**: Try `--model_name microsoft/Phi-3-mini-4k-instruct`
3. **Run on GPU**: Add `--device cuda` for HuggingFace models
4. **Run specific conditions**: Modify the script to run only certain task/goal pairs

## Troubleshooting

### Out of Memory (GPU)
- Use smaller model or reduce batch size
- Try CPU: `--device cpu`

### Rate Limits (OpenAI)
- Reduce `--n_samples`
- Add delays in the model client
- Use tier-appropriate rate limits

### Slow Progress
- Check if model is running on GPU (`--device cuda`)
- Reduce problem complexity (fewer samples)
- Progress bars show real-time status

## Implementation Details

### New Features Added
1. **New attacker goal**: `answer_times_7` (harder than plus_1)
2. **MATH problems**: Algebra, multi-step arithmetic, word problems
3. **Grid plotting**: Custom 3×3 heatmap layout matching paper format
4. **Log-scale parameters**: k and attacker_strength use log-spaced values

### Key Files Modified/Created
- `attacks/many_shot.py`: Added `answer_times_7` goal
- `data/gen_math.py`: Added `sample_math()` for harder problems
- `eval/plotting.py`: Added `plot_figure2_grid()` for 3×3 layout
- `run_figure2.py`: New script to run all 9 experiments
- `config/figure2.yaml`: Configuration template

## Citation

If using this implementation, please cite the original paper:

```
Anil et al., "Many-shot jailbreaking", 2024
arXiv:2501.18841v1
```

