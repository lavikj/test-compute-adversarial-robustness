# Quick Start: Reproducing Figure 2

## What is Figure 2?

Figure 2 is a **3×3 grid of heatmaps** from the paper showing how inference-time compute affects adversarial robustness across different:
- **Tasks** (rows): Addition → Multiplication → MATH problems (easy to hard)
- **Goals** (columns): Output 42 → Answer+1 → Answer×7 (easy to hard)

**Color**: Purple = defense works, Yellow = attack succeeds

## Installation (First Time Only)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key (only if using OpenAI)
export OPENAI_API_KEY="your-key-here"
```

## Quick Test (5 minutes)

Run tests to verify everything works:
```bash
python test_figure2.py
```

Expected output:
```
✅ All tests passed! Implementation is ready.
```

## Generate Demo Figure (30 seconds)

Generate Figure 2 with dummy data to see the layout:
```bash
python demo_figure2.py
```

Output: `results/figure2_demo.png`

## Run Small Experiment (10-30 minutes)

Generate Figure 2 with real model but few samples:
```bash
python run_figure2.py --n_samples 10
```

Output: `results/figure2.png` + 9 CSV files

## Run Full Experiment (2-8 hours)

Generate publication-quality Figure 2:
```bash
# With o1-mini (as in paper - recommended for exact reproduction)
export OPENAI_API_KEY="your-key"
python run_figure2.py --backend openai \
                      --model_name o1-mini \
                      --n_samples 100 \
                      --use_hendrycks_math

# With HuggingFace model (free, local)
python run_figure2.py --backend huggingface \
                      --model_name microsoft/Phi-3-mini-4k-instruct \
                      --device cuda \
                      --n_samples 100

# With GPT-4 (faster than o1-mini)
python run_figure2.py --backend openai \
                      --model_name gpt-4 \
                      --n_samples 100
```

**For exact paper reproduction with o1-mini**, see: `REPRODUCE_FIGURE2_O1MINI.md`

## What You Get

After running, check the `results/` directory:

```
results/
├── figure2.png                              # Main 3×3 grid figure
├── figure2_addition_output_42.csv           # Individual results
├── figure2_addition_answer_plus_1.csv
├── figure2_addition_answer_times_7.csv
├── figure2_multiplication_output_42.csv
├── figure2_multiplication_answer_plus_1.csv
├── figure2_multiplication_answer_times_7.csv
├── figure2_math_output_42.csv
├── figure2_math_answer_plus_1.csv
└── figure2_math_answer_times_7.csv
```

## Command Line Options

```bash
python run_figure2.py [OPTIONS]

Options:
  --backend {openai,huggingface}  Model backend (default: huggingface)
  --model_name TEXT              Model name (default: microsoft/Phi-3-mini-4k-instruct)
  --device {cuda,cpu}            Device for HF models (default: auto)
  --n_samples INT                Samples per condition (default: 100)
  --output_dir TEXT              Output directory (default: results)
  --seed INT                     Random seed (default: 42)
```

## Expected Results

Your figure should show:

- **Top-left** (addition + output_42): Dark purple → Easy to defend
- **Bottom-right** (MATH + answer×7): Yellow/green → Hard to defend
- **Diagonal gradient**: Increasing difficulty from top-left to bottom-right

This demonstrates that **scaling inference compute has limits** - it helps for simple attacks but struggles with harder tasks/goals.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Use `--device cpu` or smaller model |
| Very slow | Add `--device cuda` or reduce `--n_samples 10` |
| Import errors | Run from repo root directory |
| Rate limits (OpenAI) | Reduce `--n_samples` or add delays |
| Tests fail | Check dependencies: `pip install -r requirements.txt` |

## Next Steps

1. ✅ **Run demo**: `python demo_figure2.py`
2. ✅ **Quick test**: `python run_figure2.py --n_samples 10`
3. **Full run**: `python run_figure2.py --n_samples 100`
4. **Analyze**: Open `results/figure2.png` and CSV files
5. **Customize**: Edit `run_figure2.py` or `config/figure2.yaml`

## More Information

- **Detailed guide**: See `FIGURE2_GUIDE.md`
- **Implementation details**: See `IMPLEMENTATION_SUMMARY.md`
- **Original paper**: arXiv:2501.18841v1

## Support

If you encounter issues:
1. Run `python test_figure2.py` to diagnose
2. Check error messages carefully
3. Review documentation in `FIGURE2_GUIDE.md`
4. Verify API keys are set (for OpenAI)

