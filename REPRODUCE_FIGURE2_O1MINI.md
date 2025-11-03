# Reproducing Figure 2 with o1-mini

This guide shows you exactly how to reproduce **Figure 2** from the paper using **o1-mini** (OpenAI's model).

## About Figure 2

**Figure 2** demonstrates many-shot jailbreaking attacks on o1-mini across:
- **3 Tasks** (rows): 2-digit addition → 2-digit multiplication → MATH problems
- **3 Goals** (columns): output_42 → answer_plus_1 → answer_times_7
- **Defense**: Self-consistency voting with varying k (inference-time compute)
- **Attack**: Many-shot examples with varying strength (number of tokens)

**Key Finding**: Attack success tends to zero as inference-time compute increases, BUT some hard task+goal combinations remain vulnerable even with high compute.

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install datasets library (for real MATH dataset - optional)

```bash
pip install datasets
```

### 3. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
```

## Quick Test (5-10 minutes)

First, verify everything works with a small test:

```bash
python run_figure2.py \
    --backend openai \
    --model_name o1-mini \
    --n_samples 5 \
    --output_dir results/test
```

This runs with only 5 samples per condition for quick verification.

## Full Reproduction (4-8 hours)

### Option 1: With Synthetic MATH Problems (Simpler, Faster)

```bash
python run_figure2.py \
    --backend openai \
    --model_name o1-mini \
    --n_samples 100 \
    --digits 2 \
    --output_dir results
```

**Estimated Time**: 4-8 hours
**Cost**: ~$50-100 in API credits (depends on token usage)
**Output**: `results/figure2.png` + 9 CSV files

### Option 2: With Real Hendrycks MATH Dataset (Paper-Accurate)

```bash
python run_figure2.py \
    --backend openai \
    --model_name o1-mini \
    --n_samples 100 \
    --digits 2 \
    --use_hendrycks_math \
    --output_dir results
```

**Difference**: Uses real [Hendrycks MATH dataset](https://huggingface.co/datasets/EleutherAI/hendrycks_math) for the "MATH problems" task (row 3)

## Command-Line Options

```bash
python run_figure2.py [OPTIONS]

Required for o1-mini:
  --backend openai              Use OpenAI API
  --model_name o1-mini          Specify o1-mini model

Optional:
  --n_samples INT               Samples per condition (default: 100)
  --digits {2,3,4}              Digits for add/mult (default: 2)
  --use_hendrycks_math          Use real MATH dataset
  --output_dir TEXT             Output directory (default: results)
  --seed INT                    Random seed (default: 42)
```

## Understanding the Parameters

### Experimental Grid

The script runs 9 experiments with:

| Parameter | Values | Description |
|-----------|--------|-------------|
| **k** (x-axis) | [316, 1000, 3162, 10000] | Inference-time compute (log-scale: 10^2.5 to 10^4) |
| **Attack Strength** (y-axis) | [316, 1000, 3162, 10000] | Number of adversarial tokens (log-scale) |
| **Tasks** | addition, multiplication, math | 3 rows in Figure 2 |
| **Goals** | output_42, answer_plus_1, answer_times_7 | 3 columns in Figure 2 |

### Total Model Calls

For `--n_samples 100`:
- **Per condition**: 100 samples × 4 k-values × 4 attack strengths = 1,600 calls
- **All 9 conditions**: 1,600 × 9 = **14,400 total calls**
- **With k=10,000**: Each call samples 10,000 times → millions of tokens!

⚠️ **Note**: o1-mini uses **reasoning tokens**, so total token usage is higher than prompt+response tokens alone.

## Expected Output

After completion, you get:

### Main Figure
```
results/figure2.png
```

A 3×3 grid of heatmaps showing:
- **Purple**: Low attack success (defense works)
- **Yellow**: High attack success (defense fails)
- **Gradient**: Top-left (easy to defend) → bottom-right (hard to defend)

### Individual CSV Files (9 total)
```
results/figure2_addition_output_42.csv
results/figure2_addition_answer_plus_1.csv
results/figure2_addition_answer_times_7.csv
results/figure2_multiplication_output_42.csv
results/figure2_multiplication_answer_plus_1.csv
results/figure2_multiplication_answer_times_7.csv
results/figure2_math_output_42.csv
results/figure2_math_answer_plus_1.csv
results/figure2_math_answer_times_7.csv
```

Each CSV contains:
- `k`: Inference compute level
- `attacker_strength`: Attack token count
- `attacker_goal`: Goal type
- `attack_success_rate`: P[attack succeeds] (0-1)
- `accuracy`: Model accuracy (0-1)

## Monitoring Progress

The script shows real-time progress:

```
================================================================================
Running experiment: figure2_addition_output_42
================================================================================
  Generating addition problems...
  Task: addition, Goal: output_42

Running figure2_addition_output_42 experiment: 100%|████████| 1600/1600 [00:30<00:00]
Results saved to results/figure2_addition_output_42.csv
...
```

## Cost Estimation

### o1-mini Pricing (as of Nov 2024)
- **Input tokens**: $3 per 1M tokens
- **Output tokens**: $12 per 1M tokens
- **Reasoning tokens** (internal): Not charged separately but increase latency

### Estimated Costs for n_samples=100
- **Prompt tokens**: ~500 tokens × 14,400 calls = 7.2M tokens → ~$22
- **Output tokens**: ~100 tokens × 14,400 calls = 1.44M tokens → ~$17
- **Total**: ~$40-60 (varies with actual token usage)

For `n_samples=10` (quick test): ~$4-6

## Rate Limits

OpenAI has rate limits on o1-mini:
- **Free tier**: Very limited
- **Tier 1+**: 10K+ requests per day

If you hit rate limits:
1. The script will automatically retry
2. Or reduce `--n_samples`
3. Or run experiments separately (see below)

## Running Experiments Separately

If you want to run conditions individually (useful for debugging or parallel execution):

```bash
# Just run addition + output_42
python run.py --config config/figure2.yaml \
    --override task=addition \
    --override goals='[output_42]'
```

Or modify `run_figure2.py` to run specific task/goal combinations.

## Troubleshooting

### Issue: Rate limit exceeded
**Solution**: 
- Wait and retry
- Reduce `--n_samples`
- Upgrade OpenAI tier

### Issue: Authentication error
**Solution**: 
- Check `export OPENAI_API_KEY="..."`
- Verify key is valid
- Check `.env` file

### Issue: Model not found
**Solution**: 
- Verify you have access to o1-mini
- Try `--model_name o1-preview` or `gpt-4`

### Issue: Very slow
**Solution**: 
- This is expected! o1-mini with k=10,000 is computationally expensive
- Each high-k sample takes longer due to reasoning
- Consider starting with `--n_samples 10`

### Issue: Import error for datasets
**Solution**: 
- Don't use `--use_hendrycks_math`
- Or install: `pip install datasets`

## Differences from Paper

### What's the Same
✅ Model: o1-mini (when you use `--model_name o1-mini`)
✅ Tasks: Addition, multiplication, MATH problems
✅ Goals: output_42, answer_plus_1, answer_times_7
✅ Parameters: Same log-scale k and attack strength values

### What's Different
⚠️ **MATH problems**: Uses synthetic problems by default (simpler)
  - **Fix**: Add `--use_hendrycks_math` for real dataset
  
⚠️ **Digit count**: Default is 2-digit (Figure 2 shows "2 digit")
  - Figure 20 (o1-preview) uses 4-digit
  - **Fix**: Add `--digits 4` if you want 4-digit

⚠️ **Sample size**: Paper likely uses more samples
  - **Fix**: Increase `--n_samples` (but costs more)

## Validation

To verify your results match the paper:

1. **Top-left (addition + output_42)**: Should be mostly **purple** (low attack success)
2. **Bottom-right (MATH + answer_times_7)**: Should have more **yellow/green** (higher attack success)
3. **Pattern**: Attack success should generally **decrease** as k increases (left to right in each heatmap)
4. **Pattern**: Attack success should generally **increase** as attacker strength increases (bottom to top in each heatmap)

## Next Steps

1. ✅ **Quick test**: Run with `--n_samples 5` first
2. **Pilot run**: Try `--n_samples 10` (~30-60 min, $5)
3. **Full run**: Use `--n_samples 100` (~4-8 hours, $50)
4. **Analysis**: Compare your `results/figure2.png` with paper's Figure 2
5. **Iterate**: Adjust parameters, try different models

## Alternative Models

If you don't have access to o1-mini or want to test faster:

### OpenAI GPT-4
```bash
python run_figure2.py --backend openai --model_name gpt-4 --n_samples 10
```

### OpenAI GPT-3.5-turbo (Fastest/Cheapest)
```bash
python run_figure2.py --backend openai --model_name gpt-3.5-turbo --n_samples 10
```

### HuggingFace Model (Local/Free)
```bash
python run_figure2.py --backend huggingface \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --device cuda --n_samples 10
```

⚠️ Results will differ from paper with different models!

## Paper Reference

**Title**: "Many-shot Jailbreaking"  
**Authors**: Anil et al., 2024  
**arXiv**: 2501.18841v1  
**Figure 2**: Many-shot attack on o1-mini  
**Figure 20**: Same experiments on o1-preview (4-digit instead of 2-digit)

**Section A.1**: Details on MULTIPLICATION, ADDITION, MATH DATASET  
**MATH Dataset**: [Hendrycks et al. on HuggingFace](https://huggingface.co/datasets/EleutherAI/hendrycks_math)

## Support

For questions or issues:
1. Check this guide
2. Review `QUICK_START.md` and `FIGURE2_GUIDE.md`
3. Run `python test_figure2.py` to verify setup
4. Check OpenAI API status and rate limits
5. Review error messages carefully

## Complete Example Session

```bash
# Step 1: Setup
export OPENAI_API_KEY="sk-..."
pip install -r requirements.txt
pip install datasets

# Step 2: Quick test (5 min, $1)
python run_figure2.py --backend openai --model_name o1-mini --n_samples 5

# Step 3: Check output
ls results/figure2*.png results/figure2*.csv

# Step 4: Full run (4-8 hours, $50)
python run_figure2.py \
    --backend openai \
    --model_name o1-mini \
    --n_samples 100 \
    --digits 2 \
    --use_hendrycks_math \
    --output_dir results

# Step 5: View results
open results/figure2.png
```

---

**Ready to reproduce Figure 2? Start with the quick test!** 🚀

