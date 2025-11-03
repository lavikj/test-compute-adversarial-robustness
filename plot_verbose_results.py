"""Plot verbose results as heatmaps."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load results
df = pd.read_csv('results/verbose_results.csv')

print("Loaded results:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Goals: {df['goal'].unique()}")
print(f"Reasoning efforts: {df['reasoning_effort'].unique()}")
print(f"Strengths: {sorted(df['strength'].unique())}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY BY GOAL")
print("="*80)
for goal in df['goal'].unique():
    goal_df = df[df['goal'] == goal]
    print(f"\n{goal}:")
    print(f"  Attack Success Rate: {goal_df['attack_success'].mean():.3f}")
    print(f"  Accuracy: {goal_df['accuracy'].mean():.3f}")

# Create 1x3 grid of heatmaps (one per goal)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Colormap (viridis: purple=0, yellow=1)
colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde724']
cmap = LinearSegmentedColormap.from_list('viridis', colors, N=100)

goals = ['output_42', 'answer_plus_1', 'answer_times_7']
goal_labels = ['Output 42', 'Answer + 1', 'Answer × 7']

for idx, (goal, label) in enumerate(zip(goals, goal_labels)):
    ax = axes[idx]
    
    # Filter data for this goal
    goal_df = df[df['goal'] == goal]
    
    # Pivot for heatmap: rows=strength, cols=reasoning_effort
    pivot = goal_df.pivot_table(
        index='strength',
        columns='reasoning_effort',
        values='attack_success',
        aggfunc='mean'
    )
    
    # Reorder columns: low → medium → high
    pivot = pivot[['low', 'medium', 'high']]
    
    print(f"\n{goal} pivot table:")
    print(pivot)
    
    # Plot heatmap
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=1, origin='lower')
    
    # Set ticks
    x_ticks = np.arange(len(pivot.columns))
    y_ticks = np.arange(len(pivot.index))
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticklabels(pivot.index, fontsize=10)
    
    ax.set_title(f"{label}\n2-digit Addition (o1)", fontsize=12, fontweight='bold')
    ax.set_xlabel("k (inference compute)", fontsize=10)
    
    if idx == 0:
        ax.set_ylabel("Attack strength (tokens)", fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(len(pivot.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            color = 'white' if value > 0.5 else 'black'
            text = ax.text(j, i, f"{value:.1f}",
                          ha="center", va="center", color=color, 
                          fontsize=9, fontweight='bold')

# Add colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('P[attack success]', fontsize=11)

plt.suptitle('Many-shot Attack vs Test-Time Compute (o1)\nAttack Success Should Decrease: low → medium → high', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('results/verbose_results_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved to: results/verbose_results_heatmap.png")

plt.close()

# Also create a summary table
print("\n" + "="*80)
print("ATTACK SUCCESS RATE BY REASONING EFFORT AND GOAL")
print("="*80)
summary = df.groupby(['goal', 'reasoning_effort'])['attack_success'].mean().unstack()
# Reorder columns: low → medium → high
if 'low' in summary.columns and 'medium' in summary.columns and 'high' in summary.columns:
    summary = summary[['low', 'medium', 'high']]
print(summary)
print("\nInterpretation:")
print("- 1.0 = Attack always succeeds")
print("- 0.0 = Defense always works")
print("- Attack success should DECREASE from low → high (more test-time compute = more robust)")

