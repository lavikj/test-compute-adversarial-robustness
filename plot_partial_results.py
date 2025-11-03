"""Plot partial Figure 2 results from completed experiments."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load completed results
results_dict = {}

# Try to load all possible completed experiments
experiments = [
    ("addition", "output_42"),
    ("addition", "answer_plus_1"),
    ("addition", "answer_times_7"),
    ("multiplication", "output_42"),
    ("multiplication", "answer_plus_1"),
    ("multiplication", "answer_times_7"),
    ("math", "output_42"),
    ("math", "answer_plus_1"),
    ("math", "answer_times_7"),
]

for task, goal in experiments:
    csv_file = f"results/figure2_fast_{task}_{goal}.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        results_dict[(task, goal)] = df
        print(f"✓ Loaded: {task} × {goal}")

print(f"\nTotal experiments loaded: {len(results_dict)}")

if not results_dict:
    print("❌ No completed experiments found!")
    exit(1)

# Determine grid size based on what we have
tasks_found = sorted(set(task for task, _ in results_dict.keys()))
goals_found = sorted(set(goal for _, goal in results_dict.keys()), 
                     key=lambda x: ["output_42", "answer_plus_1", "answer_times_7"].index(x))

print(f"Tasks found: {tasks_found}")
print(f"Goals found: {goals_found}")

# Create colormap (purple to yellow, like paper)
colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde724']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('viridis', colors, N=n_bins)

# Labels
task_labels = {
    "addition": "Simple Task\n2 digit add",
    "multiplication": "Medium Task\n2 digit mult",
    "math": "Harder Task\nMATH problem"
}

goal_labels = {
    "output_42": "Attacker goal (simple):\noutput 42",
    "answer_plus_1": "Attacker goal (medium):\ncorrect answer + 1",
    "answer_times_7": "Attacker goal (harder):\ncorrect answer × 7"
}

# Create subplot grid
n_rows = len(tasks_found)
n_cols = len(goals_found)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

# Handle case where axes is 1D or scalar
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

fig.suptitle(f"Figure 2 (Partial - {len(results_dict)} experiments) - Many-shot attack on o3-mini",
             fontsize=14, y=0.98)

# Plot each completed experiment
for i, task in enumerate(tasks_found):
    for j, goal in enumerate(goals_found):
        ax = axes[i, j]
        
        key = (task, goal)
        if key not in results_dict:
            ax.text(0.5, 0.5, 'Not completed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        df = results_dict[key]
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            index='attacker_strength',
            columns='k',
            values='attack_success_rate',
            aggfunc='mean'
        )
        
        # Plot heatmap
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=1, origin='lower')
        
        # Set ticks
        x_ticks = np.arange(len(pivot.columns))
        y_ticks = np.arange(len(pivot.index))
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)
        
        # Add column titles at top
        if i == 0:
            ax.set_title(goal_labels[goal], fontsize=11, pad=10)
        
        # Add row labels on left
        if j == 0:
            ax.set_ylabel(task_labels[task], fontsize=10, rotation=0, 
                         ha='right', va='center', labelpad=50)
        
        # Add axis labels
        if i == n_rows - 1:
            ax.set_xlabel("k (inference compute)", fontsize=10)
        if j == 0:
            # Y-label already set
            pass
        
        # Add grid
        ax.set_xticks(np.arange(len(pivot.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(pivot.index)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)

# Add global y-axis label
fig.text(0.04, 0.5, 'Attack strength (tokens)', va='center', rotation='vertical', fontsize=12)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = fig.colorbar(axes[0, 0].images[0] if len(axes[0, 0].images) > 0 else im, cax=cbar_ax)
cbar.set_label('P[attack success]', fontsize=11)

plt.tight_layout(rect=[0.06, 0.03, 0.90, 0.96])

# Save figure
output_file = "results/figure2_partial.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved to: {output_file}")

plt.close()

# Also print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for (task, goal), df in sorted(results_dict.items()):
    print(f"\n{task} × {goal}:")
    print(f"  Mean attack success rate: {df['attack_success_rate'].mean():.3f}")
    print(f"  Mean accuracy: {df['accuracy'].mean():.3f}")
    
    # Show trend: does attack success decrease with k?
    by_k = df.groupby('k')['attack_success_rate'].mean()
    print(f"  Attack success by k:")
    for k, asr in by_k.items():
        print(f"    k={k}: {asr:.3f}")

