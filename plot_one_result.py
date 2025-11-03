"""Plot single completed result."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load data
df = pd.read_csv("results/addition_output_42.csv")

print("Data loaded:")
print(df)
print(f"\nMean attack success rate: {df['attack_success_rate'].mean():.3f}")
print(f"Mean accuracy: {df['accuracy'].mean():.3f}")

# Create colormap
colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde724']
cmap = LinearSegmentedColormap.from_list('viridis', colors, N=100)

# Pivot for heatmap
pivot = df.pivot_table(
    index='attacker_strength',
    columns='k',
    values='attack_success_rate',
    aggfunc='mean'
)

print("\nPivot table:")
print(pivot)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot heatmap
im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=1, origin='lower')

# Set ticks
x_ticks = np.arange(len(pivot.columns))
y_ticks = np.arange(len(pivot.index))

ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticklabels(pivot.index, fontsize=11)

ax.set_title("Addition (2-digit) - Goal: output 42\nMany-shot attack on o1", fontsize=14, pad=15)
ax.set_xlabel("k (inference compute)", fontsize=12)
ax.set_ylabel("Attack strength (tokens)", fontsize=12)

# Add grid
ax.set_xticks(np.arange(len(pivot.columns)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(pivot.index)) - 0.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

# Add value annotations
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        text = ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                      ha="center", va="center", color="white", fontsize=10, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('P[attack success]', fontsize=12)

plt.tight_layout()
plt.savefig("results/addition_output_42_heatmap.png", dpi=300, bbox_inches='tight')
print("\n✅ Plot saved to: results/addition_output_42_heatmap.png")

plt.close()

