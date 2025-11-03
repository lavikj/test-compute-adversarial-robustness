"""Plotting utilities for experiment results."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List


def plot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str,
    output_file: str,
    cmap: str = "RdYlGn_r",
):
    """
    Create heatmap of attack success rate.
    
    Args:
        df: Results DataFrame
        x_col: Column for x-axis (attacker_strength)
        y_col: Column for y-axis (k)
        value_col: Column for heat values (attack_success_rate)
        title: Plot title
        output_file: Output file path
        cmap: Colormap
    """
    # Pivot for heatmap
    pivot = df.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc="mean",
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Labels
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(
                j, i, f"{pivot.values[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_file}")


def plot_line_plot(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    group_col: Optional[str],
    title: str,
    output_file: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """
    Create line plot showing monotone decay.
    
    Args:
        df: Results DataFrame
        x_col: Column for x-axis (k or attacker_strength)
        value_col: Column for y-axis (attack_success_rate)
        group_col: Optional column to group by (attacker_strength or attacker_goal)
        title: Plot title
        output_file: Output file path
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group_col:
        for group_val in df[group_col].unique():
            group_df = df[df[group_col] == group_val]
            group_agg = group_df.groupby(x_col)[value_col].mean().sort_index()
            ax.plot(
                group_agg.index,
                group_agg.values,
                marker="o",
                label=f"{group_col}={group_val}",
                linewidth=2,
            )
        ax.legend()
    else:
        agg = df.groupby(x_col)[value_col].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", linewidth=2)
    
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or value_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Line plot saved to {output_file}")


def generate_plots(
    df: pd.DataFrame,
    variation: str,
    output_dir: str = "results",
):
    """
    Generate all plots for a variation.
    
    Args:
        df: Results DataFrame
        variation: Variation name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap: attacker_strength vs k
    heatmap_file = os.path.join(output_dir, f"{variation}_heatmap.png")
    plot_heatmap(
        df=df,
        x_col="attacker_strength",
        y_col="k",
        value_col="attack_success_rate",
        title=f"{variation}: Attack Success Rate",
        output_file=heatmap_file,
    )
    
    # Line plot: k vs attack_success_rate (grouped by attacker_strength)
    line_file = os.path.join(output_dir, f"{variation}_line_k.png")
    plot_line_plot(
        df=df,
        x_col="k",
        value_col="attack_success_rate",
        group_col="attacker_strength",
        title=f"{variation}: ASR vs Compute (k)",
        output_file=line_file,
        xlabel="k (self-consistency samples)",
        ylabel="Attack Success Rate",
    )
    
    # Line plot: attacker_strength vs attack_success_rate (grouped by k)
    line_file2 = os.path.join(output_dir, f"{variation}_line_strength.png")
    plot_line_plot(
        df=df,
        x_col="attacker_strength",
        value_col="attack_success_rate",
        group_col="k",
        title=f"{variation}: ASR vs Attacker Strength",
        output_file=line_file2,
        xlabel="Attacker Strength (tokens)",
        ylabel="Attack Success Rate",
    )


def plot_figure2_grid(
    results_dict: dict,
    output_file: str,
    tasks: List[str] = ["addition", "multiplication", "math"],
    goals: List[str] = ["output_42", "answer_plus_1", "answer_times_7"],
):
    """
    Create Figure 2: 3x3 grid of heatmaps showing attack success rates.
    
    Args:
        results_dict: Dictionary mapping (task, goal) -> DataFrame
        output_file: Output file path for the figure
        tasks: List of task names (rows)
        goals: List of goal names (columns)
    """
    # Create custom colormap (purple to yellow)
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde724']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('viridis', colors, N=n_bins)
    
    # Task and goal labels
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
    
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Figure 2: Many-shot attack (Anil et al., 2024) on a variety of math tasks and adversary goals for o1-mini.",
                 fontsize=12, y=0.995)
    
    # For each task (row) and goal (column)
    for i, task in enumerate(tasks):
        for j, goal in enumerate(goals):
            ax = axes[i, j]
            
            # Get data for this condition
            key = (task, goal)
            if key not in results_dict or results_dict[key].empty:
                # No data - show empty plot
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            df = results_dict[key]
            
            # Pivot data for heatmap
            pivot = df.pivot_table(
                index='attacker_strength',
                columns='k',
                values='attack_success_rate',
                aggfunc='mean'
            )
            
            # Plot heatmap
            im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=1, origin='lower')
            
            # Set ticks with log scale labels
            x_ticks = np.arange(len(pivot.columns))
            y_ticks = np.arange(len(pivot.index))
            
            # Format as powers of 10 if values are large
            k_labels = [f"$10^{{{np.log10(k):.1f}}}$" if k >= 100 else str(k) for k in pivot.columns]
            strength_labels = [f"$10^{{{np.log10(s):.1f}}}$" if s >= 100 else str(s) for s in pivot.index]
            
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels(k_labels, fontsize=8)
            ax.set_yticklabels(strength_labels, fontsize=8)
            
            # Add column titles at top
            if i == 0:
                ax.set_title(goal_labels[goal], fontsize=10, pad=10)
            
            # Add row labels on left
            if j == 0:
                ax.set_ylabel(task_labels[task], fontsize=10, rotation=0, ha='right', va='center', labelpad=40)
            
            # Only show axis labels on edges
            if i == 2:  # Bottom row
                ax.set_xlabel("Inference time compute (log-scale)", fontsize=9)
            if j == 0:  # Left column  
                # Y-label already set above
                pass
            
            # Add grid
            ax.set_xticks(np.arange(len(pivot.columns)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(pivot.index)) - 0.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Add global y-axis label
    fig.text(0.02, 0.5, 'Attack length (tokens)', va='center', rotation='vertical', fontsize=11)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axes[0, 0].images[0], cax=cbar_ax)
    cbar.set_label('P[attack success]', fontsize=11)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.90, 0.98])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved to {output_file}")

