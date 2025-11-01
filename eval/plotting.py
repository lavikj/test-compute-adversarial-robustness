"""Plotting utilities for experiment results."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


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

