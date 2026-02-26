#!/usr/bin/env python3
"""
Generate publication-quality plots for semantic alignment analysis results.

Usage:
    python analysis.py

Output: Saves plots to results/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"


# Creator metadata from Table 1
creator_info = {
    "zackdfilms": ("Short Curiosity", "#1E88E5"),
    "camillaara": ("Lifestyle/Short", "#43A047"),
    "thetrenchfamily": ("Family Vlogs", "#FB8C00"),
    "markrober": ("Engineering/Science", "#E53935"),
    "nickdigiovanni": ("Cooking", "#8E24AA"),
    "jordanmatter": ("Family Vlogs", "#FDD835"),
    "cristiano": ("Sports/Lifestyle", "#00ACC1"),
    "stokestwins": ("Pranks/Challenges", "#5E35B1"),
}


def load_data(results_dir: Path) -> pd.DataFrame:
    """Load topic alignment data."""
    df = pd.read_csv(results_dir / "topic_alignment.csv")
    df["content_type"] = df["youtuber"].map(lambda x: creator_info.get(x, ("Unknown", "gray"))[0])
    df["color"] = df["youtuber"].map(lambda x: creator_info.get(x, ("Unknown", "gray"))[1])
    return df


def create_figures_dir(results_dir: Path) -> Path:
    """Create figures directory if it doesn't exist."""
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_similarity_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 1: Distribution of cosine similarity scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    sns.histplot(df["cosine_similarity"], kde=True, bins=30, ax=ax1, color="#2E86AB", alpha=0.7)
    ax1.axvline(df["cosine_similarity"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['cosine_similarity'].mean():.3f}")
    ax1.axvline(df["cosine_similarity"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {df['cosine_similarity'].median():.3f}")
    ax1.axvline(df["cosine_similarity"].quantile(0.1), color="orange", linestyle=":", alpha=0.7, label=f"10th: {df['cosine_similarity'].quantile(0.1):.3f}")
    ax1.axvline(df["cosine_similarity"].quantile(0.9), color="orange", linestyle=":", alpha=0.7, label=f"90th: {df['cosine_similarity'].quantile(0.9):.3f}")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Semantic Alignment Scores")
    ax1.legend(loc="upper left", fontsize=9)
    
    # Cumulative distribution
    sorted_scores = np.sort(df["cosine_similarity"])
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax2.plot(sorted_scores, cdf, linewidth=2, color="#2E86AB")
    ax2.axvline(0.302, color="red", linestyle=":", alpha=0.7)
    ax2.axvline(0.545, color="orange", linestyle="--", alpha=0.7)
    ax2.axvline(0.726, color="green", linestyle="-", alpha=0.7, label="Median")
    ax2.axvline(0.855, color="orange", linestyle="--", alpha=0.7)
    ax2.axvline(0.964, color="red", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Cumulative Distribution")
    ax2.grid(True, alpha=0.3)
    
    # Add percentile annotations
    ax2.annotate("10th\n0.302", xy=(0.302, 0.1), xytext=(0.1, 0.3), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))
    ax2.annotate("90th\n0.964", xy=(0.964, 0.9), xytext=(0.8, 0.7), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_similarity_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig1_similarity_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: fig1_similarity_distribution")


def plot_creator_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 2: Creator rankings with error bars."""
    creator_stats = df.groupby("youtuber")["cosine_similarity"].agg(["mean", "std", "count"]).reset_index()
    creator_stats["se"] = creator_stats["std"] / np.sqrt(creator_stats["count"])
    creator_stats["ci_lower"] = creator_stats["mean"] - 1.96 * creator_stats["se"]
    creator_stats["ci_upper"] = creator_stats["mean"] + 1.96 * creator_stats["se"]
    creator_stats["content_type"] = creator_stats["youtuber"].map(lambda x: creator_info.get(x, ("Unknown", "gray"))[0])
    creator_stats["color"] = creator_stats["youtuber"].map(lambda x: creator_info.get(x, ("Unknown", "gray"))[1])
    
    # Sort by mean
    creator_stats = creator_stats.sort_values("mean", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(creator_stats))
    bars = ax.barh(y_pos, creator_stats["mean"], 
                   xerr=[creator_stats["mean"] - creator_stats["ci_lower"], 
                         creator_stats["ci_upper"] - creator_stats["mean"]], 
                   color=creator_stats["color"], alpha=0.8, capsize=4)
    
    # Add labels
    for i, (idx, row) in enumerate(creator_stats.iterrows()):
        ax.text(row["mean"] + 0.05, i, f"{row['mean']:.3f}", va="center", fontsize=10)
        ax.text(0.02, i, f"{row['youtuber']}\n({row['content_type']})", va="center", fontsize=9)
    
    ax.set_xlabel("Mean Cosine Similarity")
    ax.set_title("Creator Rankings by Semantic Alignment (with 95% CI)")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.axvline(df["cosine_similarity"].mean(), color="black", linestyle="--", alpha=0.5, label="Overall Mean")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_creator_rankings.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig2_creator_rankings.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: fig2_creator_rankings")


def plot_alignment_tiers(df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 3: Alignment tier classification."""
    # Define tiers based on analysis.md
    low_threshold = 0.545
    high_threshold = 0.855
    
    df["tier"] = pd.cut(df["cosine_similarity"], 
                        bins=[0, low_threshold, high_threshold, 1],
                        labels=["Low", "Medium", "High"])
    
    tier_counts = df["tier"].value_counts().sort_index()
    tier_pct = (tier_counts / len(df) * 100).round(1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    colors = ["#E53935", "#FDD835", "#43A047"]
    bars = ax1.bar(tier_counts.index, tier_counts.values, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Alignment Tier")
    ax1.set_ylabel("Number of Videos")
    ax1.set_title("Distribution of Alignment Tiers")
    
    # Add percentage labels
    for bar, pct in zip(bars, tier_pct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f"{int(height)}\n({pct}%)", ha="center", va="bottom", fontweight="bold")
    
    # Box plot by tier
    sns.boxplot(data=df, x="tier", y="cosine_similarity", ax=ax2, hue="tier", palette=colors, legend=False)
    ax2.set_xlabel("Alignment Tier")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Similarity Distribution by Tier")
    ax2.axhline(0.545, color="orange", linestyle="--", alpha=0.5)
    ax2.axhline(0.855, color="orange", linestyle="--", alpha=0.5)
    
    # Add mean annotations
    for tier in ["Low", "Medium", "High"]:
        tier_mean = df[df["tier"] == tier]["cosine_similarity"].mean()
        tier_idx = {"Low": 0, "Medium": 1, "High": 2}[tier]
        ax2.text(tier_idx, tier_mean + 0.02, f"μ={tier_mean:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_alignment_tiers.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig3_alignment_tiers.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: fig3_alignment_tiers")


def plot_similarity_vs_topics(df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 4: Scatter plot of similarity vs number of topics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1.scatter(df["n_topics"], df["cosine_similarity"], alpha=0.5, s=50, color="#2E86AB")
    
    # Add trend line
    z = np.polyfit(df["n_topics"], df["cosine_similarity"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["n_topics"].min(), df["n_topics"].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Trend (r={df['n_topics'].corr(df['cosine_similarity']):.3f})")
    
    # Add correlation test
    corr, pvalue = stats.pearsonr(df["n_topics"], df["cosine_similarity"])
    ax1.text(0.95, 0.05, f"r={corr:.3f}, p={pvalue:.4f}", transform=ax1.transAxes,
            ha="right", va="bottom", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax1.set_xlabel("Number of Topics")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Semantic Alignment vs Content Complexity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Hexbin for density
    hb = ax2.hexbin(df["n_topics"], df["cosine_similarity"], gridsize=20, cmap="Blues", mincnt=1)
    ax2.set_xlabel("Number of Topics")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Density Plot: Alignment vs Topic Count")
    plt.colorbar(hb, ax=ax2, label="Count")
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_similarity_vs_topics.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig4_similarity_vs_topics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: fig4_similarity_vs_topics")


def plot_creator_violin(df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 5: Violin plot showing full distribution by creator."""
    creator_order = df.groupby("youtuber")["cosine_similarity"].mean().sort_values(ascending=False).index.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create violin plot
    sns.violinplot(data=df, x="youtuber", y="cosine_similarity", order=creator_order, ax=ax, inner="box", hue="youtuber", palette="husl", legend=False)
    
    # Add mean points
    means = df.groupby("youtuber")["cosine_similarity"].mean().reindex(creator_order)
    ax.scatter(range(len(means)), means, color="red", s=50, zorder=3, label="Mean")
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Creator")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Distribution of Alignment Scores by Creator")
    ax.legend()
    ax.axhline(df["cosine_similarity"].mean(), color="black", linestyle="--", alpha=0.3, label="Overall Mean")
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_creator_violin.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "fig5_creator_violin.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: fig5_creator_violin")


def generate_topics_table(results_dir: Path, fig_dir: Path) -> None:
    """Generate table of top 3 topics per youtuber ranked by frequency.
    
    Returns LaTeX and CSV formats.
    """
    print("\nGenerating topics table...")
    
    # Collect all topics per youtuber
    youtuber_topics: Dict[str, List[Tuple[str, int, str]]] = {}
    
    for youtuber_dir in results_dir.iterdir():
        if not youtuber_dir.is_dir():
            continue
        
        youtuber = youtuber_dir.name
        topics_file = youtuber_dir / "topics.csv"
        
        if not topics_file.exists():
            continue
        
        # Load topics for this youtuber
        topics_df = pd.read_csv(topics_file)
        
        # Count topic frequencies (using keywords as unique identifier)
        topic_counts = topics_df["keywords"].value_counts()
        total_videos = topics_df["video_id"].nunique()
        
        # Get top 3 topics with percentage
        top_topics = []
        for keywords, count in topic_counts.head(3).items():
            pct = (count / total_videos) * 100
            # Clean and truncate keywords to first 5 for readability
            kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()][:5]
            short_kw = ", ".join(kw_list) if kw_list else "N/A"
            top_topics.append((short_kw, count, f"{pct:.1f}%"))
        
        youtuber_topics[youtuber] = top_topics
    
    # Create DataFrame for table
    table_data = []
    for youtuber in sorted(youtuber_topics.keys()):
        topics = youtuber_topics[youtuber]
        row = {"Youtuber": youtuber}
        for i, (kw, count, pct) in enumerate(topics, 1):
            row[f"Topic_{i}"] = f"{kw}"
            row[f"Topic_{i}_count"] = count
            row[f"Topic_{i}_pct"] = pct
        table_data.append(row)
    
    topics_table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = fig_dir / "top_topics_per_youtuber.csv"
    topics_table_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Generate LaTeX table with frequency info
    latex = []
    latex.append("\\begin{table*}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Top 3 Topics per Creator by Frequency}")
    latex.append("\\label{tab:top_topics}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lp{3.5cm}p{3.5cm}p{3.5cm}}")
    latex.append("\\hline")
    latex.append("\\textbf{Creator} & \\textbf{Topic 1 (n, \\%)} & \\textbf{Topic 2 (n, \\%)} & \\textbf{Topic 3 (n, \\%)} \\\\")
    latex.append("\\hline")
    
    for _, row in topics_table_df.iterrows():
        youtuber = row["Youtuber"]
        t1 = row.get("Topic_1", "N/A").replace("&", "\\&").replace("_", "\\_")
        t2 = row.get("Topic_2", "N/A").replace("&", "\\&").replace("_", "\\_")
        t3 = row.get("Topic_3", "N/A").replace("&", "\\&").replace("_", "\\_")
        c1 = row.get("Topic_1_count", 0)
        p1 = row.get("Topic_1_pct", "0%")
        c2 = row.get("Topic_2_count", 0)
        p2 = row.get("Topic_2_pct", "0%")
        c3 = row.get("Topic_3_count", 0)
        p3 = row.get("Topic_3_pct", "0%")
        latex.append(f"{youtuber} & {t1} ({c1}, {p1}) & {t2} ({c2}, {p2}) & {t3} ({c3}, {p3}) \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    
    latex_path = fig_dir / "top_topics_table.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex))
    print(f"✓ Saved: {latex_path}")
    
    # Print preview
    print("\n" + "="*80)
    print("TOPICS TABLE PREVIEW")
    print("="*80)
    print(topics_table_df[["Youtuber", "Topic_1", "Topic_2", "Topic_3"]].to_string(index=False))
    print("="*80 + "\n")


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for the paper."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total videos: {len(df)}")
    print(f"Mean cosine similarity: {df['cosine_similarity'].mean():.3f}")
    print(f"Median cosine similarity: {df['cosine_similarity'].median():.3f}")
    print(f"Standard deviation: {df['cosine_similarity'].std():.3f}")
    print(f"\nPercentiles:")
    print(f"  10th: {df['cosine_similarity'].quantile(0.1):.3f}")
    print(f"  25th: {df['cosine_similarity'].quantile(0.25):.3f}")
    print(f"  50th: {df['cosine_similarity'].quantile(0.5):.3f}")
    print(f"  75th: {df['cosine_similarity'].quantile(0.75):.3f}")
    print(f"  90th: {df['cosine_similarity'].quantile(0.9):.3f}")
    
    print(f"\nCreator Rankings (by mean similarity):")
    creator_means = df.groupby("youtuber")["cosine_similarity"].agg(["mean", "std", "count"])
    creator_means["se"] = creator_means["std"] / np.sqrt(creator_means["count"])
    creator_means["ci_lower"] = creator_means["mean"] - 1.96 * creator_means["se"]
    creator_means["ci_upper"] = creator_means["mean"] + 1.96 * creator_means["se"]
    creator_means = creator_means.sort_values("mean", ascending=False)
    
    for i, (creator, row) in enumerate(creator_means.iterrows(), 1):
        print(f"  {i}. {creator}: {row['mean']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] (n={row['count']})")
    
    print(f"\nTier Distribution:")
    low_threshold = 0.545
    high_threshold = 0.855
    n_low = (df["cosine_similarity"] < low_threshold).sum()
    n_high = (df["cosine_similarity"] >= high_threshold).sum()
    n_medium = len(df) - n_low - n_high
    print(f"  Low (< {low_threshold}): {n_low} ({100*n_low/len(df):.1f}%)")
    print(f"  Medium: {n_medium} ({100*n_medium/len(df):.1f}%)")
    print(f"  High (>= {high_threshold}): {n_high} ({100*n_high/len(df):.1f}%)")
    print("="*60 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate analysis plots for semantic alignment study")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for figures")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    results_dir = Path(args.output_dir)
    
    if not (results_dir / "topic_alignment.csv").exists():
        print(f"❌ Error: topic_alignment.csv not found in {results_dir}")
        return
    
    print("Loading data...")
    df = load_data(results_dir)
    
    print("Creating figures directory...")
    fig_dir = create_figures_dir(results_dir)
    
    print("\nGenerating plots...")
    plot_similarity_distribution(df, fig_dir)
    plot_creator_comparison(df, fig_dir)
    plot_alignment_tiers(df, fig_dir)
    plot_similarity_vs_topics(df, fig_dir)
    plot_creator_violin(df, fig_dir)
    
    # Generate topics table
    generate_topics_table(results_dir, fig_dir)
    
    print_summary_stats(df)
    
    print(f"\n✓ All plots saved to: {fig_dir}")


if __name__ == "__main__":
    main()
