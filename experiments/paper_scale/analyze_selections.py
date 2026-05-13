#!/usr/bin/env python3
"""
Opt-GCS Selection Analysis & Visualization.

Generates paper-quality analysis of data selection results:
1. Selection overlap matrix (Jaccard similarity between methods)
2. Domain composition analysis
3. 2D PCA visualization of selected vs unselected
4. Data efficiency curves
5. Score distribution analysis

Usage:
    python experiments/paper_scale/analyze_selections.py \
        --save_dir /jizhicfs/karonhe/dataflex_saves/paper_experiments \
        --output_dir /jizhicfs/karonhe/dataflex_saves/paper_experiments/analysis
"""

import argparse
import json
import os
import glob
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_selections(save_dir: str) -> dict:
    """Load all selection results from the experiment directory."""
    selections = {}
    for method_dir in sorted(glob.glob(os.path.join(save_dir, "*/"))):
        method_name = os.path.basename(method_dir.rstrip('/'))
        if method_name in ('logs', 'configs', 'analysis'):
            continue
        for budget_dir in sorted(glob.glob(os.path.join(method_dir, "budget_*/"))):
            budget = os.path.basename(budget_dir.rstrip('/'))
            # Find selection cache files
            cache_dir_pattern = os.path.join(save_dir, f"../*{method_name}*output/step_*.json")
            for cache_file in glob.glob(os.path.join(budget_dir, "**/*.json"), recursive=True):
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    if 'indices' in data:
                        key = f"{method_name}/{budget}"
                        selections[key] = {
                            'indices': set(data['indices']),
                            'metrics': data.get('metrics', {}),
                        }
                except:
                    pass
    return selections


def compute_overlap_matrix(selections: dict, output_dir: str):
    """Compute and plot Jaccard similarity between all method pairs."""
    methods = sorted(selections.keys())
    n = len(methods)
    if n < 2:
        print("  Not enough methods to compute overlap. Skipping.")
        return

    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s_i = selections[methods[i]]['indices']
            s_j = selections[methods[j]]['indices']
            if len(s_i | s_j) > 0:
                overlap[i, j] = len(s_i & s_j) / len(s_i | s_j)
            else:
                overlap[i, j] = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(overlap, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [m.split('/')[0][:15] for m in methods]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title('Selection Overlap (Jaccard Similarity)')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{overlap[i,j]:.2f}', ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selection_overlap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved overlap matrix to {output_dir}/selection_overlap.png")


def analyze_domain_composition(data_path: str, selections: dict, output_dir: str):
    """Analyze what types of data each method selects."""
    # Load training data
    with open(data_path) as f:
        data = json.load(f)

    # Simple domain classification by keywords
    def classify_sample(item):
        text = (item.get('instruction', '') + ' ' + item.get('output', '')).lower()
        if any(w in text for w in ['code', 'python', 'function', 'programming', 'algorithm']):
            return 'code'
        elif any(w in text for w in ['math', 'calculate', 'equation', 'theorem', 'proof']):
            return 'math'
        elif any(w in text for w in ['reason', 'logic', 'deduce', 'infer', 'conclusion']):
            return 'reasoning'
        elif any(w in text for w in ['translate', 'language', 'grammar', 'word']):
            return 'language'
        elif any(w in text for w in ['science', 'physics', 'chemistry', 'biology']):
            return 'science'
        else:
            return 'general'

    # Classify all samples
    domains = [classify_sample(item) for item in data]
    domain_counter = Counter(domains)
    print(f"  Full dataset domain distribution: {dict(domain_counter)}")

    # Analyze each method's selection
    method_compositions = {}
    for method, sel_data in selections.items():
        indices = sel_data['indices']
        selected_domains = [domains[i] for i in indices if i < len(domains)]
        method_compositions[method] = Counter(selected_domains)

    if not method_compositions:
        return

    # Plot stacked bar chart
    all_domains = sorted(domain_counter.keys())
    fig, ax = plt.subplots(figsize=(12, 6))

    methods_list = sorted(method_compositions.keys())
    x = np.arange(len(methods_list))
    width = 0.8

    bottom = np.zeros(len(methods_list))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_domains)))

    for domain_idx, domain in enumerate(all_domains):
        counts = []
        for method in methods_list:
            total = sum(method_compositions[method].values())
            count = method_compositions[method].get(domain, 0)
            counts.append(count / total * 100 if total > 0 else 0)
        counts = np.array(counts)
        ax.bar(x, counts, width, bottom=bottom, label=domain, color=colors[domain_idx])
        bottom += counts

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Domain Composition of Selected Data')
    ax.set_xticks(x)
    short_names = [m.split('/')[0][:15] for m in methods_list]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_composition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved domain composition to {output_dir}/domain_composition.png")


def plot_data_efficiency_curves(results_file: str, output_dir: str):
    """Plot data efficiency curves (selection ratio vs benchmark score)."""
    if not os.path.exists(results_file):
        print(f"  Results file not found: {results_file}")
        print("  Skipping data efficiency curves. Run evaluation first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    for idx, (method, scores) in enumerate(sorted(results.items())):
        budgets = sorted(scores.keys(), key=lambda x: int(x))
        x_vals = [int(b) / 100000 * 100 for b in budgets]  # as percentage
        y_vals = [scores[b] for b in budgets]
        ax.plot(x_vals, y_vals, marker=markers[idx % len(markers)],
                label=method, linewidth=2, markersize=8)

    ax.set_xlabel('Selection Ratio (%)')
    ax.set_ylabel('MMLU Accuracy')
    ax.set_title('Data Efficiency Curves')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_efficiency_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved data efficiency curves to {output_dir}/data_efficiency_curves.png")


def analyze_score_distributions(selections: dict, output_dir: str):
    """Analyze and plot score distributions from selection metadata."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean scores (selected vs all)
    methods = []
    scores_selected = []
    scores_all = []

    for method, sel_data in sorted(selections.items()):
        metrics = sel_data.get('metrics', {})
        if 'scores_selected_mean' in metrics and 'scores_all_mean' in metrics:
            methods.append(method.split('/')[0][:15])
            scores_selected.append(metrics['scores_selected_mean'])
            scores_all.append(metrics['scores_all_mean'])

    if methods:
        x = np.arange(len(methods))
        width = 0.35
        axes[0].bar(x - width/2, scores_all, width, label='All data', alpha=0.7)
        axes[0].bar(x + width/2, scores_selected, width, label='Selected', alpha=0.7)
        axes[0].set_xlabel('Method')
        axes[0].set_ylabel('Mean Projection Score')
        axes[0].set_title('Score: Selected vs All')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[0].legend()

    # Plot 2: Effective rank and whitening beta comparison
    methods2 = []
    eff_ranks = []
    betas = []
    for method, sel_data in sorted(selections.items()):
        metrics = sel_data.get('metrics', {})
        if 'effective_rank' in metrics:
            methods2.append(method.split('/')[0][:15])
            eff_ranks.append(metrics['effective_rank'])
            betas.append(metrics.get('whitening_beta', 0))

    if methods2:
        x = np.arange(len(methods2))
        axes[1].bar(x, eff_ranks, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Method')
        axes[1].set_ylabel('Effective Rank')
        axes[1].set_title('Effective Rank Used')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods2, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved score analysis to {output_dir}/score_analysis.png")


def main():
    parser = argparse.ArgumentParser(description='Opt-GCS Selection Analysis')
    parser.add_argument('--save_dir', type=str,
                        default='/jizhicfs/karonhe/dataflex_saves/paper_experiments')
    parser.add_argument('--data_path', type=str,
                        default='/jizhicfs/karonhe/DataFlex/data/Openhermes_train.json')
    parser.add_argument('--results_file', type=str,
                        default='/jizhicfs/karonhe/dataflex_saves/paper_experiments/mmlu_results.json')
    parser.add_argument('--output_dir', type=str,
                        default='/jizhicfs/karonhe/dataflex_saves/paper_experiments/analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Opt-GCS Selection Analysis")
    print("=" * 60)

    # Load selections
    print("\n[1] Loading selection results...")
    selections = load_selections(args.save_dir)
    print(f"  Found {len(selections)} method/budget combinations")

    # Overlap analysis
    print("\n[2] Computing selection overlap...")
    compute_overlap_matrix(selections, args.output_dir)

    # Domain composition
    print("\n[3] Analyzing domain composition...")
    if os.path.exists(args.data_path):
        analyze_domain_composition(args.data_path, selections, args.output_dir)
    else:
        print(f"  Data not found at {args.data_path}")

    # Score distributions
    print("\n[4] Analyzing score distributions...")
    analyze_score_distributions(selections, args.output_dir)

    # Data efficiency curves
    print("\n[5] Plotting data efficiency curves...")
    plot_data_efficiency_curves(args.results_file, args.output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
