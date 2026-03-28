# -*- coding: utf-8 -*-
"""
Visualization Module - VERSION 12
===================================
CHANGES (v5 → v6):
    - plot_constraint_analysis() KALDIRILDI
    - plot_pareto_with_constraints() KALDIRILDI
    - plot_pareto_pairwise_projections(): Z1/Z2, Z1/Z3, Z4/Z2, Z4/Z3, Z2/Z3 pairs REMOVED
      → Sadece Z1 vs Z4 tutulur
    - plot_pareto_2d_v1(): Selected solution marked with (★ star, red)
    - plot_pareto_2d_v1(): Z1 and Z4 value labels added to each Pareto point
    - _sort_pareto_for_plot(): Sort by Z4 (fixes linear appearance)
"""

import os
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Any, Optional

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'pareto_black':  '#1a1a1a',
    'pareto_purple': '#7B2D8E',
    'dominated':     '#C0C0C0',
    'pareto_line':   '#2ECC40',
    'selected':      '#E84646',   # Selected solution color (red)
    'ga':      '#F18F01',
    'nsga3':   '#2E86AB',
    'best':    '#C73E1D',
    'grid':    '#E8E8E8',
    'label_bg': 'white'
}

LABELS = list(string.ascii_uppercase)


def _sort_pareto_for_plot(pareto_objectives: np.ndarray):
    """
    Sorts Pareto points in ascending order of Z4.
    (v6 fix: sorting by Z1 caused linear appearance)
    """
    if len(pareto_objectives) == 0:
        return np.array([])
    return np.argsort(pareto_objectives[:, 1])  # Z4 ascending → left=low Z4, right=high Z4


def _select_best_solution(pareto_objectives: np.ndarray,
                          selected_idx: Optional[int] = None) -> int:
    """
    Determines the 'selected solution' index from the Pareto front.
    Uses selected_idx if provided; otherwise selects the max Z1 point.
    """
    if selected_idx is not None and 0 <= selected_idx < len(pareto_objectives):
        return selected_idx
    # Default: highest Z1
    return int(np.argmax(pareto_objectives[:, 0]))


def plot_pareto_2d_v1(
    pareto_objectives: np.ndarray,
    algorithm: str,
    scenario: str,
    output_path: str,
    all_solutions: List[Dict] = None,
    show_dominated: bool = True,
    selected_solution_idx: Optional[int] = None
) -> str:
    """
    Main Pareto plot (Z₁ vs Z₄) — v6 updates.

    Yenilikler:
        - Selected solution: marked with large red star (★)
        - Each Pareto point: Z1 and Z4 values shown as text
        - Sorting: by Z4 (linear appearance fix)
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    is_nsga      = 'NSGA' in algorithm.upper()
    pareto_color = COLORS['pareto_purple'] if is_nsga else COLORS['pareto_black']

    # ── 1. Dominated points (grey) — shown for both GA and NSGA-III ──
    if show_dominated and all_solutions:
        all_obj = np.array([[s.get('Z1', 0), s.get('Z4', 0)] for s in all_solutions])
        if len(all_obj) > 0:
            plot_obj = all_obj[np.random.choice(len(all_obj), min(500, len(all_obj)), replace=False)]
            ax.scatter(plot_obj[:, 1], plot_obj[:, 0],
                       c=COLORS['dominated'], alpha=0.4, s=50,
                       edgecolors='#AAAAAA', linewidths=0.4, zorder=1)

    # ── 2. Pareto front ──────────────────────────────────────────────────
    if len(pareto_objectives) > 0:
        sorted_idx    = _sort_pareto_for_plot(pareto_objectives)
        sorted_pareto = pareto_objectives[sorted_idx]
        sel_in_orig   = _select_best_solution(pareto_objectives, selected_solution_idx)
        # Find corresponding index in sorted array
        sel_in_sorted = int(np.where(sorted_idx == sel_in_orig)[0][0])

        # Green connection line
        if len(sorted_pareto) > 1:
            ax.plot(sorted_pareto[:, 1], sorted_pareto[:, 0],
                    c=COLORS['pareto_line'], linewidth=2.5, zorder=2,
                    solid_capstyle='round')

        # Pareto points
        for i, (z1, z4) in enumerate(sorted_pareto):
            is_selected = (i == sel_in_sorted)
            if is_selected:
                # Selected solution: large red star
                ax.scatter(z4, z1, marker='*',
                           s=550, c=COLORS['selected'],
                           edgecolors='#8B0000', linewidths=1.5, zorder=5)
            else:
                ax.scatter(z4, z1, marker='o',
                           s=180, c=pareto_color,
                           edgecolors='white' if is_nsga else '#333333',
                           linewidths=1.5, zorder=3)

            # Letter label (inside point)
            if i < len(LABELS):
                label = LABELS[i]
                ax.annotate(label, (z4, z1),
                            fontsize=9, fontweight='bold', color='white',
                            ha='center', va='center', zorder=6)

        # ── 3. Z1 and Z4 value labels (only selected + endpoints to avoid overlap) ──
        for i, (z1, z4) in enumerate(sorted_pareto):
            is_selected = (i == sel_in_sorted)
            # Only label: selected solution + first point + last point
            show_label = is_selected or i == 0 or i == len(sorted_pareto) - 1
            if not show_label:
                continue

            # Place label slightly above the point
            offset_y = (sorted_pareto[:, 0].max() - sorted_pareto[:, 0].min()) * 0.025 + 0.3
            offset_y = max(offset_y, 0.5)

            value_text = f"Z₁={z1:.1f}%\nZ₄={z4:.0f}"
            txt_color  = COLORS['selected'] if is_selected else '#333333'
            txt_weight = 'bold' if is_selected else 'normal'
            txt_size   = 8.5 if is_selected else 7.5

            ax.annotate(
                value_text,
                xy=(z4, z1),
                xytext=(z4, z1 + offset_y),
                fontsize=txt_size, fontweight=txt_weight, color=txt_color,
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=COLORS['selected'] if is_selected else '#CCCCCC',
                          alpha=0.85),
                zorder=7
            )

    # ── 4. Axes and title ────────────────────────────────────────────────
    ax.set_xlabel('Z₄: Capacity Deviation [Objective 2]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Z₁: Average Satisfaction (%) [Objective 1]', fontsize=13, fontweight='bold')
    algo_label = 'NSGA-III' if is_nsga else 'GA'
    ax.set_title(f"Pareto Front — {algo_label} ({scenario})", fontsize=15, fontweight='bold', pad=15)

    # ── 5. Legend ─────────────────────────────────────────────────────────
    legend_elements = []
    if not is_nsga:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['dominated'],
                   markersize=10, markeredgecolor='#AAAAAA', label='All Solutions'))
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=pareto_color,
               markersize=12, markeredgecolor='white' if is_nsga else '#333',
               label='Pareto Front'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['selected'],
               markersize=16, markeredgecolor='#8B0000', label='Selected Solution'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              framealpha=0.9, edgecolor='#CCCCCC')

    # ── 6. Grid and style ───────────────────────────────────────────────────
    ax.grid(True, alpha=0.15, color='#999999', linestyle='-')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5); spine.set_color('#333333')
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)

    # ── 7. Statistics box ──────────────────────────────────────────────
    if len(pareto_objectives) > 0:
        sel_obj   = pareto_objectives[sel_in_orig]
        stats_txt = (
            f"Pareto Solutions: {len(pareto_objectives)}\n"
            f"Z₄: [{pareto_objectives[:,1].min():.0f} — {pareto_objectives[:,1].max():.0f}]\n"
            f"Z₁: [{pareto_objectives[:,0].min():.1f}% — {pareto_objectives[:,0].max():.1f}%]\n"
            f"★ Selected: Z₁={sel_obj[0]:.1f}%, Z₄={sel_obj[1]:.0f}"
        )
        ax.text(0.02, 0.02, stats_txt, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8F8',
                          edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def plot_pareto_with_feasibility(
    pareto_objectives: np.ndarray,
    algorithm: str,
    scenario: str,
    output_path: str,
    all_solutions: List[Dict] = None,
    selected_solution_idx: Optional[int] = None
) -> str:
    """
    Pareto plot with feasibility coloring.
    Shows ALL explored solutions with color coding:
        - Light blue: Feasible but dominated
        - Light red/orange: Infeasible (constraint violation)
        - Pareto front: colored markers + green line
        - Selected solution: red star

    This is the companion plot to plot_pareto_2d_v1 (which shows clean Pareto only).
    """
    if len(pareto_objectives) == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 9))
    is_nsga = 'NSGA' in algorithm.upper()
    pareto_color = COLORS['pareto_purple'] if is_nsga else COLORS['pareto_black']

    # ── 1. Background: ALL solutions with feasibility coloring ────────
    if all_solutions:
        feasible_pts = []
        infeasible_pts = []
        for s in all_solutions:
            pt = [s.get('Z4', 0), s.get('Z1', 0)]
            if s.get('feasible', False):
                feasible_pts.append(pt)
            else:
                infeasible_pts.append(pt)

        if infeasible_pts:
            inf_arr = np.array(infeasible_pts)
            # Sample if too many
            if len(inf_arr) > 600:
                idx = np.random.choice(len(inf_arr), 600, replace=False)
                inf_arr = inf_arr[idx]
            ax.scatter(inf_arr[:, 0], inf_arr[:, 1],
                       c='#FFCCCC', alpha=0.35, s=35,
                       edgecolors='#FF9999', linewidths=0.3, zorder=1,
                       label=f'Infeasible ({len(infeasible_pts)})')

        if feasible_pts:
            feas_arr = np.array(feasible_pts)
            if len(feas_arr) > 600:
                idx = np.random.choice(len(feas_arr), 600, replace=False)
                feas_arr = feas_arr[idx]
            ax.scatter(feas_arr[:, 0], feas_arr[:, 1],
                       c='#CCE5FF', alpha=0.4, s=40,
                       edgecolors='#99CCFF', linewidths=0.3, zorder=2,
                       label=f'Feasible Dominated ({len(feasible_pts)})')

    # ── 2. Pareto front ──────────────────────────────────────────────
    sorted_idx    = _sort_pareto_for_plot(pareto_objectives)
    sorted_pareto = pareto_objectives[sorted_idx]
    sel_in_orig   = _select_best_solution(pareto_objectives, selected_solution_idx)
    sel_in_sorted = int(np.where(sorted_idx == sel_in_orig)[0][0]) if sel_in_orig in sorted_idx else 0

    # Green connection line
    if len(sorted_pareto) > 1:
        ax.plot(sorted_pareto[:, 1], sorted_pareto[:, 0],
                c=COLORS['pareto_line'], linewidth=2.5, zorder=4,
                solid_capstyle='round')

    # Pareto points
    for i, (z1, z4) in enumerate(sorted_pareto):
        is_selected = (i == sel_in_sorted)
        if is_selected:
            ax.scatter(z4, z1, marker='*', s=550, c=COLORS['selected'],
                       edgecolors='#8B0000', linewidths=1.5, zorder=7)
        else:
            ax.scatter(z4, z1, marker='o', s=180, c=pareto_color,
                       edgecolors='white' if is_nsga else '#333333',
                       linewidths=1.5, zorder=5)

        # Labels: only selected + endpoints
        show_label = is_selected or i == 0 or i == len(sorted_pareto) - 1
        if show_label:
            offset_y = (sorted_pareto[:, 0].max() - sorted_pareto[:, 0].min()) * 0.025 + 0.3
            offset_y = max(offset_y, 0.5)
            value_text = f"Z₁={z1:.1f}%\nZ₄={z4:.0f}"
            txt_color = COLORS['selected'] if is_selected else '#333333'
            txt_weight = 'bold' if is_selected else 'normal'
            ax.annotate(value_text, xy=(z4, z1), xytext=(z4, z1 + offset_y),
                        fontsize=8 if is_selected else 7.5, fontweight=txt_weight,
                        color=txt_color, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor=COLORS['selected'] if is_selected else '#CCCCCC',
                                  alpha=0.85), zorder=8)

    # ── 3. Axes ──────────────────────────────────────────────────────
    ax.set_xlabel('Z₄: Capacity Deviation [Objective 2]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Z₁: Average Satisfaction (%) [Objective 1]', fontsize=13, fontweight='bold')
    algo_label = 'NSGA-III' if is_nsga else 'GA'
    ax.set_title(f"Pareto Front with Feasibility — {algo_label} ({scenario})",
                 fontsize=15, fontweight='bold', pad=15)

    # ── 4. Legend ────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFCCCC',
               markersize=9, markeredgecolor='#FF9999', label='Infeasible'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCE5FF',
               markersize=9, markeredgecolor='#99CCFF', label='Feasible (dominated)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=pareto_color,
               markersize=12, markeredgecolor='white' if is_nsga else '#333',
               label='Pareto Front'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['selected'],
               markersize=16, markeredgecolor='#8B0000', label='Selected Solution'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.9, edgecolor='#CCCCCC')

    # ── 5. Grid + stats ──────────────────────────────────────────────
    ax.grid(True, alpha=0.15, color='#999999', linestyle='-')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5); spine.set_color('#333333')
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)

    if all_solutions:
        n_feas = sum(1 for s in all_solutions if s.get('feasible', False))
        n_inf  = len(all_solutions) - n_feas
        sel_obj = pareto_objectives[sel_in_orig]
        stats_txt = (
            f"Pareto: {len(pareto_objectives)} | Feasible: {n_feas} | Infeasible: {n_inf}\n"
            f"★ Selected: Z₁={sel_obj[0]:.1f}%, Z₄={sel_obj[1]:.0f}"
        )
        ax.text(0.02, 0.02, stats_txt, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8F8',
                          edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def plot_convergence_v1(
    history: Dict[str, List],
    algorithm: str,
    scenario: str,
    output_path: str
) -> str:
    """Convergence plots — shows Z1 and Z4 evolution."""
    generations = history.get('generation', [])
    if not generations:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics = [
        ('best_Z1', 'Z₁: Avg. Satisfaction (%) [OBJ-MAX]', '#2E86AB', True),
        ('best_Z4', 'Z₄: Capacity Deviation [OBJ-MIN]',    '#F18F01', False),
    ]

    for idx, (key, title, color, is_max) in enumerate(metrics):
        ax     = axes[idx]
        values = history.get(key, [])
        if values:
            ax.plot(generations, values, c=color, linewidth=2)
            ax.fill_between(generations, values, alpha=0.15, color=color)
            ax.axhline(y=values[-1], color=color, linestyle='--', alpha=0.5)
            imp = values[-1] - values[0] if is_max else values[0] - values[-1]
            imp_str = f"+{imp:.2f}" if imp > 0 else f"{imp:.2f}"
            ax.text(0.98, 0.98,
                    f"Start: {values[0]:.4f}\nFinal: {values[-1]:.4f}\nChange: {imp_str}",
                    transform=ax.transAxes, fontsize=9, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Adaptive rate plot (optional)
    if 'p_shift' in history and len(history.get('p_shift', [])) > 0:
        # Third plot optional
        pass

    fig.suptitle(f'Convergence Analysis — {algorithm} ({scenario})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def plot_parallel_coordinates_v1(
    pareto_objectives: np.ndarray,
    constraint_values: List[Dict],
    algorithm: str,
    scenario: str,
    output_path: str
) -> str:
    """Parallel coordinates plot (Z1 and Z4 only)."""
    if len(pareto_objectives) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(pareto_objectives)

    data = np.zeros((n, 2))
    data[:, 0] = pareto_objectives[:, 0]
    data[:, 1] = pareto_objectives[:, 1]

    norm = data.copy()
    for i in range(2):
        lo, hi = norm[:, i].min(), norm[:, i].max()
        norm[:, i] = (norm[:, i] - lo) / (hi - lo) if hi > lo else 0.5

    # Z4: high value = BAD → invert
    norm[:, 1] = 1 - norm[:, 1]

    alpha = max(0.2, min(0.7, 10 / n))
    for row in norm:
        ax.plot([0, 1], row, c=COLORS['nsga3'], alpha=alpha, linewidth=1)

    best_idx = int(np.argmax(pareto_objectives[:, 0]))
    ax.plot([0, 1], norm[best_idx], c=COLORS['selected'], linewidth=3,
            label='Selected (max Z₁)', marker='*', markersize=12)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Z₁ Satisfaction\n(MAX)', 'Z₄ Cap. Deviation\n(MIN)'], fontsize=12)
    ax.set_ylabel('Normalized Value (Higher = Better)', fontsize=11)
    ax.set_title(f'Parallel Coordinates — {algorithm} ({scenario})\n{n} Pareto Solutions',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def plot_satisfaction_distribution(
    p_values: Dict[str, float],
    algorithm: str,
    scenario: str,
    output_path: str
) -> str:
    """Satisfaction distribution histogram and bar chart."""
    if not p_values:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    values = list(p_values.values())

    ax1.hist(values, bins=10, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax1.axvline(np.mean(values), color=COLORS['best'], linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(values)*100:.1f}%')
    ax1.set_xlabel('Satisfaction Rate', fontsize=11)
    ax1.set_ylabel('Staff Count', fontsize=11)
    ax1.set_title('Satisfaction Distribution', fontsize=12, fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    sorted_items  = sorted(p_values.items(), key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 0)
    sorted_labels = [i[0] for i in sorted_items]
    sorted_vals   = [i[1] * 100 for i in sorted_items]

    bar_colors = ['#27AE60' if v >= 80 else '#2ECC71' if v >= 60 else
                  '#F1C40F' if v >= 40 else '#E74C3C' if v > 0 else '#95A5A6'
                  for v in sorted_vals]

    ax2.bar(sorted_labels, sorted_vals, color=bar_colors, edgecolor='white')
    ax2.axhline(y=np.mean(values)*100, color=COLORS['best'], linestyle='--', linewidth=2)
    ax2.set_xlabel('Staff', fontsize=11)
    ax2.set_ylabel('Satisfaction (%)', fontsize=11)
    ax2.set_title('Satisfaction by Staff Member', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(handles=[
        mpatches.Patch(facecolor='#27AE60', label='≥80%'),
        mpatches.Patch(facecolor='#2ECC71', label='60-79%'),
        mpatches.Patch(facecolor='#F1C40F', label='40-59%'),
        mpatches.Patch(facecolor='#E74C3C', label='1-39%'),
        mpatches.Patch(facecolor='#95A5A6', label='0%'),
    ], loc='upper right', fontsize=8)

    fig.suptitle(f'Satisfaction Analysis — {algorithm} ({scenario})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def plot_algorithm_comparison_v1(
    ga_results: Dict[str, Any],
    nsga3_results: Dict[str, Any],
    scenario: str,
    output_path: str
) -> str:
    """GA vs NSGA-III comparison plot (Z1, Z4, runtime, solution count)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = ['Z₁ (%)', 'Z₄']
    x = np.arange(len(labels)); w = 0.35

    ga_vals = [ga_results.get('Z1', 0), ga_results.get('Z4', 0)]

    po = nsga3_results.get('pareto_objectives', [])
    if po and len(po) > 0:
        best_idx  = int(np.argmax([o[0] for o in po]))
        nsga_vals = [po[best_idx][0], po[best_idx][1]]
    else:
        nsga_vals = [0, 0]

    axes[0].bar(x - w/2, ga_vals,   w, label='GA',      color=COLORS['ga'])
    axes[0].bar(x + w/2, nsga_vals, w, label='NSGA-III', color=COLORS['nsga3'])
    axes[0].set_title('Objective Values', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    times = [ga_results.get('elapsed_time', 0), nsga3_results.get('elapsed_time', 0)]
    axes[1].bar(['GA', 'NSGA-III'], times, color=[COLORS['ga'], COLORS['nsga3']])
    axes[1].set_ylabel('Runtime (seconds)', fontsize=11)
    axes[1].set_title('Runtime', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(['GA\n(Single Solution)', 'NSGA-III\n(Pareto)'],
                [1, nsga3_results.get('pareto_size', 0)],
                color=[COLORS['ga'], COLORS['nsga3']])
    axes[2].set_ylabel('Solution Count', fontsize=11)
    axes[2].set_title('Solution Diversity', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Algorithm Comparison — {scenario}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def generate_all_visualizations_v1(
    pareto_objectives: np.ndarray,
    results: Dict[str, Any],
    scenario: str,
    algorithm: str,
    output_dir: str,
    all_solutions: List[Dict] = None,
    constraint_values: List[Dict] = None,
    p_values: Dict[str, float] = None,
    selected_solution_idx: Optional[int] = None
) -> List[str]:
    """Generate all visualizations (v6: Z1/Z4 plots only)."""
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    if len(pareto_objectives) > 0:
        pareto_path = os.path.join(output_dir, f"{algorithm}_{scenario}_pareto_Z1_Z4.png")
        plot_pareto_2d_v1(pareto_objectives, algorithm, scenario, pareto_path,
                          all_solutions, show_dominated=True,
                          selected_solution_idx=selected_solution_idx)
        output_files.append(pareto_path)

    if len(pareto_objectives) > 0:
        parallel_path = os.path.join(output_dir, f"{algorithm}_{scenario}_parallel.png")
        plot_parallel_coordinates_v1(pareto_objectives, constraint_values,
                                     algorithm, scenario, parallel_path)
        output_files.append(parallel_path)

    if 'history' in results:
        conv_path = os.path.join(output_dir, f"{algorithm}_{scenario}_convergence.png")
        plot_convergence_v1(results['history'], algorithm, scenario, conv_path)
        output_files.append(conv_path)

    if p_values:
        sat_path = os.path.join(output_dir, f"{algorithm}_{scenario}_satisfaction.png")
        plot_satisfaction_distribution(p_values, algorithm, scenario, sat_path)
        output_files.append(sat_path)

    return output_files


def _smooth_stepwise(generations, values, num_points=500):
    """
    Convert stepwise (staircase) data into smooth convex curves.
    Uses segment-midpoint approach: places control points at the center
    of each constant-value segment, then applies PCHIP interpolation.
    This produces genuinely smooth transitions instead of sharp steps.
    """
    import numpy as np

    gens = np.array(generations, dtype=float)
    vals = np.array(values, dtype=float)
    n = len(gens)

    if n < 3:
        return gens, vals

    # ── Extract constant segments ────────────────────────────────────
    segments = []
    seg_start = 0
    for i in range(1, n):
        if abs(vals[i] - vals[i-1]) > 1e-9:
            segments.append((seg_start, i - 1, vals[seg_start]))
            seg_start = i
    segments.append((seg_start, n - 1, vals[seg_start]))

    # ── Build control points at segment midpoints ────────────────────
    mid_gens = []
    mid_vals = []
    for start_idx, end_idx, val in segments:
        mid_idx = (start_idx + end_idx) / 2.0
        mid_gen = gens[int(start_idx)] + (gens[min(int(end_idx), n-1)] - gens[int(start_idx)]) / 2.0
        mid_gens.append(mid_gen)
        mid_vals.append(val)

    # Pin endpoints to exact start/end values
    mid_gens[0] = gens[0]
    mid_gens[-1] = gens[-1]

    mid_gens = np.array(mid_gens, dtype=float)
    mid_vals = np.array(mid_vals, dtype=float)

    # Remove duplicate x values
    unique_mask = np.concatenate(([True], np.diff(mid_gens) > 1e-9))
    mid_gens = mid_gens[unique_mask]
    mid_vals = mid_vals[unique_mask]

    # ── Smooth interpolation ──────────────────────────────────────────
    x_smooth = np.linspace(gens[0], gens[-1], num_points)

    if len(mid_gens) >= 4:
        try:
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(mid_gens, mid_vals)
            y_smooth = interp(x_smooth)
        except ImportError:
            degree = min(5, len(mid_gens) - 1)
            coeffs = np.polyfit(mid_gens, mid_vals, degree)
            y_smooth = np.polyval(coeffs, x_smooth)
    elif len(mid_gens) >= 2:
        y_smooth = np.interp(x_smooth, mid_gens, mid_vals)
    else:
        return gens, vals

    # Clamp to floor (minimum value in data) and ceiling
    floor = min(vals)
    y_smooth = np.clip(y_smooth, floor, 1.0)

    return x_smooth, y_smooth


def plot_mutation_rates(
    history: Dict[str, List],
    algorithm: str,
    scenario: str,
    output_path: str
) -> str:
    """
    Adaptive mutation operator probability evolution plot (smooth curves).
    Shows p_shift, p_swap, p_blockswap changes across generations
    with PCHIP interpolation for smooth, parabolic transitions.
    Separate plot for GA and NSGA-III.
    """
    import numpy as np

    generations = history.get('generation', [])
    if not generations:
        return None

    has_data = any(len(history.get(k, [])) > 0 for k in ('p_shift', 'p_swap', 'p_blockswap'))
    if not has_data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    NUM_SMOOTH = 500  # interpolation resolution

    # ── Left panel: Individual operator rates (smooth) ─────────────────
    ax1 = axes[0]
    op_config = [
        ('p_shift',     'Shift Mutation (p_shift)',       '#2E86AB', '-'),
        ('p_swap',      'Swap Mutation (p_swap)',         '#F18F01', '--'),
        ('p_blockswap', 'Block-Swap Mutation (p_blockswap)', '#C73E1D', '-.'),
    ]

    for key, label, color, ls in op_config:
        values = history.get(key, [])
        if values and len(values) == len(generations):
            x_sm, y_sm = _smooth_stepwise(generations, values, NUM_SMOOTH)
            ax1.plot(x_sm, y_sm, c=color, linewidth=2.2,
                     linestyle=ls, label=label, zorder=3)
            ax1.fill_between(x_sm, y_sm, alpha=0.08, color=color)
            # Start/end annotations (use original values for accuracy)
            ax1.annotate(f'{values[0]:.3f}',
                         xy=(generations[0], values[0]),
                         xytext=(generations[0] + len(generations)*0.02, values[0]),
                         fontsize=8, color=color, fontweight='bold')
            ax1.annotate(f'{values[-1]:.3f}',
                         xy=(generations[-1], values[-1]),
                         xytext=(generations[-1] - len(generations)*0.02, values[-1]),
                         fontsize=8, color=color, fontweight='bold', ha='right')

    ax1.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mutation Probability', fontsize=11, fontweight='bold')
    ax1.set_title('Operator Probability Evolution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2, color='#999999')
    ax1.set_ylim(bottom=0)
    ax1.set_axisbelow(True)

    # ── Right panel: Combined mutation pressure (smooth) ───────────────
    ax2 = axes[1]
    p_shift_vals     = history.get('p_shift', [])
    p_swap_vals      = history.get('p_swap', [])
    p_blockswap_vals = history.get('p_blockswap', [])

    if p_shift_vals and p_swap_vals and p_blockswap_vals:
        n = min(len(p_shift_vals), len(p_swap_vals), len(p_blockswap_vals), len(generations))
        gens_n = generations[:n]

        # Smooth each operator separately
        x_sm, shift_sm = _smooth_stepwise(gens_n, p_shift_vals[:n], NUM_SMOOTH)
        _,    swap_sm   = _smooth_stepwise(gens_n, p_swap_vals[:n], NUM_SMOOTH)
        _,    bs_sm     = _smooth_stepwise(gens_n, p_blockswap_vals[:n], NUM_SMOOTH)

        # Total mutation pressure (smooth)
        total_sm = 1.0 - (1.0 - shift_sm) * (1.0 - swap_sm) * (1.0 - bs_sm)

        ax2.plot(x_sm, total_sm, c='#333333', linewidth=2.5,
                 label='P(at least one mutation)', zorder=4)
        ax2.fill_between(x_sm, total_sm, alpha=0.12, color='#333333')

        # Stacked contribution (smooth)
        ax2.fill_between(x_sm, 0, shift_sm,
                         alpha=0.3, color='#2E86AB', label='Shift contribution')
        shift_swap_sm = shift_sm + swap_sm
        ax2.fill_between(x_sm, shift_sm, shift_swap_sm,
                         alpha=0.3, color='#F18F01', label='Swap contribution')
        all_sum_sm = shift_swap_sm + bs_sm
        ax2.fill_between(x_sm, shift_swap_sm, all_sum_sm,
                         alpha=0.3, color='#C73E1D', label='Block-Swap contribution')

        # Stats box (use original values for accuracy)
        total_pressure_raw = [
            1.0 - (1.0 - p_shift_vals[i]) * (1.0 - p_swap_vals[i]) * (1.0 - p_blockswap_vals[i])
            for i in range(n)
        ]
        stats_txt = (
            f"Start total: {total_pressure_raw[0]:.3f}\n"
            f"Final total: {total_pressure_raw[-1]:.3f}\n"
            f"Max total:   {max(total_pressure_raw):.3f}\n"
            f"Min total:   {min(total_pressure_raw):.3f}"
        )
        ax2.text(0.02, 0.98, stats_txt, transform=ax2.transAxes, fontsize=8.5,
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#CCCCCC', alpha=0.9))

    ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Combined Mutation Pressure', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.2, color='#999999')
    ax2.set_ylim(bottom=0)
    ax2.set_axisbelow(True)

    is_nsga = 'NSGA' in algorithm.upper()
    algo_label = 'NSGA-III' if is_nsga else 'GA'
    fig.suptitle(f'Adaptive Mutation Rates — {algo_label} ({scenario})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


# Backward compatibility stubs (silently return None if called)
def plot_constraint_analysis(*args, **kwargs):
    return None

def plot_pareto_with_constraints(*args, **kwargs):
    return None

def plot_pareto_pairwise_projections(*args, **kwargs):
    return []

def plot_pareto_pairwise_scatter(*args, **kwargs):
    return None


if __name__ == "__main__":
    print("Visualization Module v6 | Selected ★ | Z1/Z4 labels | Z2/Z3 plots removed")
