# -*- coding: utf-8 -*-
"""
VERSION 12 - Main Run Script
=====================================
CHANGES (v11 → v12):
    - Multi-complexity experiment: MEDIUM, HARD, EXTREME in single run
    - 18 total runs: 3 complexity × 3 scenarios × 2 TOL modes
    - Cross-complexity summary table for thesis comparison
    - Algorithm parameters unchanged from V11

KULLANIM:
    # Single scenario (TOL1 = ±1 flexibility ON)
    python run_v2.py -s 1

    # Single scenario, no flexibility (TOL0)
    python run_v2.py -s 1 --no-flexibility

    # Run all scenarios
    python run_v2.py -a

    # Full experiment: S1-S3 × (TOL0 + TOL1), --z2 and --z3 required
    python run_v2.py -e --z2 0.15 --z3 0.015
"""

import os
import sys
import json
import time
import math
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from data_loader import load_scenario
from ga_algorithm import GeneticAlgorithmV2
from nsga3_algorithm import NSGA3V2
from objectives import (
    evaluate_objectives_v2,
    print_objectives_summary_v2,
    get_constraint_thresholds,
    set_constraint_thresholds,
    calculate_individual_satisfaction,
    calculate_working_hours,
    check_Z2_constraint,
    check_Z3_constraint,
)
from constraints import set_capacity_flexibility, get_capacity_flexibility, FLEXIBILITY_MIN_REQ
from complexity_config import set_complexity, get_complexity, print_complexity_info, COMPLEXITY_LEVELS

try:
    from reports import generate_all_reports
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False
    print("⚠️ Reports module could not be loaded.")

try:
    from visualization import (
        plot_pareto_2d_v1,
        plot_pareto_with_feasibility,
        plot_parallel_coordinates_v1,
        plot_convergence_v1,
        plot_satisfaction_distribution,
        plot_algorithm_comparison_v1,
        generate_all_visualizations_v1,
        plot_mutation_rates,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization module could not be loaded.")


# ── Helper: Pareto solution closest to ideal point ─────────────────────────

def _pick_selected_solution(pareto_objectives, pareto_solutions, scenario_data):
    """
    Determines the 'selected solution' from the Pareto front.
    Criterion: minimum Euclidean distance to ideal point (max Z1, min Z4).
    Searches feasible solutions first; falls back to full front.
    Returns: (index_in_pareto, Z1, Z4, Z2, Z3)
    """
    if len(pareto_objectives) == 0:
        return -1, 0, 0, 0, 0

    z1_min = pareto_objectives[:, 0].min(); z1_max = pareto_objectives[:, 0].max()
    z4_min = pareto_objectives[:, 1].min(); z4_max = pareto_objectives[:, 1].max()
    best_idx = -1; best_dist = float('inf')

    for feasible_only in (True, False):
        for i, sol in enumerate(pareto_solutions):
            res = evaluate_objectives_v2(
                sol, scenario_data['preferences'], scenario_data['manager_requirements'],
                scenario_data['doctors'], scenario_data['days'], scenario_data['num_shifts']
            )
            _, const, const_sat, _, _, _ = res
            if feasible_only and not (const_sat[0] and const_sat[1]):
                continue
            z1n = (pareto_objectives[i, 0] - z1_min) / (z1_max - z1_min + 1e-9)
            z4n = (pareto_objectives[i, 1] - z4_min) / (z4_max - z4_min + 1e-9)
            d   = math.sqrt((1 - z1n) ** 2 + z4n ** 2)
            if d < best_dist:
                best_dist = d; best_idx = i
        if best_idx >= 0:
            break

    if best_idx < 0:
        best_idx = int(np.argmax(pareto_objectives[:, 0]))

    res = evaluate_objectives_v2(
        pareto_solutions[best_idx], scenario_data['preferences'],
        scenario_data['manager_requirements'], scenario_data['doctors'],
        scenario_data['days'], scenario_data['num_shifts']
    )
    _, const, _, _, _, _ = res
    return best_idx, float(pareto_objectives[best_idx, 0]), float(pareto_objectives[best_idx, 1]), float(const[0]), float(const[1])


# ── Main comparison function ─────────────────────────────────────────────

def run_comparison(
    scenario_num: int = 1,
    complexity: str = 'HARD',
    custom_z2: float = None,
    custom_z3: float = None,
    ga_params: Dict = None,
    nsga_params: Dict = None,
    verbose: bool = True,
    generate_plots: bool = True,
    capacity_flexibility: bool = True,   # True=TOL1, False=TOL0
    run_label: str = None
) -> Optional[Dict]:
    """
    GA vs NSGA-III comparison for the specified scenario.

    capacity_flexibility=True  → TOL1: ±1 flexibility for shifts with required≥3
    capacity_flexibility=False → TOL0: exact match for all shifts
    """
    # Set flexibility
    set_capacity_flexibility(capacity_flexibility)

    # Complexity
    if custom_z2 is not None and custom_z3 is not None:
        complexity_info = set_complexity('CUSTOM', z2=custom_z2, z3=custom_z3)
    else:
        complexity_info = set_complexity(complexity)

    z2_thresh, z3_thresh = get_constraint_thresholds()
    flex_str = f"±1 (required≥{FLEXIBILITY_MIN_REQ})" if capacity_flexibility else "NONE (exact match)"

    print("\n" + "█" * 75)
    print("█" + f"  VERSION 12 — SCENARIO {scenario_num} — {complexity_info['level']}  ".center(73) + "█")
    print("█" * 75)
    print(f"   Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   OBJECTIVES : Z₁ (Satisfaction MAX), Z₄ (Capacity Deviation MIN)")
    print(f"   CONSTRAINTS: Z₂ ≤ {z2_thresh:.4f},  Z₃ ≤ {z3_thresh:.6f}  (inequalizlik)")
    print(f"   FLEXIBILITY: {flex_str}")
    if run_label:
        print(f"   LABEL      : {run_label}")
    print("█" * 75)

    # Load data
    print(f"\n📂 Loading scenario {scenario_num}...")
    try:
        script_dir    = os.path.dirname(os.path.abspath(__file__))
        excel_path    = os.path.join(script_dir, 'Database.xlsx')
        scenario_name = f'S{scenario_num}'
        scenario_data = load_scenario(excel_path, scenario_name)
        print(f"   ✓ Staff: {len(scenario_data['doctors'])} | Days: {scenario_data['days']} | Vardiya: {scenario_data['num_shifts']}")
    except Exception as e:
        print(f"   ✗ Hata: {e}")
        import traceback; traceback.print_exc()
        return None

    results = {
        'scenario': scenario_num,
        'version': 7,
        'complexity': complexity_info,
        'timestamp': datetime.now().isoformat(),
        'run_config': {
            'capacity_flexibility': capacity_flexibility,
            'flexibility_min_req':  FLEXIBILITY_MIN_REQ,
            'z2_threshold': z2_thresh,
            'z3_threshold': z3_thresh,
            'run_label': run_label,
        },
        'ga': None,
        'nsga3': None
    }

    # ────────────────────────────────────────────────────────────────────────
    # GA
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*75}\n🧬 GA VERSION 12 Running...\n{'─'*75}")
    default_ga = {
        'population_size': 300,
        'generations': 500,
        'crossover_rate': 0.85,
        'p_shift': 0.15,
        'p_swap':  0.12,
        'p_blockswap': 0.08,
        'elitism_rate': 0.05,
        'w1': 0.7,
        'w4': 0.3,
        'seed': 42,
    }
    if ga_params:
        default_ga.update(ga_params)

    ga             = GeneticAlgorithmV2(scenario_data=scenario_data, **default_ga)
    ga_solution, ga_objectives, ga_results = ga.run(verbose=verbose)
    results['ga']  = ga_results

    # GA detailed summary
    ga_eval = evaluate_objectives_v2(
        ga_solution, scenario_data['preferences'], scenario_data['manager_requirements'],
        scenario_data['doctors'], scenario_data['days'], scenario_data['num_shifts']
    )
    ga_obj, ga_const, ga_const_sat, _, ga_p_values, ga_details = ga_eval
    print_objectives_summary_v2(ga_obj, ga_const, ga_const_sat, ga_p_values, ga_details)

    # ────────────────────────────────────────────────────────────────────────
    # NSGA-III
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*75}\n🔬 NSGA-III VERSION 12 Running...\n{'─'*75}")
    default_nsga = {
        'population_size': 300,
        'generations': 500,
        'crossover_rate': 0.90,
        'p_shift': 0.15,
        'p_swap':  0.12,
        'p_blockswap': 0.08,
        'n_divisions': 50,
        'seed': 42,
    }
    if nsga_params:
        default_nsga.update(nsga_params)

    nsga3 = NSGA3V2(scenario_data=scenario_data, **default_nsga)
    pareto_solutions, pareto_objectives_arr, nsga3_results = nsga3.run(verbose=verbose)
    results['nsga3'] = nsga3_results

    # Selected solution
    best_idx, nsga_z1, nsga_z4, nsga_z2, nsga_z3 = _pick_selected_solution(
        pareto_objectives_arr, pareto_solutions, scenario_data)

    # NSGA-III Pareto tablosu
    if len(pareto_solutions) > 0:
        print(f"\n📊 NSGA-III Pareto Front ({len(pareto_solutions)} solutions) — sorted by distance to ideal point:")
        print(f"   {'No':>3} {'Z₁ (%)':>10} {'Z₄':>8} {'Z₂':>10} {'Z₃':>12} {'Status':>10}")
        print(f"   {'─'*3}─{'─'*10}─{'─'*8}─{'─'*10}─{'─'*12}─{'─'*10}")

        z1_min = pareto_objectives_arr[:, 0].min(); z1_max = pareto_objectives_arr[:, 0].max()
        z4_min = pareto_objectives_arr[:, 1].min(); z4_max = pareto_objectives_arr[:, 1].max()
        dists  = [
            (i, math.sqrt(
                (1 - (pareto_objectives_arr[i,0] - z1_min)/(z1_max - z1_min + 1e-9))**2 +
                ((pareto_objectives_arr[i,1] - z4_min)/(z4_max - z4_min + 1e-9))**2
            ))
            for i in range(len(pareto_solutions))
        ]
        for rank, (i, _) in enumerate(sorted(dists, key=lambda x: x[1])[:10]):
            res = evaluate_objectives_v2(
                pareto_solutions[i], scenario_data['preferences'], scenario_data['manager_requirements'],
                scenario_data['doctors'], scenario_data['days'], scenario_data['num_shifts']
            )
            ov, cv, csat, _, _, _ = res
            status  = "✓ Feasible" if (csat[0] and csat[1]) else "✗ Infeas."
            marker  = " ★" if i == best_idx else ""
            print(f"   {rank+1:3d} {ov[0]:10.2f} {ov[1]:8.0f} {cv[0]:10.4f} {cv[1]:12.6f} {status:>10}{marker}")
        if len(pareto_solutions) > 10:
            print(f"   ... and {len(pareto_solutions)-10} more solutions")
        print(f"   ★ = Selected solution (closest to ideal point)")

    # ────────────────────────────────────────────────────────────────────────
    # COMPARISON SUMMARY
    # ────────────────────────────────────────────────────────────────────────
    ga_z1_v, ga_z4_v   = ga_results['best_objectives']
    ga_z2_v, ga_z3_v   = ga_results['best_constraints']
    ga_feas_str = "✓" if (ga_const_sat[0] and ga_const_sat[1]) else "✗"
    nsga_feas_str = "✓" if best_idx >= 0 else "✗"

    print(f"\n{'█'*75}")
    print("█" + " COMPARISON SUMMARY ".center(73, "═") + "█")
    print(f"{'█'*75}")
    print(f"   {'Metrik':<32} {'GA':>18} {'NSGA-III ★':>18}")
    print(f"   {'─'*32}─{'─'*18}─{'─'*18}")
    print(f"   {'Z₁ Satisfaction (%)':<32} {ga_z1_v:>17.2f}% {nsga_z1:>17.2f}%")
    print(f"   {'Z₄ Capacity Deviation':<32} {ga_z4_v:>18.0f} {nsga_z4:>18.0f}")
    print(f"   {'Z₂ (≤{:.4f})'.format(z2_thresh):<32} {ga_z2_v:>18.4f} {nsga_z2:>18.4f}")
    print(f"   {'Z₃ (≤{:.6f})'.format(z3_thresh):<32} {ga_z3_v:>18.6f} {nsga_z3:>18.6f}")
    print(f"   {'Constraint Status':<32} {ga_feas_str:>18} {nsga_feas_str:>18}")
    print(f"   {'Runtime (s)':<32} {ga_results['elapsed_time']:>18.2f} {nsga3_results['elapsed_time']:>18.2f}")
    print(f"   {'Pareto Solutions':<32} {'1':>18} {len(pareto_solutions):>18}")
    if len(pareto_objectives_arr) > 0:
        print(f"\n   Pareto range: Z₁=[{pareto_objectives_arr[:,0].min():.1f}% — {pareto_objectives_arr[:,0].max():.1f}%]  "
              f"Z₄=[{pareto_objectives_arr[:,1].min():.0f} — {pareto_objectives_arr[:,1].max():.0f}]")
    print(f"{'█'*75}\n")

    # ────────────────────────────────────────────────────────────────────────
    # JSON save
    # ────────────────────────────────────────────────────────────────────────
    output_dir  = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    label_suffix = f"_{run_label}" if run_label else ""
    output_file  = os.path.join(output_dir, f"v12_results_S{scenario_num}_{complexity_info['level']}{label_suffix}.json")

    ga_working_hours = calculate_working_hours(
        ga_solution, scenario_data['doctors'], scenario_data['days'],
        scenario_data['num_shifts'], scenario_data['shift_durations'], scenario_data['contract_hours'])

    nsga3_p_values = {}; nsga3_working_hours = {}
    if len(pareto_solutions) > 0 and best_idx >= 0:
        best_nsga3_sol = pareto_solutions[best_idx]
        nsga3_p_values = calculate_individual_satisfaction(
            best_nsga3_sol, scenario_data['preferences'], scenario_data['doctors'],
            scenario_data['days'], scenario_data['num_shifts'])
        nsga3_working_hours = calculate_working_hours(
            best_nsga3_sol, scenario_data['doctors'], scenario_data['days'],
            scenario_data['num_shifts'], scenario_data['shift_durations'], scenario_data['contract_hours'])

    best_z1_idx = int(np.argmax(pareto_objectives_arr[:, 0])) if len(pareto_objectives_arr) > 0 else -1
    best_z4_idx = int(np.argmin(pareto_objectives_arr[:, 1])) if len(pareto_objectives_arr) > 0 else -1

    save_results = {
        'scenario': scenario_num, 'version': 7,
        'complexity': complexity_info,
        'timestamp': results['timestamp'],
        'run_config': results['run_config'],
        'constraint_thresholds': {'Z2': z2_thresh, 'Z3': z3_thresh},
        'scenario_info': {
            'personnel_count': len(scenario_data['doctors']),
            'days': scenario_data['days'],
            'num_shifts': scenario_data['num_shifts'],
            'shift_durations': scenario_data['shift_durations'],
            'contract_hours': scenario_data['contract_hours']
        },
        'ga': {
            'best_objectives': list(ga_results['best_objectives']),
            'best_constraints': list(ga_results['best_constraints']),
            'constraints_satisfied': list(ga_const_sat),
            'elapsed_time': ga_results['elapsed_time'],
            'generations': ga_results['generations'],
            'p_shift_final':     ga_results.get('p_shift_final', 0),
            'p_swap_final':      ga_results.get('p_swap_final', 0),
            'p_blockswap_final': ga_results.get('p_blockswap_final', 0),
            'individual_satisfaction': {k: round(v*100,2) for k,v in ga_p_values.items()},
            'working_hours': ga_working_hours,
        },
        'nsga3': {
            'pareto_size': nsga3_results['pareto_size'],
            'pareto_objectives': nsga3_results['pareto_objectives'],
            'constraint_values': nsga3_results['constraint_values'],
            'elapsed_time': nsga3_results['elapsed_time'],
            'generations': nsga3_results['generations'],
            'n_reference_points': nsga3_results['n_reference_points'],
            'pareto_ranges': {
                'Z1_min': float(pareto_objectives_arr[:,0].min()) if len(pareto_objectives_arr)>0 else 0,
                'Z1_max': float(pareto_objectives_arr[:,0].max()) if len(pareto_objectives_arr)>0 else 0,
                'Z4_min': float(pareto_objectives_arr[:,1].min()) if len(pareto_objectives_arr)>0 else 0,
                'Z4_max': float(pareto_objectives_arr[:,1].max()) if len(pareto_objectives_arr)>0 else 0,
            },
            'selected_solution': {
                'method': 'ideal_point_distance',
                'Z1': nsga_z1, 'Z4': nsga_z4, 'Z2': nsga_z2, 'Z3': nsga_z3,
                'index': int(best_idx) if best_idx >= 0 else -1,
            },
            'extreme_solutions': {
                'max_Z1': {'Z1': float(pareto_objectives_arr[best_z1_idx,0]), 'Z4': float(pareto_objectives_arr[best_z1_idx,1]), 'index': best_z1_idx} if best_z1_idx >= 0 else {},
                'min_Z4': {'Z1': float(pareto_objectives_arr[best_z4_idx,0]), 'Z4': float(pareto_objectives_arr[best_z4_idx,1]), 'index': best_z4_idx} if best_z4_idx >= 0 else {},
            },
            'p_shift_final':     nsga3_results.get('p_shift_final', 0),
            'p_swap_final':      nsga3_results.get('p_swap_final', 0),
            'p_blockswap_final': nsga3_results.get('p_blockswap_final', 0),
            'individual_satisfaction': {k: round(v*100,2) for k,v in nsga3_p_values.items()},
            'working_hours': nsga3_working_hours,
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON kaydedildi: {output_file}")

    # ────────────────────────────────────────────────────────────────────────
    # PLOTS
    # ────────────────────────────────────────────────────────────────────────
    if generate_plots and VISUALIZATION_AVAILABLE:
        print(f"\n📊 Generating plots...")
        plots_dir = os.path.join(output_dir, 'plots', f'S{scenario_num}_{complexity_info["level"]}{label_suffix}')
        os.makedirs(plots_dir, exist_ok=True)
        sname = f'S{scenario_num}'

        try:
            if len(pareto_objectives_arr) > 0 and best_idx >= 0:
                # Pareto plot — selected solution ★
                pp = os.path.join(plots_dir, f"NSGA3_{sname}_pareto_Z1_Z4.png")
                plot_pareto_2d_v1(pareto_objectives_arr, 'NSGA-III', sname, pp,
                                  nsga3_results.get('all_solutions', []),
                                  show_dominated=True,
                                  selected_solution_idx=best_idx)
                print(f"   ✓ Pareto: {pp}")

                # Pareto plot with feasibility coloring
                ppf = os.path.join(plots_dir, f"NSGA3_{sname}_pareto_feasibility.png")
                plot_pareto_with_feasibility(pareto_objectives_arr, 'NSGA-III', sname, ppf,
                                              nsga3_results.get('all_solutions', []),
                                              selected_solution_idx=best_idx)
                print(f"   ✓ Pareto (feasibility): {ppf}")

                # Paralel koordinat
                par = os.path.join(plots_dir, f"NSGA3_{sname}_parallel.png")
                plot_parallel_coordinates_v1(pareto_objectives_arr,
                                             nsga3_results.get('constraint_values', []),
                                             'NSGA-III', sname, par)
                print(f"   ✓ Paralel koordinat: {par}")
            elif len(pareto_objectives_arr) == 0:
                print(f"   ⚠️ No Pareto solutions found — Pareto plots skipped")

            # GA memnuniyet
            if ga_p_values:
                sp = os.path.join(plots_dir, f"GA_{sname}_satisfaction.png")
                plot_satisfaction_distribution(ga_p_values, 'GA', sname, sp)
                print(f"   ✓ Satisfaction distribution: {sp}")

            # Comparison
            cp = os.path.join(plots_dir, f"Comparison_{sname}.png")
            plot_algorithm_comparison_v1(
                {'Z1': ga_z1_v, 'Z4': ga_z4_v, 'elapsed_time': ga_results['elapsed_time']},
                {'pareto_objectives': nsga3_results['pareto_objectives'],
                 'elapsed_time': nsga3_results['elapsed_time'],
                 'pareto_size': nsga3_results['pareto_size']},
                sname, cp)
            print(f"   ✓ Comparison: {cp}")

            # Convergence
            if ga_results.get('history'):
                gcp = os.path.join(plots_dir, f"GA_{sname}_convergence.png")
                plot_convergence_v1(ga_results['history'], 'GA', sname, gcp)
                print(f"   ✓ GA convergence: {gcp}")

            if nsga3_results.get('history'):
                ncp = os.path.join(plots_dir, f"NSGA3_{sname}_convergence.png")
                plot_convergence_v1(nsga3_results['history'], 'NSGA-III', sname, ncp)
                print(f"   ✓ NSGA-III convergence: {ncp}")

            # Mutation Rate Evolution (GA)
            if ga_results.get('history') and ga_results['history'].get('p_shift'):
                gmp = os.path.join(plots_dir, f"GA_{sname}_mutation_rates.png")
                plot_mutation_rates(ga_results['history'], 'GA', sname, gmp)
                print(f"   ✓ GA mutation rates: {gmp}")

            # Mutation Rate Evolution (NSGA-III)
            if nsga3_results.get('history') and nsga3_results['history'].get('p_shift'):
                nmp = os.path.join(plots_dir, f"NSGA3_{sname}_mutation_rates.png")
                plot_mutation_rates(nsga3_results['history'], 'NSGA-III', sname, nmp)
                print(f"   ✓ NSGA-III mutation rates: {nmp}")

            print(f"   📁 Grafikler: {plots_dir}")

        except Exception as e:
            print(f"   ⚠️ Plot error: {e}")
            import traceback; traceback.print_exc()

    # ────────────────────────────────────────────────────────────────────────
    # EXCEL RAPORLARI
    # ────────────────────────────────────────────────────────────────────────
    if REPORTS_AVAILABLE:
        reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        try:
            ga_rr = {
                'Z1': ga_z1_v, 'Z4': ga_z4_v, 'Z2': ga_z2_v, 'Z3': ga_z3_v,
                'min_satisfaction': min(ga_p_values.values()) if ga_p_values else 0,
                'max_satisfaction': max(ga_p_values.values()) if ga_p_values else 0,
                'over_staffed':  ga_details.get('over_staffed', 0),
                'under_staffed': ga_details.get('under_staffed', 0),
                'best_fitness':  ga_results.get('best_fitness', 0),
                'weights':       {'w1': default_ga.get('w1', 0.7), 'w4': default_ga.get('w4', 0.3)},
                'population_size': 120,
                'generations':   ga_results['generations'],
                'elapsed_time':  ga_results['elapsed_time'],
            }
            ga_rp = os.path.join(reports_dir, f"GA_S{scenario_num}_{complexity_info['level']}{label_suffix}_report.xlsx")
            generate_all_reports(ga_solution, scenario_data, ga_rr, 'GA', ga_rp,
                                history=ga_results.get('history'))
            print(f"   ✓ GA raporu: {ga_rp}")

            if len(pareto_solutions) > 0 and best_idx >= 0:
                nsga_rr = {
                    'Z1': nsga_z1, 'Z4': nsga_z4, 'Z2': nsga_z2, 'Z3': nsga_z3,
                    'min_satisfaction': min(nsga3_p_values.values()) if nsga3_p_values else 0,
                    'max_satisfaction': max(nsga3_p_values.values()) if nsga3_p_values else 0,
                    'pareto_size': nsga3_results['pareto_size'],
                    'n_reference_points': nsga3_results['n_reference_points'],
                    'population_size': 120,
                    'generations': nsga3_results['generations'],
                    'elapsed_time': nsga3_results['elapsed_time'],
                }
                nsga_rp = os.path.join(reports_dir, f"NSGA3_S{scenario_num}_{complexity_info['level']}{label_suffix}_report.xlsx")
                generate_all_reports(pareto_solutions[best_idx], scenario_data, nsga_rr, 'NSGA-III', nsga_rp,
                                    history=nsga3_results.get('history'))
                print(f"   ✓ NSGA-III raporu: {nsga_rp}")
        except Exception as e:
            print(f"   ⚠️ Report error: {e}"); import traceback; traceback.print_exc()

    results.update({'ga': ga_results, 'nsga3': nsga3_results,
                    'best_nsga3_idx': best_idx,
                    'final_z1': {'ga': ga_z1_v, 'nsga': nsga_z1},
                    'final_z4': {'ga': ga_z4_v, 'nsga': nsga_z4}})
    return results


# ── All scenarios ────────────────────────────────────────────────────────────

def run_all_scenarios(complexity='HARD', verbose=True, generate_plots=True, capacity_flexibility=True):
    all_results = {}
    for s in [1, 2, 3]:
        result = run_comparison(s, complexity=complexity, verbose=verbose,
                                generate_plots=generate_plots,
                                capacity_flexibility=capacity_flexibility)
        all_results[s] = result
    return all_results


# ── Full experiment: 3 scenarios × (TOL0 + TOL1) ─────────────────────────────────────

def run_full_experiment(z2_threshold, z3_threshold, verbose=True, generate_plots=True, level_name=None):
    """
    6 runs: S1-S3 × (TOL0=exact, TOL1=±1).
    level_name: if provided, included in file names to prevent overwriting (e.g., 'MEDIUM', 'HARD', 'EXTREME').
    """
    print("\n" + "█"*80)
    level_display = f" [{level_name}]" if level_name else ""
    print("█" + f"   FULL EXPERIMENT{level_display} — ALL SCENARIOS × (TOL0 + TOL1)   ".center(78) + "█")
    print("█"*80)
    print(f"   Z₂ ≤ {z2_threshold:.4f},  Z₃ ≤ {z3_threshold:.6f}")
    print(f"   TOL0: Exact match  |  TOL1: ±1 flexibility (required≥{FLEXIBILITY_MIN_REQ})")
    print("█"*80 + "\n")

    all_results = {}
    tol_configs = [(False, 'TOL0_exact'), (True, 'TOL1_flex')]

    for scenario_num in [1, 2, 3]:
        for flex, tol_name in tol_configs:
            label = f"S{scenario_num}_{tol_name}"
            flex_desc = f"±1 (required≥{FLEXIBILITY_MIN_REQ})" if flex else "Exact Match"
            print("\n" + "▓"*80)
            print(f"▓  SENARYO {scenario_num} — {tol_name}  ({flex_desc})".ljust(79) + "▓")
            print("▓"*80)
            # Include level_name in run_label for unique file names
            if level_name:
                full_label = f"{level_name}_{tol_name}"
            else:
                full_label = tol_name
            try:
                result = run_comparison(
                    scenario_num=scenario_num,
                    complexity='CUSTOM',
                    custom_z2=z2_threshold,
                    custom_z3=z3_threshold,
                    verbose=verbose,
                    generate_plots=generate_plots,
                    capacity_flexibility=flex,
                    run_label=full_label
                )
                all_results[label] = result
            except Exception as e:
                print(f"   ❌ Hata: {e}")
                import traceback; traceback.print_exc()

    # Summary table
    print("\n\n" + "█"*100)
    print("█" + " FULL EXPERIMENT RESULT SUMMARY ".center(98, "═") + "█")
    print("█"*100)
    print(f"   Z₂ ≤ {z2_threshold:.4f},  Z₃ ≤ {z3_threshold:.6f}")
    print(f"\n   {'Run':<22} │ {'GA Z₁%':>7} {'GA Z₄':>6} {'GA F.':>5} │ {'NSGA★ Z₁%':>9} {'NSGA★ Z₄':>8} {'ParP':>5} │ {'Time(s)':>7}")
    print(f"   {'─'*22}─┼─{'─'*20}─┼─{'─'*24}─┼─{'─'*7}")

    for scenario_num in [1, 2, 3]:
        for flex, tol_name in tol_configs:
            label = f"S{scenario_num}_{tol_name}"
            r = all_results.get(label)
            if not r: continue
            ga   = r.get('ga', {})
            nsga = r.get('nsga3', {})
            ga_z1  = ga.get('best_objectives', [0,0])[0]
            ga_z4  = ga.get('best_objectives', [0,0])[1]
            ga_cs  = ga.get('constraints_satisfied', (False, False))
            ga_f   = "✓" if (ga_cs[0] and ga_cs[1]) else "✗"
            po     = nsga.get('pareto_objectives', [])
            bi     = r.get('best_nsga3_idx', -1)
            if po and bi >= 0:
                nsga_z1 = po[bi][0]; nsga_z4 = po[bi][1]
            elif po:
                nsga_z1 = max(p[0] for p in po); nsga_z4 = min(p[1] for p in po)
            else:
                nsga_z1 = nsga_z4 = 0
            pareto_n  = nsga.get('pareto_size', 0)
            total_t   = ga.get('elapsed_time', 0) + nsga.get('elapsed_time', 0)
            flex_mark = "±1" if flex else "±0"
            row_name  = f"S{scenario_num} ({flex_mark})"
            print(f"   {row_name:<22} │ {ga_z1:>6.1f}% {ga_z4:>6.0f} {ga_f:>5} │ {nsga_z1:>8.1f}% {nsga_z4:>8.0f} {pareto_n:>5} │ {total_t:>6.1f}s")
        if scenario_num < 3:
            print(f"   {'·'*22}─┼─{'·'*20}─┼─{'·'*24}─┼─{'·'*7}")

    print(f"\n   NSGA★ = Selected solution from Pareto front (closest to ideal point)")
    print(f"   ParP  = Number of Pareto front solutions")
    print(f"{'█'*100}\n")
    return all_results


# ── Multi-Complexity Experiment ──────────────────────────────────────────────

THESIS_COMPLEXITY_LEVELS = {
    'MEDIUM':  {'z2': 0.25,  'z3': 0.025,  'desc': 'Dengeli kısıt yapısı'},
    'HARD':    {'z2': 0.15,  'z3': 0.015,  'desc': 'Sıkı kısıtlar — önerilen test seviyesi'},
    'EXTREME': {'z2': 0.05,  'z3': 0.003,  'desc': 'Neredeyse mükemmel eşitlik zorunluluğu'},
}


def run_multi_complexity_experiment(verbose=True, generate_plots=True):
    """
    18 runs: 3 complexity levels × 3 scenarios × 2 TOL modes.
    MEDIUM → HARD → EXTREME sequential execution with cross-complexity summary.
    """
    print("\n" + "█"*100)
    print("█" + "   MULTI-COMPLEXITY EXPERIMENT — MEDIUM × HARD × EXTREME   ".center(98) + "█")
    print("█"*100)
    print(f"   3 complexity levels × 3 scenarios × 2 TOL modes = 18 runs")
    for lvl, cfg in THESIS_COMPLEXITY_LEVELS.items():
        print(f"   {lvl:<10}: Z₂ ≤ {cfg['z2']:.4f},  Z₃ ≤ {cfg['z3']:.6f}  — {cfg['desc']}")
    print("█"*100 + "\n")

    grand_results = {}

    for level_name, level_cfg in THESIS_COMPLEXITY_LEVELS.items():
        z2t = level_cfg['z2']
        z3t = level_cfg['z3']

        print("\n" + "╔" + "═"*98 + "╗")
        print("║" + f"  COMPLEXITY: {level_name} — Z₂ ≤ {z2t:.4f}, Z₃ ≤ {z3t:.6f}  ".center(98) + "║")
        print("╚" + "═"*98 + "╝")

        level_results = run_full_experiment(
            z2_threshold=z2t,
            z3_threshold=z3t,
            verbose=verbose,
            generate_plots=generate_plots,
            level_name=level_name
        )
        grand_results[level_name] = level_results

    # ── Cross-Complexity Summary Table ─────────────────────────────────────
    print("\n\n" + "█"*120)
    print("█" + " CROSS-COMPLEXITY COMPARISON — ALL RESULTS ".center(118, "═") + "█")
    print("█"*120)

    header = (f"   {'Level':<10} {'Run':<12} │ "
              f"{'GA Z₁%':>7} {'GA Z₄':>6} {'GA F.':>5} │ "
              f"{'NSGA★ Z₁%':>9} {'NSGA★ Z₄':>8} {'ParP':>5} │ "
              f"{'Time(s)':>8}")
    print(header)
    print(f"   {'─'*10} {'─'*12}─┼─{'─'*20}─┼─{'─'*24}─┼─{'─'*8}")

    tol_configs = [(False, 'TOL0_exact', '±0'), (True, 'TOL1_flex', '±1')]

    for level_name in THESIS_COMPLEXITY_LEVELS:
        level_results = grand_results.get(level_name, {})
        for scenario_num in [1, 2, 3]:
            for flex, tol_name, flex_mark in tol_configs:
                label = f"S{scenario_num}_{tol_name}"
                r = level_results.get(label)
                if not r:
                    continue
                ga   = r.get('ga', {})
                nsga = r.get('nsga3', {})
                ga_z1  = ga.get('best_objectives', [0,0])[0]
                ga_z4  = ga.get('best_objectives', [0,0])[1]
                ga_cs  = ga.get('constraints_satisfied', (False, False))
                ga_f   = "✓" if (ga_cs[0] and ga_cs[1]) else "✗"
                po     = nsga.get('pareto_objectives', [])
                bi     = r.get('best_nsga3_idx', -1)
                if po and bi >= 0:
                    nsga_z1 = po[bi][0]; nsga_z4 = po[bi][1]
                elif po:
                    nsga_z1 = max(p[0] for p in po); nsga_z4 = min(p[1] for p in po)
                else:
                    nsga_z1 = nsga_z4 = 0
                pareto_n  = nsga.get('pareto_size', 0)
                total_t   = ga.get('elapsed_time', 0) + nsga.get('elapsed_time', 0)
                row_name  = f"S{scenario_num}({flex_mark})"
                print(f"   {level_name:<10} {row_name:<12} │ "
                      f"{ga_z1:>6.1f}% {ga_z4:>6.0f} {ga_f:>5} │ "
                      f"{nsga_z1:>8.1f}% {nsga_z4:>8.0f} {pareto_n:>5} │ "
                      f"{total_t:>7.1f}s")
        print(f"   {'·'*10} {'·'*12}─┼─{'·'*20}─┼─{'·'*24}─┼─{'·'*8}")

    # ── Per-level summary stats ────────────────────────────────────────────
    print(f"\n   {'Level':<10} │ {'GA Feasible':>12} │ {'Avg Pareto':>11} │ {'Avg NSGA Z₁':>12}")
    print(f"   {'─'*10}─┼─{'─'*12}─┼─{'─'*11}─┼─{'─'*12}")

    for level_name in THESIS_COMPLEXITY_LEVELS:
        level_results = grand_results.get(level_name, {})
        ga_feas_count = 0
        total_runs = 0
        pareto_sum = 0
        nsga_z1_sum = 0
        nsga_count = 0

        for scenario_num in [1, 2, 3]:
            for flex, tol_name, flex_mark in tol_configs:
                label = f"S{scenario_num}_{tol_name}"
                r = level_results.get(label)
                if not r:
                    continue
                total_runs += 1
                ga = r.get('ga', {})
                nsga = r.get('nsga3', {})
                ga_cs = ga.get('constraints_satisfied', (False, False))
                if ga_cs[0] and ga_cs[1]:
                    ga_feas_count += 1
                pareto_n = nsga.get('pareto_size', 0)
                pareto_sum += pareto_n
                po = nsga.get('pareto_objectives', [])
                bi = r.get('best_nsga3_idx', -1)
                if po and bi >= 0:
                    nsga_z1_sum += po[bi][0]
                    nsga_count += 1

        avg_pareto = pareto_sum / max(total_runs, 1)
        avg_nsga_z1 = nsga_z1_sum / max(nsga_count, 1)
        print(f"   {level_name:<10} │ {ga_feas_count:>5}/{total_runs:<6} │ {avg_pareto:>10.1f} │ {avg_nsga_z1:>11.1f}%")

    print(f"\n{'█'*120}\n")
    return grand_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='VERSION 12 — GA and NSGA-III')
    parser.add_argument('--scenario', '-s', type=int, default=1, choices=[1,2,3])
    parser.add_argument('--all', '-a', action='store_true', help='All scenarios')
    parser.add_argument('--complexity', '-c', type=str, default='HARD',
                        choices=['EASY','MEDIUM','HARD','VERY_HARD','EXTREME'])
    parser.add_argument('--z2', type=float, default=None)
    parser.add_argument('--z3', type=float, default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--experiment', '-e', action='store_true',
                        help='Full experiment: S1-S3 × (TOL0+TOL1). --z2 and --z3 required.')
    parser.add_argument('--multi', '-m', action='store_true',
                        help='Multi-complexity experiment: MEDIUM×HARD×EXTREME × S1-S3 × TOL0+TOL1 (18 runs)')
    parser.add_argument('--no-flexibility', action='store_true',
                        help='Disable capacity flexibility (TOL0: exact match)')
    parser.add_argument('--label', type=str, default=None)

    args = parser.parse_args()
    print_complexity_info()

    if args.multi:
        run_multi_complexity_experiment(
            verbose=not args.quiet,
            generate_plots=not args.no_plots
        )
    elif args.experiment:
        if args.z2 is None or args.z3 is None:
            print("❌ --z2 and --z3 are required for --experiment!")
            print("   Example: python run_v2.py -e --z2 0.15 --z3 0.015")
            sys.exit(1)
        run_full_experiment(args.z2, args.z3, verbose=not args.quiet,
                            generate_plots=not args.no_plots)
    elif args.all:
        run_all_scenarios(complexity=args.complexity, verbose=not args.quiet,
                          generate_plots=not args.no_plots,
                          capacity_flexibility=not args.no_flexibility)
    else:
        run_comparison(
            scenario_num=args.scenario,
            complexity=args.complexity,
            custom_z2=args.z2,
            custom_z3=args.z3,
            verbose=not args.quiet,
            generate_plots=not args.no_plots,
            capacity_flexibility=not args.no_flexibility,
            run_label=args.label
        )
