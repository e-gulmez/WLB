# -*- coding: utf-8 -*-
"""
Objectives Module - VERSION 12
================================
CHANGES (v5 → v6):
    - Equality constraint mode REMOVED (inequality only)
    - Z4 uses raw deviation; flexibility managed in constraints.py

OBJECTIVE FUNCTIONS:
    Z1: Ortalama Memnuniyet (MAXIMIZE)
    Z4: Capacity Deviation    (MINIMIZE)

CONSTRAINTS (inequality ≤):
    Z2: Fairness Gap     ≤ Z2_THRESHOLD
    Z3: Workload Variance ≤ Z3_THRESHOLD
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# ── Thresholds ──────────────────────────────────────────────────────────
Z2_THRESHOLD = 0.20
Z3_THRESHOLD = 0.003


def set_constraint_thresholds(z2_threshold=None, z3_threshold=None):
    global Z2_THRESHOLD, Z3_THRESHOLD
    if z2_threshold is not None:
        Z2_THRESHOLD = z2_threshold
    if z3_threshold is not None:
        Z3_THRESHOLD = z3_threshold


def get_constraint_thresholds():
    return Z2_THRESHOLD, Z3_THRESHOLD


def set_z4_tolerance(tolerance: int):
    """Backward compatibility — no-op in v6."""
    pass


# ── Bireysel hesaplamalar ────────────────────────────────────────────────────

def calculate_individual_satisfaction(assignments, preferences, doctors, days, num_shifts):
    total_req = defaultdict(int)
    satisfied  = defaultdict(int)
    for day in range(days):
        day_key = day + 1
        for shift in range(num_shifts):
            requesting = set(preferences.get(day_key, [[]] * num_shifts)[shift])
            assigned   = set(assignments[day][shift]) if day < len(assignments) and shift < len(assignments[day]) else set()
            for p in requesting:
                if p and p in doctors:
                    total_req[p] += 1
                    if p in assigned:
                        satisfied[p] += 1
    return {doc: (satisfied[doc] / total_req[doc]) if total_req[doc] > 0 else 0.0 for doc in doctors}


def calculate_working_hours(assignments, doctors, days, num_shifts, shift_durations, contract_hours):
    info = {}
    for doc in doctors:
        n_shifts = 0; worked = 0
        for day in range(days):
            if day < len(assignments):
                for shift in range(num_shifts):
                    if shift < len(assignments[day]) and doc in assignments[day][shift]:
                        n_shifts += 1
                        worked   += shift_durations[shift] if shift < len(shift_durations) else 6
        weeks    = days / 7
        contract = contract_hours.get(doc, 40) * weeks
        info[doc] = {
            'assigned_shifts':  n_shifts,
            'worked_hours':     worked,
            'contract_hours':   contract,
            'fulfillment_rate': round(worked / contract * 100, 2) if contract > 0 else 0.0
        }
    return info


def calculate_Z1_average_satisfaction(p_values):
    if not p_values: return 0.0
    return sum(p_values.values()) / len(p_values) * 100.0


def calculate_Z2_fairness_gap(p_values):
    if not p_values or len(p_values) < 2: return 0.0
    v = list(p_values.values())
    return max(v) - min(v)


def calculate_Z3_workload_variance(p_values):
    if not p_values: return 0.0
    v = list(p_values.values()); mean = sum(v) / len(v)
    return sum((p - mean) ** 2 for p in v) / len(v)


def calculate_Z3_standard_deviation(p_values):
    return np.sqrt(calculate_Z3_workload_variance(p_values))


def calculate_Z4_capacity_deviation(assignments, manager_requirements, days, num_shifts):
    """
    Z4 = Σ |assigned - required|  (raw, no tolerance)
    Flexibility (±1 required≥3) applied as feasibility in constraints.py.
    """
    total = 0.0; over = 0; under = 0
    for day in range(days):
        day_key = day + 1
        for shift in range(num_shifts):
            required = manager_requirements.get(day_key, [0] * num_shifts)[shift]
            assigned = len(set(assignments[day][shift])) if day < len(assignments) and shift < len(assignments[day]) else 0
            diff = assigned - required
            if diff < 0:   under += abs(diff); total += abs(diff)
            elif diff > 0: over  += diff;      total += diff
    return total, over, under


# ── Constraint check ────────────────────────────────────────────────────────────

def check_Z2_constraint(z2_value):
    EPS = 1e-6
    if z2_value <= Z2_THRESHOLD + EPS: return True, 0.0
    return False, z2_value - Z2_THRESHOLD


def check_Z3_constraint(z3_value):
    EPS = 1e-6
    if z3_value <= Z3_THRESHOLD + EPS: return True, 0.0
    return False, z3_value - Z3_THRESHOLD


def calculate_constraint_penalty(z2_value, z3_value, penalty_weight=1000.0):
    _, z2v = check_Z2_constraint(z2_value)
    _, z3v = check_Z3_constraint(z3_value)
    penalty = 0.0
    if z2v > 0: penalty += penalty_weight * z2v ** 2
    if z3v > 0: penalty += penalty_weight * z3v ** 2
    return penalty


# ── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_objectives_v2(assignments, preferences, manager_requirements, doctors, days, num_shifts):
    p_values = calculate_individual_satisfaction(assignments, preferences, doctors, days, num_shifts)
    Z1 = calculate_Z1_average_satisfaction(p_values)
    Z4, over, under = calculate_Z4_capacity_deviation(assignments, manager_requirements, days, num_shifts)
    Z2 = calculate_Z2_fairness_gap(p_values)
    Z3 = calculate_Z3_workload_variance(p_values)
    Z3_std = calculate_Z3_standard_deviation(p_values)
    Z2_ok, Z2_viol = check_Z2_constraint(Z2)
    Z3_ok, Z3_viol = check_Z3_constraint(Z3)
    vals = list(p_values.values())
    details = {
        'over_staffed': over, 'under_staffed': under,
        'min_satisfaction': min(vals) if vals else 0,
        'max_satisfaction': max(vals) if vals else 0,
        'std_satisfaction': Z3_std, 'variance': Z3, 'fairness_gap': Z2,
        'z2_threshold': Z2_THRESHOLD, 'z3_threshold': Z3_THRESHOLD,
        'z2_satisfied': Z2_ok, 'z3_satisfied': Z3_ok,
        'z2_violation': Z2_viol, 'z3_violation': Z3_viol,
        'all_constraints_satisfied': Z2_ok and Z3_ok,
    }
    return (Z1, float(Z4)), (Z2, Z3), (Z2_ok, Z3_ok), (Z2_viol, Z3_viol), p_values, details


def evaluate_all_objectives(assignments, preferences, manager_requirements, doctors, days, num_shifts):
    result = evaluate_objectives_v2(assignments, preferences, manager_requirements, doctors, days, num_shifts)
    (Z1, Z4), (Z2, Z3), _, _, p_values, details = result
    return (Z1, Z2, Z3, Z4), p_values, details


def print_objectives_summary_v2(objectives, constraints, constraint_satisfied, p_values, details):
    Z1, Z4 = objectives; Z2, Z3 = constraints; Z2_ok, Z3_ok = constraint_satisfied
    print(f"\n{'='*65}\n[*] VERSION 12: OBJECTIVE FUNCTIONS AND CONSTRAINTS\n{'='*65}")
    print(f"   Z1 Satisfaction: {Z1:.2f}%  |  Z4 Capacity Deviation: {Z4:.1f} (raw)")
    print(f"   Z2: {Z2:.4f} ≤ {details['z2_threshold']:.4f} {'✓' if Z2_ok else '✗'}  |  "
          f"Z3: {Z3:.6f} ≤ {details['z3_threshold']:.6f} {'✓' if Z3_ok else '✗'}")
    print(f"   Durum: {'✓ ALL CONSTRAINTS SATISFIED' if details['all_constraints_satisfied'] else '✗ CONSTRAINT VIOLATION'}")
    print("=" * 65)


evaluate_objectives_v1      = evaluate_objectives_v2
print_objectives_summary_v1 = print_objectives_summary_v2

if __name__ == "__main__":
    print(f"Objectives v6 | Z2≤{Z2_THRESHOLD} | Z3≤{Z3_THRESHOLD} | Inequality Only")
