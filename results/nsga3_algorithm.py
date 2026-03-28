# -*- coding: utf-8 -*-
"""
NSGA-III Algorithm — VERSION 12.1
================================================
Standard NSGA-III (Deb & Jain, 2014) with problem-specific adaptations.

V12.1 FIX: CDP Pareto tiebreaker for equal-cv infeasible solutions.
    When both solutions are infeasible with identical cv, Pareto dominance
    is now applied as tiebreaker. Previously, equal cv returned False
    (non-comparable), causing degenerate Pareto fronts with dominated points.

STANDARD COMPONENTS (from literature):
    - Das-Dennis reference point generation
    - Constrained Domination Principle (CDP)
    - Non-dominated sorting with constraint handling
    - Crowding distance for last-front selection
    - Binary tournament selection with CDP

PROBLEM-SPECIFIC ADAPTATIONS:
    - Constraint model: Z2 ≤ threshold, Z3 ≤ threshold
    - TOL0/TOL1 capacity flexibility
    - Shift, Swap, Block-Swap mutation operators (domain-specific)
    - Adaptive mutation rates (success-based, per operator)
    - Preference-aware individual creation

REMOVED (non-standard, caused diversity collapse loop in V6):
    - Diversity collapse detection + immigrant injection
    - Z4/Z1 elitist protection
"""

import time
import random
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Any

from objectives import (
    evaluate_objectives_v2,
    get_constraint_thresholds,
    check_Z2_constraint,
    check_Z3_constraint,
)
from constraints import is_valid_solution, _flexibility_bounds
from operators import (
    ensure_structure,
    get_available_doctors,
    crossover_day_slice,
    crossover_uniform,
    mutate_shift,
    mutate_swap,
    mutate_block_swap,
)


# ── Standard NSGA-III Helper Functions ────────────────────────────────────

def generate_reference_points(n_obj, n_divisions):
    """Das-Dennis reference point generation."""
    def recurse(n_obj, left, total, current, result):
        if n_obj == 1:
            result.append(current + [left / total])
            return
        for i in range(left + 1):
            recurse(n_obj - 1, left - i, total, current + [i / total], result)
    result = []
    recurse(n_obj, n_divisions, n_divisions, [], result)
    return np.array(result)


def dominates(obj1, obj2, directions):
    """Standard Pareto dominance check."""
    better_in_any = False
    for i, d in enumerate(directions):
        if d == 'max':
            if obj1[i] < obj2[i]: return False
            if obj1[i] > obj2[i]: better_in_any = True
        else:
            if obj1[i] > obj2[i]: return False
            if obj1[i] < obj2[i]: better_in_any = True
    return better_in_any


def constrained_dominates(obj1, cv1, feas1, obj2, cv2, feas2, directions):
    """
    Constrained Domination Principle (CDP) — Deb 2002, with Pareto tiebreaker.
    1. Feasible beats infeasible
    2. Both feasible → normal Pareto dominance
    3. Both infeasible → lower cv wins; equal cv → Pareto dominance tiebreaker
    """
    if feas1 and not feas2: return True
    if not feas1 and feas2: return False
    if feas1 and feas2:     return dominates(obj1, obj2, directions)
    # Both infeasible: compare cv first, then Pareto tiebreaker
    if cv1 < cv2: return True
    if cv1 > cv2: return False
    return dominates(obj1, obj2, directions)  # same cv → Pareto decides


def constrained_non_dominated_sort(objectives, violations, feasibilities, directions):
    """Non-dominated sorting with CDP."""
    n = len(objectives)
    if n == 0:
        return []

    dom_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if constrained_dominates(objectives[i], violations[i], feasibilities[i],
                                     objectives[j], violations[j], feasibilities[j], directions):
                dominated_by[i].append(j)
                dom_count[j] += 1
            elif constrained_dominates(objectives[j], violations[j], feasibilities[j],
                                       objectives[i], violations[i], feasibilities[i], directions):
                dominated_by[j].append(i)
                dom_count[i] += 1

    fronts = []
    current = [i for i in range(n) if dom_count[i] == 0]
    if not current:
        return [list(range(n))]

    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt

    return fronts


def crowding_distance(objectives, front_indices):
    """Standard crowding distance calculation."""
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    front_obj = objectives[front_indices]

    for m in range(objectives.shape[1]):
        sorted_idx = np.argsort(front_obj[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = front_obj[sorted_idx[-1], m] - front_obj[sorted_idx[0], m]
        if obj_range > 0:
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    front_obj[sorted_idx[i + 1], m] - front_obj[sorted_idx[i - 1], m]
                ) / obj_range

    return distances


# ── Main NSGA-III Class ───────────────────────────────────────────────────

class NSGA3V2:
    """
    Standard NSGA-III with problem-specific constraint model.

    Objectives: Z1 (Satisfaction, MAX), Z4 (Capacity Deviation, MIN)
    Constraints: Z2 ≤ threshold (inequality), Z3 ≤ threshold (inequality)
    """

    def __init__(
        self,
        scenario_data,
        population_size: int = 300,
        generations:     int = 500,
        crossover_rate:  float = 0.90,
        # Independent dice: separate probability per operator (adaptive)
        p_shift:         float = 0.15,
        p_swap:          float = 0.12,
        p_blockswap:     float = 0.08,
        n_divisions:     int = 50,
        seed:            int = None,
        verbose_interval: int = 25,
        adaptive_interval: int = 20,
        **kwargs  # Accept and ignore legacy params
    ):
        # Problem data
        self.data              = scenario_data
        self.scenario          = scenario_data['scenario']
        self.days              = scenario_data['days']
        self.num_shifts        = scenario_data['num_shifts']
        self.shift_durations   = scenario_data['shift_durations']
        self.doctors           = scenario_data['doctors']
        self.contract_hours    = scenario_data['contract_hours']
        self.preferences       = scenario_data['preferences']
        self.manager_requirements = scenario_data['manager_requirements']

        # Algorithm parameters
        self.population_size = population_size
        self.generations     = generations
        self.crossover_rate  = crossover_rate
        self.p_shift         = p_shift
        self.p_swap          = p_swap
        self.p_blockswap     = p_blockswap
        self.n_divisions     = n_divisions
        self.directions      = ['max', 'min']
        self.verbose_interval = verbose_interval

        # Adaptive mutation config
        self.adaptive_interval = adaptive_interval
        self.P_MIN_SHIFT  = 0.08;  self.P_MAX_SHIFT  = 0.30
        self.P_MIN_SWAP   = 0.08;  self.P_MAX_SWAP   = 0.25
        self.P_MIN_BS     = 0.05;  self.P_MAX_BS     = 0.15
        self.ADAPT_DELTA  = 0.02
        self._shift_success = 0; self._shift_total = 0
        self._swap_success  = 0; self._swap_total  = 0
        self._bs_success    = 0; self._bs_total    = 0

        # Reference points
        self.reference_points = generate_reference_points(2, n_divisions)
        self.n_ref = len(self.reference_points)

        # Constraint thresholds
        self.z2_threshold, self.z3_threshold = get_constraint_thresholds()

        # History tracking
        self.history = {
            'generation': [], 'pareto_size': [],
            'best_Z1': [], 'best_Z4': [], 'best_Z2': [], 'best_Z3': [],
            'feasible_count': [], 'hypervolume': [],
            'p_shift': [], 'p_swap': [], 'p_blockswap': [],
        }
        self.all_solutions = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, individual):
        """
        Returns: (obj_array, const_array, constraint_satisfied, total_cv, is_valid, details)
        total_cv: sum of normalized constraint violations (0 = feasible)
        """
        is_valid, _, _ = is_valid_solution(
            individual, self.shift_durations, self.contract_hours,
            self.manager_requirements, self.preferences,
            self.doctors, self.days, self.num_shifts
        )
        result = evaluate_objectives_v2(
            individual, self.preferences, self.manager_requirements,
            self.doctors, self.days, self.num_shifts
        )
        objectives, constraints, constraint_satisfied, constraint_violations, p_values, details = result
        z2_viol, z3_viol = constraint_violations
        total_cv = z2_viol + z3_viol
        if not is_valid:
            total_cv += 1000

        is_feasible = is_valid and constraint_satisfied[0] and constraint_satisfied[1]

        return (np.array(objectives), np.array(constraints),
                constraint_satisfied, total_cv, is_valid, is_feasible, details)

    # ── Individual Creation ───────────────────────────────────────────

    def create_individual(self):
        """Random individual with flexibility bounds for Z4 diversity."""
        assignments = [[[] for _ in range(self.num_shifts)] for _ in range(self.days)]
        MAX_DAILY = 12
        n_weeks = (self.days + 6) // 7
        week_hrs = [{d: 0 for d in self.doctors} for _ in range(n_weeks)]

        for day in range(self.days):
            day_key = day + 1
            week_idx = day // 7
            daily = {d: 0 for d in self.doctors}

            for shift in range(self.num_shifts):
                required = self.manager_requirements.get(day_key, [0] * self.num_shifts)[shift]
                dur = self.shift_durations[shift] if shift < len(self.shift_durations) else 0
                req_list = self.preferences.get(day_key, [[]] * self.num_shifts)[shift]
                req_count = len([p for p in req_list if p])

                lower, upper = _flexibility_bounds(required, req_count)
                target = random.randint(lower, upper)

                requesting = [d for d in req_list
                              if d and d in self.doctors
                              and daily[d] + dur <= MAX_DAILY
                              and week_hrs[week_idx].get(d, 0) + dur <= self.contract_hours.get(d, 40)]
                others = [d for d in self.doctors
                          if d not in requesting
                          and daily.get(d, 0) + dur <= MAX_DAILY
                          and week_hrs[week_idx].get(d, 0) + dur <= self.contract_hours.get(d, 40)]
                random.shuffle(others)

                if len(requesting) >= target:
                    assigned = random.sample(requesting, target)
                else:
                    assigned = requesting[:]
                    need = target - len(assigned)
                    if others and need > 0:
                        assigned += random.sample(others, min(need, len(others)))

                for d in assigned:
                    daily[d] += dur
                    week_hrs[week_idx][d] = week_hrs[week_idx].get(d, 0) + dur
                assignments[day][shift] = assigned
        return assignments

    def create_z4_optimal_individual(self):
        """Z4-optimal: assigns exactly required (Z4≈0, diversity seed)."""
        assignments = [[[] for _ in range(self.num_shifts)] for _ in range(self.days)]
        MAX_DAILY = 12
        n_weeks = (self.days + 6) // 7
        week_hrs = [{d: 0 for d in self.doctors} for _ in range(n_weeks)]

        for day in range(self.days):
            day_key = day + 1
            week_idx = day // 7
            daily = {d: 0 for d in self.doctors}

            for shift in range(self.num_shifts):
                required = self.manager_requirements.get(day_key, [0] * self.num_shifts)[shift]
                dur = self.shift_durations[shift] if shift < len(self.shift_durations) else 0
                requesting = [d for d in self.preferences.get(day_key, [[]] * self.num_shifts)[shift]
                              if d and d in self.doctors
                              and daily[d] + dur <= MAX_DAILY
                              and week_hrs[week_idx].get(d, 0) + dur <= self.contract_hours.get(d, 40)]
                others = [d for d in self.doctors
                          if d not in requesting
                          and daily.get(d, 0) + dur <= MAX_DAILY
                          and week_hrs[week_idx].get(d, 0) + dur <= self.contract_hours.get(d, 40)]
                random.shuffle(others)

                assigned = requesting[:required]
                if len(assigned) < required:
                    assigned += others[:required - len(assigned)]

                for d in assigned:
                    daily[d] += dur
                    week_hrs[week_idx][d] = week_hrs[week_idx].get(d, 0) + dur
                assignments[day][shift] = assigned
        return assignments

    def initialize_population(self):
        """Mixed initialization: 20% Z4-optimal + 80% random with flexibility."""
        n_z4 = max(1, int(self.population_size * 0.2))
        pop = [self.create_z4_optimal_individual() for _ in range(n_z4)]
        pop += [self.create_individual() for _ in range(self.population_size - n_z4)]
        return pop

    # ── Selection ─────────────────────────────────────────────────────

    def binary_tournament(self, population, objectives, ranks, distances,
                          feasibilities, violations):
        """Binary tournament with CDP."""
        i, j = random.sample(range(len(population)), 2)

        if feasibilities[i] and not feasibilities[j]:
            return deepcopy(population[i])
        elif feasibilities[j] and not feasibilities[i]:
            return deepcopy(population[j])
        elif not feasibilities[i] and not feasibilities[j]:
            return deepcopy(population[i] if violations[i] < violations[j] else population[j])
        else:
            # Both feasible: rank first, then crowding distance
            if ranks[i] < ranks[j]:
                return deepcopy(population[i])
            elif ranks[j] < ranks[i]:
                return deepcopy(population[j])
            else:
                return deepcopy(population[i] if distances[i] > distances[j] else population[j])

    # ── Pareto Front Extraction ───────────────────────────────────────

    def get_pareto_front(self, population, objectives, feasibilities, violations, valid_mask):
        """Extract Pareto front: prioritize feasible solutions."""
        # Try feasible + valid first
        fv_mask = feasibilities & valid_mask
        indices = np.where(fv_mask)[0]

        # Fallback to valid only
        if len(indices) == 0:
            indices = np.where(valid_mask)[0]
        if len(indices) == 0:
            return [], np.array([])

        fronts = constrained_non_dominated_sort(
            objectives[indices], violations[indices],
            feasibilities[indices], self.directions)

        if not fronts or not fronts[0]:
            return [], np.array([])

        pareto_idx = [indices[i] for i in fronts[0]]

        # Deduplicate by (Z1 rounded to 0.1, Z4 integer)
        seen = set()
        unique_idx = []
        for i in pareto_idx:
            key = (round(float(objectives[i, 0]), 1), round(float(objectives[i, 1]), 0))
            if key not in seen:
                seen.add(key)
                unique_idx.append(i)

        return [population[i] for i in unique_idx], objectives[unique_idx]

    # ── Adaptive Mutation Rate Update ──────────────────────────────

    def _adaptive_update_rates(self):
        """
        Update mutation rates based on success ratios.
        Called every adaptive_interval generations.
        """
        def _update(p, success, total, p_min, p_max):
            if total == 0:
                return p
            ratio = success / total
            if ratio > 0.2:
                p = min(p_max, p + self.ADAPT_DELTA)
            elif ratio < 0.05:
                p = max(p_min, p - self.ADAPT_DELTA)
            return p

        self.p_shift = _update(self.p_shift, self._shift_success, self._shift_total,
                               self.P_MIN_SHIFT, self.P_MAX_SHIFT)
        self.p_swap = _update(self.p_swap, self._swap_success, self._swap_total,
                              self.P_MIN_SWAP, self.P_MAX_SWAP)
        self.p_blockswap = _update(self.p_blockswap, self._bs_success, self._bs_total,
                                   self.P_MIN_BS, self.P_MAX_BS)

        # Reset counters
        self._shift_success = 0; self._shift_total = 0
        self._swap_success  = 0; self._swap_total  = 0
        self._bs_success    = 0; self._bs_total    = 0

    # ── Main Evolution Loop ───────────────────────────────────────────

    def run(self, verbose=True):
        start = time.time()
        z2t, z3t = get_constraint_thresholds()

        if verbose:
            from constraints import CAPACITY_FLEXIBILITY, FLEXIBILITY_MIN_REQ
            flex_str = f"±1 (required≥{FLEXIBILITY_MIN_REQ})" if CAPACITY_FLEXIBILITY else "None (exact)"
            print(f"\n{'='*65}")
            print(f"🔬 NSGA-III VERSION 12 — Scenario {self.scenario}")
            print(f"{'='*65}")
            print(f"   OBJECTIVES: Z₁ (MAX), Z₄ (MIN)")
            print(f"   CONSTRAINTS: Z₂ ≤ {z2t:.4f}, Z₃ ≤ {z3t:.6f}")
            print(f"   FLEXIBILITY: {flex_str}")
            print(f"   Pop: {self.population_size} | Gen: {self.generations} | Ref: {self.n_ref}")
            print(f"   p_shift={self.p_shift:.2f} | p_swap={self.p_swap:.2f} | p_bs={self.p_blockswap:.2f} (adaptive)")
            print(f"{'='*65}")

        # Initialize
        population = self.initialize_population()
        n = self.population_size

        # Evaluate initial population
        objectives    = np.zeros((n, 2))
        constraints   = np.zeros((n, 2))
        violations    = np.zeros(n)
        valid_mask    = np.zeros(n, dtype=bool)
        feasibilities = np.zeros(n, dtype=bool)

        for i, ind in enumerate(population):
            obj, const, _, cv, valid, feas, _ = self.evaluate(ind)
            objectives[i] = obj
            constraints[i] = const
            violations[i] = cv
            valid_mask[i] = valid
            feasibilities[i] = feas

        # ── Main loop ─────────────────────────────────────────────────
        for gen in range(self.generations):

            # ── Compute ranks and distances for selection ──────────
            valid_idx = np.where(valid_mask)[0]
            ranks = np.full(n, 999)
            distances = np.zeros(n)

            if len(valid_idx) > 0:
                fronts = constrained_non_dominated_sort(
                    objectives[valid_idx], violations[valid_idx],
                    feasibilities[valid_idx], self.directions)

                for rank, front in enumerate(fronts):
                    front_global = [valid_idx[i] for i in front]
                    for idx in front_global:
                        ranks[idx] = rank
                    cd = crowding_distance(objectives, front_global)
                    for i, idx in enumerate(front_global):
                        distances[idx] = cd[i]

            # ── Generate offspring ─────────────────────────────────
            offspring = []
            offspring_op_info = []  # track which operators were applied
            while len(offspring) < n:
                p1 = self.binary_tournament(population, objectives, ranks,
                                            distances, feasibilities, violations)
                p2 = self.binary_tournament(population, objectives, ranks,
                                            distances, feasibilities, violations)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = (crossover_day_slice if random.random() < 0.5
                             else crossover_uniform)(p1, p2, self.days, self.num_shifts)
                else:
                    child = deepcopy(p1)

                # Mutation: independent dice per operator (adaptive rates)
                applied_shift = applied_swap = applied_bs = False
                if random.random() < self.p_shift:
                    child = mutate_shift(
                        child, self.preferences, self.manager_requirements,
                        self.shift_durations, self.contract_hours,
                        self.doctors, self.days, self.num_shifts)
                    applied_shift = True
                if random.random() < self.p_swap:
                    child = mutate_swap(
                        child, self.preferences, self.shift_durations,
                        self.contract_hours, self.doctors,
                        self.days, self.num_shifts)
                    applied_swap = True
                if random.random() < self.p_blockswap:
                    child = mutate_block_swap(
                        child, self.preferences, self.shift_durations,
                        self.contract_hours, self.doctors,
                        self.days, self.num_shifts)
                    applied_bs = True

                offspring.append(child)
                offspring_op_info.append((applied_shift, applied_swap, applied_bs))

            # ── Evaluate offspring ─────────────────────────────────
            off_obj  = np.zeros((n, 2))
            off_con  = np.zeros((n, 2))
            off_viol = np.zeros(n)
            off_val  = np.zeros(n, dtype=bool)
            off_feas = np.zeros(n, dtype=bool)

            for i, ind in enumerate(offspring):
                obj, const, _, cv, valid, feas, _ = self.evaluate(ind)
                off_obj[i] = obj
                off_con[i] = const
                off_viol[i] = cv
                off_val[i] = valid
                off_feas[i] = feas

            # ── Combined population (μ + λ) ────────────────────────
            combined_pop  = population + offspring
            combined_obj  = np.vstack([objectives, off_obj])
            combined_con  = np.vstack([constraints, off_con])
            combined_viol = np.concatenate([violations, off_viol])
            combined_val  = np.concatenate([valid_mask, off_val])
            combined_feas = np.concatenate([feasibilities, off_feas])

            # ── Environmental selection via CDP + crowding ──────────
            c_valid = np.where(combined_val)[0]
            selected = []

            if len(c_valid) > 0:
                fronts = constrained_non_dominated_sort(
                    combined_obj[c_valid], combined_viol[c_valid],
                    combined_feas[c_valid], self.directions)

                for front in fronts:
                    front_global = [c_valid[i] for i in front]

                    if len(selected) + len(front_global) <= n:
                        selected.extend(front_global)
                    else:
                        remaining = n - len(selected)
                        cd = crowding_distance(combined_obj, front_global)
                        sorted_by_cd = sorted(range(len(front_global)),
                                              key=lambda i: cd[i], reverse=True)
                        selected.extend([front_global[i] for i in sorted_by_cd[:remaining]])
                        break

            # Fill if not enough valid solutions
            while len(selected) < n:
                if len(c_valid) > 0:
                    selected.append(random.choice(c_valid.tolist()))
                else:
                    selected.append(random.randint(0, len(combined_pop) - 1))

            # ── Update population ──────────────────────────────────
            population    = [combined_pop[i] for i in selected]
            objectives    = combined_obj[selected]
            constraints   = combined_con[selected]
            violations    = combined_viol[selected]
            valid_mask    = combined_val[selected]
            feasibilities = combined_feas[selected]

            # ── Track operator success (offspring that survived selection) ──
            survived_offspring = set(i - n for i in selected if i >= n)
            for oi in survived_offspring:
                if oi < len(offspring_op_info):
                    a_sh, a_sw, a_bs = offspring_op_info[oi]
                    if a_sh: self._shift_success += 1
                    if a_sw: self._swap_success += 1
                    if a_bs: self._bs_success += 1
            for a_sh, a_sw, a_bs in offspring_op_info:
                if a_sh: self._shift_total += 1
                if a_sw: self._swap_total += 1
                if a_bs: self._bs_total += 1

            # ── Adaptive mutation rate update ─────────────────────
            if (gen + 1) % self.adaptive_interval == 0:
                self._adaptive_update_rates()

            # ── Statistics ─────────────────────────────────────────
            pareto_sol, pareto_obj = self.get_pareto_front(
                population, objectives, feasibilities, violations, valid_mask)
            pareto_size = len(pareto_sol)
            feasible_count = int(np.sum(feasibilities))

            fi = np.where(feasibilities)[0] if feasible_count > 0 else np.where(valid_mask)[0]
            if len(fi) > 0:
                best_z1 = float(objectives[fi, 0].max())
                best_z4 = float(objectives[fi, 1].min())
                bz1_idx = fi[int(np.argmax(objectives[fi, 0]))]
                best_z2 = float(constraints[bz1_idx, 0])
                best_z3 = float(constraints[bz1_idx, 1])
            else:
                best_z1, best_z4, best_z2, best_z3 = 0, 100, 1, 1

            # History
            self.history['generation'].append(gen)
            self.history['pareto_size'].append(pareto_size)
            self.history['best_Z1'].append(best_z1)
            self.history['best_Z4'].append(best_z4)
            self.history['best_Z2'].append(best_z2)
            self.history['best_Z3'].append(best_z3)
            self.history['feasible_count'].append(feasible_count)
            self.history['hypervolume'].append(0)
            self.history['p_shift'].append(self.p_shift)
            self.history['p_swap'].append(self.p_swap)
            self.history['p_blockswap'].append(self.p_blockswap)

            # All solutions (for scatter plot)
            for i in range(len(objectives)):
                self.all_solutions.append({
                    'generation': gen, 'Z1': float(objectives[i, 0]),
                    'Z4': float(objectives[i, 1]),
                    'Z2': float(constraints[i, 0]), 'Z3': float(constraints[i, 1]),
                    'feasible': bool(feasibilities[i])
                })

            # Log
            if verbose and (gen % self.verbose_interval == 0 or gen == self.generations - 1):
                z2_ok, _ = check_Z2_constraint(best_z2)
                z3_ok, _ = check_Z3_constraint(best_z3)
                status = "✓" if (z2_ok and z3_ok) else "✗"
                print(f"   Nesil {gen:3d}: Pareto={pareto_size:2d} "
                      f"maxZ₁={best_z1:.1f}% minZ₄={best_z4:.0f} "
                      f"Z₂={best_z2:.3f} Z₃={best_z3:.4f} {status} "
                      f"Feasible={feasible_count}/{n} "
                      f"p_s={self.p_shift:.3f} p_sw={self.p_swap:.3f} p_bs={self.p_blockswap:.3f}")

        elapsed = time.time() - start

        # ── Final Pareto front ─────────────────────────────────────
        pareto_solutions, pareto_objectives = self.get_pareto_front(
            population, objectives, feasibilities, violations, valid_mask)

        pareto_constraints = []
        for sol in pareto_solutions:
            _, const, _, _, _, _, _ = self.evaluate(sol)
            pareto_constraints.append({'Z2': float(const[0]), 'Z3': float(const[1])})

        if verbose:
            print(f"\n{'='*65}")
            print(f"✅ NSGA-III v12 Completed — {elapsed:.2f}s")
            print(f"   Pareto Solutions: {len(pareto_solutions)}")
            if len(pareto_objectives) > 0:
                print(f"   Z₁: [{pareto_objectives[:,0].min():.2f}% — {pareto_objectives[:,0].max():.2f}%]")
                print(f"   Z₄: [{pareto_objectives[:,1].min():.0f} — {pareto_objectives[:,1].max():.0f}]")
            print(f"{'='*65}\n")

        return pareto_solutions, pareto_objectives, {
            'scenario': self.scenario,
            'algorithm': 'NSGA-III-V7-Standard',
            'version': 7,
            'pareto_size': len(pareto_solutions),
            'pareto_objectives': pareto_objectives.tolist() if len(pareto_objectives) > 0 else [],
            'constraint_values': pareto_constraints,
            'constraint_thresholds': {'Z2': z2t, 'Z3': z3t},
            'elapsed_time': elapsed,
            'generations': self.generations,
            'population_size': self.population_size,
            'n_divisions': self.n_divisions,
            'n_reference_points': self.n_ref,
            'history': self.history,
            'all_solutions': self.all_solutions,
            'p_shift_final':     self.p_shift,
            'p_swap_final':      self.p_swap,
            'p_blockswap_final': self.p_blockswap,
        }


NSGA3 = NSGA3V2

if __name__ == "__main__":
    print("NSGA-III v12 | CDP + Crowding | Adaptive Mutation (shift+swap+blockswap) | No Diversity Collapse")
