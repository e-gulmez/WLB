# -*- coding: utf-8 -*-
"""
GA Algorithm — VERSION 12
==========================================
Standard Genetic Algorithm with weighted-sum fitness and penalty-based
constraint handling.

STANDARD COMPONENTS:
    - Weighted-sum fitness: w1*Z1_norm - w4*Z4_norm
    - Penalty for constraint violations (Z2, Z3 thresholds)
    - Rank-based selection
    - Elitism
    - Shift, Swap, Block-Swap mutation (domain-specific operators)
    - Adaptive mutation rates (success-based, per operator)

PROBLEM-SPECIFIC:
    - Constraint model: Z2 ≤ threshold, Z3 ≤ threshold
    - TOL0/TOL1 capacity flexibility
    - Preference-aware individual creation

REMOVED (non-standard):
    - Diversity collapse detection
    - Immigrant injection
    - Stagnation boost
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
    calculate_individual_satisfaction,
    calculate_working_hours,
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


class GeneticAlgorithmV2:
    """
    Standard GA with weighted-sum fitness and penalty constraints.

    Objectives: Z1 (Satisfaction, MAX), Z4 (Capacity Deviation, MIN)
    Constraints: Z2 ≤ threshold, Z3 ≤ threshold (penalty-based)
    """

    def __init__(
        self,
        scenario_data: Dict[str, Any],
        population_size: int = 300,
        generations: int = 500,
        crossover_rate: float = 0.85,
        # Independent dice: separate probability per operator (adaptive)
        p_shift: float = 0.15,
        p_swap:  float = 0.12,
        p_blockswap: float = 0.08,
        elitism_rate: float = 0.05,
        w1: float = 0.7,
        w4: float = 0.3,
        seed: int = None,
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
        self.preferences       = scenario_data['preferences']
        self.manager_requirements = scenario_data['manager_requirements']
        self.contract_hours    = scenario_data['contract_hours']

        # Algorithm parameters
        self.population_size = population_size
        self.generations     = generations
        self.crossover_rate  = crossover_rate
        self.p_shift         = p_shift
        self.p_swap          = p_swap
        self.p_blockswap     = p_blockswap
        self.elitism_count   = max(1, int(population_size * elitism_rate))
        self.w1, self.w4     = w1, w4
        self.verbose_interval = verbose_interval

        # Adaptive mutation config
        self.adaptive_interval = adaptive_interval
        self.P_MIN_SHIFT  = 0.08;  self.P_MAX_SHIFT  = 0.30
        self.P_MIN_SWAP   = 0.08;  self.P_MAX_SWAP   = 0.25
        self.P_MIN_BS     = 0.05;  self.P_MAX_BS     = 0.15
        self.ADAPT_DELTA  = 0.02   # step size per update
        # Success counters (reset every adaptive_interval generations)
        self._shift_success = 0; self._shift_total = 0
        self._swap_success  = 0; self._swap_total  = 0
        self._bs_success    = 0; self._bs_total    = 0

        # Constraint thresholds
        self.z2_threshold, self.z3_threshold = get_constraint_thresholds()

        # Best tracking
        self.best_fitness    = float('-inf')
        self.best_solution   = None
        self.best_objectives = (0.0, 0.0)
        self.best_constraints = (0.0, 0.0)

        # History
        self.history = {
            'generation': [], 'best_Z1': [], 'best_Z4': [],
            'best_Z2': [], 'best_Z3': [],
            'p_shift': [], 'p_swap': [], 'p_blockswap': [],
        }
        self.all_solutions = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── Individual Creation ───────────────────────────────────────────

    def create_individual(self):
        """Random individual with flexibility bounds."""
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
        """Z4-optimal: assigns exactly required count."""
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
        """1/3 Z4-optimal + 2/3 random with flexibility."""
        n_pref = self.population_size // 3
        n_rnd = self.population_size - n_pref
        pop = [self.create_z4_optimal_individual() for _ in range(n_pref)]
        pop += [self.create_individual() for _ in range(n_rnd)]
        return pop

    # ── Fitness ───────────────────────────────────────────────────────

    def calculate_fitness(self, objectives, constraints, is_valid):
        """
        Weighted-sum fitness with constraint penalty.
        fitness = w1 * Z1_norm - w4 * Z4_norm - penalty(Z2, Z3) - penalty(hard)
        """
        Z1, Z4 = objectives
        Z2, Z3 = constraints

        Z1_norm = Z1 / 100.0
        n_slots = self.days * self.num_shifts
        Z4_norm = min(2.0, Z4 / max(1.0, n_slots * 0.5))

        base_fitness = self.w1 * Z1_norm - self.w4 * Z4_norm

        # Soft constraint penalty (Z2, Z3)
        EPS = 1e-6
        z2_viol = max(0, Z2 - self.z2_threshold - EPS)
        z3_viol = max(0, Z3 - self.z3_threshold - EPS)
        total_cv = (z2_viol / self.z2_threshold) + (z3_viol / self.z3_threshold)

        # Hard constraint penalty
        hard_penalty = 50.0 if not is_valid else 0.0

        PENALTY_WEIGHT = 50.0
        return base_fitness - hard_penalty - PENALTY_WEIGHT * total_cv

    def evaluate(self, individual):
        """
        Returns: (fitness, objectives, constraints, constraint_satisfied, is_valid, details)
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
        objectives, constraints, constraint_satisfied, _, p_values, details = result
        fitness = self.calculate_fitness(objectives, constraints, is_valid)
        return (fitness, objectives, constraints, constraint_satisfied, is_valid, details)

    # ── Selection ─────────────────────────────────────────────────────

    def rank_selection(self, population, evaluations):
        """Rank-based selection."""
        n = len(evaluations)
        sorted_order = sorted(range(n), key=lambda i: evaluations[i][0])
        ranks = [0] * n
        for rank, idx in enumerate(sorted_order):
            ranks[idx] = rank + 1
        total = sum(ranks)
        probs = [r / total for r in ranks]
        idx = int(np.random.choice(n, p=probs))
        return idx

    # ── Adaptive Mutation Rate Update ──────────────────────────────

    def _adaptive_update_rates(self):
        """
        Update mutation rates based on success ratios.
        Called every adaptive_interval generations.
        Successful operator → increase rate; unsuccessful → decrease.
        """
        def _update(p, success, total, p_min, p_max):
            if total == 0:
                return p
            ratio = success / total
            if ratio > 0.2:     # >20% success → increase
                p = min(p_max, p + self.ADAPT_DELTA)
            elif ratio < 0.05:  # <5% success → decrease
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
        start_time = time.time()
        z2t, z3t = get_constraint_thresholds()

        if verbose:
            from constraints import CAPACITY_FLEXIBILITY, FLEXIBILITY_MIN_REQ
            flex_str = f"±1 (required≥{FLEXIBILITY_MIN_REQ})" if CAPACITY_FLEXIBILITY else "None (exact match)"
            print(f"\n{'='*65}")
            print(f"🧬 GA VERSION 12 — Scenario {self.scenario}")
            print(f"{'='*65}")
            print(f"   OBJECTIVES: Z₁ (MAX, w={self.w1}), Z₄ (MIN, w={self.w4})")
            print(f"   CONSTRAINTS: Z₂ ≤ {z2t:.4f}, Z₃ ≤ {z3t:.6f}")
            print(f"   FLEXIBILITY: {flex_str}")
            print(f"   Pop: {self.population_size} | Gen: {self.generations}")
            print(f"   p_shift={self.p_shift:.2f} | p_swap={self.p_swap:.2f} | p_bs={self.p_blockswap:.2f} (adaptive)")
            print(f"{'='*65}")

        population = self.initialize_population()
        evaluations = [self.evaluate(ind) for ind in population]
        self.best_fitness = float('-inf')
        self.best_solution = None

        for gen in range(self.generations):
            n = len(population)

            # ── Elitism ────────────────────────────────────────────
            sorted_idx = sorted(range(n), key=lambda i: evaluations[i][0], reverse=True)
            elites = [deepcopy(population[i]) for i in sorted_idx[:self.elitism_count]]
            elite_evals = [evaluations[i] for i in sorted_idx[:self.elitism_count]]

            new_population = elites[:]
            new_evaluations = list(elite_evals)

            # ── Breed offspring ─────────────────────────────────────
            while len(new_population) < n:
                p1_idx = self.rank_selection(population, evaluations)
                p2_idx = self.rank_selection(population, evaluations)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = (crossover_day_slice if random.random() < 0.5
                             else crossover_uniform)(
                        population[p1_idx], population[p2_idx],
                        self.days, self.num_shifts)
                else:
                    child = deepcopy(population[p1_idx])

                # Mutation: independent dice per operator (adaptive rates)
                parent_fitness = evaluations[p1_idx][0]
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

                child_eval = self.evaluate(child)
                new_population.append(child)
                new_evaluations.append(child_eval)

                # Track success: child fitness > parent fitness
                improved = child_eval[0] > parent_fitness
                if applied_shift:
                    self._shift_total += 1
                    if improved: self._shift_success += 1
                if applied_swap:
                    self._swap_total += 1
                    if improved: self._swap_success += 1
                if applied_bs:
                    self._bs_total += 1
                    if improved: self._bs_success += 1

            population = new_population[:n]
            evaluations = new_evaluations[:n]

            # ── Track best solution ─────────────────────────────────
            # Priority: feasible > infeasible; among feasible: highest fitness
            feasible_evals = [(i, e) for i, e in enumerate(evaluations)
                              if e[3][0] and e[3][1]]  # Z2_ok and Z3_ok
            best_is_feasible = (self.best_solution is not None and
                                self.best_constraints[0] <= self.z2_threshold and
                                self.best_constraints[1] <= self.z3_threshold)

            if feasible_evals:
                # Among feasible: pick highest fitness (accounts for both Z1 and Z4)
                best_feas_idx, best_feas_eval = max(feasible_evals, key=lambda x: x[1][0])
                hist_obj = best_feas_eval[1]
                hist_con = best_feas_eval[2]
                best_feas_fitness = best_feas_eval[0]

                if not best_is_feasible:
                    # First feasible ever → always update
                    self.best_solution    = deepcopy(population[best_feas_idx])
                    self.best_objectives  = hist_obj
                    self.best_constraints = hist_con
                    self.best_fitness     = best_feas_fitness
                else:
                    # Already had feasible → update if this gen's best feasible has higher fitness
                    if best_feas_fitness > self.best_fitness:
                        self.best_solution    = deepcopy(population[best_feas_idx])
                        self.best_objectives  = hist_obj
                        self.best_constraints = hist_con
                        self.best_fitness     = best_feas_fitness
            else:
                # No feasible in this gen
                best_idx = max(range(len(evaluations)), key=lambda i: evaluations[i][0])
                hist_obj = evaluations[best_idx][1]
                hist_con = evaluations[best_idx][2]
                if self.best_solution is None:
                    self.best_solution    = deepcopy(population[best_idx])
                    self.best_objectives  = hist_obj
                    self.best_constraints = hist_con

            # ── Adaptive mutation rate update ─────────────────────
            if (gen + 1) % self.adaptive_interval == 0:
                self._adaptive_update_rates()

            # ── History ─────────────────────────────────────────────
            self.history['generation'].append(gen)
            self.history['best_Z1'].append(hist_obj[0])
            self.history['best_Z4'].append(hist_obj[1])
            self.history['best_Z2'].append(hist_con[0])
            self.history['best_Z3'].append(hist_con[1])
            self.history['p_shift'].append(self.p_shift)
            self.history['p_swap'].append(self.p_swap)
            self.history['p_blockswap'].append(self.p_blockswap)

            # Track all solutions for scatter plot
            for e in evaluations:
                _, obj, con, csat, _, _ = e
                self.all_solutions.append({
                    'generation': gen, 'Z1': float(obj[0]), 'Z4': float(obj[1]),
                    'Z2': float(con[0]), 'Z3': float(con[1]),
                    'feasible': bool(csat[0] and csat[1])
                })

            # ── Log ─────────────────────────────────────────────────
            if verbose and (gen % self.verbose_interval == 0 or gen == self.generations - 1):
                z2_ok, _ = check_Z2_constraint(hist_con[0])
                z3_ok, _ = check_Z3_constraint(hist_con[1])
                status = "✓" if (z2_ok and z3_ok) else "✗"
                feasible_count = len(feasible_evals)
                print(f"   Nesil {gen:3d}: Z₁={hist_obj[0]:.1f}%, Z₄={hist_obj[1]:.1f} "
                      f"Z₂={hist_con[0]:.3f} Z₃={hist_con[1]:.4f} {status} "
                      f"Feasible={feasible_count}/{n} "
                      f"p_s={self.p_shift:.3f} p_sw={self.p_swap:.3f} p_bs={self.p_blockswap:.3f}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n   ✅ GA completed: {elapsed:.1f}s")
            print(f"   En iyi Z₁={self.best_objectives[0]:.2f}%, Z₄={self.best_objectives[1]:.1f}")

        # ── Build result ───────────────────────────────────────────
        z2_ok, _ = check_Z2_constraint(self.best_constraints[0])
        z3_ok, _ = check_Z3_constraint(self.best_constraints[1])

        _, _, _, _, _, details = evaluate_objectives_v2(
            self.best_solution, self.preferences, self.manager_requirements,
            self.doctors, self.days, self.num_shifts)

        p_vals = calculate_individual_satisfaction(
            self.best_solution, self.preferences, self.doctors,
            self.days, self.num_shifts)

        wh = calculate_working_hours(
            self.best_solution, self.doctors, self.days, self.num_shifts,
            self.shift_durations, self.contract_hours)

        return self.best_solution, self.best_objectives, {
            'history':          self.history,
            'elapsed_time':     elapsed,
            'best_objectives':  self.best_objectives,
            'best_constraints': self.best_constraints,
            'Z1': self.best_objectives[0],
            'Z4': self.best_objectives[1],
            'Z2': self.best_constraints[0],
            'Z3': self.best_constraints[1],
            'generations':      self.generations,
            'population_size':  self.population_size,
            'best_fitness':     self.best_fitness,
            'weights':          {'w1': self.w1, 'w4': self.w4},
            'constraints_satisfied': (z2_ok, z3_ok),
            'all_constraints_satisfied': details['all_constraints_satisfied'],
            'individual_satisfaction': p_vals,
            'working_hours': wh,
            'p_shift_final':     self.p_shift,
            'p_swap_final':      self.p_swap,
            'p_blockswap_final': self.p_blockswap,
            'all_solutions':     self.all_solutions,
        }


if __name__ == "__main__":
    print("GA v12 | Weighted-Sum | Adaptive Mutation (shift+swap+blockswap) | No Diversity Collapse")
