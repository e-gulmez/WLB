# -*- coding: utf-8 -*-
"""
Operators Module - VERSION 6
==============================
CHANGES (v5 → v6):
    - mutate_shift: candidates[:target] → random.sample(candidates, target)
    - Flexibility threshold 2 → 3 (tied to FLEXIBILITY_MIN_REQ)
    - repair_assignment: KALDIRILDI (v12'de kullanılmıyor)
"""

import random
from copy import deepcopy
from typing import Dict, List, Tuple, Any

from constraints import (calculate_working_hours, _flexibility_bounds,
                         get_capacity_flexibility, FLEXIBILITY_MIN_REQ)


def _compute_week_hours(assignments, shift_durations, doctors, days, num_shifts):
    n_weeks = (days + 6) // 7
    week_hours = [{doc: 0 for doc in doctors} for _ in range(max(1, n_weeks))]
    for day in range(min(days, len(assignments))):
        week_idx = min(day // 7, len(week_hours) - 1)
        for shift in range(min(num_shifts, len(assignments[day]))):
            duration = shift_durations[shift] if shift < len(shift_durations) else 0
            for doc in set(assignments[day][shift]):
                if doc in doctors:
                    week_hours[week_idx][doc] += duration
    return week_hours


def ensure_structure(assignments, days, num_shifts):
    if not isinstance(assignments, list):
        assignments = []
    while len(assignments) < days:
        assignments.append([])
    for d in range(days):
        if not isinstance(assignments[d], list):
            assignments[d] = []
        while len(assignments[d]) < num_shifts:
            assignments[d].append([])
        for s in range(num_shifts):
            if not isinstance(assignments[d][s], list):
                assignments[d][s] = []
    return assignments


def doctor_prefers(doc, day, shift, preferences):
    day_key = day + 1
    prefs = preferences.get(day_key, [])
    return doc in prefs[shift] if shift < len(prefs) else False


def get_available_doctors(day, shift, assignments, shift_durations, contract_hours,
                          doctors, days, num_shifts, preferences=None,
                          max_daily_hours=12, prioritize_preferences=True):
    week_hours_list = _compute_week_hours(assignments, shift_durations, doctors, days, num_shifts)
    week_idx = min(day // 7, len(week_hours_list) - 1)
    weekly_hours = week_hours_list[week_idx]

    # FIX 4: Target shift excluded from daily hours calculation.
    # Old code summed all shifts (including target) → double-assignment
    # or wrong capacity calculation on the same day.
    # Fix: flat dict + all shifts except target shift.
    daily_hours = {doc: 0 for doc in doctors}
    if day < len(assignments):
        for s in range(min(num_shifts, len(assignments[day]))):
            if s == shift:
                continue  # skip target shift
            dur = shift_durations[s] if s < len(shift_durations) else 0
            for doc in set(assignments[day][s]):
                if doc in doctors:
                    daily_hours[doc] += dur

    shift_duration = shift_durations[shift] if shift < len(shift_durations) else 0
    already_assigned = set(assignments[day][shift]) if day < len(assignments) and shift < len(assignments[day]) else set()

    candidates = []
    for doc in doctors:
        if doc in already_assigned: continue
        if weekly_hours.get(doc, 0) + shift_duration > contract_hours.get(doc, 0): continue
        if daily_hours.get(doc, 0) + shift_duration > max_daily_hours: continue
        candidates.append(doc)

    if prioritize_preferences and preferences:
        candidates.sort(key=lambda d: (
            -int(doctor_prefers(d, day, shift, preferences)),
            -(contract_hours.get(d, 0) - weekly_hours.get(d, 0))
        ))
    else:
        candidates.sort(key=lambda d: -(contract_hours.get(d, 0) - weekly_hours.get(d, 0)))

    return candidates


# ── Crossover ────────────────────────────────────────────────────────────────

def crossover_day_slice(parent1, parent2, days, num_shifts):
    """Day-based single-point crossover."""
    p1 = ensure_structure(deepcopy(parent1), days, num_shifts)
    p2 = ensure_structure(deepcopy(parent2), days, num_shifts)
    cut = random.randint(1, days - 1)
    return [deepcopy(p1[d]) if d < cut else deepcopy(p2[d]) for d in range(days)]


def crossover_uniform(parent1, parent2, days, num_shifts, prob=0.5):
    """Uniform crossover: independent parent selection per (day, shift)."""
    p1 = ensure_structure(deepcopy(parent1), days, num_shifts)
    p2 = ensure_structure(deepcopy(parent2), days, num_shifts)
    child = [[[] for _ in range(num_shifts)] for _ in range(days)]
    for d in range(days):
        for s in range(num_shifts):
            child[d][s] = deepcopy(p1[d][s]) if random.random() < prob else deepcopy(p2[d][s])
    return child


# ── Mutation Operators ─────────────────────────────────────────────────────

def mutate_shift(assignments, preferences, manager_requirements,
                 shift_durations, contract_hours, doctors, days, num_shifts):
    """
    Shift Mutation — Re-samples a random (day, shift) slot.

    FIX (v6): candidates[:target] → random.sample(candidates, target)
    True random sampling; different staff can be assigned each call.

    Flexibility threshold: compatible with FLEXIBILITY_MIN_REQ (=3).
    """
    assignments = ensure_structure(deepcopy(assignments), days, num_shifts)
    day   = random.randint(0, days - 1)
    shift = random.randint(0, num_shifts - 1)
    day_key = day + 1

    required      = manager_requirements.get(day_key, [0] * num_shifts)[shift]
    requesting    = preferences.get(day_key, [[]] * num_shifts)[shift]
    request_count = len([p for p in requesting if p])

    lower, upper  = _flexibility_bounds(required, request_count)
    target        = random.randint(lower, upper)

    assignments[day][shift] = []
    candidates = get_available_doctors(day, shift, assignments, shift_durations,
                                       contract_hours, doctors, days, num_shifts, preferences)

    # ▶ FIX: deterministic [:target] replaced with random.sample
    k = min(target, len(candidates))
    assignments[day][shift] = random.sample(candidates, k) if k > 0 else []
    return assignments


def mutate_swap(assignments, preferences, shift_durations, contract_hours,
                doctors, days, num_shifts):
    """
    Swap Mutation — Replace non-requesting staff with requesting staff.
    Capacity unchanged; focused on improving Z1 quality.
    """
    assignments = ensure_structure(deepcopy(assignments), days, num_shifts)
    day   = random.randint(0, days - 1)
    shift = random.randint(0, num_shifts - 1)
    current = sorted(set(assignments[day][shift]))

    non_preferring = [doc for doc in current if not doctor_prefers(doc, day, shift, preferences)]
    if not non_preferring:
        return assignments

    out_doc = random.choice(non_preferring)
    temp = deepcopy(assignments)
    temp[day][shift] = [d for d in current if d != out_doc]

    candidates = get_available_doctors(day, shift, temp, shift_durations,
                                       contract_hours, doctors, days, num_shifts, preferences)
    preferring = [d for d in candidates if doctor_prefers(d, day, shift, preferences) and d not in current]

    if preferring:
        in_doc = random.choice(preferring)
        assignments[day][shift] = sorted(set([d for d in current if d != out_doc] + [in_doc]))
    return assignments


def mutate_block_swap(assignments, preferences, shift_durations, contract_hours,
                      doctors, days, num_shifts, block_len_range=(2, 3)):
    """
    Block-Swap Mutation — Bulk swap of consecutive shifts between two days.
    Provides large-scale jumps to escape local optima.
    """
    assignments = ensure_structure(deepcopy(assignments), days, num_shifts)
    if days < 2 or num_shifts < 2:
        return assignments

    d1, d2 = random.sample(range(days), 2)
    min_len, max_len = block_len_range
    max_len   = min(max_len, num_shifts)
    min_len   = max(1, min(min_len, max_len))
    block_len = random.randint(min_len, max_len)
    start     = random.randint(0, num_shifts - block_len)

    for s in range(start, start + block_len):
        assignments[d1][s], assignments[d2][s] = assignments[d2][s], assignments[d1][s]
    return assignments



# NOTE: repair_assignment KALDIRILDI (v12'de kullanılmıyor)


if __name__ == "__main__":
    from constraints import FLEXIBILITY_MIN_REQ as FMR
    flex = get_capacity_flexibility()
    print(f"Operators v6 | CAPACITY_FLEXIBILITY={flex} | FLEXIBILITY_MIN_REQ={FMR}")
