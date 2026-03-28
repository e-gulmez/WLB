# -*- coding: utf-8 -*-
"""
Constraints Module - VERSION 6
=================================
CHANGES (v5 → v6):
    - Flexibility threshold 2 → 3 (±1 flexibility if required ≥ 3)
    - CAPACITY_FLEXIBILITY global flag added (TOL0 / TOL1 distinction)
    - Adjustable at runtime via set_capacity_flexibility()

Flexibility Rules (TOL1 — CAPACITY_FLEXIBILITY=True):
    required = 1 or 2  → exact match required  [required, required]
    required ≥ 3       → smart ±1 flexibility
        talep > required  → [required, required+1]
        talep < required  → [required-1, required]
        talep = required  → [required-1, required+1]

Esneklik Yok (TOL0 — CAPACITY_FLEXIBILITY=False):
    Exact match for every required value: [required, required]
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict

# ── Global flexibility setting ───────────────────────────────────────────────────
CAPACITY_FLEXIBILITY    = True   # True=TOL1 (required≥3 for ±1), False=TOL0 (exact)
FLEXIBILITY_MIN_REQ     = 3      # Minimum required value for flexibility


def set_capacity_flexibility(enabled: bool):
    """
    Sets capacity flexibility.
    enabled=True  → TOL1: ±1 flexibility in shifts where required≥3
    enabled=False → TOL0: exact match in all shifts
    """
    global CAPACITY_FLEXIBILITY
    CAPACITY_FLEXIBILITY = enabled


def get_capacity_flexibility() -> bool:
    return CAPACITY_FLEXIBILITY


def _flexibility_bounds(required: int, request_count: int) -> Tuple[int, int]:
    """
    Returns [lower, upper] bounds based on given required and demand count.
    Operates according to global CAPACITY_FLEXIBILITY and FLEXIBILITY_MIN_REQ.
    """
    if not CAPACITY_FLEXIBILITY or required < FLEXIBILITY_MIN_REQ:
        # TOL0 or low required → exact match
        lower = required
        upper = required
    else:
        # TOL1: required ≥ FLEXIBILITY_MIN_REQ → ±1 smart flexibility
        if request_count > required:
            lower = required
            upper = required + 1
        elif request_count < required:
            lower = max(1, required - 1)
            upper = required
        else:
            lower = max(1, required - 1)
            upper = required + 1
    return lower, upper


# ── Working hours calculation ─────────────────────────────────────────────────────

def calculate_working_hours(assignments, shift_durations, doctors, days, num_shifts):
    n_weeks = (days + 6) // 7
    week_hours = [{doc: 0 for doc in doctors} for _ in range(max(1, n_weeks))]
    daily_hours = {day: {doc: 0 for doc in doctors} for day in range(days)}

    for day in range(days):
        if day >= len(assignments): continue
        for shift in range(num_shifts):
            if shift >= len(assignments[day]): continue
            duration = shift_durations[shift] if shift < len(shift_durations) else 0
            week_idx = min(day // 7, len(week_hours) - 1)
            for doc in set(assignments[day][shift]):
                if doc in doctors:
                    week_hours[week_idx][doc] += duration
                    daily_hours[day][doc]      += duration

    weekly_hours = {doc: max(week_hours[w][doc] for w in range(len(week_hours))) for doc in doctors}
    return weekly_hours, daily_hours


def check_weekly_hours(weekly_hours, contract_hours):
    violations = [f"{doc}: {h}h > {contract_hours.get(doc,0)}h"
                  for doc, h in weekly_hours.items() if h > contract_hours.get(doc, 0)]
    return len(violations) == 0, violations


def check_daily_hours(daily_hours, max_daily=12):
    violations = [f"Day {day+1}, {doc}: {h}h > {max_daily}h"
                  for day, dh in daily_hours.items()
                  for doc, h  in dh.items() if h > max_daily]
    return len(violations) == 0, violations


def check_no_duplicate(assignments, days, num_shifts):
    violations = []
    for day in range(days):
        if day >= len(assignments): continue
        for shift in range(num_shifts):
            if shift >= len(assignments[day]): continue
            lst = assignments[day][shift]
            if len(lst) != len(set(lst)):
                seen = set(); dups = []
                for doc in lst:
                    if doc in seen: dups.append(doc)
                    seen.add(doc)
                violations.append(f"Day {day+1}, Shift{shift+1}: Duplicate={dups}")
    return len(violations) == 0, violations


def check_capacity_smart(assignments, manager_requirements, preferences, days, num_shifts):
    """
    Capacity constraint check (v6: CAPACITY_FLEXIBILITY + FLEXIBILITY_MIN_REQ).
    """
    violations = []
    stats = {'exact_match': 0, 'plus_one_used': 0, 'minus_one_used': 0, 'violations': 0}

    for day in range(days):
        day_key = day + 1
        if day >= len(assignments): continue
        for shift in range(num_shifts):
            if shift >= len(assignments[day]): continue

            required      = manager_requirements.get(day_key, [0] * num_shifts)[shift]
            assigned_count = len(set(assignments[day][shift]))
            requesting    = preferences.get(day_key, [[]] * num_shifts)[shift]
            request_count = len([p for p in requesting if p])

            lower, upper = _flexibility_bounds(required, request_count)

            if lower <= assigned_count <= upper:
                if assigned_count == required:        stats['exact_match'] += 1
                elif assigned_count > required:        stats['plus_one_used'] += 1
                else:                                  stats['minus_one_used'] += 1
            else:
                stats['violations'] += 1
                violations.append(
                    f"Day {day+1}, Shift{shift+1}: Assigned={assigned_count}, "
                    f"Gerekli={required}, Talep={request_count}, "
                    f"Allowed=[{lower}-{upper}]"
                )

    return len(violations) == 0, violations, stats


def is_valid_solution(assignments, shift_durations, contract_hours,
                      manager_requirements, preferences, doctors, days, num_shifts,
                      max_daily_hours=12):
    details = {'weekly_hours': {}, 'violations': [], 'flexibility_stats': {}}
    total_violations = 0

    weekly_hours, daily_hours = calculate_working_hours(
        assignments, shift_durations, doctors, days, num_shifts)
    details['weekly_hours'] = weekly_hours

    valid1, viol1 = check_weekly_hours(weekly_hours, contract_hours)
    if not valid1:
        total_violations += len(viol1)
        details['violations'].extend([f"[Weekly] {v}" for v in viol1])

    valid2, viol2 = check_daily_hours(daily_hours, max_daily_hours)
    if not valid2:
        total_violations += len(viol2)
        details['violations'].extend([f"[Daily] {v}" for v in viol2])

    valid3, viol3 = check_no_duplicate(assignments, days, num_shifts)
    if not valid3:
        total_violations += len(viol3)
        details['violations'].extend([f"[Tekrar] {v}" for v in viol3])

    valid4, viol4, flex_stats = check_capacity_smart(
        assignments, manager_requirements, preferences, days, num_shifts)
    details['flexibility_stats'] = flex_stats
    if not valid4:
        total_violations += len(viol4)
        details['violations'].extend([f"[Kapasite] {v}" for v in viol4])

    return total_violations == 0, float(total_violations), details


if __name__ == "__main__":
    print(f"Constraints v6 | CAPACITY_FLEXIBILITY={CAPACITY_FLEXIBILITY} | MIN_REQ={FLEXIBILITY_MIN_REQ}")
