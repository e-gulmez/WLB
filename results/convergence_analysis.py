# -*- coding: utf-8 -*-
"""
YAKINSAMA ANALİZİ - VERSION 6 Uyumlu
=====================================
Bu script erken yakınsama probleminin kök nedenini tespit eder.

VERSION 6 UYUMLULUK:
    - repair_assignment KALDIRILDI (v6'da yok)
    - evaluate_objectives_v2 kullanılıyor (6 dönüş değeri)
    - Independent dice mutation (shift, swap, blockswap ayrı ayrı)
    - Adaptive mutation rate analizi eklendi
    - mutate_block_swap operatör testine dahil edildi

Analizler:
1. Multi-Seed Testi: 10 farklı seed ile tutarlılık kontrolü
2. Popülasyon Çeşitliliği: Nesil bazlı diversity ölçümü
3. Operatör Etkinliği: shift/swap/blockswap başarı oranları
4. Constraint Severity: Kısıtların daraltıcı etkisi
"""

import os
import sys
import json
import time
import random
import numpy as np
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

# Modülleri import et (v6 uyumlu)
from data_loader import load_scenario
from objectives import (
    evaluate_objectives_v2,
    calculate_individual_satisfaction,
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


class ConvergenceAnalyzer:
    """Yakınsama analizi için araçlar (v6 uyumlu)."""

    def __init__(self, scenario_data):
        self.data = scenario_data
        self.scenario = scenario_data['scenario']
        self.days = scenario_data['days']
        self.num_shifts = scenario_data['num_shifts']
        self.shift_durations = scenario_data['shift_durations']
        self.doctors = scenario_data['doctors']
        self.contract_hours = scenario_data['contract_hours']
        self.preferences = scenario_data['preferences']
        self.manager_requirements = scenario_data['manager_requirements']

    def solution_to_vector(self, solution):
        """Çözümü karşılaştırılabilir vektöre çevir."""
        vector = []
        for day in range(self.days):
            for shift in range(self.num_shifts):
                if day < len(solution) and shift < len(solution[day]):
                    personnel = sorted(solution[day][shift])
                    vector.append(tuple(personnel))
                else:
                    vector.append(tuple())
        return tuple(vector)

    def calculate_hamming_distance(self, sol1, sol2):
        """İki çözüm arasındaki Hamming mesafesi."""
        distance = 0
        for day in range(self.days):
            for shift in range(self.num_shifts):
                set1 = set(sol1[day][shift]) if day < len(sol1) and shift < len(sol1[day]) else set()
                set2 = set(sol2[day][shift]) if day < len(sol2) and shift < len(sol2[day]) else set()
                distance += len(set1.symmetric_difference(set2))
        return distance

    def calculate_population_diversity(self, population):
        """Popülasyon çeşitliliğini hesapla."""
        n = len(population)
        if n < 2:
            return {'unique_ratio': 0.0, 'unique_count': 0,
                    'avg_hamming_distance': 0, 'normalized_diversity': 0.0,
                    'population_size': n}

        unique_vectors = set()
        for sol in population:
            unique_vectors.add(self.solution_to_vector(sol))
        unique_ratio = len(unique_vectors) / n

        sample_size = min(50, n)
        indices = random.sample(range(n), sample_size)
        total_distance = 0
        pair_count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_distance += self.calculate_hamming_distance(
                    population[indices[i]], population[indices[j]])
                pair_count += 1

        avg_distance = total_distance / pair_count if pair_count > 0 else 0
        max_distance = self.days * self.num_shifts * len(self.doctors)
        normalized_distance = avg_distance / max_distance if max_distance > 0 else 0

        return {
            'unique_ratio': unique_ratio,
            'unique_count': len(unique_vectors),
            'avg_hamming_distance': avg_distance,
            'normalized_diversity': normalized_distance,
            'population_size': n
        }

    def evaluate(self, individual):
        """Bireyi değerlendir (v6: evaluate_objectives_v2 — 6 dönüş değeri)."""
        is_valid, _, _ = is_valid_solution(
            individual, self.shift_durations, self.contract_hours,
            self.manager_requirements, self.preferences, self.doctors,
            self.days, self.num_shifts
        )
        result = evaluate_objectives_v2(
            individual, self.preferences, self.manager_requirements,
            self.doctors, self.days, self.num_shifts
        )
        objectives, constraints, constraint_satisfied, constraint_violations, p_values, details = result
        z2_ok, z3_ok = constraint_satisfied
        return {
            'Z1': objectives[0],
            'Z4': objectives[1],
            'Z2': constraints[0],
            'Z3': constraints[1],
            'is_valid': is_valid,
            'feasible': is_valid and z2_ok and z3_ok,
            'p_values': p_values,
            'constraint_satisfied': constraint_satisfied,
        }

    def create_random_individual(self):
        """
        Rastgele birey oluştur (v6: flexibility bounds kullanır, repair yok).
        """
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

    def apply_v6_mutation(self, child, p_shift=0.15, p_swap=0.12, p_blockswap=0.08):
        """
        V6 Independent Dice Mutation: her operatör bağımsız olasılıkla uygulanır.
        Repair yok — operatörler kendi içinde tutarlı çözüm üretir.
        """
        ops_applied = []

        if random.random() < p_shift:
            child = mutate_shift(
                child, self.preferences, self.manager_requirements,
                self.shift_durations, self.contract_hours, self.doctors,
                self.days, self.num_shifts)
            ops_applied.append('shift')

        if random.random() < p_swap:
            child = mutate_swap(
                child, self.preferences, self.shift_durations,
                self.contract_hours, self.doctors, self.days, self.num_shifts)
            ops_applied.append('swap')

        if random.random() < p_blockswap:
            child = mutate_block_swap(
                child, self.preferences, self.shift_durations,
                self.contract_hours, self.doctors, self.days, self.num_shifts)
            ops_applied.append('blockswap')

        return child, ops_applied


def analyze_multi_seed(scenario_data, n_seeds=10, generations=100, pop_size=50):
    """
    ANALİZ 1: Multi-Seed Testi
    V6: repair kaldırıldı, independent dice mutation kullanılıyor.
    """
    print("\n" + "=" * 70)
    print("📊 ANALİZ 1: MULTI-SEED TESTİ (v6)")
    print("=" * 70)
    print(f"   Seed Sayısı: {n_seeds} | Nesil: {generations} | Pop: {pop_size}")

    analyzer = ConvergenceAnalyzer(scenario_data)
    results = []

    for seed in range(n_seeds):
        print(f"\n   Seed {seed}: ", end="", flush=True)
        random.seed(seed)
        np.random.seed(seed)

        population = [analyzer.create_random_individual() for _ in range(pop_size)]
        best_z1 = 0
        best_z4 = 1000
        convergence_gen = 0

        for gen in range(generations):
            evals = [analyzer.evaluate(ind) for ind in population]
            feasible_evals = [e for e in evals if e['feasible']]
            valid_evals = [e for e in evals if e['is_valid']]

            target = feasible_evals if feasible_evals else valid_evals
            if target:
                current_best_z1 = max(e['Z1'] for e in target)
                current_best_z4 = min(e['Z4'] for e in target)
                if current_best_z1 > best_z1 + 0.1:
                    best_z1 = current_best_z1
                    convergence_gen = gen
                if current_best_z4 < best_z4 - 0.1:
                    best_z4 = current_best_z4
                    convergence_gen = gen

            fitnesses = []
            for e in evals:
                if e['feasible']:
                    fitnesses.append(e['Z1'] - e['Z4'] * 0.3)
                elif e['is_valid']:
                    fitnesses.append(e['Z1'] - e['Z4'] * 0.3 - 100)
                else:
                    fitnesses.append(-10000)

            elite_idx = int(np.argmax(fitnesses))
            new_population = [deepcopy(population[elite_idx])]

            while len(new_population) < pop_size:
                idx1, idx2 = random.sample(range(pop_size), 2)
                p1 = population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]
                idx1, idx2 = random.sample(range(pop_size), 2)
                p2 = population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]

                if random.random() < 0.85:
                    child = (crossover_day_slice if random.random() < 0.5 else crossover_uniform)(
                        p1, p2, analyzer.days, analyzer.num_shifts)
                else:
                    child = deepcopy(p1)

                child, _ = analyzer.apply_v6_mutation(child)
                new_population.append(child)

            population = new_population

        results.append({'seed': seed, 'best_Z1': best_z1,
                        'best_Z4': best_z4, 'convergence_gen': convergence_gen})
        print(f"Z₁={best_z1:.2f}%, Z₄={best_z4:.0f}, Conv@Gen{convergence_gen}")

    z1_values = [r['best_Z1'] for r in results]
    z4_values = [r['best_Z4'] for r in results]
    conv_gens = [r['convergence_gen'] for r in results]

    print(f"\n   {'─' * 60}")
    print(f"   📈 SONUÇLAR:")
    print(f"   Z₁ Aralığı: [{min(z1_values):.2f}% - {max(z1_values):.2f}%]  Std: {np.std(z1_values):.4f}")
    print(f"   Z₄ Aralığı: [{min(z4_values):.0f} - {max(z4_values):.0f}]")
    print(f"   Ort. Yakınsama Nesli: {np.mean(conv_gens):.1f}")

    z1_std = np.std(z1_values)
    if z1_std < 1.0:
        conclusion = "✅ TUTARLI: Farklı seed'ler benzer sonuç → Global optimuma yakın"
    elif z1_std < 3.0:
        conclusion = "⚠️ ORTA: Bazı varyans var → Lokal optimum riski"
    else:
        conclusion = "❌ TUTARSIZ: Yüksek varyans → Kesin lokal optimum problemi"
    print(f"\n   🎯 YORUM: {conclusion}")
    return results


def analyze_diversity_over_generations(scenario_data, generations=100, pop_size=50, seed=42):
    """
    ANALİZ 2: Nesil Bazlı Çeşitlilik Kaybı (v6)
    """
    print("\n" + "=" * 70)
    print("📊 ANALİZ 2: NESİL BAZLI ÇEŞİTLİLİK ANALİZİ (v6)")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    analyzer = ConvergenceAnalyzer(scenario_data)
    population = [analyzer.create_random_individual() for _ in range(pop_size)]

    diversity_history = []
    fitness_history = []

    for gen in range(generations):
        diversity = analyzer.calculate_population_diversity(population)
        diversity_history.append(diversity)

        evals = [analyzer.evaluate(ind) for ind in population]
        feasible_evals = [e for e in evals if e['feasible']]
        valid_evals = [e for e in evals if e['is_valid']]
        target = feasible_evals if feasible_evals else valid_evals

        if target:
            best_z1 = max(e['Z1'] for e in target)
            avg_z1 = np.mean([e['Z1'] for e in target])
            best_z4 = min(e['Z4'] for e in target)
        else:
            best_z1 = avg_z1 = 0
            best_z4 = 100

        fitness_history.append({
            'best_Z1': best_z1, 'avg_Z1': avg_z1, 'best_Z4': best_z4,
            'valid_count': len(valid_evals), 'feasible_count': len(feasible_evals)
        })

        if gen % 20 == 0 or gen == generations - 1:
            print(f"   Nesil {gen:3d}: Unique={diversity['unique_ratio'] * 100:.1f}%, "
                  f"Diversity={diversity['normalized_diversity']:.4f}, "
                  f"Z₁={best_z1:.2f}%, Feasible={len(feasible_evals)}/{pop_size}")

        fitnesses = []
        for e in evals:
            if e['feasible']:
                fitnesses.append(e['Z1'] - e['Z4'] * 0.3)
            elif e['is_valid']:
                fitnesses.append(e['Z1'] - e['Z4'] * 0.3 - 100)
            else:
                fitnesses.append(-10000)

        elite_idx = int(np.argmax(fitnesses))
        new_population = [deepcopy(population[elite_idx])]

        while len(new_population) < pop_size:
            idx1, idx2 = random.sample(range(pop_size), 2)
            p1 = population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]
            idx1, idx2 = random.sample(range(pop_size), 2)
            p2 = population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]

            if random.random() < 0.85:
                child = (crossover_day_slice if random.random() < 0.5 else crossover_uniform)(
                    p1, p2, analyzer.days, analyzer.num_shifts)
            else:
                child = deepcopy(p1)
            child, _ = analyzer.apply_v6_mutation(child)
            new_population.append(child)

        population = new_population

    initial_div = diversity_history[0]['normalized_diversity']
    final_div = diversity_history[-1]['normalized_diversity']
    loss = (initial_div - final_div) / initial_div * 100 if initial_div > 0 else 0

    print(f"\n   {'─' * 60}")
    print(f"   📈 ÇEŞİTLİLİK KAYBI:")
    print(f"   Başlangıç: Unique={diversity_history[0]['unique_ratio'] * 100:.1f}%, Diversity={initial_div:.4f}")
    print(f"   Final:     Unique={diversity_history[-1]['unique_ratio'] * 100:.1f}%, Diversity={final_div:.4f}")
    print(f"   Kayıp: {loss:.1f}%")

    final_unique = diversity_history[-1]['unique_ratio']
    if final_unique < 0.2:
        conclusion = "❌ KRİTİK: Popülasyon homojenleşti!"
    elif final_unique < 0.5:
        conclusion = "⚠️ UYARI: Önemli çeşitlilik kaybı var."
    else:
        conclusion = "✅ İYİ: Çeşitlilik makul düzeyde korunuyor."
    print(f"\n   🎯 YORUM: {conclusion}")
    return diversity_history, fitness_history


def analyze_operator_effectiveness(scenario_data, n_tests=100, seed=42):
    """
    ANALİZ 3: Operatör Etkinliği (v6)
    shift, swap, blockswap bağımsız başarı oranları.
    """
    print("\n" + "=" * 70)
    print("📊 ANALİZ 3: OPERATÖR ETKİNLİĞİ ANALİZİ (v6)")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    analyzer = ConvergenceAnalyzer(scenario_data)

    parents = [analyzer.create_random_individual() for _ in range(20)]
    parent_vectors = set(analyzer.solution_to_vector(p) for p in parents)

    # Crossover testi
    cx_results = {'new': 0, 'same': 0, 'improved': 0}
    print(f"\n   Crossover Testi ({n_tests} deneme)...")
    for _ in range(n_tests):
        p1, p2 = random.sample(parents, 2)
        child = (crossover_day_slice if random.random() < 0.5 else crossover_uniform)(
            p1, p2, analyzer.days, analyzer.num_shifts)
        cv = analyzer.solution_to_vector(child)
        if cv not in parent_vectors:
            cx_results['new'] += 1
        else:
            cx_results['same'] += 1
        ce = analyzer.evaluate(child)
        pe = analyzer.evaluate(p1)
        if ce['feasible'] and ce['Z1'] > pe['Z1']:
            cx_results['improved'] += 1

    print(f"   → Yeni çözüm:        {cx_results['new'] / n_tests * 100:.1f}%")
    print(f"   → Parent kopyası:     {cx_results['same'] / n_tests * 100:.1f}%")
    print(f"   → İyileştirme oranı:  {cx_results['improved'] / n_tests * 100:.1f}%")

    # Her operatör bağımsız testi
    op_tests = {
        'shift': lambda c: mutate_shift(
            c, analyzer.preferences, analyzer.manager_requirements,
            analyzer.shift_durations, analyzer.contract_hours,
            analyzer.doctors, analyzer.days, analyzer.num_shifts),
        'swap': lambda c: mutate_swap(
            c, analyzer.preferences, analyzer.shift_durations,
            analyzer.contract_hours, analyzer.doctors,
            analyzer.days, analyzer.num_shifts),
        'blockswap': lambda c: mutate_block_swap(
            c, analyzer.preferences, analyzer.shift_durations,
            analyzer.contract_hours, analyzer.doctors,
            analyzer.days, analyzer.num_shifts),
    }
    op_stats = {k: {'new': 0, 'z1_up': 0, 'z4_down': 0} for k in op_tests}

    for op_name, op_func in op_tests.items():
        print(f"\n   {op_name.title()} Mutation Testi ({n_tests} deneme)...")
        for _ in range(n_tests):
            p = random.choice(parents)
            pe = analyzer.evaluate(p)
            pv = analyzer.solution_to_vector(p)
            child = op_func(deepcopy(p))
            ce = analyzer.evaluate(child)
            if analyzer.solution_to_vector(child) != pv:
                op_stats[op_name]['new'] += 1
            if ce['Z1'] > pe['Z1']:
                op_stats[op_name]['z1_up'] += 1
            if ce['Z4'] < pe['Z4']:
                op_stats[op_name]['z4_down'] += 1

        s = op_stats[op_name]
        print(f"   → Değişiklik:  {s['new'] / n_tests * 100:.1f}%")
        print(f"   → Z₁ artış:   {s['z1_up'] / n_tests * 100:.1f}%")
        print(f"   → Z₄ azalma:  {s['z4_down'] / n_tests * 100:.1f}%")

    # Combined independent dice
    print(f"\n   Combined V6 Independent Dice ({n_tests} deneme)...")
    combo = {'any': 0, 'improved': 0, 'counts': defaultdict(int)}
    for _ in range(n_tests):
        p = random.choice(parents)
        pe = analyzer.evaluate(p)
        child, ops = analyzer.apply_v6_mutation(deepcopy(p))
        ce = analyzer.evaluate(child)
        if ops:
            combo['any'] += 1
            for op in ops:
                combo['counts'][op] += 1
        if ce['Z1'] > pe['Z1']:
            combo['improved'] += 1
    print(f"   → Mutasyon uygulanan: {combo['any'] / n_tests * 100:.1f}%")
    print(f"   → Z₁ iyileştirme:    {combo['improved'] / n_tests * 100:.1f}%")
    print(f"   → Dağılım: shift={combo['counts']['shift']}, "
          f"swap={combo['counts']['swap']}, blockswap={combo['counts']['blockswap']}")

    print(f"\n   {'─' * 60}")
    print(f"   🎯 OPERATÖR ANALİZİ YORUMU:")
    if cx_results['new'] / n_tests < 0.3:
        print(f"   ❌ Crossover yetersiz çeşitlilik üretiyor")
    else:
        print(f"   ✅ Crossover yeterli çeşitlilik üretiyor")
    for k, s in op_stats.items():
        r = s['new'] / n_tests
        if r < 0.3:
            print(f"   ⚠️ {k}: Düşük değişiklik ({r * 100:.0f}%)")
        else:
            print(f"   ✅ {k}: Yeterli ({r * 100:.0f}%)")

    return {
        'crossover_new_ratio': cx_results['new'] / n_tests,
        'operator_new_ratios': {k: v['new'] / n_tests for k, v in op_stats.items()},
        'operator_z1_improvement': {k: v['z1_up'] / n_tests for k, v in op_stats.items()},
        'combined_mutation_ratio': combo['any'] / n_tests,
    }


def analyze_feasible_space(scenario_data, n_samples=500, seed=42):
    """
    ANALİZ 4: Feasible Uzay Analizi (v6)
    """
    print("\n" + "=" * 70)
    print("📊 ANALİZ 4: FEASİBLE UZAY ANALİZİ (v6)")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    analyzer = ConvergenceAnalyzer(scenario_data)
    z2t, z3t = get_constraint_thresholds()

    raw_valid = 0
    raw_feasible = 0
    z1_vals = []
    z4_vals = []
    unique_sols = set()

    print(f"\n   {n_samples} rastgele çözüm (v6, repair yok)...")
    print(f"   Kısıtlar: Z₂ ≤ {z2t:.4f}, Z₃ ≤ {z3t:.6f}")

    for _ in range(n_samples):
        ind = analyzer.create_random_individual()
        ev = analyzer.evaluate(ind)
        if ev['is_valid']:
            raw_valid += 1
            z1_vals.append(ev['Z1'])
            z4_vals.append(ev['Z4'])
            unique_sols.add(analyzer.solution_to_vector(ind))
            if ev['feasible']:
                raw_feasible += 1

    print(f"\n   → Hard-constraint geçerli: {raw_valid / n_samples * 100:.2f}%")
    print(f"   → Tam feasible:           {raw_feasible / n_samples * 100:.2f}%")
    print(f"   → Unique çözüm:           {len(unique_sols)}/{n_samples}")

    if z1_vals:
        print(f"\n   Z₁: [{min(z1_vals):.2f}% — {max(z1_vals):.2f}%]  "
              f"Ort={np.mean(z1_vals):.2f}% Std={np.std(z1_vals):.2f}%")
    if z4_vals:
        print(f"   Z₄: [{min(z4_vals):.0f} — {max(z4_vals):.0f}]  "
              f"Ort={np.mean(z4_vals):.1f} Std={np.std(z4_vals):.1f}")

    unique_ratio = len(unique_sols) / n_samples if n_samples > 0 else 0
    print(f"\n   {'─' * 60}")
    print(f"   🎯 FEASİBLE UZAY YORUMU:")
    fr = raw_feasible / n_samples
    if fr < 0.05:
        print(f"   ❌ Feasible uzay çok dar ({fr * 100:.1f}%)")
    elif fr < 0.20:
        print(f"   ⚠️ Feasible oranı düşük ({fr * 100:.1f}%)")
    else:
        print(f"   ✅ Feasible uzay yeterli ({fr * 100:.1f}%)")

    return {
        'hard_valid_ratio': raw_valid / n_samples,
        'feasible_ratio': fr,
        'unique_ratio': unique_ratio,
        'unique_count': len(unique_sols),
        'z1_std': np.std(z1_vals) if z1_vals else 0,
        'z4_std': np.std(z4_vals) if z4_vals else 0,
    }


def run_full_analysis(scenario_num=1):
    """Tüm analizleri çalıştır (v6 uyumlu)."""
    print("\n" + "█" * 70)
    print("█" + "   KAPSAMLI YAKINSAMA ANALİZİ (v6)   ".center(68) + "█")
    print("█" + f"   Senaryo {scenario_num}   ".center(68) + "█")
    print("█" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, 'Database.xlsx')
    scenario_data = load_scenario(excel_path, f'S{scenario_num}')

    z2t, z3t = get_constraint_thresholds()
    print(f"\n📂 Senaryo {scenario_num}: {len(scenario_data['doctors'])} personel, "
          f"{scenario_data['days']} gün, {scenario_data['num_shifts']} vardiya")
    print(f"   Kısıtlar: Z₂ ≤ {z2t:.4f}, Z₃ ≤ {z3t:.6f}")

    results = {}
    results['multi_seed'] = analyze_multi_seed(scenario_data, n_seeds=10, generations=50, pop_size=30)
    results['diversity'], results['fitness'] = analyze_diversity_over_generations(
        scenario_data, generations=50, pop_size=30, seed=42)
    results['operators'] = analyze_operator_effectiveness(scenario_data, n_tests=100, seed=42)
    results['feasible_space'] = analyze_feasible_space(scenario_data, n_samples=300, seed=42)

    print("\n" + "█" * 70)
    print("█" + " ÖZET VE ÖNERİLER ".center(68, "═") + "█")
    print("█" * 70)

    problems = []
    recommendations = []

    z1_values = [r['best_Z1'] for r in results['multi_seed']]
    if np.std(z1_values) >= 1.0:
        problems.append("Farklı seed'lerde farklı sonuçlar")
        recommendations.append("Stagnation detection + immigrant injection güçlendir")

    final_unique = results['diversity'][-1]['unique_ratio']
    if final_unique < 0.2:
        problems.append(f"Kritik çeşitlilik kaybı (unique={final_unique * 100:.0f}%)")
        recommendations.append("Diversity collapse tespiti + agresif enjeksiyon")

    op_new = results['operators'].get('operator_new_ratios', {})
    for op_name, ratio in op_new.items():
        if ratio < 0.3:
            problems.append(f"{op_name}: düşük çeşitlilik ({ratio * 100:.0f}%)")

    fs = results['feasible_space']
    if fs['feasible_ratio'] < 0.05:
        problems.append(f"Feasible uzay çok dar ({fs['feasible_ratio'] * 100:.1f}%)")
        recommendations.append("Kısıt eşiklerini gevşet veya complexity düşür")

    print("\n🔴 TESPİT EDİLEN PROBLEMLER:")
    for p in (problems or ["Kritik problem tespit edilmedi"]):
        print(f"   • {p}")
    print("\n🟢 ÖNERİLER:")
    for r in (recommendations or ["Mevcut yapı yeterli"]):
        print(f"   • {r}")
    print("\n" + "█" * 70)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Yakınsama Analizi (v6)')
    parser.add_argument('--scenario', '-s', type=int, default=1)
    args = parser.parse_args()
    run_full_analysis(args.scenario)
