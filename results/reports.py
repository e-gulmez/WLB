# -*- coding: utf-8 -*-
"""
Reports Module - VERSİYON 1
===========================
Excel raporları oluşturma (2 Amaç + 2 Kısıt modeli).

7 Rapor:
    1. Talep Karşılama Oranı
    2. Talepsiz Atamalar
    3. Yönetici Talepleri
    4. Personel Atamaları
    5. Vardiya Atamaları
    6. Çok Amaçlı Performans (2 Amaç + 2 Kısıt)
    7. Adalet Analizi
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from objectives import calculate_individual_satisfaction
from constraints import calculate_working_hours


def doctor_prefers(doc: str, day: int, shift: int, preferences: Dict) -> bool:
    """Personelin vardiyayı tercih edip etmediğini kontrol eder."""
    day_key = day + 1
    prefs = preferences.get(day_key, [])
    if shift < len(prefs):
        return doc in prefs[shift]
    return False


def report1_satisfaction(
    assignments: List[List[List[str]]],
    preferences: Dict[int, List[List[str]]],
    doctors: List[str],
    days: int,
    num_shifts: int
) -> pd.DataFrame:
    """Rapor 1: Talep Karşılama Oranı"""
    p_values = calculate_individual_satisfaction(
        assignments, preferences, doctors, days, num_shifts
    )
    
    data = []
    for doc in sorted(doctors, key=lambda x: int(x[1:])):
        requested = 0
        assigned_pref = 0
        total_assigned = 0
        
        for day in range(days):
            day_key = day + 1
            for shift in range(num_shifts):
                if doc in preferences.get(day_key, [[]] * num_shifts)[shift]:
                    requested += 1
                    if day < len(assignments) and shift < len(assignments[day]):
                        if doc in assignments[day][shift]:
                            assigned_pref += 1
                
                if day < len(assignments) and shift < len(assignments[day]):
                    if doc in assignments[day][shift]:
                        total_assigned += 1
        
        data.append({
            'Personel': doc,
            'Talep Edilen': requested,
            'Toplam Atanan': total_assigned,
            'Tercih Karşılanan': assigned_pref,
            'Karşılama Oranı (%)': f"{p_values.get(doc, 0) * 100:.1f}"
        })
    
    return pd.DataFrame(data)


def report2_non_preferred(
    assignments: List[List[List[str]]],
    preferences: Dict[int, List[List[str]]],
    shift_times: List[str],
    doctors: List[str],
    days: int,
    num_shifts: int
) -> pd.DataFrame:
    """Rapor 2: Talepsiz Atamalar"""
    data = []
    
    for day in range(days):
        for shift in range(num_shifts):
            if day >= len(assignments) or shift >= len(assignments[day]):
                continue
            
            for doc in set(assignments[day][shift]):
                if doc not in doctors:
                    continue
                
                if not doctor_prefers(doc, day, shift, preferences):
                    shift_time = shift_times[shift] if shift < len(shift_times) else f"V{shift+1}"
                    data.append({
                        'Personel': doc,
                        'Gün': day + 1,
                        'Vardiya': shift + 1,
                        'Saat': shift_time
                    })
    
    if not data:
        return pd.DataFrame(columns=['Personel', 'Gün', 'Vardiya', 'Saat'])
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Personel', 'Gün', 'Vardiya'])
    return df


def report3_manager_requirements(
    assignments: List[List[List[str]]],
    manager_requirements: Dict[int, List[int]],
    preferences: Dict[int, List[List[str]]],
    shift_times: List[str],
    days: int,
    num_shifts: int
) -> pd.DataFrame:
    """Rapor 3: Yönetici Talepleri"""
    data = []
    
    for day in range(days):
        day_key = day + 1
        
        for shift in range(num_shifts):
            required = manager_requirements.get(day_key, [0] * num_shifts)[shift]
            
            if day < len(assignments) and shift < len(assignments[day]):
                assigned_count = len(set(assignments[day][shift]))
            else:
                assigned_count = 0
            
            requesting = preferences.get(day_key, [[]] * num_shifts)[shift]
            request_count = len([p for p in requesting if p])
            
            if required >= 2:
                if request_count > required:
                    lower, upper = required, required + 1
                elif request_count < required:
                    lower, upper = max(1, required - 1), required
                else:
                    lower, upper = max(1, required - 1), required + 1
            else:
                lower, upper = 1, 1
            
            if assigned_count == required:
                status = "✓ Tam"
            elif lower <= assigned_count <= upper:
                if assigned_count > required:
                    status = f"↑ +1 ({assigned_count})"
                else:
                    status = f"↓ -1 ({assigned_count})"
            else:
                status = f"✗ Hata ({assigned_count})"
            
            shift_time = shift_times[shift] if shift < len(shift_times) else f"V{shift+1}"
            
            data.append({
                'Gün': day + 1,
                'Vardiya': shift_time,
                'Gerekli': required,
                'Talep Eden': request_count,
                'Atanan': assigned_count,
                'Durum': status
            })
    
    return pd.DataFrame(data)


def report4_personnel_assignments(
    assignments: List[List[List[str]]],
    preferences: Dict[int, List[List[str]]],
    shift_times: List[str],
    doctors: List[str],
    days: int,
    num_shifts: int
) -> pd.DataFrame:
    """Rapor 4: Personel Atamaları"""
    data = []
    
    for day in range(days):
        for shift in range(num_shifts):
            if day >= len(assignments) or shift >= len(assignments[day]):
                continue
            
            for doc in set(assignments[day][shift]):
                if doc not in doctors:
                    continue
                
                is_pref = doctor_prefers(doc, day, shift, preferences)
                status = "✓ Tercih" if is_pref else "○ Talepsiz"
                shift_time = shift_times[shift] if shift < len(shift_times) else f"V{shift+1}"
                
                data.append({
                    'Personel': doc,
                    'Gün': day + 1,
                    'Vardiya': shift_time,
                    'Durum': status
                })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df['sort_key'] = df['Personel'].apply(lambda x: int(x[1:]))
        df = df.sort_values(['sort_key', 'Gün']).drop('sort_key', axis=1)
    return df


def report5_shift_assignments(
    assignments: List[List[List[str]]],
    preferences: Dict[int, List[List[str]]],
    shift_times: List[str],
    days: int,
    num_shifts: int
) -> pd.DataFrame:
    """Rapor 5: Vardiya Atamaları"""
    data = []
    
    for day in range(days):
        day_key = day + 1
        
        for shift in range(num_shifts):
            requesting = preferences.get(day_key, [[]] * num_shifts)[shift]
            requested = [p for p in requesting if p]
            
            if day < len(assignments) and shift < len(assignments[day]):
                assigned = sorted(set(assignments[day][shift]))
            else:
                assigned = []
            
            shift_time = shift_times[shift] if shift < len(shift_times) else f"V{shift+1}"
            
            data.append({
                'Gün': day + 1,
                'Vardiya': shift_time,
                'Talep Edenler': ', '.join(sorted(requested, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)) if requested else '-',
                'Atananlar': ', '.join(sorted(assigned, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)) if assigned else '-',
                'Talep Sayısı': len(requested),
                'Atanan Sayısı': len(assigned)
            })
    
    return pd.DataFrame(data)


def report6_multi_objective_performance_v1(
    results: Dict[str, Any],
    algorithm: str
) -> pd.DataFrame:
    """Rapor 6: Çok Amaçlı Performans (VERSİYON 1 - 2 Amaç + 2 Kısıt)"""
    data = []
    
    data.append({'Metrik': 'ALGORİTMA', 'Değer': algorithm})
    data.append({'Metrik': 'VERSİYON', 'Değer': 'V1 (2 Amaç + 2 Kısıt)'})
    data.append({'Metrik': '─' * 30, 'Değer': '─' * 30})
    
    data.append({'Metrik': '═══ AMAÇ FONKSİYONLARI ═══', 'Değer': ''})
    data.append({'Metrik': 'Z₁ Ort. Memnuniyet (%) [MAX]', 'Değer': f"{results.get('Z1', 0):.2f}"})
    data.append({'Metrik': 'Z₄ Kapasite Sapması [MIN]', 'Değer': f"{results.get('Z4', 0):.0f}"})
    
    data.append({'Metrik': '─' * 30, 'Değer': '─' * 30})
    data.append({'Metrik': '═══ KISITLAR (Raporlama) ═══', 'Değer': ''})
    data.append({'Metrik': 'Z₂ Adalet Farkı', 'Değer': f"{results.get('Z2', 0):.4f}"})
    data.append({'Metrik': 'Z₃ İş Yükü Varyansı', 'Değer': f"{results.get('Z3', 0):.6f}"})
    
    std_dev = np.sqrt(results.get('Z3', 0))
    data.append({'Metrik': 'Standart Sapma (%)', 'Değer': f"{std_dev * 100:.2f}"})
    
    data.append({'Metrik': '─' * 30, 'Değer': '─' * 30})
    data.append({'Metrik': '═══ DETAYLAR ═══', 'Değer': ''})
    data.append({'Metrik': 'Min Memnuniyet (%)', 'Değer': f"{results.get('min_satisfaction', 0) * 100:.1f}"})
    data.append({'Metrik': 'Max Memnuniyet (%)', 'Değer': f"{results.get('max_satisfaction', 0) * 100:.1f}"})
    data.append({'Metrik': 'Fazla Personel', 'Değer': f"+{results.get('over_staffed', 0)}"})
    data.append({'Metrik': 'Eksik Personel', 'Değer': f"-{results.get('under_staffed', 0)}"})
    
    data.append({'Metrik': '─' * 30, 'Değer': '─' * 30})
    data.append({'Metrik': '═══ ALGORİTMA PARAMETRELERİ ═══', 'Değer': ''})
    
    if algorithm == 'GA':
        data.append({'Metrik': 'Fitness', 'Değer': f"{results.get('best_fitness', 0):.4f}"})
        weights = results.get('weights', {})
        if isinstance(weights, dict):
            data.append({'Metrik': 'w₁ (Z₁ Ağırlığı)', 'Değer': f"{weights.get('w1', 0):.2f}"})
            data.append({'Metrik': 'w₄ (Z₄ Ağırlığı)', 'Değer': f"{weights.get('w4', 0):.2f}"})
    
    data.append({'Metrik': 'Popülasyon', 'Değer': str(results.get('population_size', '-'))})
    data.append({'Metrik': 'Nesil', 'Değer': str(results.get('generations', '-'))})
    data.append({'Metrik': 'Süre (sn)', 'Değer': f"{results.get('elapsed_time', 0):.2f}"})
    
    if algorithm in ['NSGA-III', 'NSGA3']:
        data.append({'Metrik': '─' * 30, 'Değer': '─' * 30})
        data.append({'Metrik': '═══ PARETO FRONT ═══', 'Değer': ''})
        data.append({'Metrik': 'Pareto Çözüm Sayısı', 'Değer': str(results.get('pareto_size', 0))})
        data.append({'Metrik': 'Referans Noktası Sayısı', 'Değer': str(results.get('n_reference_points', '-'))})
        
        pareto_obj = results.get('pareto_objectives', [])
        # Numpy array veya list olabilir, her ikisi için de çalışan kontrol
        if pareto_obj is not None and len(pareto_obj) > 0:
            obj_array = np.array(pareto_obj)
            data.append({'Metrik': 'Z₁ Aralığı (%)', 'Değer': f"[{obj_array[:,0].min():.2f} - {obj_array[:,0].max():.2f}]"})
            data.append({'Metrik': 'Z₄ Aralığı', 'Değer': f"[{obj_array[:,1].min():.0f} - {obj_array[:,1].max():.0f}]"})
    
    return pd.DataFrame(data)


def report7_fairness_analysis(
    assignments: List[List[List[str]]],
    preferences: Dict[int, List[List[str]]],
    contract_hours: Dict[str, int],
    shift_durations: List[int],
    doctors: List[str],
    days: int,
    num_shifts: int
) -> pd.DataFrame:
    """Rapor 7: Adalet Analizi"""
    p_values = calculate_individual_satisfaction(
        assignments, preferences, doctors, days, num_shifts
    )
    
    weekly_hours, _ = calculate_working_hours(
        assignments, shift_durations, doctors, days, num_shifts
    )
    
    data = []
    for doc in sorted(doctors, key=lambda x: int(x[1:])):
        p = p_values.get(doc, 0)
        hours = weekly_hours.get(doc, 0)
        contract = contract_hours.get(doc, 0)
        
        if p >= 0.8:
            cat = "🟢 Yüksek"
        elif p >= 0.5:
            cat = "🟡 Orta"
        elif p > 0:
            cat = "🟠 Düşük"
        else:
            cat = "🔴 Yok"
        
        usage = (hours / contract * 100) if contract > 0 else 0
        
        data.append({
            'Personel': doc,
            'Memnuniyet (%)': f"{p * 100:.1f}",
            'Kategori': cat,
            'Çalışma (saat)': hours,
            'Sözleşme (saat)': contract,
            'Kullanım (%)': f"{usage:.1f}"
        })
    
    values = list(p_values.values())
    if values:
        data.append({'Personel': '─' * 15, 'Memnuniyet (%)': '─' * 10, 'Kategori': '─' * 12, 'Çalışma (saat)': '─', 'Sözleşme (saat)': '─', 'Kullanım (%)': '─'})
        data.append({'Personel': 'ORTALAMA', 'Memnuniyet (%)': f"{np.mean(values) * 100:.1f}", 'Kategori': '', 'Çalışma (saat)': '', 'Sözleşme (saat)': '', 'Kullanım (%)': ''})
        data.append({'Personel': 'STD SAPMA', 'Memnuniyet (%)': f"{np.std(values) * 100:.1f}", 'Kategori': '', 'Çalışma (saat)': '', 'Sözleşme (saat)': '', 'Kullanım (%)': ''})
        data.append({'Personel': 'MIN', 'Memnuniyet (%)': f"{min(values) * 100:.1f}", 'Kategori': '', 'Çalışma (saat)': '', 'Sözleşme (saat)': '', 'Kullanım (%)': ''})
        data.append({'Personel': 'MAX', 'Memnuniyet (%)': f"{max(values) * 100:.1f}", 'Kategori': '', 'Çalışma (saat)': '', 'Sözleşme (saat)': '', 'Kullanım (%)': ''})
        data.append({'Personel': 'ADALET FARKI (Z₂)', 'Memnuniyet (%)': f"{(max(values) - min(values)):.4f}", 'Kategori': '', 'Çalışma (saat)': '', 'Sözleşme (saat)': '', 'Kullanım (%)': ''})
        data.append({'Personel': 'VARYANS (Z₃)', 'Memnuniyet (%)': f"{np.var(values):.6f}", 'Kategori': '', 'Çalışma (saat)': '', 'Sözleşme (saat)': '', 'Kullanım (%)': ''})
    
    return pd.DataFrame(data)


def report8_mutation_rates(
    history: Dict[str, List],
    algorithm: str
) -> pd.DataFrame:
    """Rapor 8: Adaptive Mutasyon Oranları Değişimi"""
    generations = history.get('generation', [])
    p_shift = history.get('p_shift', [])
    p_swap = history.get('p_swap', [])
    p_bs = history.get('p_blockswap', [])

    if not generations or not p_shift:
        return pd.DataFrame({'Bilgi': ['Mutasyon verisi bulunamadı']})

    n = min(len(generations), len(p_shift), len(p_swap), len(p_bs))

    combined = [
        1.0 - (1.0 - p_shift[i]) * (1.0 - p_swap[i]) * (1.0 - p_bs[i])
        for i in range(n)
    ]

    # Detaylı tablo: her 25 nesilde bir + ilk ve son
    sample_indices = [0]
    step = max(1, n // 20)
    for i in range(step, n - 1, step):
        sample_indices.append(i)
    if n - 1 not in sample_indices:
        sample_indices.append(n - 1)

    data = []
    for i in sample_indices:
        data.append({
            'Nesil': generations[i],
            'p_shift': round(p_shift[i], 4),
            'p_swap': round(p_swap[i], 4),
            'p_blockswap': round(p_bs[i], 4),
            'Combined P': round(combined[i], 4),
        })

    # Özet satırları
    data.append({})
    data.append({
        'Nesil': 'ÖZET',
        'p_shift': '',
        'p_swap': '',
        'p_blockswap': '',
        'Combined P': '',
    })
    stats = [
        ('Başlangıç', 0),
        ('Bitiş', n - 1),
    ]
    for label, idx in stats:
        data.append({
            'Nesil': label,
            'p_shift': round(p_shift[idx], 4),
            'p_swap': round(p_swap[idx], 4),
            'p_blockswap': round(p_bs[idx], 4),
            'Combined P': round(combined[idx], 4),
        })

    # Max/min
    data.append({
        'Nesil': 'Max p_shift',
        'p_shift': round(max(p_shift[:n]), 4),
        'p_swap': '',
        'p_blockswap': '',
        'Combined P': '',
    })
    data.append({
        'Nesil': 'Max p_swap',
        'p_shift': '',
        'p_swap': round(max(p_swap[:n]), 4),
        'p_blockswap': '',
        'Combined P': '',
    })
    data.append({
        'Nesil': 'Max p_blockswap',
        'p_shift': '',
        'p_swap': '',
        'p_blockswap': round(max(p_bs[:n]), 4),
        'Combined P': '',
    })
    data.append({
        'Nesil': 'Max Combined',
        'p_shift': '',
        'p_swap': '',
        'p_blockswap': '',
        'Combined P': round(max(combined), 4),
    })

    # Dominant operator
    shift_area = sum(p_shift[:n])
    swap_area = sum(p_swap[:n])
    bs_area = sum(p_bs[:n])
    total_area = shift_area + swap_area + bs_area
    if total_area > 0:
        data.append({
            'Nesil': 'Shift Katkısı (%)',
            'p_shift': f"{shift_area / total_area * 100:.1f}%",
            'p_swap': '',
            'p_blockswap': '',
            'Combined P': '',
        })
        data.append({
            'Nesil': 'Swap Katkısı (%)',
            'p_shift': '',
            'p_swap': f"{swap_area / total_area * 100:.1f}%",
            'p_blockswap': '',
            'Combined P': '',
        })
        data.append({
            'Nesil': 'BlockSwap Katkısı (%)',
            'p_shift': '',
            'p_swap': '',
            'p_blockswap': f"{bs_area / total_area * 100:.1f}%",
            'Combined P': '',
        })

    # Floor ulaşım nesli
    floor_shift = next((generations[i] for i in range(n) if p_shift[i] <= 0.08 + 1e-9), '-')
    floor_swap = next((generations[i] for i in range(n) if p_swap[i] <= 0.08 + 1e-9), '-')
    floor_bs = next((generations[i] for i in range(n) if p_bs[i] <= 0.05 + 1e-9), '-')
    data.append({
        'Nesil': 'Floor Ulaşım Nesli',
        'p_shift': floor_shift,
        'p_swap': floor_swap,
        'p_blockswap': floor_bs,
        'Combined P': '',
    })

    return pd.DataFrame(data)


def generate_all_reports(
    assignments: List[List[List[str]]],
    scenario_data: Dict[str, Any],
    results: Dict[str, Any],
    algorithm: str,
    output_path: str,
    history: Dict[str, List] = None
) -> str:
    """Tüm raporları Excel dosyasına yazar."""
    days = scenario_data['days']
    num_shifts = scenario_data['num_shifts']
    shift_times = scenario_data['shift_times']
    doctors = scenario_data['doctors']
    preferences = scenario_data['preferences']
    manager_requirements = scenario_data['manager_requirements']
    contract_hours = scenario_data['contract_hours']
    shift_durations = scenario_data['shift_durations']
    
    reports = {
        '1_Talep_Karsilama': report1_satisfaction(
            assignments, preferences, doctors, days, num_shifts
        ),
        '2_Talepsiz_Atamalar': report2_non_preferred(
            assignments, preferences, shift_times, doctors, days, num_shifts
        ),
        '3_Yonetici_Talepleri': report3_manager_requirements(
            assignments, manager_requirements, preferences, shift_times, days, num_shifts
        ),
        '4_Personel_Atamalari': report4_personnel_assignments(
            assignments, preferences, shift_times, doctors, days, num_shifts
        ),
        '5_Vardiya_Atamalari': report5_shift_assignments(
            assignments, preferences, shift_times, days, num_shifts
        ),
        '6_Cok_Amacli_Performans': report6_multi_objective_performance_v1(
            results, algorithm
        ),
        '7_Adalet_Analizi': report7_fairness_analysis(
            assignments, preferences, contract_hours, shift_durations,
            doctors, days, num_shifts
        )
    }

    if history and any(history.get(k) for k in ('p_shift', 'p_swap', 'p_blockswap')):
        reports['8_Mutasyon_Oranlari'] = report8_mutation_rates(history, algorithm)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in reports.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return output_path


if __name__ == "__main__":
    print("Reports Module V1 - 2 Amaç + 2 Kısıt Modeli")
