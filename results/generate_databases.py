# -*- coding: utf-8 -*-
"""
Database Generator - 8 Farklı Test Senaryosu
=============================================
Mevcut data_loader.py formatına uygun Excel dosyaları oluşturur.

Test Matrisi:
  - Personel: 24, 30, 75, 100, 250
  - Gün: 14, 21, 30
  
8 Senaryo:
  DB1: 24 personel, 14 gün  (Küçük/Kısa)
  DB2: 24 personel, 30 gün  (Küçük/Uzun)
  DB3: 30 personel, 21 gün  (Orta/Orta)
  DB4: 30 personel, 30 gün  (Orta/Uzun) - Mevcut benchmark
  DB5: 75 personel, 21 gün  (Büyük/Orta)
  DB6: 75 personel, 30 gün  (Büyük/Uzun)
  DB7: 100 personel, 30 gün (Çok Büyük)
  DB8: 250 personel, 30 gün (Ekstrem)

Excel Format:
  - Sheet 1: 'Talep Tablosu' (Senaryo, Gün, Vardiya Saati, Yönetici Talebi (PS), Personel Talep Edenler (PID))
  - Sheet 2: 'Personel Çalışma Saatleri' (PID, Haftalık Çalışma Süresi)
  - Sheet 3: 'Senaryo Vardiya Kırılımları' (Senaryo, Vardiya Saati)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime

# Sabit seed for reproducibility
random.seed(42)
np.random.seed(42)


# ============================================================================
# SENARYO KONFİGÜRASYONLARI
# ============================================================================

SCENARIOS = {
    'DB1': {
        'name': 'Küçük/Kısa',
        'personnel': 24,
        'days': 14,
        'weekly_hours': 40,
        'description': '24 personel, 14 gün'
    },
    'DB2': {
        'name': 'Küçük/Uzun',
        'personnel': 24,
        'days': 30,
        'weekly_hours': 40,
        'description': '24 personel, 30 gün'
    },
    'DB3': {
        'name': 'Orta/Orta',
        'personnel': 30,
        'days': 21,
        'weekly_hours': 40,
        'description': '30 personel, 21 gün'
    },
    'DB4': {
        'name': 'Orta/Uzun',
        'personnel': 30,
        'days': 30,
        'weekly_hours': 40,
        'description': '30 personel, 30 gün (Mevcut Benchmark)'
    },
    'DB5': {
        'name': 'Büyük/Orta',
        'personnel': 75,
        'days': 21,
        'weekly_hours': 40,
        'description': '75 personel, 21 gün'
    },
    'DB6': {
        'name': 'Büyük/Uzun',
        'personnel': 75,
        'days': 30,
        'weekly_hours': 40,
        'description': '75 personel, 30 gün'
    },
    'DB7': {
        'name': 'Çok Büyük',
        'personnel': 100,
        'days': 30,
        'weekly_hours': 40,
        'description': '100 personel, 30 gün'
    },
    'DB8': {
        'name': 'Ekstrem',
        'personnel': 250,
        'days': 30,
        'weekly_hours': 40,
        'description': '250 personel, 30 gün'
    }
}

# Vardiya tanımları - KULLANICININ GERÇEK YAPISI
SHIFT_CONFIGS = {
    # S1: 4 vardiya - 6'şar saat
    'S1': {
        'shifts': ['00:00-06:00', '06:00-12:00', '12:00-18:00', '18:00-24:00'],
        'durations': [6, 6, 6, 6],
        'popularity': [0.75, 1.35, 1.20, 0.85]  # Gece az, gündüz çok tercih
    },
    # S2: 5 vardiya - farklı süreler
    'S2': {
        'shifts': ['00:00-08:00', '08:00-13:00', '13:00-17:00', '17:00-20:00', '20:00-00:00'],
        'durations': [8, 5, 4, 3, 4],
        'popularity': [0.70, 1.40, 1.30, 1.10, 0.80]  # Sabah/öğlen yüksek
    },
    # S3: 6 vardiya - 4'er saat
    'S3': {
        'shifts': ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00'],
        'durations': [4, 4, 4, 4, 4, 4],
        'popularity': [0.65, 0.80, 1.30, 1.25, 1.10, 0.85]
    }
}


def is_weekend(day: int) -> bool:
    return (day % 7) in [5, 6]


def generate_manager_requirements(n_personnel: int, n_days: int, shift_config: dict, seed_offset: int = 0) -> dict:
    random.seed(42 + seed_offset)
    
    shifts = shift_config['shifts']
    popularity = shift_config['popularity']
    n_shifts = len(shifts)
    
    requirements = {}
    
    for day in range(1, n_days + 1):
        day_reqs = {}
        daily_ratio = random.uniform(0.65, 0.75)
        total_daily = int(n_personnel * daily_ratio)
        
        inverse_pop = [1.0 / p for p in popularity]
        total_inv = sum(inverse_pop)
        
        remaining = total_daily
        for i, shift in enumerate(shifts):
            if i == n_shifts - 1:
                req = remaining
            else:
                ratio = inverse_pop[i] / total_inv
                req = int(total_daily * ratio) + random.randint(-1, 1)
                remaining -= req
            
            req = max(1, min(req, n_personnel // 2))
            day_reqs[shift] = req
        
        requirements[day] = day_reqs
    
    return requirements


def generate_personnel_preferences(
    n_personnel: int, 
    n_days: int, 
    shift_config: dict,
    manager_requirements: dict,
    seed_offset: int = 0
) -> dict:
    """
    Personel tercihlerini oluşturur.
    
    Hedef: 
    - Toplam personel tercihi ≈ %105-110 Yönetici talebi
    - Her personel benzer sayıda tercih yapar (Z2 için)
    
    Returns:
        {day: {shift_time: [P1, P2, ...]}}
    """
    random.seed(42 + seed_offset)
    
    shifts = shift_config['shifts']
    popularity = shift_config['popularity']
    n_shifts = len(shifts)
    
    # Toplam yönetici talebi
    total_manager_demand = sum(
        sum(manager_requirements[day].values()) 
        for day in range(1, n_days + 1)
    )
    
    # Hedef toplam personel tercihi
    target_total_prefs = int(total_manager_demand * 1.08)
    
    # Her personel için hedef tercih sayısı (dengeli dağılım)
    base_prefs_per_person = target_total_prefs // n_personnel
    
    # Personel tipleri (vardiya tercihi için)
    personnel_types = {}
    for p in range(1, n_personnel + 1):
        ptype = random.choices(
            ['morning', 'afternoon', 'evening', 'flexible'],
            weights=[0.35, 0.30, 0.15, 0.20]
        )[0]
        personnel_types[f'P{p}'] = ptype
    
    type_shift_weights = {
        'morning': {0: 0.5, 1: 1.8, 2: 1.2, 3: 0.5, 4: 0.4, 5: 0.3},
        'afternoon': {0: 0.4, 1: 1.0, 2: 1.6, 3: 1.2, 4: 0.8, 5: 0.5},
        'evening': {0: 0.6, 1: 0.6, 2: 0.8, 3: 1.4, 4: 1.5, 5: 1.3},
        'flexible': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
    }
    
    # Her personel için tercih sayısı (±%15 varyasyon - Z2 için kontrollü)
    personnel_pref_counts = {}
    for p in range(1, n_personnel + 1):
        # Varyasyon sınırlı: ±15%
        variation = random.uniform(0.85, 1.15)
        personnel_pref_counts[f'P{p}'] = int(base_prefs_per_person * variation)
    
    # Gün-vardiya slotlarını oluştur
    all_slots = []
    for day in range(1, n_days + 1):
        for shift in shifts:
            # Her slot için ağırlık (yönetici talebiyle orantılı)
            weight = manager_requirements[day][shift]
            all_slots.append((day, shift, weight))
    
    # Tercihleri oluştur
    preferences = {day: {shift: [] for shift in shifts} for day in range(1, n_days + 1)}
    
    for p in range(1, n_personnel + 1):
        pid = f'P{p}'
        ptype = personnel_types[pid]
        target_count = personnel_pref_counts[pid]
        
        # Bu personel için slot ağırlıkları
        weighted_slots = []
        for day, shift, base_weight in all_slots:
            shift_idx = shifts.index(shift)
            type_weight = type_shift_weights[ptype].get(shift_idx, 1.0)
            pop_weight = popularity[shift_idx]
            
            # Hafta sonu azalt
            weekend_factor = 0.7 if is_weekend(day) else 1.0
            
            final_weight = base_weight * type_weight * pop_weight * weekend_factor
            weighted_slots.append((day, shift, final_weight))
        
        # Ağırlıklı rastgele seçim
        total_weight = sum(w for _, _, w in weighted_slots)
        
        selected_count = 0
        selected_slots = set()
        
        # Hedef sayıya ulaşana kadar seç
        attempts = 0
        while selected_count < target_count and attempts < target_count * 3:
            attempts += 1
            
            # Rastgele seçim (ağırlıklı)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for day, shift, weight in weighted_slots:
                cumulative += weight
                if r <= cumulative:
                    slot_key = (day, shift)
                    if slot_key not in selected_slots:
                        selected_slots.add(slot_key)
                        preferences[day][shift].append(pid)
                        selected_count += 1
                    break
    
    return preferences


def create_database_excel(scenario_key: str, output_dir: str = None) -> str:
    config = SCENARIOS[scenario_key]
    
    n_personnel = config['personnel']
    n_days = config['days']
    weekly_hours = config['weekly_hours']
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    filename = f"Database_{scenario_key}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    # ========== TALEP TABLOSU ==========
    talep_rows = []
    
    # S1: 4 vardiya (6'şar saat)
    shift_config_s1 = SHIFT_CONFIGS['S1']
    manager_reqs_s1 = generate_manager_requirements(n_personnel, n_days, shift_config_s1, seed_offset=0)
    prefs_s1 = generate_personnel_preferences(n_personnel, n_days, shift_config_s1, manager_reqs_s1, seed_offset=0)
    
    for day in range(1, n_days + 1):
        for shift in shift_config_s1['shifts']:
            req = manager_reqs_s1[day][shift]
            personnel_list = prefs_s1[day][shift]
            personnel_str = ', '.join(personnel_list) if personnel_list else 'Personel talebi yok'
            
            talep_rows.append({
                'Senaryo': 'S1',
                'Gün': day,
                'Vardiya Saati': shift,
                'Yönetici Talebi (PS)': req,
                'Personel Talep Edenler (PID)': personnel_str
            })
    
    # S2: 5 vardiya (8, 5, 4, 3, 4 saat)
    shift_config_s2 = SHIFT_CONFIGS['S2']
    manager_reqs_s2 = generate_manager_requirements(n_personnel, n_days, shift_config_s2, seed_offset=100)
    prefs_s2 = generate_personnel_preferences(n_personnel, n_days, shift_config_s2, manager_reqs_s2, seed_offset=100)
    
    for day in range(1, n_days + 1):
        for shift in shift_config_s2['shifts']:
            req = manager_reqs_s2[day][shift]
            personnel_list = prefs_s2[day][shift]
            personnel_str = ', '.join(personnel_list) if personnel_list else 'Personel talebi yok'
            
            talep_rows.append({
                'Senaryo': 'S2',
                'Gün': day,
                'Vardiya Saati': shift,
                'Yönetici Talebi (PS)': req,
                'Personel Talep Edenler (PID)': personnel_str
            })
    
    # S3: 6 vardiya (4'er saat)
    shift_config_s3 = SHIFT_CONFIGS['S3']
    manager_reqs_s3 = generate_manager_requirements(n_personnel, n_days, shift_config_s3, seed_offset=200)
    prefs_s3 = generate_personnel_preferences(n_personnel, n_days, shift_config_s3, manager_reqs_s3, seed_offset=200)
    
    for day in range(1, n_days + 1):
        for shift in shift_config_s3['shifts']:
            req = manager_reqs_s3[day][shift]
            personnel_list = prefs_s3[day][shift]
            personnel_str = ', '.join(personnel_list) if personnel_list else 'Personel talebi yok'
            
            talep_rows.append({
                'Senaryo': 'S3',
                'Gün': day,
                'Vardiya Saati': shift,
                'Yönetici Talebi (PS)': req,
                'Personel Talep Edenler (PID)': personnel_str
            })
    
    df_talep = pd.DataFrame(talep_rows)
    
    # ========== PERSONEL ÇALIŞMA SAATLERİ ==========
    # Gerçekçi sözleşme saatleri: 36, 38, 40, 42, 45
    # Dağılım: 36 (%21), 38 (%8), 40 (%42), 42 (%25), 45 (%4)
    CONTRACT_HOURS = [36, 38, 40, 42, 45]
    CONTRACT_WEIGHTS = [0.21, 0.08, 0.42, 0.25, 0.04]
    
    personel_rows = [{'PID': 'PID', 'Haftalık Çalışma Süresi': 'Haftalık Çalışma Süresi'}]
    
    random.seed(42)
    for p in range(1, n_personnel + 1):
        hours = random.choices(CONTRACT_HOURS, weights=CONTRACT_WEIGHTS, k=1)[0]
        personel_rows.append({
            'PID': f'P{p}',
            'Haftalık Çalışma Süresi': hours
        })
    
    df_personel = pd.DataFrame(personel_rows)
    df_personel.columns = ['Personel Sözleşme Çalışma Saatleri', 'Unnamed: 1']
    
    # ========== SENARYO VARDIYA KIRIMLARI ==========
    kirilim_rows = []
    
    for shift in shift_config_s1['shifts']:
        kirilim_rows.append({'Senaryo': 'S1', 'Vardiya Saati': shift})
    
    for shift in shift_config_s2['shifts']:
        kirilim_rows.append({'Senaryo': 'S2', 'Vardiya Saati': shift})
    
    for shift in shift_config_s3['shifts']:
        kirilim_rows.append({'Senaryo': 'S3', 'Vardiya Saati': shift})
    
    df_kirilim = pd.DataFrame(kirilim_rows)
    
    # ========== EXCEL'E KAYDET ==========
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df_talep.to_excel(writer, sheet_name='Talep Tablosu', index=False)
        df_personel.to_excel(writer, sheet_name='Personel Çalışma Saatleri', index=False)
        df_kirilim.to_excel(writer, sheet_name='Senaryo Vardiya Kırılımları', index=False)
    
    return filepath


def calculate_statistics(filepath: str) -> dict:
    df_talep = pd.read_excel(filepath, sheet_name='Talep Tablosu')
    
    stats = {}
    
    for scenario in ['S1', 'S2', 'S3']:
        df_s = df_talep[df_talep['Senaryo'] == scenario]
        
        manager_demand = df_s['Yönetici Talebi (PS)'].sum()
        
        personnel_demand = 0
        for _, row in df_s.iterrows():
            prefs = row['Personel Talep Edenler (PID)']
            if prefs != 'Personel talebi yok' and pd.notna(prefs):
                personnel_demand += len([p for p in str(prefs).split(',') if p.strip().startswith('P')])
        
        ratio = personnel_demand / manager_demand if manager_demand > 0 else 0
        
        stats[scenario] = {
            'manager_demand': manager_demand,
            'personnel_demand': personnel_demand,
            'ratio': ratio
        }
    
    return stats


def generate_all_databases(output_dir: str = None) -> dict:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'databases')
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print("\n" + "="*70)
    print("📊 DATABASE GENERATOR - 8 Test Senaryosu")
    print("="*70)
    
    for key, config in SCENARIOS.items():
        print(f"\n🔄 {key}: {config['description']}")
        
        random.seed(42)
        np.random.seed(42)
        
        filepath = create_database_excel(key, output_dir)
        
        stats = calculate_statistics(filepath)
        
        results[key] = {
            'filepath': filepath,
            'personnel': config['personnel'],
            'days': config['days'],
            'stats': stats
        }
        
        print(f"   ✓ Personel: {config['personnel']}, Gün: {config['days']}")
        for scenario, s in stats.items():
            print(f"   ✓ {scenario}: Yön={s['manager_demand']}, Per={s['personnel_demand']}, Oran={s['ratio']:.1%}")
        print(f"   ✓ Dosya: {filepath}")
    
    print("\n" + "="*70)
    print("✅ Tüm database'ler oluşturuldu!")
    print("="*70)
    
    print("\n📋 ÖZET TABLO (S1 Senaryosu):")
    print("-" * 80)
    print(f"{'DB':<6} {'Personel':>10} {'Gün':>6} {'Yön.Talep':>12} {'Per.Talep':>12} {'Oran':>10}")
    print("-" * 80)
    for key, data in results.items():
        s1 = data['stats']['S1']
        print(f"{key:<6} {data['personnel']:>10} {data['days']:>6} {s1['manager_demand']:>12} {s1['personnel_demand']:>12} {s1['ratio']:>10.1%}")
    print("-" * 80)
    
    return results


if __name__ == '__main__':
    results = generate_all_databases()
