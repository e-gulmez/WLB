# -*- coding: utf-8 -*-
"""
Data Loader Module - ADIM 1
===========================
Excel veritabanından senaryo verilerini yükler.

Versiyon: 1.0
Tarih: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import os


def parse_shift_duration(shift_time: str) -> int:
    """
    Vardiya süresini saat olarak hesaplar.
    
    Örnek:
        "06:00-12:00" -> 6 saat
        "18:00-00:00" -> 6 saat (gece yarısı geçişi)
    """
    try:
        parts = shift_time.split('-')
        start_hour = int(parts[0].split(':')[0])
        end_hour = int(parts[1].split(':')[0])
        
        # Gece yarısı geçişi (örn: 18:00-00:00)
        if end_hour == 0:
            end_hour = 24
            
        duration = end_hour - start_hour
        return duration if duration > 0 else 24 + duration
    except:
        return 6  # Default 6 saat


def parse_personnel_requests(request_str: str) -> List[str]:
    """
    Personel taleplerini parse eder.
    
    Örnek:
        "P1, P2, P3" -> ['P1', 'P2', 'P3']
        "Personel talebi yok" -> []
    """
    if pd.isna(request_str) or str(request_str).strip() == '' or request_str == 'Personel talebi yok':
        return []
    
    personnel = []
    for p in str(request_str).split(','):
        p = p.strip()
        if p.startswith('P') and len(p) > 1:
            # P1, P2, ..., P12 formatını kabul et
            try:
                int(p[1:])  # Sayı mı kontrol et
                personnel.append(p)
            except ValueError:
                continue
    return personnel


def get_available_scenarios(excel_path: str) -> List[str]:
    """
    Excel dosyasındaki mevcut senaryoları döndürür.
    """
    df = pd.read_excel(excel_path, sheet_name='Talep Tablosu')
    scenarios = sorted(df['Senaryo'].unique().tolist())
    return scenarios


def load_scenario(excel_path: str, scenario: str) -> Dict[str, Any]:
    """
    Belirli bir senaryo için tüm verileri yükler.
    
    Args:
        excel_path: Excel dosya yolu
        scenario: Senaryo adı ('S1', 'S2', 'S3')
        
    Returns:
        Dictionary:
            - scenario: Senaryo adı
            - days: Gün sayısı (7)
            - num_shifts: Vardiya sayısı
            - shift_times: Vardiya saat aralıkları
            - shift_durations: Vardiya süreleri (saat)
            - doctors: Personel listesi
            - contract_hours: {personel: haftalık_saat}
            - preferences: {gün: [[vardiya0_talepleri], ...]}
            - manager_requirements: {gün: [vardiya0_kapasite, ...]}
            - total_capacity: Toplam kapasite
    """
    
    # 1. Talep Tablosu
    df_demand = pd.read_excel(excel_path, sheet_name='Talep Tablosu')
    df_scenario = df_demand[df_demand['Senaryo'] == scenario].copy()
    
    if df_scenario.empty:
        raise ValueError(f"Senaryo '{scenario}' bulunamadı!")
    
    # 2. Personel Çalışma Saatleri
    df_hours = pd.read_excel(excel_path, sheet_name='Personel Çalışma Saatleri', header=1)
    df_hours = df_hours[df_hours['PID'].astype(str).str.startswith('P', na=False)].copy()
    
    contract_hours = {}
    for _, row in df_hours.iterrows():
        pid = str(row['PID'])
        hours = int(row['Haftalık Çalışma Süresi'])
        contract_hours[pid] = hours
    
    doctors = sorted(contract_hours.keys(), key=lambda x: int(x[1:]))
    
    # 3. Vardiya Kırılımları
    df_shifts = pd.read_excel(excel_path, sheet_name='Senaryo Vardiya Kırılımları')
    df_shifts_scenario = df_shifts[df_shifts['Senaryo'] == scenario].copy()
    
    shift_times = df_shifts_scenario['Vardiya Saati'].tolist()
    shift_durations = [parse_shift_duration(st) for st in shift_times]
    num_shifts = len(shift_times)
    
    # 4. Gün sayısı
    days = int(df_scenario['Gün'].max())
    
    # 5. Preferences ve Manager Requirements
    preferences = {day: [[] for _ in range(num_shifts)] for day in range(1, days + 1)}
    manager_requirements = {day: [0] * num_shifts for day in range(1, days + 1)}
    
    # Vardiya saati -> indeks mapping
    shift_to_idx = {st: idx for idx, st in enumerate(shift_times)}
    
    for _, row in df_scenario.iterrows():
        day = int(row['Gün'])
        shift_time = row['Vardiya Saati']
        shift_idx = shift_to_idx.get(shift_time, 0)
        
        # Yönetici talebi (kapasite)
        manager_req = int(row['Yönetici Talebi (PS)'])
        manager_requirements[day][shift_idx] = manager_req
        
        # Personel talepleri
        personnel_requests = parse_personnel_requests(row['Personel Talep Edenler (PID)'])
        preferences[day][shift_idx] = personnel_requests
    
    # 6. Toplam kapasite
    total_capacity = sum(sum(reqs) for reqs in manager_requirements.values())
    
    return {
        'scenario': scenario,
        'days': days,
        'num_shifts': num_shifts,
        'shift_times': shift_times,
        'shift_durations': shift_durations,
        'doctors': doctors,
        'contract_hours': contract_hours,
        'preferences': preferences,
        'manager_requirements': manager_requirements,
        'total_capacity': total_capacity
    }


def print_scenario_summary(data: Dict[str, Any]) -> None:
    """
    Senaryo özetini yazdırır.
    """
    print(f"\n{'='*60}")
    print(f"📊 SENARYO {data['scenario']} ÖZETİ")
    print(f"{'='*60}")
    
    print(f"\n📅 Temel Bilgiler:")
    print(f"   Gün sayısı: {data['days']}")
    print(f"   Vardiya sayısı: {data['num_shifts']}")
    print(f"   Personel sayısı: {len(data['doctors'])}")
    print(f"   Toplam kapasite: {data['total_capacity']} kişi-vardiya")
    
    print(f"\n⏰ Vardiya Yapısı:")
    for idx, (time, duration) in enumerate(zip(data['shift_times'], data['shift_durations'])):
        print(f"   V{idx+1}: {time} ({duration} saat)")
    
    print(f"\n👥 Personel Sözleşme Saatleri:")
    for doc in data['doctors']:
        print(f"   {doc}: {data['contract_hours'][doc]} saat/hafta")
    
    print(f"\n📋 Günlük Yönetici Talepleri:")
    for day in range(1, data['days'] + 1):
        reqs = data['manager_requirements'][day]
        print(f"   Gün {day}: {reqs} (Toplam: {sum(reqs)})")
    
    print(f"\n🙋 Günlük Personel Talep Sayıları:")
    for day in range(1, data['days'] + 1):
        prefs = data['preferences'][day]
        counts = [len(p) for p in prefs]
        print(f"   Gün {day}: {counts}")
    
    print(f"{'='*60}\n")


# ============================================
# TEST FONKSİYONU
# ============================================
def test_data_loader(excel_path: str):
    """
    Data loader modülünü test eder.
    """
    print("\n" + "="*60)
    print("🧪 DATA LOADER TEST")
    print("="*60)
    
    # Test 1: Mevcut senaryolar
    print("\n✅ Test 1: Mevcut senaryolar")
    scenarios = get_available_scenarios(excel_path)
    print(f"   Bulunan senaryolar: {scenarios}")
    assert len(scenarios) == 3, "3 senaryo olmalı!"
    assert 'S1' in scenarios and 'S2' in scenarios and 'S3' in scenarios
    print("   ✓ BAŞARILI")
    
    # Test 2: Her senaryo için veri yükleme
    for scenario in scenarios:
        print(f"\n✅ Test 2.{scenario}: {scenario} senaryosu yükleme")
        data = load_scenario(excel_path, scenario)
        
        # Kontroller
        assert data['days'] == 7, "7 gün olmalı!"
        assert len(data['doctors']) == 12, "12 personel olmalı!"
        assert data['num_shifts'] > 0, "En az 1 vardiya olmalı!"
        
        print(f"   Vardiya sayısı: {data['num_shifts']}")
        print(f"   Toplam kapasite: {data['total_capacity']}")
        print(f"   Vardiya süreleri: {data['shift_durations']}")
        print("   ✓ BAŞARILI")
    
    # Test 3: Detaylı S1 kontrolü
    print("\n✅ Test 3: S1 detaylı kontrol")
    s1 = load_scenario(excel_path, 'S1')
    
    # S1: 4 vardiya, her biri 6 saat
    assert s1['num_shifts'] == 4, "S1'de 4 vardiya olmalı!"
    assert all(d == 6 for d in s1['shift_durations']), "S1'de her vardiya 6 saat olmalı!"
    
    # Toplam kapasite kontrolü
    expected_capacity = 66  # Excel'den
    assert s1['total_capacity'] == expected_capacity, f"Toplam kapasite {expected_capacity} olmalı!"
    
    # Gün 1, Vardiya 2 (06:00-12:00) için talep kontrolü
    day1_v2_prefs = s1['preferences'][1][1]  # Gün 1, Vardiya indeks 1
    print(f"   Gün 1, V2 talepleri: {day1_v2_prefs}")
    assert 'P1' in day1_v2_prefs, "P1, Gün 1 V2'yi talep etmeli!"
    
    print("   ✓ BAŞARILI")
    
    # Test 4: Sözleşme saatleri kontrolü
    print("\n✅ Test 4: Sözleşme saatleri")
    expected_hours = {
        'P1': 24, 'P2': 24, 'P3': 24, 'P4': 32, 'P5': 32, 'P6': 40,
        'P7': 45, 'P8': 40, 'P9': 45, 'P10': 32, 'P11': 32, 'P12': 32
    }
    for pid, hours in expected_hours.items():
        actual = s1['contract_hours'].get(pid)
        assert actual == hours, f"{pid} için {hours} saat olmalı, {actual} bulundu!"
    print(f"   Toplam sözleşme: {sum(s1['contract_hours'].values())} saat")
    print("   ✓ BAŞARILI")
    
    print("\n" + "="*60)
    print("✅ TÜM TESTLER BAŞARILI!")
    print("="*60 + "\n")
    
    return True


# Ana çalıştırma
if __name__ == "__main__":
    excel_path = "/mnt/user-data/uploads/Database_28_12_2025.xlsx"
    
    if os.path.exists(excel_path):
        # Testleri çalıştır
        test_data_loader(excel_path)
        
        # Özet göster
        print("\n" + "="*60)
        print("📋 TÜM SENARYOLARIN ÖZETİ")
        print("="*60)
        
        for scenario in ['S1', 'S2', 'S3']:
            data = load_scenario(excel_path, scenario)
            print(f"\n{scenario}:")
            print(f"  Vardiya: {data['num_shifts']} adet, Süreler: {data['shift_durations']}")
            print(f"  Kapasite: {data['total_capacity']} kişi-vardiya")
    else:
        print(f"❌ Excel dosyası bulunamadı: {excel_path}")
