# WLB Optimization - Versiyon 3 (Dinamik Eliminasyon)

## 🎯 Amaç

Bu proje, personel vardiya çizelgeleme optimizasyonu için GA (Genetik Algoritma) ve NSGA-III algoritmalarını kullanır.

## ⚠️ Versiyon Geçmişi

### V1 → V2 (Model Düzeltme):
- Z1, Z2, Z4 → 3 Amaç ❌
- Z1, Z4 → **2 Amaç** | Z2, Z3 → **2 Kısıt** ✓

### V2 → V3 (Algoritma Düzeltme):
- Ada (Island) modeli **KALDIRILDI** (kaçış mekanizması yerine kök neden çözümü)
- Kötü kromozom eliminasyonu **statik → dinamik** (Softmax + Sıcaklık)
- Belirli oranda kötü kromozoma izin verilir (çeşitlilik koruması)
- Pareto grafikleri **akademik formata** güncellendi

---

## 📊 Model Tanımı

### Amaç Fonksiyonları

| Fonksiyon | Açıklama | Yön |
|-----------|----------|-----|
| **Z1** | Ortalama Memnuniyet | MAXIMIZE |
| **Z4** | Kapasite Sapması | MINIMIZE |

### Kısıtlar (Eşik Değerli)

| Kısıt | Açıklama | Koşul |
|-------|----------|-------|
| **Z2** | Adalet Farkı (max-min) | Z2 ≤ threshold |
| **Z3** | İş Yükü Varyansı | Z3 ≤ threshold |

---

## 🧬 V3: Softmax + Sıcaklık Dinamik Eliminasyon

### Neden Ada Modeli Kaldırıldı?

Ada modeli, erken yakınsama sorununu "maskeliyordu" ancak çözmüyordu:
- 120 bireyi 5 adaya bölmek her adada yalnızca 24 birey bırakıyordu
- Küçük alt-popülasyonlar zaten çeşitlilik açısından zayıftı
- Gerçek sorun: kötü kromozomların agresif eliminasyonu → çeşitlilik kaybı

### Softmax + Sıcaklık Yaklaşımı

**Temel Fikir:** Simulated Annealing mantığıyla kromozom eliminasyonu.

**Sıcaklık Zamanlaması:**
```
T(gen) = T_min + (T_max - T_min) × (1 - gen/max_gen)^α
```

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| T_max | 5.0 | Başlangıç sıcaklığı (yüksek → uniform eleme) |
| T_min | 0.1 | Son sıcaklık (düşük → keskin eleme) |
| α | 2.0 | Soğuma hızı |

**Eliminasyon Olasılığı:**
```
P_elim(i) = softmax(-fitness(i) / T)
```

- **Yüksek T** → Kötü ve iyi arasında fark az → Çeşitlilik korunur
- **Düşük T** → Kötüler çok yüksek olasılıkla elenir → Yakınsama

**Dinamik Kötü Kromozom Oranı:**
```
İzin verilen kötü oran: %30 (başlangıç) → %5 (son)
```

### Süreç

1. Sıcaklık ve izin verilen kötü oranı hesapla
2. Medyan altı "kötü" kromozomları belirle
3. Fazlalıkları softmax olasılığına göre stokastik seç
4. Seçilenleri yeni bireylerle değiştir
5. **Elitler ASLA elenmez**

---

## 🔧 Complexity Seviyeleri

| Seviye | Z2 Eşik | Z3 Eşik | Açıklama |
|--------|---------|---------|----------|
| EASY | 0.35 | 0.040 | Kolay |
| MEDIUM | 0.25 | 0.025 | Orta |
| **HARD** | 0.15 | 0.015 | Zor (Varsayılan) |
| VERY_HARD | 0.10 | 0.008 | Çok Zor |
| EXTREME | 0.05 | 0.003 | Ekstrem |

---

## 🚀 Kullanım

```bash
# Senaryo 1'i HARD complexity ile çalıştır
python run_v2.py -s 1

# Tüm senaryoları çalıştır
python run_v2.py -a

# Özel eşik değerleri ile
python run_v2.py -s 1 --z2 0.12 --z3 0.010
```

---

## 📁 Dosya Yapısı

```
wlb_v2_optimized/
├── ga_algorithm.py        # GA V3 (Softmax + Sıcaklık, Ada modeli yok)
├── nsga3_algorithm.py     # NSGA-III (2 amaç + 2 kısıt)
├── visualization.py       # V3 Pareto grafikleri (akademik format)
├── objectives.py          # Amaç fonksiyonları ve kısıt hesaplamaları
├── constraints.py         # Hard kısıtlar
├── complexity_config.py   # Complexity ayarları
├── run_v2.py              # Ana çalıştırma scripti
├── operators.py           # Genetik operatörler
├── data_loader.py         # Veri yükleme
├── Database.xlsx          # Senaryo verileri
└── README.md              # Bu dosya
```

---

## 🔄 V3 Erken Yakınsama Önleme Mekanizmaları

### GA'da:
- ✅ **Softmax + Sıcaklık Dinamik Eliminasyon** (YENİ)
- ✅ **Dinamik Kötü Kromozom Toleransı**: %30 → %5 (YENİ)
- ✅ **Adaptive Mutation**: Stagnation durumunda mutasyon oranı artırılır
- ✅ **Çeşitlilik Tabanlı Immigrant Injection**: Sadece çeşitlilik düşükse
- ❌ ~~Island Model~~: Kaldırıldı (V3)
- ❌ ~~Agresif Immigrant Injection~~: Kaldırıldı (V3)

### NSGA-III'de:
- ✅ **Reference Points**: Das-Dennis referans noktaları
- ✅ **Crowding Distance**: Pareto front dağılımı
- ✅ **Constraint Handling**: Feasibility-based dominance
- ✅ **Stagnation Detection**: İyileşme durduğunda müdahale

---

## 📈 Pareto Grafik Formatı (V3)

Akademik standart formatta:
- **Gri daireler**: Tüm çözümler (domine edilen)
- **Siyah daireler** (GA): Executable Policy
- **Mor daireler** (NSGA-III): Executable Policy
- **Harf etiketleri**: A, B, C, ... (sıralı)
- **Yeşil çizgi**: Pareto cephesi bağlantısı

---

## 📝 Notlar

1. GA artık tek popülasyon kullanır (120 birey, bölünmüyor)
2. Softmax eliminasyon her nesilde otomatik çalışır
3. Sıcaklık eğrisi `α=2.0` ile kuadratik soğuma yapar
4. `use_islands` parametresi geriye uyumluluk için var ama görmezden gelinir

---

## 🔗 Bağımlılıklar

```bash
pip install numpy pandas openpyxl matplotlib
```
