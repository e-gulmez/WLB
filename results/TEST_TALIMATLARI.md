# 🧪 WLB v2 - 8 Database Test Talimatları

## 📊 Yapılan Düzeltmeler

### 1. GA Algoritması
- **Ağırlıklar:** w1=0.7 (Z1 öncelik), w4=0.3 (Z4 ikincil)
- **Z4 Normalizasyonu:** Esnek ceza sistemi
  - Z4 < soft_Z4: Düşük ceza (Z1 artışına izin ver)
  - Z4 > soft_Z4: Artan ceza
  - Z4 > max_Z4: Ağır ceza

### 2. Database'ler
8 farklı test senaryosu oluşturuldu:

| DB | Personel | Gün | Yön.Talep | Per.Talep | Oran |
|----|----------|-----|-----------|-----------|------|
| DB1 | 24 | 14 | 224 | 237 | 105.8% |
| DB2 | 24 | 30 | 484 | 505 | 104.3% |
| DB3 | 30 | 21 | 426 | 415 | 97.4% |
| DB4 | 30 | 30 | 609 | 642 | 105.4% |
| DB5 | 75 | 21 | 1075 | 1131 | 105.2% |
| DB6 | 75 | 30 | 1540 | 1657 | 107.6% |
| DB7 | 100 | 30 | 2063 | 2186 | 106.0% |
| DB8 | 250 | 30 | 5178 | 5512 | 106.5% |

---

## 🚀 Çalıştırma Komutları

### Hazırlık
```bash
unzip wlb_v2_8DB_FINAL.zip
cd wlb_v2_optimized
```

### Tek Database Test
```bash
# DB4 (30 personel, 30 gün) - Varsayılan
cp databases/Database_DB4.xlsx Database.xlsx
python run_v2.py -s 1 -c HARD

# DB1 (24 personel, 14 gün) - Küçük
cp databases/Database_DB1.xlsx Database.xlsx
python run_v2.py -s 1 -c HARD

# DB8 (250 personel, 30 gün) - Ekstrem
cp databases/Database_DB8.xlsx Database.xlsx
python run_v2.py -s 1 -c HARD
```

### Tüm Senaryolar
```bash
python run_v2.py -a -c HARD
```

### Complexity Seviyeleri
```bash
# EASY (Gevşek kısıtlar)
python run_v2.py -s 1 -c EASY

# MEDIUM (Orta)
python run_v2.py -s 1 -c MEDIUM

# HARD (Sıkı) - Varsayılan
python run_v2.py -s 1 -c HARD

# VERY_HARD (Çok sıkı)
python run_v2.py -s 1 -c VERY_HARD
```

---

## 📋 Tam Test Matrisi

8 Database × 3 Senaryo × 4 Complexity = 96 Test

### Önerilen Test Planı

**Faz 1: Küçük Ölçek Validasyonu**
```bash
cp databases/Database_DB1.xlsx Database.xlsx
python run_v2.py -a -c EASY
python run_v2.py -a -c MEDIUM
python run_v2.py -a -c HARD
```

**Faz 2: Orta Ölçek**
```bash
cp databases/Database_DB4.xlsx Database.xlsx
python run_v2.py -a -c HARD
```

**Faz 3: Büyük Ölçek**
```bash
cp databases/Database_DB6.xlsx Database.xlsx
python run_v2.py -a -c HARD
```

**Faz 4: Ekstrem**
```bash
cp databases/Database_DB8.xlsx Database.xlsx
python run_v2.py -s 1 -c MEDIUM  # Önce MEDIUM ile başla
```

---

## 📈 Beklenen Sonuçlar

### GA (w1=0.7, w4=0.3)
- Z1: %85-92 arası (öncelikli)
- Z4: 5-30 arası (makul trade-off)
- Z4=0 problemi çözülmeli

### NSGA-III
- 120 Pareto çözüm
- Z1: %85-90 arası
- Z4: 0-40 arası dağılım
- Karar verici trade-off seçebilir

---

## ⚠️ Önemli Notlar

1. **DB3 Dikkat:** Oran %97.4 - bazı vardiyalarda understaffing olabilir
2. **DB8 Süre:** 250 personel için çalışma süresi 30-60 dakika olabilir
3. **Sonuç Kayıt:** `results/` klasörüne JSON olarak kaydedilir

---

## 📁 Dosya Yapısı

```
wlb_v2_optimized/
├── databases/
│   ├── Database_DB1.xlsx  (24p/14g)
│   ├── Database_DB2.xlsx  (24p/30g)
│   ├── Database_DB3.xlsx  (30p/21g)
│   ├── Database_DB4.xlsx  (30p/30g)
│   ├── Database_DB5.xlsx  (75p/21g)
│   ├── Database_DB6.xlsx  (75p/30g)
│   ├── Database_DB7.xlsx  (100p/30g)
│   └── Database_DB8.xlsx  (250p/30g)
├── ga_algorithm.py        (Düzeltilmiş)
├── nsga3_algorithm.py
├── run_v2.py             (Düzeltilmiş)
├── generate_databases.py  (Database generator)
└── ...
```
