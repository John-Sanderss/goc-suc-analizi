# Göç ve Suç Analizi – Avrupa Ülkeleri

Avrupa ülkelerinde **göç verileri** ile **suç oranları** arasındaki ilişkiyi inceleyen ve gelecek için suç eğilimlerini tahmin eden bir çalışma.

-  **Random Forest (RF)**: Göç verisinin suç oranlarını açıklamadaki gücünü **R²** skorlarıyla ölçer.
-  **ARIMA (Zaman Serisi)**: Geçmiş verilerden hareketle **3 yıllık** toplam suç tahminleri üretir.
-  Sonuçlar **ülke bazında**, **suç türü bazında** ve **Avrupa geneli** için görselleştirilir.
 Ayrıntılı rapor: [report/RAPOR_SON.pdf](report/RAPOR_SON.pdf)

---

## Yöntem ve Veriler

**Veri Kaynakları**
- Suç verileri: Eurostat – <https://ec.europa.eu/eurostat/web/crime/database>  
- Göç verileri: OECD – <https://www.oecd.org/en/data/datasets/database-on-immigrants-in-oecd-and-non-oecd-countries.html>

> Not: Telif/kullanım koşulları nedeniyle tam veri paylaşılmamıştır. Proje için “`data/merged_goc_suc.csv`” dosyası gereklidir. İsterseniz yukardaki kaynaklardan verilere ulaşıp bir csv dosyası oluşturabilirsiniz.

**Modeller**
- **RF (scikit-learn)**:  
  - *Göç → Toplam Suç* (ülke bazında)  
  - *Göç → Suç Türleri* (Avrupa geneli)  
  - *Göç → Suç Türleri* (ülke + suç türü düzeyi)
- **ARIMA (statsmodels)**:  
  - *Tek ülke*, *iki ülke karşılaştırmalı* ve *Avrupa toplamı* için 3 yıllık toplam suç tahmini

---

## Hızlı Başlangıç

> Python 3.10+ önerilir.

1) Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```
2) Proje kökünden çalıştırın:
```bash
# Random Forest analizleri
python src/random_forest_analysis.py

# ARIMA analizleri
python src/arima_analysis.py
```
## Yazarlar

- **Yiğithan BAŞAĞA** – Ankara Üniversitesi, Bilgisayar Mühendisliği  
- **Esra KURNAZ** – Ankara Üniversitesi, Bilgisayar Mühendisliği
