# Portföy Optimizasyonu Projesi

## Proje Hakkında
Bu proje, modern portföy teorisi ve yapay zeka tekniklerini kullanarak optimal yatırım portföyü oluşturmayı amaçlamaktadır. Markowitz'in Modern Portföy Teorisi temel alınarak, yapay zeka ve makine öğrenimi teknikleri ile geliştirilmiş bir portföy optimizasyon sistemi oluşturulacaktır.

## Teknik Gereksinimler
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Yfinance
- Plotly
- SciPy

## Proje Adımları

### 1. Veri Toplama ve Ön İşleme
- **Veri Kaynakları:**
  - Yahoo Finance API (yfinance) kullanarak hisse senedi verilerinin çekilmesi
  - Finansal göstergelerin toplanması
  - Piyasa endeks verilerinin elde edilmesi

- **Veri Ön İşleme:**
  - Eksik verilerin temizlenmesi
  - Aykırı değerlerin tespiti ve işlenmesi
  - Verilerin normalize edilmesi
  - Teknik göstergelerin hesaplanması (RSI, MACD, vb.)

### 2. Risk ve Getiri Analizi
- Tarihsel getiri hesaplamaları
- Volatilite analizi
- Kovaryans matrisinin oluşturulması
- Sharpe oranı hesaplaması
- Beta katsayısı analizi

### 3. Portföy Optimizasyon Modelleri
- **Klasik Optimizasyon:**
  - Markowitz Ortalama-Varyans Optimizasyonu
  - Minimum Varyans Portföyü
  - Maksimum Sharpe Oranı Portföyü

- **Yapay Zeka Tabanlı Optimizasyon:**
  - Derin Öğrenme Modelleri (LSTM, GRU)
  - Gradient Boosting Algoritmaları
  - Monte Carlo Simülasyonları

### 4. Risk Yönetimi
- VaR (Value at Risk) hesaplaması
- CVaR (Conditional Value at Risk) analizi
- Stres testleri
- Senaryo analizleri

### 5. Model Değerlendirme ve Backtesting
- Out-of-sample testleri
- Çapraz doğrulama
- Performans metrikleri hesaplama
- Backtest sonuçlarının analizi

### 6. Görselleştirme ve Raporlama
- Etkin sınır (Efficient Frontier) grafikleri
- Portföy performans grafikleri
- Risk-getiri dağılım grafikleri
- Interaktif dashboard oluşturma

## Proje Yapısı
```
portfolio_optimization/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── classical/
│   └── ai_models/
├── notebooks/
│   ├── 1_data_collection.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_model_development.ipynb
│   └── 4_evaluation.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
```

## Kurulum
1. Gerekli Python paketlerinin kurulumu:
```bash
pip install -r requirements.txt
```

2. Veri kaynaklarının yapılandırılması
3. Modellerin eğitilmesi
4. Backtesting ve optimizasyon

## Kullanım
1. Veri toplama scriptlerinin çalıştırılması
2. Model eğitimi
3. Portföy optimizasyonu
4. Sonuçların görselleştirilmesi

## Önemli Notlar
- Finansal veriler günlük olarak güncellenmeli
- Risk parametreleri düzenli olarak kalibre edilmeli
- Model performansı sürekli izlenmeli
- Piyasa koşullarına göre parametreler güncellenmelidir

## Gelecek Geliştirmeler
- Gerçek zamanlı veri entegrasyonu
- Duygu analizi entegrasyonu
- Blockchain entegrasyonu
- Otomatik alım-satım stratejileri 

## Final Ürün ve Özellikleri

### 1. Interaktif Web Arayüzü
- Modern ve kullanıcı dostu dashboard
- Portföy performansının gerçek zamanlı takibi
- Özelleştirilebilir grafik ve tablolar
- Mobil uyumlu tasarım

### 2. Portföy Analiz Araçları
- Risk-getiri analiz raporları
- Portföy çeşitlendirme önerileri
- Otomatik rebalancing tavsiyeleri
- Tarihsel performans analizleri

### 3. Yapay Zeka Destekli Özellikler
- Gelecek trend tahminleri
- Optimal portföy ağırlıkları önerileri
- Risk uyarı sistemi
- Anomali tespiti ve uyarıları

### 4. Raporlama Sistemi
- PDF formatında detaylı portföy raporları
- Haftalık/aylık performans özetleri
- Vergi raporlaması için gerekli dökümanlar
- Özelleştirilebilir rapor şablonları

### 5. Veri Entegrasyonu
- Gerçek zamanlı piyasa verileri
- Ekonomik göstergeler ve haberler
- Teknik analiz göstergeleri
- Temel analiz metrikleri

### 6. Risk Yönetimi Araçları
- Stop-loss önerileri
- Portföy sigorta stratejileri
- Stres testi senaryoları
- Risk limiti uyarıları

### 7. Kullanıcı Yönetimi
- Çoklu portföy yönetimi
- Kişiselleştirilmiş risk profili
- Yatırım hedefi takibi
- Performans karşılaştırma araçları 