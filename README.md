# Gelişmiş Portföy Optimizasyonu

Bu proje, modern portföy teorisi ve gelişmiş risk yönetimi tekniklerini kullanarak optimal portföy oluşturma sürecini otomatize eder.

## Özellikler

### 1. Veri Toplama ve Ön İşleme
- Yahoo Finance API üzerinden otomatik veri toplama
- Getirilerin hesaplanması (basit/logaritmik)
- Teknik göstergelerin hesaplanması (SMA, RSI, vb.)

### 2. Risk ve Getiri Analizi
- Gelişmiş risk metrikleri:
  - Value at Risk (VaR) - Tarihsel, Parametrik ve Monte Carlo
  - Expected Shortfall (CVaR)
  - Maksimum Drawdown
  - Volatilite analizi
  - Kuyruk riski metrikleri
- Makroekonomik faktör analizi:
  - Döviz kuru hassasiyeti
  - Faiz oranı hassasiyeti
  - Piyasa betası

### 3. Portföy Optimizasyonu
- Farklı optimizasyon hedefleri:
  - Maksimum Sharpe Oranı
  - Minimum Volatilite
  - Maksimum Sortino Oranı
- Kısıtlamalar:
  - Maksimum hisse ağırlığı
  - Sektör limitleri
  - Minimum hisse sayısı
- Risk limitleri:
  - VaR limiti
  - Expected Shortfall limiti

### 4. Performans ve Güvenilirlik
- Numba ile hızlandırılmış hesaplamalar
- Önbellekleme ve bellek optimizasyonu
- Kapsamlı hata yönetimi
- Detaylı loglama

## Kurulum

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt
```

## Kullanım

```python
from src.data.data_collector import DataCollector
from src.models.portfolio_optimizer import PortfolioOptimizer
from src.models.enhanced_risk_manager import EnhancedRiskManager

# Veri topla
collector = DataCollector()
data = collector.fetch_stock_data(symbols=['THYAO.IS', 'GARAN.IS'], 
                                start_date='2022-01-01', 
                                end_date='2023-12-31')

# Portföy optimize edici oluştur
optimizer = PortfolioOptimizer(
    returns_data=data,
    market_returns=market_data,
    risk_free_rate=0.15
)

# Risk limitleri tanımla
risk_constraints = {
    'var_95': 0.02,  # Maksimum %2 VaR
    'expected_shortfall': 0.025  # Maksimum %2.5 Expected Shortfall
}

# Optimizasyon yap
optimal_weights, risk_analysis = optimizer.optimize_portfolio(
    optimization_target="Maksimum Sharpe Oranı",
    max_weight=0.2,
    sector_limit=0.3,
    min_stocks=5,
    risk_constraints=risk_constraints
)

# Risk raporunu incele
print("Risk Analizi:")
print("VaR Metrikleri:", risk_analysis['var_metrics'])
print("Expected Shortfall:", risk_analysis['expected_shortfall'])
print("Kuyruk Riski:", risk_analysis['tail_risk'])
print("Stres Testi Sonuçları:", risk_analysis['stress_test'])
print("Makroekonomik Etkiler:", risk_analysis['macro_impact'])
```

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın. 