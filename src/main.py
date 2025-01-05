from src.data.data_collector import DataCollector
from src.models.portfolio_optimizer import PortfolioOptimizer
import pandas as pd
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # BIST-30 hisseleri
        symbols = [
            'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS',
            'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HEKTS.IS', 'KCHOL.IS',
            'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS',
            'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TAVHL.IS', 'THYAO.IS',
            'TKFEN.IS', 'TOASO.IS', 'TUPRS.IS', 'VESTL.IS', 'YKBNK.IS'
        ]

        # Veri toplama
        logger.info("Veri toplama başlıyor...")
        collector = DataCollector()
        data = collector.fetch_stock_data(
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )

        # Piyasa verisi (BIST-100)
        market_data = collector.fetch_market_index(
            index_symbol='XU100.IS',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )

        # Getirileri hesapla
        returns_data = pd.DataFrame()
        for symbol, stock_data in data.items():
            returns_data[symbol] = collector.calculate_returns(stock_data, method='log')

        # Piyasa getirilerini hesapla
        if market_data is not None:
            market_returns = pd.Series(collector.calculate_returns(market_data, method='log').values.flatten())
        else:
            logger.warning("Piyasa verisi alınamadı, piyasa getirisi olmadan devam ediliyor...")
            market_returns = None

        # Portföy optimize edici oluştur
        logger.info("Portföy optimizasyonu başlıyor...")
        optimizer = PortfolioOptimizer(
            returns_data=returns_data,
            market_returns=market_returns,
            risk_free_rate=0.45  # %45 (güncel TCMB faizi)
        )

        # Risk limitleri
        risk_constraints = {
            'var_95': 0.02,  # Maksimum %2 VaR
            'expected_shortfall': 0.025  # Maksimum %2.5 Expected Shortfall
        }

        # Optimizasyon yap
        optimal_weights, risk_analysis = optimizer.optimize_portfolio(
            optimization_target="Maksimum Sharpe Oranı",
            max_weight=0.20,  # Maksimum %20 ağırlık
            sector_limit=0.30,  # Maksimum %30 sektör ağırlığı
            min_stocks=8,  # Minimum 8 hisse
            risk_constraints=risk_constraints
        )

        # Sonuçları yazdır
        logger.info("\nOptimal Portföy Ağırlıkları:")
        for symbol, weight in optimal_weights[optimal_weights > 0.01].items():
            logger.info(f"{symbol}: %{weight*100:.2f}")

        logger.info("\nRisk Analizi:")
        if risk_analysis and 'var_metrics' in risk_analysis:
            logger.info(f"VaR (95%): %{risk_analysis['var_metrics']['historical_var_95']*100:.2f}")
            logger.info(f"Expected Shortfall: %{risk_analysis['expected_shortfall']*100:.2f}")
        
        logger.info("\nMakroekonomik Etki Analizi:")
        if risk_analysis and 'macro_impact' in risk_analysis and risk_analysis['macro_impact']:
            macro_impact = risk_analysis['macro_impact']
            logger.info(f"Döviz Kuru Hassasiyeti: {macro_impact['FX_SENSITIVITY']:.2f}")
            logger.info(f"Piyasa Betası: {macro_impact['MARKET_BETA']:.2f}")
            logger.info(f"Faiz Hassasiyeti: {macro_impact['RATE_SENSITIVITY']:.2f}")
        else:
            logger.warning("Makroekonomik etki analizi yapılamadı.")

        # Belleği temizle
        optimizer.cleanup()

    except Exception as e:
        logger.error(f"Hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    main() 