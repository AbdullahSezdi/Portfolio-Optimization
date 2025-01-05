import numpy as np
import pandas as pd
from scipy import stats
import logging
from functools import lru_cache
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedRiskManager:
    def __init__(self, returns, weights, market_returns=None, risk_free_rate=0.15):
        """
        Gelişmiş risk yönetimi sınıfı.
        
        Args:
            returns: Hisse senedi getirileri (pd.DataFrame veya np.ndarray)
            weights: Portföy ağırlıkları (pd.Series, np.ndarray veya list)
            market_returns: Piyasa endeks getirileri (Beta hesaplaması için)
            risk_free_rate: Risksiz faiz oranı
        """
        try:
            # Veri tiplerini kontrol et ve dönüştür
            if isinstance(returns, pd.DataFrame):
                self.returns = returns
            elif isinstance(returns, np.ndarray):
                self.returns = pd.DataFrame(returns)
            else:
                raise ValueError("returns parametresi DataFrame veya ndarray olmalıdır")

            # Ağırlıkları numpy array'e çevir
            if isinstance(weights, (pd.Series, list)):
                self.weights = np.array(weights).flatten()
            elif isinstance(weights, np.ndarray):
                self.weights = weights.flatten()
            else:
                raise ValueError("weights parametresi Series, list veya ndarray olmalıdır")

            # Boyut kontrolü
            if len(self.weights) != self.returns.shape[1]:
                raise ValueError(f"Ağırlık sayısı ({len(self.weights)}) ve hisse senedi sayısı ({self.returns.shape[1]}) eşleşmiyor")

            self.market_returns = market_returns
            self.risk_free_rate = risk_free_rate

            # Portföy getirilerini hesapla
            try:
                # Matris çarpımı ile portföy getirilerini hesapla
                portfolio_values = np.dot(self.returns.values, self.weights)
                
                # Series'e çevir
                if isinstance(returns, pd.DataFrame):
                    self.portfolio_returns = pd.Series(
                        portfolio_values,
                        index=returns.index,
                        name='Portfolio Returns'
                    )
                else:
                    self.portfolio_returns = portfolio_values

                # NaN değerleri temizle
                if isinstance(self.portfolio_returns, pd.Series):
                    self.portfolio_returns = self.portfolio_returns.dropna()

                logger.info("Portföy getirileri başarıyla hesaplandı")
                logger.info(f"Portföy getiri sayısı: {len(self.portfolio_returns)}")
                
            except Exception as e:
                logger.error(f"Portföy getirileri hesaplanırken hata: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Risk yöneticisi başlatılırken hata: {str(e)}")
            logger.error(f"Hata detayları: {e.__class__.__name__}")
            raise

    @lru_cache(maxsize=None)
    def get_macro_data(self):
        """
        Makroekonomik verileri çeker ve önbellekte saklar.
        """
        try:
            # Portföy verilerinin tarih aralığını kullan
            if isinstance(self.portfolio_returns, pd.Series):
                start_date = self.portfolio_returns.index.min()
                end_date = self.portfolio_returns.index.max()
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
            
            macro_data = {}
            
            # USDTRY kuru
            try:
                usdtry = yf.download('USDTRY=X', start=start_date, end=end_date)['Close']
                if not usdtry.empty:
                    macro_data['USDTRY'] = usdtry
                else:
                    logger.warning("USD/TRY verisi boş")
                    macro_data['USDTRY'] = None
            except Exception as e:
                logger.warning(f"USD/TRY verisi alınamadı: {str(e)}")
                macro_data['USDTRY'] = None
            
            # BIST-100 endeksi
            try:
                bist100 = yf.download('XU100.IS', start=start_date, end=end_date)['Close']
                if not bist100.empty:
                    macro_data['BIST100'] = bist100
                else:
                    logger.warning("BIST-100 verisi boş")
                    macro_data['BIST100'] = None
            except Exception as e:
                logger.warning(f"BIST-100 verisi alınamadı: {str(e)}")
                macro_data['BIST100'] = None
            
            # En az bir veri başarıyla alındıysa devam et
            if any(v is not None for v in macro_data.values()):
                return macro_data
            else:
                logger.error("Hiçbir makroekonomik veri alınamadı")
                return None
            
        except Exception as e:
            logger.error(f"Makro veri çekme hatası: {str(e)}")
            return None

    def calculate_macro_impact(self):
        """
        Makroekonomik faktörlerin portföy üzerindeki etkisini hesaplar.
        """
        try:
            # Input validasyonu
            if not isinstance(self.portfolio_returns, pd.Series) or self.portfolio_returns.empty:
                logger.error("Geçerli portföy getirisi bulunamadı")
                return None
            
            macro_data = self.get_macro_data()
            if macro_data is None:
                return None
            
            impacts = {}
            
            # Döviz kuru etkisi
            if macro_data.get('USDTRY') is not None and not macro_data['USDTRY'].empty:
                try:
                    # Veri hazırlama
                    portfolio_returns = self.portfolio_returns.astype(float).values.flatten()
                    usdtry_returns = macro_data['USDTRY'].pct_change().fillna(0).values.flatten()
                    
                    # Veri uzunluklarını eşitle
                    min_len = min(len(portfolio_returns), len(usdtry_returns))
                    portfolio_returns = portfolio_returns[:min_len]
                    usdtry_returns = usdtry_returns[:min_len]
                    
                    if min_len > 1:
                        # Debug bilgisi
                        logger.info(f"Portfolio returns shape: {portfolio_returns.shape}")
                        logger.info(f"USDTRY returns shape: {usdtry_returns.shape}")
                        
                        # Korelasyon hesapla
                        fx_correlation = np.corrcoef(portfolio_returns, usdtry_returns)[0,1]
                        impacts['FX_SENSITIVITY'] = fx_correlation
                        
                        logger.info(f"Calculated FX correlation: {fx_correlation}")
                    else:
                        logger.warning("USD/TRY için yeterli ortak tarih bulunamadı")
                        impacts['FX_SENSITIVITY'] = 0
                except Exception as e:
                    logger.warning(f"Döviz kuru etkisi hesaplanamadı: {str(e)}")
                    logger.warning(f"Error details: {e.__class__.__name__}")
                    impacts['FX_SENSITIVITY'] = 0
            else:
                impacts['FX_SENSITIVITY'] = 0
            
            # Piyasa etkisi
            if macro_data.get('BIST100') is not None and not macro_data['BIST100'].empty:
                try:
                    # Veri hazırlama
                    portfolio_returns = self.portfolio_returns.astype(float).values.flatten()
                    bist_returns = macro_data['BIST100'].pct_change().fillna(0).values.flatten()
                    
                    # Veri uzunluklarını eşitle
                    min_len = min(len(portfolio_returns), len(bist_returns))
                    portfolio_returns = portfolio_returns[:min_len]
                    bist_returns = bist_returns[:min_len]
                    
                    if min_len > 1:
                        # Debug bilgisi
                        logger.info(f"Portfolio returns shape: {portfolio_returns.shape}")
                        logger.info(f"BIST returns shape: {bist_returns.shape}")
                        
                        # Beta hesapla
                        cov = np.cov(portfolio_returns, bist_returns)[0,1]
                        var = np.var(bist_returns)
                        
                        if var != 0:
                            market_beta = cov / var
                            impacts['MARKET_BETA'] = market_beta
                            logger.info(f"Calculated market beta: {market_beta}")
                        else:
                            logger.warning("BIST-100 varyansı sıfır")
                            impacts['MARKET_BETA'] = 1
                    else:
                        logger.warning("BIST-100 için yeterli ortak tarih bulunamadı")
                        impacts['MARKET_BETA'] = 1
                except Exception as e:
                    logger.warning(f"Piyasa etkisi hesaplanamadı: {str(e)}")
                    logger.warning(f"Error details: {e.__class__.__name__}")
                    impacts['MARKET_BETA'] = 1
            else:
                impacts['MARKET_BETA'] = 1
            
            # Faiz hassasiyeti
            try:
                # Portföy volatilitesini hesapla
                portfolio_vol = np.std(self.portfolio_returns) * np.sqrt(252)
                
                # Basit bir faiz hassasiyeti hesabı
                # Portföy volatilitesi ve beta kullanarak hesapla
                rate_sensitivity = (impacts.get('MARKET_BETA', 1) * portfolio_vol) / 100
                
                # Değeri sınırla ve yuvarla
                rate_sensitivity = round(max(min(rate_sensitivity, 1), -1), 2)
                
                impacts['RATE_SENSITIVITY'] = rate_sensitivity
                logger.info(f"Calculated rate sensitivity: {rate_sensitivity}")
            except Exception as e:
                logger.warning(f"Faiz hassasiyeti hesaplanamadı: {str(e)}")
                impacts['RATE_SENSITIVITY'] = 0
            
            return impacts if impacts else None
            
        except Exception as e:
            logger.error(f"Makro etki hesaplama hatası: {str(e)}")
            logger.error(f"Error details: {e.__class__.__name__}")
            return None

    def calculate_advanced_risk_metrics(self):
        """
        Gelişmiş risk metriklerini hesaplar.
        """
        try:
            metrics = {}
            
            # Momentum göstergeleri
            returns_series = pd.Series(self.portfolio_returns)
            metrics['momentum'] = {
                '1M': returns_series.tail(21).sum(),
                '3M': returns_series.tail(63).sum(),
                '6M': returns_series.tail(126).sum(),
                '12M': returns_series.tail(252).sum()
            }
            
            # Volatilite göstergeleri
            rolling_vol = returns_series.rolling(window=21).std() * np.sqrt(252)
            metrics['volatility'] = {
                'current': rolling_vol.iloc[-1],
                'avg_3m': rolling_vol.tail(63).mean(),
                'trend': (rolling_vol.iloc[-1] / rolling_vol.tail(63).mean()) - 1
            }
            
            # Risk-adjusted metrics
            ann_return = np.mean(self.portfolio_returns) * 252
            ann_vol = np.std(self.portfolio_returns) * np.sqrt(252)
            metrics['risk_adjusted'] = {
                'calmar_ratio': ann_return / abs(self.calculate_max_drawdown()),
                'sortino_ratio': self.calculate_sortino_ratio(),
                'treynor_ratio': (ann_return - self.risk_free_rate) / self.calculate_beta()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Gelişmiş risk metrikleri hesaplama hatası: {str(e)}")
            return None

    def calculate_max_drawdown(self):
        """Maksimum drawdown hesaplar."""
        cumulative_returns = (1 + pd.Series(self.portfolio_returns)).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()

    def calculate_sortino_ratio(self):
        """Sortino oranını hesaplar."""
        ann_return = np.mean(self.portfolio_returns) * 252
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252)
        return (ann_return - self.risk_free_rate) / downside_std if downside_std != 0 else 0

    def calculate_beta(self):
        """Portföy betasını hesaplar."""
        if self.market_returns is None:
            return 1.0
        
        # Numpy array'e çevir
        portfolio_returns = np.array(self.portfolio_returns)
        market_returns = np.array(self.market_returns)
        
        # Kovaryans matrisini hesapla
        covariance = np.cov(portfolio_returns, market_returns)[0,1]
        market_variance = np.var(market_returns)
        
        # Beta hesapla
        return covariance / market_variance if market_variance != 0 else 1.0

    def calculate_var(self, confidence_level=0.95, method='historical'):
        """
        Value at Risk (VaR) hesaplama.
        """
        try:
            # Input validasyonu
            if not hasattr(self, 'portfolio_returns') or len(self.portfolio_returns) == 0:
                logger.error("Geçerli portföy getirisi bulunamadı")
                return 0
            
            # NaN değerleri temizle
            returns = pd.Series(self.portfolio_returns).dropna()
            
            if len(returns) == 0:
                logger.error("Geçerli getiri verisi bulunamadı")
                return 0
            
            if method == 'historical':
                return -np.percentile(returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                z_score = stats.norm.ppf(confidence_level)
                return -(returns.mean() + z_score * returns.std())
            else:
                logger.error(f"Geçersiz VaR hesaplama metodu: {method}")
                return 0
            
        except Exception as e:
            logger.error(f"VaR hesaplama hatası: {str(e)}")
            return 0

    def calculate_expected_shortfall(self, confidence_level=0.95):
        """
        Expected Shortfall (CVaR) hesaplama.
        """
        try:
            # Input validasyonu
            if not hasattr(self, 'portfolio_returns') or len(self.portfolio_returns) == 0:
                logger.error("Geçerli portföy getirisi bulunamadı")
                return 0
            
            # NaN değerleri temizle
            returns = pd.Series(self.portfolio_returns).dropna()
            
            if len(returns) == 0:
                logger.error("Geçerli getiri verisi bulunamadı")
                return 0
            
            # VaR hesapla
            var = self.calculate_var(confidence_level)
            if var == 0:
                logger.warning("VaR sıfır olduğu için Expected Shortfall hesaplanamıyor")
                return 0
            
            # Expected Shortfall hesapla
            threshold = -var
            downside_returns = returns[returns <= threshold]
            
            if len(downside_returns) == 0:
                logger.warning("Eşik değerinin altında getiri bulunamadı")
                return 0
            
            es = -downside_returns.mean()
            
            # Sonucu kontrol et
            if np.isnan(es) or np.isinf(es):
                logger.error("Expected Shortfall hesaplaması geçersiz sonuç verdi")
                return 0
                
            return es
            
        except Exception as e:
            logger.error(f"Expected Shortfall hesaplama hatası: {str(e)}")
            logger.error(f"Hata detayları: {e.__class__.__name__}")
            return 0

    def calculate_stress_test(self, scenario_shocks):
        """
        Stres testi uygulama.
        
        Args:
            scenario_shocks: Senaryo şokları (örn: {'THYAO.IS': -0.20, 'GARAN.IS': -0.15})
        """
        try:
            # Numpy array'lere çevir
            weights = np.array(self.weights)
            
            # Şoklanmış getirileri hesapla
            shocked_returns = self.returns.copy()
            for symbol, shock in scenario_shocks.items():
                if symbol in shocked_returns.columns:
                    shocked_returns[symbol] = shocked_returns[symbol] * (1 + shock)
            
            # Portföy getirilerini hesapla
            shocked_portfolio_returns = np.sum(shocked_returns.values * weights.reshape(-1, 1).T, axis=1)
            
            return {
                'mean_return': np.mean(shocked_portfolio_returns),
                'volatility': np.std(shocked_portfolio_returns),
                'max_loss': np.min(shocked_portfolio_returns)
            }
        except Exception as e:
            logger.error(f"Stres testi hatası: {str(e)}")
            return None

    def calculate_risk_contribution(self):
        """
        Her hissenin portföy riskine katkısını hesapla.
        """
        try:
            # Numpy array'lere çevir
            weights = np.array(self.weights)
            
            # Kovaryans matrisini hesapla
            cov_matrix = self.returns.cov().values
            
            # Portföy volatilitesini hesapla
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return pd.Series(np.zeros_like(weights), index=self.returns.columns)
            
            # Marginal risk contribution
            mrc = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Component risk contribution
            crc = np.multiply(weights, mrc)
            
            return pd.Series(crc, index=self.returns.columns)
        except Exception as e:
            logger.error(f"Risk katkısı hesaplama hatası: {str(e)}")
            return None

    def calculate_tracking_error(self):
        """
        Tracking Error hesaplama (portföyün benchmark'tan sapması).
        """
        if self.market_returns is None:
            logger.warning("Market returns not provided for tracking error calculation")
            return None
            
        try:
            return np.std(self.portfolio_returns - self.market_returns) * np.sqrt(252)
        except Exception as e:
            logger.error(f"Tracking Error hesaplama hatası: {str(e)}")
            return None

    def calculate_information_ratio(self):
        """
        Information Ratio hesaplama.
        """
        if self.market_returns is None:
            logger.warning("Market returns not provided for information ratio calculation")
            return None
            
        try:
            tracking_error = self.calculate_tracking_error()
            active_return = (self.portfolio_returns.mean() - self.market_returns.mean()) * 252
            return active_return / tracking_error if tracking_error != 0 else 0
        except Exception as e:
            logger.error(f"Information Ratio hesaplama hatası: {str(e)}")
            return None

    def calculate_tail_risk_metrics(self):
        """
        Kuyruk riski metriklerini hesapla.
        """
        try:
            return {
                'skewness': stats.skew(self.portfolio_returns),
                'kurtosis': stats.kurtosis(self.portfolio_returns),
                'tail_ratio': abs(np.percentile(self.portfolio_returns, 95) / np.percentile(self.portfolio_returns, 5))
            }
        except Exception as e:
            logger.error(f"Kuyruk riski hesaplama hatası: {str(e)}")
            return None

    def generate_risk_report(self):
        """
        Risk raporu oluştur.
        """
        try:
            # VaR ve Expected Shortfall hesapla
            historical_var = self.calculate_var(0.95, 'historical')
            parametric_var = self.calculate_var(0.95, 'parametric')
            expected_shortfall = self.calculate_expected_shortfall()

            # Sonuçları yüzde formatına çevir
            historical_var = historical_var * 100 if historical_var is not None else 0
            parametric_var = parametric_var * 100 if parametric_var is not None else 0
            expected_shortfall = expected_shortfall * 100 if expected_shortfall is not None else 0

            # Sadece gerekli metrikleri içeren basit bir rapor döndür
            return {
                'historical_var_95': historical_var,
                'parametric_var_95': parametric_var,
                'expected_shortfall': expected_shortfall
            }

        except Exception as e:
            logger.error(f"Risk raporu oluşturma hatası: {str(e)}")
            return {
                'historical_var_95': 0,
                'parametric_var_95': 0,
                'expected_shortfall': 0
            } 