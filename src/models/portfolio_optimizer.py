import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numba import jit
import logging
from functools import lru_cache
from .enhanced_risk_manager import EnhancedRiskManager

logger = logging.getLogger(__name__)

class OptimizationError(Exception):
    """Optimizasyon hatası için özel exception sınıfı."""
    pass

class PortfolioOptimizer:
    def __init__(self, returns_data, risk_free_rate=0.15, market_returns=None):
        """
        Portföy optimize edici sınıfı.
        
        Args:
            returns_data (pd.DataFrame): Hisse senedi getirileri
            risk_free_rate (float): Risksiz faiz oranı
            market_returns (pd.Series): Piyasa endeks getirileri
            
        Raises:
            ValueError: Geçersiz veri formatı veya eksik veri durumunda
        """
        # Veri doğrulama
        if not isinstance(returns_data, pd.DataFrame):
            raise ValueError("returns_data must be a pandas DataFrame")
        
        if returns_data.empty:
            raise ValueError("returns_data cannot be empty")
            
        if market_returns is not None and not isinstance(market_returns, pd.Series):
            raise ValueError("market_returns must be a pandas Series")
        
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.market_returns = market_returns
        self.n_assets = len(returns_data.columns)
        
        # Önbellekleme için hesaplamalar
        self._initialize_cache()
        
        # Sektör bilgilerini yükle
        self.sector_mappings = self._get_sector_mappings()
        
        logger.info(f"Portfolio optimizer initialized with {self.n_assets} assets")

    def _initialize_cache(self):
        """Sık kullanılan hesaplamaları önbelleğe al."""
        try:
            self._cov_matrix = self.returns.cov().values
            self._returns_mean = self.returns.mean().values * 252
            logger.info("Cache initialization successful")
        except Exception as e:
            logger.error(f"Cache initialization failed: {str(e)}")
            raise OptimizationError("Failed to initialize cache")

    @lru_cache(maxsize=None)
    def get_sector_exposure(self, sector):
        """
        Belirli bir sektördeki pozisyonları hesaplar (önbellekli).
        """
        return [i for i, stock in enumerate(self.returns.columns) 
                if self.sector_mappings[stock] == sector]

    def validate_weights(self, weights):
        """
        Ağırlıkların geçerliliğini kontrol eder.
        """
        if not isinstance(weights, (np.ndarray, pd.Series)):
            raise ValueError("Weights must be numpy array or pandas Series")
            
        if len(weights) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} weights, got {len(weights)}")
            
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            raise ValueError("Weights must sum to 1.0")
            
        if np.any(weights < 0):
            raise ValueError("Negative weights are not allowed")

    def get_fallback_portfolio(self, min_stocks=5):
        """
        Optimizasyon başarısız olduğunda kullanılacak yedek portföy.
        """
        weights = np.zeros(self.n_assets)
        selected = np.random.choice(self.n_assets, size=min_stocks, replace=False)
        weights[selected] = 1/min_stocks
        logger.warning("Using fallback portfolio due to optimization failure")
        return weights

    def optimize_portfolio(self, optimization_target="Maksimum Sharpe Oranı",
                         max_weight=0.2, sector_limit=0.3, min_stocks=5,
                         risk_constraints=None):
        """Portföy optimizasyonu yapar."""
        try:
            logger.info(f"Optimizasyon başlatılıyor: {optimization_target}")
            logger.info(f"Parametreler: max_weight={max_weight}, sector_limit={sector_limit}, min_stocks={min_stocks}")

            # Input validasyonu
            valid_targets = ["Maksimum Sharpe Oranı", "Minimum Volatilite", "Maksimum Sortino Oranı"]
            if optimization_target not in valid_targets:
                raise ValueError(f"Geçersiz optimizasyon hedefi. Geçerli hedefler: {valid_targets}")

            # Başlangıç tahminleri
            initial_weights = np.array([1./self.n_assets] * self.n_assets)
            
            # Ana kısıtlar
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Ağırlıklar toplamı 1
            ]

            # Her hissenin sınırları
            bounds = tuple((0, max_weight) for _ in range(self.n_assets))

            # Optimizasyon hedefine göre objective fonksiyonunu seç
            if optimization_target == "Maksimum Sortino Oranı":
                objective = self._sortino_objective
            elif optimization_target == "Minimum Volatilite":
                objective = self._volatility_objective
            else:  # Maksimum Sharpe Oranı
                objective = self._sharpe_objective

            # Birden fazla optimizasyon denemesi yap
            best_result = None
            best_score = float('inf')
            successful_attempts = 0
            
            # Farklı başlangıç noktaları dene
            for attempt in range(50):
                try:
                    # Başlangıç noktası stratejileri
                    if attempt == 0:
                        init_weights = initial_weights
                        strategy = "Eşit ağırlıklı"
                    elif attempt < 20:
                        init_weights = np.random.random(self.n_assets)
                        init_weights /= init_weights.sum()
                        strategy = "Rastgele ağırlıklar"
                    else:
                        init_weights = np.zeros(self.n_assets)
                        selected = np.random.choice(self.n_assets, size=min_stocks, replace=False)
                        init_weights[selected] = 1/min_stocks
                        strategy = "Konsantre portföy"
                    
                    logger.info(f"Deneme {attempt + 1}/50 başlatılıyor (Strateji: {strategy})")
                    
                    result = minimize(
                        objective,
                        init_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={
                            'maxiter': 3000,
                            'ftol': 1e-10,
                            'disp': False
                        }
                    )
                    
                    if result.success:
                        successful_attempts += 1
                        score = objective(result.x)
                        logger.info(f"Deneme {attempt + 1} başarılı. Skor: {score}")
                        
                        if score < best_score:
                            best_result = result
                            best_score = score
                            logger.info(f"Yeni en iyi sonuç bulundu. Skor: {score}")
                            
                            # En iyi sonucun detaylarını göster
                            weights = pd.Series(result.x, index=self.returns.columns)
                            active_positions = weights[weights > 0.01]
                            logger.info(f"Aktif pozisyon sayısı: {len(active_positions)}")
                            logger.info("En büyük 5 pozisyon:")
                            for stock, weight in active_positions.nlargest(5).items():
                                logger.info(f"{stock}: {weight:.1%}")
                    else:
                        logger.warning(f"Deneme {attempt + 1} başarısız. Neden: {result.message}")
                        
                except Exception as e:
                    logger.error(f"Deneme {attempt + 1} hata verdi: {str(e)}")
                    continue

            logger.info(f"Optimizasyon tamamlandı. Başarılı deneme sayısı: {successful_attempts}/50")

            # Eğer optimizasyon başarısız olduysa yedek portföy kullan
            if best_result is None or best_score >= 1e6:
                logger.warning("Optimizasyon başarısız oldu, yedek portföy kullanılıyor")
                return self._generate_fallback_portfolio(min_stocks)

            # Sonuçları düzenle
            weights = best_result.x.copy()
            weights[weights < 0.01] = 0  # Küçük pozisyonları temizle
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)  # Normalize et
            
            # Sonuçları pandas Series'e çevir
            result_weights = pd.Series(weights, index=self.returns.columns)
            
            # Sonuç metrikleri
            metrics = {
                'Yıllık Getiri': self.calculate_portfolio_return(result_weights),
                'Yıllık Volatilite': self.calculate_portfolio_volatility(result_weights),
                'Sharpe Oranı': self.calculate_sharpe_ratio(result_weights),
                'Sortino Oranı': self.calculate_sortino_ratio(result_weights),
                'Aktif Pozisyon Sayısı': len(result_weights[result_weights > 0.01])
            }
            
            logger.info("Optimizasyon sonuç metrikleri:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.2f}")
            
            return result_weights

        except Exception as e:
            logger.error(f"Optimizasyon hatası: {str(e)}")
            return self._generate_fallback_portfolio(min_stocks)

    def _sharpe_objective(self, weights):
        """Sharpe oranı için objective function"""
        try:
            portfolio_return = np.sum(self._returns_mean * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self._cov_matrix, weights)))
            
            if portfolio_volatility < 1e-8:
                return 1e6
                
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe if np.isfinite(sharpe) else 1e6
        except:
            return 1e6

    def _volatility_objective(self, weights):
        """Volatilite için objective function"""
        try:
            return np.sqrt(np.dot(weights.T, np.dot(self._cov_matrix, weights)))
        except:
            return 1e6

    def _sortino_objective(self, weights):
        """Sortino oranı için objective function"""
        try:
            # Portföy getirilerini hesapla
            portfolio_returns = self._calculate_portfolio_returns(weights)
            if portfolio_returns is None:
                return 1e6
            
            # Yıllık getiri
            annual_return = np.mean(portfolio_returns) * 252
            
            # Downside volatilite hesaplama
            downside_std = self._calculate_downside_std(portfolio_returns)
            
            # Sortino oranı
            sortino = (annual_return - self.risk_free_rate) / downside_std
            
            # Geçerlilik kontrolleri
            if not np.isfinite(sortino) or sortino <= 0:
                return 1e6
            
            return -sortino  # Maksimizasyon için negatif
            
        except Exception as e:
            logger.error(f"Sortino objective hesaplama hatası: {str(e)}")
            return 1e6

    def calculate_sortino_ratio(self, weights):
        """Sortino oranını hesaplar."""
        try:
            # Portföy getirilerini hesapla
            portfolio_returns = self._calculate_portfolio_returns(weights)
            if portfolio_returns is None:
                logger.error("Portföy getirileri hesaplanamadı")
                return 0.0
            
            # Yıllık getiri
            annual_return = np.mean(portfolio_returns) * 252
            
            # Downside volatilite hesaplama
            downside_std = self._calculate_downside_std(portfolio_returns)
            
            # Sortino oranı
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_std
            
            # Sonuç kontrolü
            if not np.isfinite(sortino_ratio):
                logger.warning("Sortino oranı geçersiz bir değer")
                return 0.0
            
            if abs(sortino_ratio) > 100:
                logger.warning(f"Aşırı Sortino oranı hesaplandı: {sortino_ratio}")
                return np.sign(sortino_ratio) * 100
            
            return sortino_ratio
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation error: {str(e)}")
            return 0.0

    def _calculate_sortino_ratio(self, weights):
        """Sortino oranını hesaplar"""
        try:
            portfolio_returns = np.dot(self.returns.values, weights)
            annual_return = np.mean(portfolio_returns) * 252
            negative_returns = portfolio_returns[portfolio_returns < 0]
            
            if len(negative_returns) == 0:
                downside_std = 0.0001
            else:
                downside_std = max(np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252), 0.0001)
                
            return (annual_return - self.risk_free_rate) / downside_std
        except:
            return 0

    def _generate_initial_guesses(self, min_stocks):
        """Başlangıç tahminleri üretir"""
        initial_guesses = []
        
        # Eşit ağırlıklı
        initial_guesses.append(np.array([1./self.n_assets] * self.n_assets))
        
        # Rastgele ağırlıklar
        for _ in range(10):
            weights = np.random.random(self.n_assets)
            weights = weights / weights.sum()
            initial_guesses.append(weights)
        
        # Konsantre portföyler
        for _ in range(5):
            weights = np.zeros(self.n_assets)
            selected = np.random.choice(self.n_assets, size=min_stocks, replace=False)
            weights[selected] = 1/min_stocks
            initial_guesses.append(weights)
        
        return initial_guesses

    def _generate_fallback_portfolio(self, min_stocks):
        """Yedek portföy üretir"""
        weights = np.zeros(self.n_assets)
        selected = np.random.choice(self.n_assets, size=min_stocks, replace=False)
        weights[selected] = 1/min_stocks
        return pd.Series(weights, index=self.returns.columns)

    def _process_optimization_results(self, weights, min_stocks, max_weight):
        """Optimizasyon sonuçlarını işler ve düzenler"""
        # Numpy array'e çevir
        weights = np.array(weights)
        
        # Küçük ağırlıkları temizle
        weights[weights < 0.01] = 0
        
        # Minimum hisse kontrolü
        if np.sum(weights > 0) < min_stocks:
            top_indices = np.argsort(weights)[-min_stocks:]
            weights = np.zeros_like(weights)
            weights[top_indices] = 1/min_stocks
        
        # Normalize et
        weights = weights / np.sum(weights)
        
        # Maximum ağırlık kontrolü
        if np.any(weights > max_weight):
            excess = weights[weights > max_weight] - max_weight
            weights[weights > max_weight] = max_weight
            remaining_indices = weights < max_weight
            if np.sum(remaining_indices) > 0:
                weights[remaining_indices] += np.sum(excess) * weights[remaining_indices] / np.sum(weights[remaining_indices])
        
        # Son normalizasyon
        return weights / np.sum(weights)

    def cleanup(self):
        """
        Belleği temizle ve kaynakları serbest bırak.
        """
        # Önbelleği temizle
        self.get_sector_exposure.cache_clear()
        
        # Büyük veri yapılarını sil
        del self._cov_matrix
        del self._returns_mean
        
        logger.info("Cleanup completed")

    def calculate_portfolio_metrics(self, weights):
        """
        Hızlandırılmış portföy metrik hesaplamaları.
        """
        # Pandas Series'i numpy array'e çevir
        if isinstance(weights, pd.Series):
            weights = weights.values
            
        ret, vol, sharpe = self._calculate_portfolio_metrics_fast(
            weights.astype(np.float64), 
            self._returns_mean.astype(np.float64), 
            self._cov_matrix.astype(np.float64), 
            np.float64(self.risk_free_rate)
        )
        
        return {
            "Yıllık Getiri": ret,
            "Yıllık Volatilite": vol,
            "Sharpe Oranı": sharpe
        }

    @staticmethod
    @jit(nopython=True)
    def _calculate_portfolio_metrics_fast(weights, returns_mean, cov_matrix, risk_free_rate):
        """
        Numba ile hızlandırılmış portföy metrik hesaplamaları.
        """
        # Portföy getirisi
        portfolio_return = np.sum(returns_mean * weights)
        
        # Portföy volatilitesi
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe oranı
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else -np.inf
        
        return portfolio_return, portfolio_volatility, sharpe_ratio

    @staticmethod
    @jit(nopython=True)
    def _calculate_efficient_frontier_point(weights, returns_mean, cov_matrix, target_return):
        """
        Numba ile hızlandırılmış etkin sınır noktası hesaplaması.
        """
        # Veri tiplerini numpy float64'e çevir
        weights = np.asarray(weights, dtype=np.float64)
        returns_mean = np.asarray(returns_mean, dtype=np.float64)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        target_return = np.float64(target_return)
        
        portfolio_return = np.sum(returns_mean * weights)
        return_constraint = (portfolio_return - target_return) ** 2
        
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Return constraint'i ceza puanı olarak ekle
        return portfolio_volatility + 1e6 * return_constraint

    def _get_sector_mappings(self):
        """Hisse senetlerinin sektör bilgilerini döndürür."""
        # BIST hisseleri için sektör mapping'i
        sector_map = {
            'AKBNK': 'Bankacılık', 'GARAN': 'Bankacılık', 'YKBNK': 'Bankacılık',
            'THYAO': 'Havacılık', 'PGSUS': 'Havacılık',
            'BIMAS': 'Perakende',
            'TUPRS': 'Enerji', 'EREGL': 'Demir-Çelik',
            'KCHOL': 'Holding', 'SAHOL': 'Holding',
            'SISE': 'Sanayi', 'TOASO': 'Otomotiv',
            'TAVHL': 'Havalimanı', 'TKFEN': 'İnşaat',
            'VESTL': 'Elektronik', 'ARCLK': 'Elektronik',
            'ASELS': 'Savunma', 'EKGYO': 'GYO',
            'FROTO': 'Otomotiv', 'HEKTS': 'Kimya',
            'KOZAA': 'Madencilik', 'KOZAL': 'Madencilik',
            'KRDMD': 'Demir-Çelik', 'PETKM': 'Petrokimya',
            'SASA': 'Kimya'
        }
        return {stock: sector_map.get(stock.replace('.IS', ''), 'Diğer') for stock in self.returns.columns}

    def analyze_portfolio_risk(self, weights):
        """
        Portföy için detaylı risk analizi yapar.
        """
        risk_manager = EnhancedRiskManager(
            returns=self.returns,
            weights=weights,
            market_returns=self.market_returns,
            risk_free_rate=self.risk_free_rate
        )
        
        return risk_manager.generate_risk_report()

    def calculate_portfolio_return(self, weights):
        """Yıllık portföy getirisini hesaplar."""
        returns = np.sum(self.returns.mean() * weights) * 252
        return returns

    def calculate_portfolio_volatility(self, weights):
        """Yıllık portföy volatilitesini hesaplar."""
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        return volatility

    def calculate_sharpe_ratio(self, weights):
        """Sharpe oranını hesaplar."""
        ret = self.calculate_portfolio_return(weights)
        vol = self.calculate_portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol

    def backtest_portfolio(self, weights, start_date, end_date, 
                         rebalancing_period="Yok", stop_loss=None):
        """
        Portföy performansını test eder.
        """
        try:
            # Tarihleri datetime formatına çevir
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Tarih aralığındaki veriler
            returns = self.returns[start_date:end_date].copy()
            
            if returns.empty:
                raise ValueError("Seçilen tarih aralığında veri bulunamadı.")
            
            # Başlangıç portföy değeri
            portfolio_value = 1.0
            portfolio_values = [portfolio_value]
            current_weights = weights.copy()
            
            # Rebalancing periyodunu güne çevir
            period_map = {
                "Aylık": 21,      # Yaklaşık bir ay (işgünü)
                "3 Aylık": 63,    # Yaklaşık üç ay (işgünü)
                "6 Aylık": 126,   # Yaklaşık altı ay (işgünü)
                "Yıllık": 252,    # Bir yıl (işgünü)
                "Yok": len(returns)
            }
            rebalance_days = period_map.get(rebalancing_period, len(returns))
            
            # Her gün için portföy değerini hesapla
            days_since_rebalance = 0
            max_portfolio_value = portfolio_value
            
            for date, daily_returns in returns.iterrows():
                try:
                    # Günlük portföy getirisi
                    portfolio_return = np.sum(daily_returns * current_weights)
                    portfolio_value *= (1 + portfolio_return)
                    portfolio_values.append(portfolio_value)
                    
                    # Stop-loss kontrolü
                    if stop_loss and portfolio_value < max_portfolio_value * (1 - stop_loss):
                        print(f"Stop-loss tetiklendi: {date}")
                        break
                    
                    # Maximum portföy değerini güncelle
                    max_portfolio_value = max(max_portfolio_value, portfolio_value)
                    
                    # Rebalancing kontrolü
                    days_since_rebalance += 1
                    if days_since_rebalance >= rebalance_days:
                        current_weights = weights.copy()
                        days_since_rebalance = 0
                        
                except Exception as e:
                    print(f"Günlük hesaplama hatası ({date}): {str(e)}")
                    continue
            
            # Sonuçları hesapla
            portfolio_values = pd.Series(portfolio_values, index=[returns.index[0]] + list(returns.index))
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            
            # Yıllık getiriyi hesapla
            years = (end_date - start_date).days / 365
            annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Volatilite ve drawdown hesapla
            daily_returns = portfolio_values.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            drawdown = (portfolio_values / portfolio_values.cummax() - 1).min()
            
            return {
                "Toplam Getiri": total_return,
                "Yıllık Getiri": annual_return,
                "Volatilite": volatility,
                "Maksimum Drawdown": drawdown,
                "Günlük Değerler": portfolio_values
            }
            
        except Exception as e:
            print(f"Backtest sırasında hata: {str(e)}")
            return {
                "Toplam Getiri": 0,
                "Yıllık Getiri": 0,
                "Volatilite": 0,
                "Maksimum Drawdown": 0,
                "Günlük Değerler": pd.Series([1.0])
            }

    def generate_efficient_frontier(self, points=50):
        """
        Etkin sınır noktalarını üretir.
        
        Args:
            points (int): Üretilecek nokta sayısı
            
        Returns:
            pd.DataFrame: Etkin sınır noktaları (Risk, Getiri, Sharpe)
        """
        try:
            # Minimum ve maksimum getiri aralığını belirle
            min_ret = np.float64(np.min(self.returns.mean()) * 252)
            max_ret = np.float64(np.max(self.returns.mean()) * 252)
            
            # Getiri aralığını genişlet
            min_ret = np.float64(min(0, min_ret))  # En düşük getiriyi 0 veya daha düşük yap
            max_ret = np.float64(max_ret * 1.2)  # En yüksek getiriyi %20 artır
            
            target_returns = np.linspace(min_ret, max_ret, points, dtype=np.float64)
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    # Kısıtlar
                    constraints = [
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                        {'type': 'eq', 'fun': lambda x: np.sum(self.returns.mean() * x) * 252 - target_return}  # target return
                    ]
                    
                    # Bounds - her hisse için 0-1 arası
                    bounds = tuple((0, 1) for _ in range(self.n_assets))
                    
                    # Multiple optimization attempts
                    best_result = None
                    best_risk = np.inf
                    
                    # Farklı başlangıç noktaları dene
                    for _ in range(5):
                        # Random initial weights that sum to 1
                        initial_weights = np.random.random(self.n_assets).astype(np.float64)
                        initial_weights /= initial_weights.sum()
                        
                        try:
                            result = minimize(
                                lambda x: self.calculate_portfolio_volatility(x),
                                initial_weights,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={
                                    'maxiter': 1000,
                                    'ftol': 1e-8,
                                    'disp': False
                                }
                            )
                            
                            if result.success and result.fun < best_risk:
                                best_result = result
                                best_risk = result.fun
                                
                        except Exception as e:
                            logger.error(f"Optimizasyon denemesi başarısız: {str(e)}")
                            continue
                    
                    if best_result is not None:
                        weights = best_result.x.astype(np.float64)
                        risk = np.float64(self.calculate_portfolio_volatility(weights))
                        sharpe = np.float64((target_return - self.risk_free_rate) / risk if risk > 0 else -np.inf)
                        
                        efficient_portfolios.append({
                            'Risk': float(risk),
                            'Return': float(target_return),
                            'Sharpe': float(sharpe)
                        })
                        
                except Exception as e:
                    logger.error(f"Hedef getiri {target_return:.2%} için optimizasyon başarısız: {str(e)}")
                    continue
            
            if not efficient_portfolios:
                logger.error("Etkin sınır noktaları üretilemedi.")
                return None
            
            # Sort by risk and remove duplicates
            df = pd.DataFrame(efficient_portfolios)
            df = df.sort_values('Risk').drop_duplicates()
            
            # Anlamsız değerleri filtrele
            df = df[
                (df['Risk'] > 0) & 
                (df['Risk'] < 1) & 
                (df['Return'] > -1) & 
                (df['Return'] < 2) & 
                (df['Sharpe'] > -10) & 
                (df['Sharpe'] < 10)
            ]
            
            return df
            
        except Exception as e:
            logger.error(f"Etkin sınır üretilirken hata oluştu: {str(e)}")
            return None 

    def _calculate_portfolio_returns(self, weights):
        """Portföy getirilerini hesaplar."""
        try:
            if isinstance(weights, pd.Series):
                weights = weights.values
            weights = np.array(weights, dtype=np.float64)
            return np.dot(self.returns.values, weights)
        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {str(e)}")
            return None

    def _calculate_downside_std(self, portfolio_returns):
        """Downside volatiliteyi hesaplar."""
        try:
            negative_returns = portfolio_returns[portfolio_returns < 0]
            if len(negative_returns) == 0:
                return 0.0001
            
            downside_variance = np.mean(negative_returns ** 2)
            if downside_variance < 1e-10:
                return 0.0001
            
            downside_std = np.sqrt(downside_variance) * np.sqrt(252)
            return max(downside_std, 0.0001)
        except Exception as e:
            logger.error(f"Downside std calculation error: {str(e)}")
            return 0.0001 