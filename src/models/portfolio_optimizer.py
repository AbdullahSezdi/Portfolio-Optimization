import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, returns_data, risk_free_rate=0.15):
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns_data.columns)
        self.sector_mappings = self._get_sector_mappings()

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

    def optimize_portfolio(self, optimization_target="Maksimum Sharpe Oranı",
                         max_weight=0.2, sector_limit=0.3, min_stocks=5):
        """
        Portföy optimizasyonu yapar.
        """
        try:
            def objective(weights):
                try:
                    # Sektör limitini kontrol et
                    sector_weights = {}
                    for stock, weight in zip(self.returns.columns, weights):
                        sector = self.sector_mappings[stock]
                        sector_weights[sector] = sector_weights.get(sector, 0) + weight
                    
                    # Sektör limiti aşıldıysa büyük bir ceza puanı döndür
                    if any(weight > sector_limit for weight in sector_weights.values()):
                        return 1e6

                    if optimization_target == "Maksimum Sharpe Oranı":
                        return -self.calculate_sharpe_ratio(weights)
                    elif optimization_target == "Minimum Volatilite":
                        return self.calculate_portfolio_volatility(weights)
                    else:  # Maksimum Sortino Oranı
                        return -self.calculate_sortino_ratio(weights)
                except:
                    return 1e6  # Büyük bir sayı döndür (minimize edilecek)

            # Ana kısıtlar
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Ağırlıklar toplamı 1
            ]

            # Sektör kısıtları ekle
            for sector in set(self.sector_mappings.values()):
                sector_indices = [i for i, stock in enumerate(self.returns.columns) 
                                if self.sector_mappings[stock] == sector]
                if sector_indices:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, indices=sector_indices: sector_limit - sum(x[i] for i in indices)
                    })

            # Her hissenin sınırları
            bounds = tuple((0, max_weight) for _ in range(self.n_assets))

            # Optimizasyon denemeleri
            best_result = None
            best_score = float('inf')
            
            # Farklı başlangıç noktaları
            initial_guesses = []
            
            # 1. Eşit ağırlıklı başlangıç
            initial_guesses.append(np.array([1./self.n_assets] * self.n_assets))
            
            # 2. Rastgele başlangıçlar
            for _ in range(10):
                weights = np.random.random(self.n_assets)
                weights = weights / weights.sum() * 0.8  # Toplam 0.8 olsun (margin için)
                initial_guesses.append(weights)
            
            # 3. Konsantre başlangıçlar
            for _ in range(5):
                weights = np.zeros(self.n_assets)
                selected = np.random.choice(self.n_assets, size=min_stocks, replace=False)
                weights[selected] = 1/min_stocks
                initial_guesses.append(weights)

            for init_guess in initial_guesses:
                try:
                    result = minimize(
                        objective,
                        init_guess,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={
                            'maxiter': 1000,
                            'ftol': 1e-6,
                            'disp': False
                        }
                    )
                    
                    if result.success:
                        score = objective(result.x)
                        if score < best_score:
                            best_result = result
                            best_score = score
                except:
                    continue

            if best_result is None:
                raise Exception("Optimizasyon başarısız")

            # Sonuçları düzenle
            weights = best_result.x
            
            # Küçük ağırlıkları temizle
            weights[weights < 0.01] = 0
            
            # Minimum hisse sayısını kontrol et
            if np.sum(weights > 0) < min_stocks:
                # En iyi hisseleri seç
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

            # Son bir kez sektör limitlerini kontrol et ve düzelt
            sector_weights = {}
            for stock, weight in zip(self.returns.columns, weights):
                sector = self.sector_mappings[stock]
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

            # Sektör limitini aşan sektörleri düzelt
            for sector, total_weight in sector_weights.items():
                if total_weight > sector_limit:
                    # Sektördeki hisseleri bul
                    sector_stocks = [i for i, stock in enumerate(self.returns.columns) 
                                   if self.sector_mappings[stock] == sector]
                    
                    # Fazla ağırlığı hesapla
                    excess = total_weight - sector_limit
                    
                    # Sektördeki hisselerin ağırlıklarını orantılı olarak azalt
                    for idx in sector_stocks:
                        reduction = (weights[idx] / total_weight) * excess
                        weights[idx] -= reduction
                        
                        # Azaltılan ağırlığı diğer sektörlere dağıt
                        other_stocks = [i for i in range(self.n_assets) 
                                      if i not in sector_stocks and weights[i] > 0]
                        if other_stocks:
                            weight_increase = reduction / len(other_stocks)
                            for other_idx in other_stocks:
                                weights[other_idx] += weight_increase

            # Son bir kez normalize et
            weights = weights / np.sum(weights)

            return pd.Series(weights, index=self.returns.columns)

        except Exception as e:
            print(f"Optimizasyon sırasında hata: {str(e)}")
            # Basit bir portföy oluştur
            weights = np.zeros(self.n_assets)
            selected = np.random.choice(self.n_assets, size=min_stocks, replace=False)
            weights[selected] = 1/min_stocks
            return pd.Series(weights, index=self.returns.columns)

    def calculate_portfolio_metrics(self, weights):
        """Portföy metriklerini hesaplar."""
        annual_return = self.calculate_portfolio_return(weights)
        annual_volatility = self.calculate_portfolio_volatility(weights)
        sharpe_ratio = self.calculate_sharpe_ratio(weights)
        sortino_ratio = self.calculate_sortino_ratio(weights)
        
        return {
            "Yıllık Getiri": annual_return,
            "Yıllık Volatilite": annual_volatility,
            "Sharpe Oranı": sharpe_ratio,
            "Sortino Oranı": sortino_ratio
        }

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

    def calculate_sortino_ratio(self, weights):
        """Sortino oranını hesaplar."""
        ret = self.calculate_portfolio_return(weights)
        negative_returns = self.returns[self.returns < 0].fillna(0)
        downside_std = np.sqrt(np.dot(weights.T, np.dot(negative_returns.cov() * 252, weights)))
        return (ret - self.risk_free_rate) / downside_std if downside_std != 0 else 0

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
            min_ret = np.min(self.returns.mean()) * 252
            max_ret = np.max(self.returns.mean()) * 252
            
            # Getiri aralığını genişlet
            min_ret = min(0, min_ret)  # En düşük getiriyi 0 veya daha düşük yap
            max_ret = max_ret * 1.2  # En yüksek getiriyi %20 artır
            
            target_returns = np.linspace(min_ret, max_ret, points)
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
                    for _ in range(5):  # 3'ten 5'e çıkardık
                        # Random initial weights that sum to 1
                        initial_weights = np.random.random(self.n_assets)
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
                            print(f"Optimizasyon denemesi başarısız: {str(e)}")
                            continue
                    
                    if best_result is not None:
                        weights = best_result.x
                        risk = self.calculate_portfolio_volatility(weights)
                        sharpe = (target_return - self.risk_free_rate) / risk if risk > 0 else -np.inf
                        
                        efficient_portfolios.append({
                            'Risk': risk,
                            'Return': target_return,
                            'Sharpe': sharpe
                        })
                        
                except Exception as e:
                    print(f"Hedef getiri {target_return:.2%} için optimizasyon başarısız: {str(e)}")
                    continue
            
            if not efficient_portfolios:
                print("Etkin sınır noktaları üretilemedi.")
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
            print(f"Etkin sınır üretilirken hata oluştu: {str(e)}")
            return None 