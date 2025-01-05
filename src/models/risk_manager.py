import numpy as np
import pandas as pd

class RiskManager:
    def __init__(self, returns, weights):
        self.returns = returns
        self.weights = weights
        self.portfolio_returns = np.sum(returns * weights, axis=1)
        
    def calculate_var(self, confidence_level=0.95):
        """
        Value at Risk (VaR) hesaplar.
        """
        return -np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, confidence_level=0.95):
        """
        Conditional Value at Risk (CVaR) hesaplar.
        """
        var = self.calculate_var(confidence_level)
        return -self.portfolio_returns[self.portfolio_returns <= -var].mean()
    
    def calculate_max_drawdown(self):
        """
        Maksimum düşüşü hesaplar.
        """
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return drawdowns.min()
    
    def calculate_risk_metrics(self):
        """
        Tüm risk metriklerini hesaplar.
        """
        return {
            'var_metrics': {
                'var_95': self.calculate_var(0.95),
                'var_99': self.calculate_var(0.99),
                'cvar_95': self.calculate_cvar(0.95),
                'cvar_99': self.calculate_cvar(0.99)
            },
            'volatility': self.portfolio_returns.std() * np.sqrt(252),  # Yıllık volatilite
            'skewness': self.portfolio_returns.skew(),
            'kurtosis': self.portfolio_returns.kurtosis()
        } 