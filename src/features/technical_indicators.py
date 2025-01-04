import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self):
        pass
        
    def add_sma(self, data, period=20):
        """Basit Hareketli Ortalama (SMA) hesapla"""
        return data['Close'].rolling(window=period).mean()
        
    def add_rsi(self, data, period=14):
        """Göreceli Güç Endeksi (RSI) hesapla"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def add_all_indicators(self, data):
        """Tüm teknik göstergeleri hesapla"""
        df = data.copy()
        
        # SMA'ları hesapla
        df['SMA_20'] = self.add_sma(df, 20)
        df['SMA_50'] = self.add_sma(df, 50)
        
        # RSI hesapla
        df['RSI'] = self.add_rsi(df)
        
        return df 