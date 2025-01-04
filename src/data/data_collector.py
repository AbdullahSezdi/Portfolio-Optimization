import yfinance as yf
import pandas as pd
import numpy as np
import logging

class DataCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fetch_stock_data(self, symbols, start_date, end_date, interval='1d'):
        """
        Verilen semboller için hisse senedi verilerini çeker.
        """
        data = {}
        for symbol in symbols:
            try:
                stock = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                if not stock.empty:
                    data[symbol] = stock
                else:
                    self.logger.warning(f"No data found for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return data
    
    def calculate_returns(self, stock_data, method='simple'):
        """
        Hisse senedi getirilerini hesaplar.
        """
        if method == 'simple':
            returns = stock_data['Close'].pct_change()
        elif method == 'log':
            returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        else:
            raise ValueError("Method must be either 'simple' or 'log'")
        
        return returns.dropna()
    
    def fetch_market_index(self, index_symbol='^XU100.IS', start_date=None, end_date=None):
        """
        Piyasa endeks verilerini çeker (varsayılan olarak BIST 100).
        """
        try:
            index_data = yf.download(index_symbol, start=start_date, end=end_date)
            return index_data
        except Exception as e:
            self.logger.error(f"Error fetching market index data: {str(e)}")
            return None 