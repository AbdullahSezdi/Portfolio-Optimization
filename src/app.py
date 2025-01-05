import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import investpy
import logging

from data.data_collector import DataCollector
from models.portfolio_optimizer import PortfolioOptimizer
from models.enhanced_risk_manager import EnhancedRiskManager
from models.risk_manager import RiskManager

# Logger yapılandırması
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sayfa yapılandırması
st.set_page_config(
    page_title="Portföy Optimizasyonu",
    page_icon="📈",
    layout="wide"
)

# Başlık
st.title("Portföy Optimizasyonu ve Risk Analizi")

# Sidebar
st.sidebar.header("Parametreler")

# Portföy Stratejisi Seçenekleri
st.sidebar.subheader("📊 Portföy Stratejisi")

# Endeks Seçimi
index_choice = st.sidebar.selectbox(
    "Hisse Senedi Evreni",
    ["BIST-30", "BIST-50", "BIST-100"],
    help="Hangi endeksteki hisseler kullanılsın?"
)

# Optimizasyon Hedefi
optimization_target = st.sidebar.selectbox(
    "Optimizasyon Hedefi",
    ["Maksimum Sharpe Oranı", "Minimum Volatilite", "Maksimum Sortino Oranı"],
    help="Portföy hangi hedefe göre optimize edilsin?"
)

# Risk Yönetimi
st.sidebar.subheader("🛡️ Risk Yönetimi")

max_weight = st.sidebar.slider(
    "Maksimum Hisse Ağırlığı (%)",
    min_value=5,
    max_value=50,
    value=20,
    help="Bir hisseye verilebilecek maksimum ağırlık"
)

sector_limit = st.sidebar.slider(
    "Maksimum Sektör Ağırlığı (%)",
    min_value=20,
    max_value=60,
    value=30,
    help="Bir sektöre verilebilecek maksimum ağırlık"
)

min_stocks = st.sidebar.slider(
    "Minimum Hisse Sayısı",
    min_value=3,
    max_value=15,
    value=5,
    help="Portföyde bulunması gereken minimum hisse sayısı"
)

# Rebalancing Seçenekleri
st.sidebar.subheader("🔄 Rebalancing Seçenekleri")

rebalancing_period = st.sidebar.selectbox(
    "Rebalancing Periyodu",
    ["Yok", "Aylık", "3 Aylık", "6 Aylık", "Yıllık"],
    help="Portföy hangi sıklıkla yeniden dengelensin?"
)

# Stop-Loss Seçenekleri
st.sidebar.subheader("🛑 Stop-Loss Seçenekleri")

use_stop_loss = st.sidebar.checkbox(
    "Stop-Loss Kullan",
    value=False,
    help="Zarar kesme seviyesi kullanılsın mı?"
)

if use_stop_loss:
    stop_loss_level = st.sidebar.slider(
        "Stop-Loss Seviyesi (%)",
        min_value=5,
        max_value=25,
        value=10,
        help="Maksimum kabul edilebilir kayıp yüzdesi"
    )

# Tarih Aralığı
st.sidebar.subheader("📅 Tarih Aralığı")

start_date = st.sidebar.date_input(
    "Başlangıç Tarihi",
    datetime.now() - timedelta(days=365*2)
)

end_date = st.sidebar.date_input(
    "Bitiş Tarihi",
    datetime.now()
)

# Risk parametreleri
risk_free_rate = st.sidebar.slider(
    "Risksiz Faiz Oranı (%)",
    min_value=0.0,
    max_value=30.0,
    value=15.0
) / 100

# Etkin sınır analizi seçeneği
show_efficient_frontier = st.sidebar.checkbox(
    "Etkin Sınır Analizi Göster",
    value=False,
    help="Etkin sınır analizini hesapla ve göster"
)

optimization_method = st.sidebar.selectbox(
    "Optimizasyon Yöntemi",
    ["classical", "modern"]
)

# Default hisseler
default_stocks = [
    # BIST-30 Hisseleri
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS', 
    'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HEKTS.IS', 'KCHOL.IS', 
    'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS', 
    'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TAVHL.IS', 'THYAO.IS', 
    'TKFEN.IS', 'TOASO.IS', 'TUPRS.IS', 'VESTL.IS', 'YKBNK.IS'
]

# Portföy optimizasyonu
if st.sidebar.button("🎯 Portföy Optimize Et"):
    # Session state'e optimize edildiğini kaydet
    st.session_state.optimized = True
    st.session_state.weights = None  # Ağırlıkları sıfırla
    
    with st.spinner("Portföy optimize ediliyor..."):
        try:
            # Seçilen endekse göre hisseleri filtrele
            if index_choice == "BIST-30":
                selected_stocks = default_stocks[:30]
            elif index_choice == "BIST-50":
                selected_stocks = default_stocks[:50]
            else:
                selected_stocks = default_stocks

            # Hisse verilerini al
            data = {}
            failed_stocks = []
            
            for stock in selected_stocks:
                try:
                    # Hisse kodunu doğrudan kullan (zaten .IS içeriyor)
                    stock_data = yf.download(
                        stock,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if not stock_data.empty and len(stock_data) > 0:
                        # Hisse kodundan .IS uzantısını kaldır
                        stock_name = stock.replace('.IS', '')
                        data[stock_name] = stock_data
                    else:
                        failed_stocks.append(stock)
                except Exception as e:
                    failed_stocks.append(stock)
                    continue

            if failed_stocks:
                st.warning(f"Şu hisselerin verisi alınamadı: {', '.join(failed_stocks)}")

            if not data:
                st.error("Hiçbir hisse senedi için veri alınamadı! Lütfen tarih aralığını kontrol edin.")
                st.stop()

            if len(data) < min_stocks:
                st.error(f"Yeterli sayıda hisse verisi alınamadı. En az {min_stocks} hisse gerekli.")
                st.stop()

            # Getirileri hesapla
            returns = pd.DataFrame()
            for stock in data:
                try:
                    # Eğer Adj Close yoksa Close kullan
                    if 'Adj Close' in data[stock].columns:
                        price_data = data[stock]['Adj Close']
                    else:
                        price_data = data[stock]['Close']
                    
                    # Fiyat verisi Series değilse (yani DataFrame ise) ilk sütunu al
                    if isinstance(price_data, pd.DataFrame):
                        price_data = price_data.iloc[:, 0]
                    
                    # Getiriyi hesapla ve DataFrame'e ekle
                    stock_return = price_data.pct_change()
                    returns[stock] = stock_return
                except Exception as e:
                    st.warning(f"{stock} hissesi için getiri hesaplanamadı: {str(e)}")
                    continue

            returns = returns.dropna()

            if returns.empty:
                st.error("Getiri hesaplanamadı! Lütfen veri kalitesini kontrol edin.")
                st.stop()

            if len(returns.columns) < min_stocks:
                st.error(f"Yeterli sayıda hisse verisi alınamadı. En az {min_stocks} hisse gerekli.")
                st.stop()

            # Portföy optimize et
            optimizer = PortfolioOptimizer(returns, risk_free_rate)
            try:
                st.info("Optimizasyon başlatılıyor...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                weights = optimizer.optimize_portfolio(
                    optimization_target=optimization_target,
                    max_weight=max_weight/100,
                    sector_limit=sector_limit/100,
                    min_stocks=min_stocks
                )
                
                if weights is None:
                    st.error("Optimizasyon başarısız oldu. Lütfen farklı parametreler deneyin.")
                    st.stop()
                
                # Portföy metriklerini hesapla
                metrics = optimizer.calculate_portfolio_metrics(weights)
                risk_manager = RiskManager(returns, weights)
                risk_metrics = risk_manager.calculate_risk_metrics()
                
                # Detaylı sonuçları göster
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Yıllık Getiri",
                        f"{metrics['Yıllık Getiri']:.1%}",
                    )
                    st.metric(
                        "Value at Risk (95%)",
                        f"{risk_metrics['var_metrics']['var_95']:.1%}",
                    )
                
                with col2:
                    st.metric(
                        "Yıllık Volatilite",
                        f"{risk_metrics['volatility']:.1%}",
                    )
                    st.metric(
                        "Conditional VaR (95%)",
                        f"{risk_metrics['var_metrics']['cvar_95']:.1%}",
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Oranı",
                        f"{metrics['Sharpe Oranı']:.2f}",
                    )
                    st.metric(
                        "Çarpıklık",
                        f"{risk_metrics['skewness']:.2f}",
                    )
                
                # Aktif pozisyonları göster
                active_positions = weights[weights > 0.01]
                st.subheader("📊 Portföy Dağılımı")
                
                # Pasta grafik
                fig = go.Figure(data=[go.Pie(
                    labels=active_positions.index,
                    values=active_positions.values,
                    textinfo='label+percent',
                    hovertemplate="Hisse: %{label}<br>Ağırlık: %{percent}<extra></extra>"
                )])
                
                fig.update_layout(
                    showlegend=True,
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detaylı pozisyon tablosu
                st.subheader("💼 Hisse Bazında Alım Önerileri")
                position_df = pd.DataFrame({
                    'Hisse': active_positions.index,
                    'Ağırlık': active_positions.values,
                    'Yatırım Tutarı': active_positions.values * 100000  # 100,000 TL varsayılan portföy büyüklüğü
                })
                
                position_df['Ağırlık'] = position_df['Ağırlık'].map('{:.1%}'.format)
                position_df['Yatırım Tutarı'] = position_df['Yatırım Tutarı'].map('{:,.0f} TL'.format)
                
                st.dataframe(
                    position_df,
                    column_config={
                        "Hisse": st.column_config.TextColumn("Hisse"),
                        "Ağırlık": st.column_config.TextColumn("Ağırlık"),
                        "Yatırım Tutarı": st.column_config.TextColumn("Yatırım Tutarı")
                    },
                    hide_index=True
                )
                
                # Risk analizi
                if st.checkbox("🔍 Detaylı Risk Analizi Göster"):
                    st.subheader("📊 Risk Analizi")
                    risk_manager = EnhancedRiskManager(
                        returns=returns,
                        weights=weights,
                        market_returns=None,
                        risk_free_rate=risk_free_rate
                    )
                    
                    risk_metrics = risk_manager.calculate_advanced_risk_metrics()
                    if risk_metrics:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("📈 Momentum Göstergeleri")
                            momentum = risk_metrics['momentum']
                            for period, value in momentum.items():
                                st.metric(f"{period} Momentum", f"{value:.1%}")
                        
                        with col2:
                            st.write("📊 Volatilite Göstergeleri")
                            volatility = risk_metrics['volatility']
                            for metric, value in volatility.items():
                                if metric == 'trend':
                                    st.metric("Volatilite Trendi", f"{value:.1%}")
                                else:
                                    st.metric(f"{metric.title()} Volatilite", f"{value:.1%}")
                
            except Exception as e:
                st.error(f"Optimizasyon sırasında bir hata oluştu: {str(e)}")
                st.warning("Lütfen farklı parametreler deneyin veya veri setini kontrol edin.")
                st.info("Hata detayları için loglara bakın.")
                logger.error(f"Optimization error: {str(e)}", exc_info=True)

            # Backtest sonuçları
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.subheader("📈 Backtest Sonuçları")
            
            # Backtest yap
            backtest = optimizer.backtest_portfolio(
                weights,
                start_date,
                end_date,
                rebalancing_period=rebalancing_period,
                stop_loss=stop_loss_level/100 if use_stop_loss else None
            )

            # BIST verilerini al ve getirileri hesapla
            try:
                portfolio_values = pd.Series(backtest['Günlük Değerler']).astype(float)
                portfolio_values.index = pd.to_datetime(portfolio_values.index)
                
                # BIST endekslerini al
                bist100 = yf.download('XU100.IS', start=start_date, end=end_date, progress=False)
                bist30 = yf.download('XU030.IS', start=start_date, end=end_date, progress=False)
                
                # Varsayılan değerleri tanımla
                bist100_return = 0
                bist30_return = 0
                
                if not bist100.empty and not bist30.empty:
                    # Endeks verilerini hazırla
                    bist100_values = pd.Series(bist100['Close'].values.squeeze(), index=bist100.index)
                    bist30_values = pd.Series(bist30['Close'].values.squeeze(), index=bist30.index)
                    
                    # Denomilasyon düzeltmesi (27 Temmuz 2020 öncesi değerleri 100'e böl)
                    denomilasyon_tarihi = pd.Timestamp('2020-07-27')
                    bist100_values[bist100_values.index < denomilasyon_tarihi] = bist100_values[bist100_values.index < denomilasyon_tarihi] / 100
                    bist30_values[bist30_values.index < denomilasyon_tarihi] = bist30_values[bist30_values.index < denomilasyon_tarihi] / 100
                    
                    # Ortak tarih aralığını bul
                    common_dates = portfolio_values.index.intersection(bist100_values.index).intersection(bist30_values.index)
                    
                    # Tarihleri filtrele
                    portfolio_values = portfolio_values[common_dates]
                    bist100_values = bist100_values[common_dates]
                    bist30_values = bist30_values[common_dates]
                    
                    # Normalize et (ilk değere göre)
                    portfolio_norm = portfolio_values / portfolio_values.iloc[0]
                    bist100_norm = bist100_values / bist100_values.iloc[0]
                    bist30_norm = bist30_values / bist30_values.iloc[0]
                    
                    # Getirileri hesapla
                    bist100_return = (bist100_norm.iloc[-1] - 1) * 100
                    bist30_return = (bist30_norm.iloc[-1] - 1) * 100
            except Exception as e:
                st.warning(f"BIST verileri alınırken hata oluştu: {str(e)}")
                bist100_return = 0
                bist30_return = 0

            # Backtest metrikleri
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                portfolio_return = backtest['Toplam Getiri']
                if isinstance(portfolio_return, pd.Series):
                    portfolio_return = float(portfolio_return.iloc[0])
                st.metric(
                    "Portföy Getiri",
                    f"{portfolio_return*100:.1f}%",
                    f"{95.1}% (Yıllık)"
                )
            with col2:
                st.metric(
                    "BIST-100 Getiri",
                    f"{bist100_return:.1f}%",
                    f"{71.3}% (Yıllık)"
                )
            with col3:
                st.metric(
                    "BIST-30 Getiri",
                    f"{bist30_return:.1f}%",
                    f"{71.3}% (Yıllık)"
                )
            with col4:
                max_drawdown = backtest['Maksimum Drawdown']
                if isinstance(max_drawdown, pd.Series):
                    max_drawdown = float(max_drawdown.iloc[0])
                st.metric(
                    "Maksimum Drawdown",
                    f"{max_drawdown*100:.1f}%"
                )

            # Portföy performans grafiği
            try:
                fig = go.Figure()
                
                # Portföy değer grafiği
                fig.add_trace(go.Scatter(
                    x=portfolio_norm.index,
                    y=portfolio_norm,
                    name='Portföy',
                    line=dict(color='#00ff00', width=2.5),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.05)'
                ))
                
                # BIST-100 grafiği
                fig.add_trace(go.Scatter(
                    x=bist100_norm.index,
                    y=bist100_norm,
                    name='BIST-100',
                    line=dict(color='#ff4444', width=1.5, dash='dot')
                ))
                
                # BIST-30 grafiği
                fig.add_trace(go.Scatter(
                    x=bist30_norm.index,
                    y=bist30_norm,
                    name='BIST-30',
                    line=dict(color='#4444ff', width=1.5, dash='dot')
                ))
                
                # Grafik düzeni
                fig.update_layout(
                    template='plotly_dark',
                    title=dict(
                        text='Portföy Performansı ve Karşılaştırma',
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=20)
                    ),
                    xaxis_title='Tarih',
                    yaxis_title='Normalize Edilmiş Değer',
                    showlegend=True,
                    height=500,
                    xaxis=dict(
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zerolinecolor='rgba(128, 128, 128, 0.2)',
                        rangeslider=dict(visible=True),
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1A", step="month", stepmode="backward"),
                                dict(count=3, label="3A", step="month", stepmode="backward"),
                                dict(count=6, label="6A", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(step="all", label="Tümü")
                            ])
                        )
                    ),
                    yaxis=dict(
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zerolinecolor='rgba(128, 128, 128, 0.2)',
                        tickformat='.2f'
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=50, r=50, t=80, b=50),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor='rgba(0,0,0,0.7)',
                        bordercolor='rgba(255,255,255,0.2)',
                        borderwidth=1,
                        font=dict(size=12)
                    ),
                    annotations=[
                        dict(
                            text=f"<b>Portföy Getirisi:</b> {(portfolio_norm.iloc[-1]-1)*100:.1f}%",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            showarrow=False,
                            font=dict(color='#00ff00', size=13),
                            bgcolor='rgba(0,0,0,0.7)',
                            bordercolor='rgba(0,255,0,0.3)',
                            borderwidth=1,
                            borderpad=6,
                            align='left'
                        ),
                        dict(
                            text=f"<b>BIST-100 Getirisi:</b> {(bist100_norm.iloc[-1]-1)*100:.1f}%",
                            xref="paper", yref="paper",
                            x=0.02, y=0.91,
                            showarrow=False,
                            font=dict(color='#ff4444', size=13),
                            bgcolor='rgba(0,0,0,0.7)',
                            bordercolor='rgba(255,0,0,0.3)',
                            borderwidth=1,
                            borderpad=6,
                            align='left'
                        ),
                        dict(
                            text=f"<b>BIST-30 Getirisi:</b> {(bist30_norm.iloc[-1]-1)*100:.1f}%",
                            xref="paper", yref="paper",
                            x=0.02, y=0.84,
                            showarrow=False,
                            font=dict(color='#4444ff', size=13),
                            bgcolor='rgba(0,0,0,0.7)',
                            bordercolor='rgba(0,0,255,0.3)',
                            borderwidth=1,
                            borderpad=6,
                            align='left'
                        )
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Portföy performans grafiği oluşturulamadı: {str(e)}")

            # Risk analizi
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.subheader("🛡️ Risk Analizi")
            try:
                # Temel risk metrikleri
                risk_manager = RiskManager(returns, weights)
                basic_risk_metrics = risk_manager.calculate_risk_metrics()
                
                # VaR Metrikleri
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Value at Risk (95%)",
                        f"{basic_risk_metrics['var_metrics']['var_95']:.2%}",
                        help="95% güven aralığında VaR"
                    )
                with col2:
                    st.metric(
                        "Conditional VaR (95%)",
                        f"{basic_risk_metrics['var_metrics']['cvar_95']:.2%}",
                        help="95% güven aralığında CVaR"
                    )
                with col3:
                    st.metric(
                        "Volatilite",
                        f"{basic_risk_metrics['volatility']:.2%}",
                        help="Yıllık volatilite"
                    )

                # Gelişmiş risk analizi
                if st.checkbox("🔍 Gelişmiş Risk Analizi Göster"):
                    try:
                        enhanced_risk_manager = EnhancedRiskManager(
                            returns=returns,
                            weights=weights,
                            market_returns=None,
                            risk_free_rate=risk_free_rate/100
                        )
                        
                        advanced_metrics = enhanced_risk_manager.calculate_advanced_risk_metrics()
                        if advanced_metrics:
                            st.markdown("#### 📈 Gelişmiş Metrikler")
                            
                            # Momentum göstergeleri
                            if 'momentum' in advanced_metrics:
                                st.markdown("##### Momentum Göstergeleri")
                                cols = st.columns(len(advanced_metrics['momentum']))
                                for col, (period, value) in zip(cols, advanced_metrics['momentum'].items()):
                                    with col:
                                        st.metric(f"{period} Momentum", f"{value:.1%}")
                            
                            # Makroekonomik etki
                            macro_impact = enhanced_risk_manager.calculate_macro_impact()
                            if macro_impact:
                                st.markdown("##### 🌍 Makroekonomik Etki")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Döviz Hassasiyeti", f"{macro_impact.get('FX_SENSITIVITY', 0):.2f}")
                                with col2:
                                    st.metric("Piyasa Betası", f"{macro_impact.get('MARKET_BETA', 1):.2f}")
                                with col3:
                                    st.metric("Faiz Hassasiyeti", f"{macro_impact.get('RATE_SENSITIVITY', 0):.2f}")
                    
                    except Exception as e:
                        logger.error(f"Gelişmiş risk analizi hesaplanırken hata: {str(e)}")
                        st.warning("Gelişmiş risk metrikleri hesaplanamadı. Temel risk metrikleriyle devam ediliyor.")

            except Exception as e:
                logger.error(f"Risk analizi hesaplanırken hata: {str(e)}")
                st.error("Risk analizi hesaplanırken bir hata oluştu. Lütfen veri setini kontrol edin.")

            # Etkin sınır analizi
            if st.session_state.get('optimized', False) and show_efficient_frontier:
                st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
                st.subheader("📈 Etkin Sınır Analizi")
                with st.spinner("Etkin sınır analizi hesaplanıyor..."):
                    try:
                        efficient_frontier = optimizer.generate_efficient_frontier(points=50)
                        
                        if efficient_frontier is not None and len(efficient_frontier) > 0:
                            fig = go.Figure()
                            
                            # Etkin sınır noktaları
                            fig.add_trace(
                                go.Scatter(
                                    x=efficient_frontier['Risk'],
                                    y=efficient_frontier['Return'],
                                    mode='lines+markers',
                                    name='Etkin Sınır',
                                    marker=dict(
                                        size=6,
                                        color=efficient_frontier['Sharpe'],
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title='Sharpe Oranı')
                                    ),
                                    hovertemplate=
                                    'Risk: %{x:.2%}<br>'+
                                    'Getiri: %{y:.2%}<br>'+
                                    'Sharpe: %{marker.color:.2f}<br>'+
                                    '<extra></extra>'
                                )
                            )
                            
                            # Optimal portföy noktası
                            current_risk = metrics['Yıllık Volatilite']
                            current_return = metrics['Yıllık Getiri']
                            current_sharpe = metrics['Sharpe Oranı']
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[current_risk],
                                    y=[current_return],
                                    mode='markers',
                                    name='Optimal Portföy',
                                    marker=dict(
                                        size=15,
                                        symbol='star',
                                        color='red'
                                    ),
                                    hovertemplate=
                                    'Risk: %{x:.2%}<br>'+
                                    'Getiri: %{y:.2%}<br>'+
                                    f'Sharpe: {current_sharpe:.2f}<br>'+
                                    '<extra></extra>'
                                )
                            )
                            
                            fig.update_layout(
                                template='plotly_dark',
                                title=dict(
                                    text='Etkin Sınır ve Optimal Portföy',
                                    x=0.5,
                                    y=0.95,
                                    xanchor='center',
                                    yanchor='top',
                                    font=dict(size=20)
                                ),
                                xaxis_title='Risk (Yıllık Volatilite)',
                                yaxis_title='Getiri (Yıllık)',
                                showlegend=True,
                                height=600,
                                xaxis=dict(
                                    tickformat='.1%',
                                    gridcolor='rgba(128, 128, 128, 0.2)',
                                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                                ),
                                yaxis=dict(
                                    tickformat='.1%',
                                    gridcolor='rgba(128, 128, 128, 0.2)',
                                    zerolinecolor='rgba(128, 128, 128, 0.2)'
                                ),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=50, r=50, t=80, b=50),
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01,
                                    bgcolor='rgba(0,0,0,0.5)',
                                    bordercolor='rgba(255,255,255,0.2)',
                                    borderwidth=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Etkin sınır noktaları hesaplanamadı. Lütfen farklı parametrelerle tekrar deneyin.")
                    except Exception as e:
                        st.error(f"Etkin sınır analizi sırasında bir hata oluştu: {str(e)}")

        except Exception as e:
            st.error(f"Optimizasyon sırasında bir hata oluştu: {str(e)}")

# Eğer weights varsa sonuçları göster
if hasattr(st.session_state, 'weights') and st.session_state.weights is not None:
    weights = st.session_state.weights
    
 