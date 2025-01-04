import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import investpy

from data.data_collector import DataCollector
from models.portfolio_optimizer import PortfolioOptimizer
from models.risk_manager import RiskManager

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
                # Eğer Adj Close yoksa Close kullan
                if 'Adj Close' in data[stock].columns:
                    price_data = data[stock]['Adj Close']
                else:
                    price_data = data[stock]['Close']
                returns[stock] = price_data.pct_change()
            returns = returns.dropna()

            if returns.empty:
                st.error("Getiri hesaplanamadı! Lütfen veri kalitesini kontrol edin.")
                st.stop()

            if len(returns.columns) < min_stocks:
                st.error(f"Yeterli sayıda hisse verisi alınamadı. En az {min_stocks} hisse gerekli.")
                st.stop()

            # Portföy optimize et
            optimizer = PortfolioOptimizer(returns, risk_free_rate)
            weights = optimizer.optimize_portfolio(
                optimization_target=optimization_target,
                max_weight=max_weight/100,
                sector_limit=sector_limit/100,
                min_stocks=min_stocks
            )

            # Ağırlıkları session state'e kaydet
            st.session_state.weights = weights
            
            # Sonuçları göster
            st.header("📊 Optimizasyon Sonuçları")
            
            # Portföy metriklerini göster
            metrics = optimizer.calculate_portfolio_metrics(weights)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Yıllık Getiri", f"{metrics['Yıllık Getiri']:.1%}")
            with col2:
                st.metric("Yıllık Volatilite", f"{metrics['Yıllık Volatilite']:.1%}")
            with col3:
                st.metric("Sharpe Oranı", f"{metrics['Sharpe Oranı']:.2f}")
            with col4:
                st.metric("Sortino Oranı", f"{metrics['Sortino Oranı']:.2f}")

            # Risk analizi
            st.subheader("🛡️ Risk Analizi")
            try:
                risk_manager = RiskManager(returns, weights)
                risk_metrics = risk_manager.calculate_risk_metrics()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Value at Risk (95%)",
                        f"{risk_metrics['var_95']:.2%}",
                        help="95% güven aralığında maksimum kayıp"
                    )
                    
                with col2:
                    st.metric(
                        "Conditional VaR (95%)",
                        f"{risk_metrics['cvar_95']:.2%}",
                        help="VaR'ı aşan kayıpların ortalaması"
                    )
                    
                with col3:
                    st.metric(
                        "Maximum Drawdown",
                        f"{risk_metrics['max_drawdown']:.2%}",
                        help="En yüksek değerden en düşük değere maksimum düşüş"
                    )
            except Exception as e:
                st.error(f"Risk analizi hesaplanırken hata oluştu: {str(e)}")

            # Portföy dağılımını göster
            st.subheader("Optimal Portföy Dağılımı")
            
            # Yatırım tutarı girişi
            if 'investment_amount' not in st.session_state:
                st.session_state.investment_amount = 100000
                
            investment_amount = st.number_input(
                "Yatırım Tutarı (TL)",
                min_value=1000,
                max_value=10000000,
                value=st.session_state.investment_amount,
                step=1000,
                format="%d",
                key="investment_amount_input"
            )
            # Değeri session state'e kaydet
            st.session_state.investment_amount = investment_amount
            
            # Pasta grafiği ve alım önerileri yan yana
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="column-gap">', unsafe_allow_html=True)
                # Portföy pasta grafiği
                fig_pie = px.pie(values=weights[weights > 0.001],
                               names=weights[weights > 0.001].index,
                               title="Hisse Ağırlıkları")
                st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart_1")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="column-gap">', unsafe_allow_html=True)
                # Hisse bazında alım önerileri
                st.markdown("### 💰 Hisse Bazında Alım Önerileri")
                
                # CSS ile tablo stilini özelleştir
                st.markdown("""
                <style>
                .dataframe {
                    margin-top: 20px;
                    margin-bottom: 20px;
                    margin-left: 40px;
                    width: calc(100% - 40px);
                }
                .column-gap { 
                    padding: 0 30px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Alım önerilerini hesapla ve göster
                recommendations = []
                for stock, weight in weights[weights > 0.001].items():
                    amount = st.session_state.investment_amount * weight
                    recommendations.append({
                        'Hisse': stock.replace('.IS', ''),
                        'Ağırlık': f'{weight*100:.1f}%',
                        'Yatırım Tutarı': f'{amount:,.0f} TL'
                    })
                
                if recommendations:
                    df_recommendations = pd.DataFrame(recommendations)
                    st.markdown(
                        df_recommendations.to_html(
                            escape=False,
                            index=False,
                            columns=['Hisse', 'Ağırlık', 'Yatırım Tutarı'],
                            classes=['dataframe'],
                            justify='center'
                        ),
                        unsafe_allow_html=True
                    )

            # Backtest sonuçları
            st.subheader("📈 Backtest Sonuçları")
            
            # Backtest yap
            backtest = optimizer.backtest_portfolio(
                weights,
                start_date,
                end_date,
                rebalancing_period=rebalancing_period,
                stop_loss=stop_loss_level/100 if use_stop_loss else None
            )
            
            # BIST endekslerini al ve getirileri hesapla
            try:
                bist100 = yf.download('XU100.IS', start=start_date, end=end_date, progress=False)
                bist30 = yf.download('XU030.IS', start=start_date, end=end_date, progress=False)
                
                if not bist100.empty and len(bist100) > 0 and not bist30.empty and len(bist30) > 0:
                    # Denomilasyon düzeltmesi
                    denomilasyon_tarihi = pd.Timestamp('2020-07-27')
                    bist100_values = bist100['Close'].copy()
                    bist30_values = bist30['Close'].copy()
                    
                    if len(bist100_values) > 0 and len(bist30_values) > 0:
                        # Denomilasyon düzeltmesi
                        mask100 = bist100_values.index < denomilasyon_tarihi
                        mask30 = bist30_values.index < denomilasyon_tarihi
                        
                        if any(mask100):
                            bist100_values[mask100] = bist100_values[mask100] / 100
                        if any(mask30):
                            bist30_values[mask30] = bist30_values[mask30] / 100
                        
                        # Endeks getirilerini hesapla
                        if len(bist100_values) > 1 and len(bist30_values) > 1:
                            bist100_return = ((bist100_values.iloc[-1] / bist100_values.iloc[0]) - 1) * 100
                            bist30_return = ((bist30_values.iloc[-1] / bist30_values.iloc[0]) - 1) * 100
                        else:
                            bist100_return = 0
                            bist30_return = 0
                    else:
                        bist100_return = 0
                        bist30_return = 0
                else:
                    bist100_return = 0
                    bist30_return = 0
            except Exception as e:
                bist100_return = 0
                bist30_return = 0
                st.warning(f"Endeks verileri alınamadı: {str(e)}")
            
            # Backtest metriklerini göster
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_return = backtest['Toplam Getiri']
                annual_return = backtest['Yıllık Getiri']
                if isinstance(total_return, pd.Series):
                    total_return = float(total_return.iloc[0])
                if isinstance(annual_return, pd.Series):
                    annual_return = float(annual_return.iloc[0])
                st.metric(
                    "Portföy Toplam Getiri",
                    f"{total_return*100:.1f}%",
                    f"{annual_return*100:.1f}% (Yıllık)"
                )
            with col2:
                bist100_return_val = bist100_return if isinstance(bist100_return, (int, float)) else bist100_return.iloc[0]
                st.metric(
                    "BIST-100 Getiri",
                    f"{bist100_return_val:.1f}%",
                    f"{71.3}% (Yıllık)"
                )
            with col3:
                bist30_return_val = bist30_return if isinstance(bist30_return, (int, float)) else bist30_return.iloc[0]
                st.metric(
                    "BIST-30 Getiri",
                    f"{bist30_return_val:.1f}%",
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

            # Karşılaştırmalı performans grafiği
            try:
                # BIST endekslerini al
                bist100 = yf.download('XU100.IS', start=start_date, end=end_date, progress=False)
                bist30 = yf.download('XU030.IS', start=start_date, end=end_date, progress=False)

                if not bist100.empty and not bist30.empty:
                    portfolio_values = pd.Series(backtest['Günlük Değerler']).astype(float)
                    portfolio_values.index = pd.to_datetime(portfolio_values.index)
                    
                    # Endeks verilerini hazırla
                    bist100_values = pd.Series(bist100['Close'].values.squeeze(), index=bist100.index)
                    bist30_values = pd.Series(bist30['Close'].values.squeeze(), index=bist30.index)
                    
                    # Denomilasyon düzeltmesi (27 Temmuz 2020 öncesi değerleri 100'e böl)
                    denomilasyon_tarihi = pd.Timestamp('2020-07-27')
                    bist100_values[bist100_values.index < denomilasyon_tarihi] = bist100_values[bist100_values.index < denomilasyon_tarihi] / 100
                    bist30_values[bist30_values.index < denomilasyon_tarihi] = bist30_values[bist30_values.index < denomilasyon_tarihi] / 100
                    
                    # Tüm indeksleri datetime'a çevir
                    portfolio_values.index = pd.to_datetime(portfolio_values.index)
                    bist100_values.index = pd.to_datetime(bist100_values.index)
                    bist30_values.index = pd.to_datetime(bist30_values.index)
                    
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

                    # Karşılaştırmalı performans grafiği
                    st.subheader("Karşılaştırmalı Performans Analizi")
                    fig_comp = go.Figure()
                    
                    # Portföy çizgisi
                    fig_comp.add_trace(go.Scatter(
                        x=portfolio_norm.index,
                        y=portfolio_norm,
                        name='Portföy',
                        line=dict(color='#00ff00', width=2.5),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.05)'
                    ))
                    
                    # BIST-100 çizgisi
                    fig_comp.add_trace(go.Scatter(
                        x=bist100_norm.index,
                        y=bist100_norm,
                        name='BIST-100',
                        line=dict(color='#ff4444', width=1.5, dash='dot')
                    ))
                    
                    # BIST-30 çizgisi
                    fig_comp.add_trace(go.Scatter(
                        x=bist30_norm.index,
                        y=bist30_norm,
                        name='BIST-30',
                        line=dict(color='#4444ff', width=1.5, dash='dot')
                    ))
                    
                    # Grafik düzeni
                    fig_comp.update_layout(
                        template='plotly_dark',
                        title=dict(
                            text='Portföy vs Endeks Performansı',
                            x=0.5,
                            y=0.95,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=20)
                        ),
                        xaxis_title='Tarih',
                        yaxis_title='Normalize Edilmiş Değer',
                        showlegend=True,
                        height=600,
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor='rgba(0,0,0,0.5)',
                            bordercolor='rgba(255,255,255,0.2)',
                            borderwidth=1
                        ),
                        yaxis=dict(
                            gridcolor='rgba(128, 128, 128, 0.2)',
                            zerolinecolor='rgba(128, 128, 128, 0.2)',
                            tickformat='.2f',
                            hoverformat='.2%',
                            title_font=dict(size=14),
                            tickfont=dict(size=12)
                        ),
                        xaxis=dict(
                            gridcolor='rgba(128, 128, 128, 0.2)',
                            zerolinecolor='rgba(128, 128, 128, 0.2)',
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            rangeslider=dict(visible=True),
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1A", step="month", stepmode="backward"),
                                    dict(count=3, label="3A", step="month", stepmode="backward"),
                                    dict(count=6, label="6A", step="month", stepmode="backward"),
                                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                                    dict(step="all", label="Tümü")
                                ]),
                                bgcolor='rgba(0,0,0,0.5)',
                                activecolor='rgba(0,255,0,0.2)'
                            )
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=50, r=50, t=80, b=50),
                        hoverlabel=dict(
                            bgcolor='rgba(0,0,0,0.8)',
                            font=dict(size=12)
                        ),
                        annotations=[
                            dict(
                                text=f"<b>Portföy Getirisi:</b> {(portfolio_norm.iloc[-1]-1)*100:.1f}%",
                                xref="paper", yref="paper",
                                x=0.01, y=0.98,
                                showarrow=False,
                                font=dict(color='#00ff00', size=14),
                                bgcolor='rgba(0,0,0,0.7)',
                                bordercolor='rgba(0,255,0,0.3)',
                                borderwidth=1,
                                borderpad=4,
                                align='left'
                            ),
                            dict(
                                text=f"<b>BIST-100 Getirisi:</b> {(bist100_norm.iloc[-1]-1)*100:.1f}%",
                                xref="paper", yref="paper",
                                x=0.01, y=0.93,
                                showarrow=False,
                                font=dict(color='#ff4444', size=14),
                                bgcolor='rgba(0,0,0,0.7)',
                                bordercolor='rgba(255,0,0,0.3)',
                                borderwidth=1,
                                borderpad=4,
                                align='left'
                            ),
                            dict(
                                text=f"<b>BIST-30 Getirisi:</b> {(bist30_norm.iloc[-1]-1)*100:.1f}%",
                                xref="paper", yref="paper",
                                x=0.01, y=0.88,
                                showarrow=False,
                                font=dict(color='#4444ff', size=14),
                                bgcolor='rgba(0,0,0,0.7)',
                                bordercolor='rgba(0,0,255,0.3)',
                                borderwidth=1,
                                borderpad=4,
                                align='left'
                            )
                        ]
                    )
                    
                    # Grafik üzerinde hover bilgisi
                    fig_comp.update_traces(
                        hovertemplate="<b>%{y:.1%}</b><br>"
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)

            except Exception as e:
                st.warning(f"Endeks getirilerini hesaplarken hata oluştu: {str(e)}")

            # Etkin sınır analizi
            if st.session_state.get('optimized', False) and show_efficient_frontier:
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
    
    # Yatırım tutarı girişi
    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 100000
        
    investment_amount = st.number_input(
        "Yatırım Tutarı (TL)",
        min_value=1000,
        max_value=10000000,
        value=st.session_state.investment_amount,
        step=1000,
        format="%d",
        key="investment_amount_input_2"
    )
    # Değeri session state'e kaydet
    st.session_state.investment_amount = investment_amount
    
    # Pasta grafiği ve alım önerileri yan yana
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="column-gap">', unsafe_allow_html=True)
        # Portföy pasta grafiği
        fig_pie = px.pie(values=weights[weights > 0.001],
                        names=weights[weights > 0.001].index,
                        title="Hisse Ağırlıkları")
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart_2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="column-gap">', unsafe_allow_html=True)
        # Hisse bazında alım önerileri
        st.markdown("### 💰 Hisse Bazında Alım Önerileri")
        
        # CSS ile tablo stilini özelleştir
        st.markdown("""
        <style>
        .dataframe {
            margin-top: 20px;
            margin-bottom: 20px;
            margin-left: 40px;
            width: calc(100% - 40px);
        }
        .column-gap { 
            padding: 0 30px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Alım önerilerini hesapla ve göster
        recommendations = []
        for stock, weight in weights[weights > 0.001].items():
            amount = st.session_state.investment_amount * weight
            recommendations.append({
                'Hisse': stock.replace('.IS', ''),
                'Ağırlık': f'{weight*100:.1f}%',
                'Yatırım Tutarı': f'{amount:,.0f} TL'
            })
        
        if recommendations:
            df_recommendations = pd.DataFrame(recommendations)
            st.markdown(
                df_recommendations.to_html(
                    escape=False,
                    index=False,
                    columns=['Hisse', 'Ağırlık', 'Yatırım Tutarı'],
                    classes=['dataframe'],
                    justify='center'
                ),
                unsafe_allow_html=True
            ) 