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

# Default hisseler
default_stocks = [
    # BIST-30 Hisseleri
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS', 
    'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HEKTS.IS', 'KCHOL.IS', 
    'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS', 
    'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TAVHL.IS', 'THYAO.IS', 
    'TKFEN.IS', 'TOASO.IS', 'TUPRS.IS', 'VESTL.IS', 'YKBNK.IS',
    # BIST-50 için ek hisseler
    'AKSEN.IS', 'ALARK.IS', 'ALBRK.IS', 'ANACM.IS', 'AYGAZ.IS',
    'CCOLA.IS', 'DOHOL.IS', 'ENKAI.IS', 'GESAN.IS', 'GUBRF.IS',
    'HALKB.IS', 'ISGYO.IS', 'KARSN.IS', 'KONTR.IS', 'MGROS.IS',
    'OYAKC.IS', 'PRKME.IS', 'SOKM.IS', 'TCELL.IS', 'VAKBN.IS',
    # BIST-100 için ek hisseler
    'AFYON.IS', 'AGESA.IS', 'AKFGY.IS', 'AKSA.IS', 'ALCTL.IS',
    'ALGYO.IS', 'ALKIM.IS', 'ARCLK.IS', 'ASTOR.IS', 'ASUZU.IS',
    'AYDEM.IS', 'BAGFS.IS', 'BAKAB.IS', 'BANVT.IS', 'BRISA.IS',
    'BRYAT.IS', 'BUCIM.IS', 'CANTE.IS', 'CEMTS.IS', 'CIMSA.IS',
    'DESA.IS', 'DGKLB.IS', 'DOAS.IS', 'ECILC.IS', 'EGEEN.IS'
]

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Ana tema renkleri */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #2ca02c;
        --background-color: #0e1117;
        --text-color: #ffffff;
    }
    
    /* Başlık stilleri */
    .main-header {
        color: var(--text-color);
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        border-radius: 10px;
    }
    
    /* Metrik kartları */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Bölüm başlıkları */
    .section-header {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Grafik container */
    .chart-container {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Tablo stilleri */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        border: none;
    }
    .dataframe th {
        background-color: rgba(255, 255, 255, 0.1);
        color: var(--text-color);
        font-weight: 600;
    }
    .dataframe td {
        color: var(--text-color);
    }
    
    /* Sidebar stilleri */
    .sidebar .sidebar-content {
        background-color: var(--background-color);
    }
    
    /* Buton stilleri */
    .stButton>button {
        width: 100%;
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
    }
    
    /* Loading spinner */
    .stSpinner>div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Checkbox stilleri */
    .stCheckbox>label {
        color: var(--text-color);
    }
    
    /* Select box stilleri */
    .stSelectbox>div>div {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Slider stilleri */
    .stSlider>div>div {
        background-color: var(--primary-color);
    }
    
    /* Tab stilleri */
    .stTabs>div>div>div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px 5px 0 0;
    }
    .stTabs>div>div>div[data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs>div>div>div[data-baseweb="tab"] {
        background-color: transparent;
        color: var(--text-color);
        border: none;
        border-radius: 5px 5px 0 0;
    }
    .stTabs>div>div>div[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Ana başlık
st.markdown('<h1 class="main-header">🚀 Portföy Optimizasyonu ve Risk Analizi</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="text-align: center; color: #1f77b4;">⚙️ Parametre Ayarları</h2>', unsafe_allow_html=True)
    
    # Portföy Stratejisi
    st.markdown('<h3 style="color: #2ca02c;">📊 Portföy Stratejisi</h3>', unsafe_allow_html=True)
    
    index_choice = st.selectbox(
        "Hisse Senedi Evreni",
        ["BIST-30", "BIST-50", "BIST-100"],
        help="Hangi endeksteki hisseler kullanılsın?"
    )
    
    optimization_target = st.selectbox(
        "Optimizasyon Hedefi",
        ["Maksimum Sharpe Oranı", "Minimum Volatilite", "Maksimum Sortino Oranı"],
        help="Portföy hangi hedefe göre optimize edilsin?"
    )
    
    # Risk Yönetimi
    st.markdown('<h3 style="color: #2ca02c;">🛡️ Risk Yönetimi</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        max_weight = st.slider(
            "Max. Hisse (%)",
            min_value=5,
            max_value=50,
            value=20,
            help="Maksimum hisse ağırlığı"
        )
    with col2:
        sector_limit = st.slider(
            "Max. Sektör (%)",
            min_value=20,
            max_value=60,
            value=30,
            help="Maksimum sektör ağırlığı"
        )
    
    min_stocks = st.slider(
        "Min. Hisse Sayısı",
        min_value=3,
        max_value=15,
        value=5,
        help="Minimum hisse sayısı"
    )
    
    # Rebalancing
    st.markdown('<h3 style="color: #2ca02c;">🔄 Rebalancing</h3>', unsafe_allow_html=True)
    
    rebalancing_period = st.selectbox(
        "Yeniden Dengeleme",
        ["Yok", "Aylık", "3 Aylık", "6 Aylık", "Yıllık"],
        help="Portföy yeniden dengeleme sıklığı"
    )
    
    # Stop-Loss
    st.markdown('<h3 style="color: #2ca02c;">🛑 Stop-Loss</h3>', unsafe_allow_html=True)
    
    use_stop_loss = st.checkbox(
        "Stop-Loss Aktif",
        value=False,
        help="Stop-loss kullanılsın mı?"
    )
    
    if use_stop_loss:
        stop_loss_level = st.slider(
            "Stop-Loss Seviyesi (%)",
            min_value=5,
            max_value=25,
            value=10,
            help="Stop-loss seviyesi"
        )
    
    # Tarih Aralığı
    st.markdown('<h3 style="color: #2ca02c;">📅 Tarih Aralığı</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Başlangıç",
            datetime.now() - timedelta(days=365*2)
        )
    with col2:
        end_date = st.date_input(
            "Bitiş",
            datetime.now()
        )
    
    # Risk parametreleri
    risk_free_rate = st.slider(
        "Risksiz Faiz (%)",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        help="Yıllık risksiz faiz oranı"
    ) / 100
    
    # Analiz seçenekleri
    st.markdown('<h3 style="color: #2ca02c;">📊 Analiz Seçenekleri</h3>', unsafe_allow_html=True)
    
    show_efficient_frontier = st.checkbox(
        "Etkin Sınır Analizi",
        value=False,
        help="Etkin sınır grafiğini göster"
    )
    
    optimization_method = st.selectbox(
        "Optimizasyon Yöntemi",
        ["Modern", "Klasik"],
        help="Kullanılacak optimizasyon yöntemi"
    )
    
    # Optimizasyon butonu
    st.markdown('<br>', unsafe_allow_html=True)
    if st.button("🎯 Portföyü Optimize Et", help="Optimizasyonu başlat"):
        st.session_state.optimized = True
        st.session_state.weights = None

# Ana içerik
if st.session_state.get('optimized', False):
    try:
        with st.spinner("🔄 Portföy optimize ediliyor..."):
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
                    stock_data = yf.download(
                        stock,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )
                    
                    if not stock_data.empty and len(stock_data) > 0:
                        stock_name = stock.replace('.IS', '')
                        data[stock_name] = stock_data
                    else:
                        failed_stocks.append(stock)
                except Exception as e:
                    failed_stocks.append(stock)
                    continue

            if failed_stocks:
                st.warning(
                    f"⚠️ Bazı hisselerin verisi alınamadı:\n"
                    f"- {', '.join(failed_stocks)}\n"
                    "Bu durum genellikle veri sağlayıcısından kaynaklı geçici bir sorundur."
                )

            if not data:
                st.error("❌ Hiçbir hisse senedi için veri alınamadı! Lütfen tarih aralığını kontrol edin.")
                st.stop()

            if len(data) < min_stocks:
                st.error(f"❌ Yeterli sayıda hisse verisi alınamadı. En az {min_stocks} hisse gerekli.")
                st.stop()

            # Getirileri hesapla
            returns = pd.DataFrame()
            for stock in data:
                try:
                    if 'Adj Close' in data[stock].columns:
                        price_data = data[stock]['Adj Close']
                    else:
                        price_data = data[stock]['Close']
                    
                    if isinstance(price_data, pd.DataFrame):
                        price_data = price_data.iloc[:, 0]
                    
                    stock_return = price_data.pct_change()
                    returns[stock] = stock_return
                except Exception as e:
                    st.warning(f"⚠️ {stock} hissesi için getiri hesaplanamadı: {str(e)}")
                    continue

            returns = returns.dropna()

            if returns.empty:
                st.error("❌ Getiri hesaplanamadı! Lütfen veri kalitesini kontrol edin.")
                st.stop()

            if len(returns.columns) < min_stocks:
                st.error(f"❌ Yeterli sayıda hisse verisi alınamadı. En az {min_stocks} hisse gerekli.")
                st.stop()

            # Portföy optimize et
            optimizer = PortfolioOptimizer(returns, risk_free_rate)
            
            weights = optimizer.optimize_portfolio(
                optimization_target=optimization_target,
                max_weight=max_weight/100,
                sector_limit=sector_limit/100,
                min_stocks=min_stocks
            )
            
            if weights is None:
                st.error("❌ Optimizasyon başarısız oldu. Lütfen farklı parametreler deneyin.")
                st.stop()
            
            # Sonuçları session state'e kaydet
            st.session_state.weights = weights
            st.session_state.returns = returns
            
            # Portföy metriklerini hesapla ve kaydet
            metrics = optimizer.calculate_portfolio_metrics(weights)
            st.session_state.metrics = metrics
            
            # Risk metriklerini hesapla ve kaydet
            risk_manager = RiskManager(returns, weights)
            risk_metrics = risk_manager.calculate_risk_metrics()
            st.session_state.risk_metrics = risk_metrics
            
            # Aktif pozisyonları hesapla ve kaydet
            active_positions = weights[weights > 0.01]
            st.session_state.active_positions = active_positions

            # Sonuçları göster
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "📈 Yıllık Getiri",
                    f"{st.session_state.metrics['Yıllık Getiri']:.1%}",
                    help="Portföyün yıllık getirisi"
                )
                st.metric(
                    "🎯 Value at Risk (95%)",
                    f"{st.session_state.risk_metrics['var_metrics']['var_95']:.1%}",
                    help="95% güven aralığında VaR"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "📊 Yıllık Volatilite",
                    f"{st.session_state.risk_metrics['volatility']:.1%}",
                    help="Portföyün yıllık volatilitesi"
                )
                st.metric(
                    "💫 Conditional VaR (95%)",
                    f"{st.session_state.risk_metrics['var_metrics']['cvar_95']:.1%}",
                    help="95% güven aralığında CVaR"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "⚡ Sharpe Oranı",
                    f"{st.session_state.metrics['Sharpe Oranı']:.2f}",
                    help="Risk ayarlı getiri oranı"
                )
                st.metric(
                    "↔️ Çarpıklık",
                    f"{st.session_state.risk_metrics['skewness']:.2f}",
                    help="Getiri dağılımının çarpıklığı"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Portföy dağılımı
            st.markdown('<h2 class="section-header">📊 Portföy Dağılımı</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Pasta grafik
                colors = [
                    '#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e74c3c',
                    '#1abc9c', '#34495e', '#16a085', '#27ae60', '#2980b9',
                    '#8e44ad', '#f39c12', '#d35400', '#c0392b', '#7f8c8d'
                ]
                fig = go.Figure(data=[go.Pie(
                    labels=active_positions.index,
                    values=active_positions.values,
                    textinfo='label+percent',
                    hovertemplate="<b>%{label}</b><br>Ağırlık: %{percent}<extra></extra>",
                    marker=dict(
                        colors=colors[:len(active_positions)],
                        line=dict(color='#ffffff', width=2)
                    ),
                    textfont=dict(size=14, color='white'),
                    hole=0.3
                )])
                
                fig.update_layout(
                    showlegend=True,
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', size=12),
                    legend=dict(
                        yanchor="top",
                        y=1.0,
                        xanchor="left",
                        x=1.0,
                        bgcolor='rgba(0,0,0,0.5)',
                        bordercolor='rgba(255,255,255,0.2)',
                        font=dict(size=12)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('### 💼 Hisse Bazında Alım Önerileri')
                
                position_df = pd.DataFrame({
                    'Hisse': active_positions.index,
                    'Ağırlık': active_positions.values,
                    'Yatırım Tutarı': active_positions.values * 100000
                })
                
                position_df['Ağırlık'] = position_df['Ağırlık'].map('{:.1%}'.format)
                position_df['Yatırım Tutarı'] = position_df['Yatırım Tutarı'].map('{:,.0f} TL'.format)
                
                st.dataframe(
                    position_df,
                    column_config={
                        "Hisse": st.column_config.TextColumn(
                            "Hisse Kodu",
                            help="Hisse senedi kodu",
                            width="medium"
                        ),
                        "Ağırlık": st.column_config.TextColumn(
                            "Portföy Ağırlığı",
                            help="Portföydeki yüzdesel ağırlığı",
                            width="medium"
                        ),
                        "Yatırım Tutarı": st.column_config.TextColumn(
                            "Önerilen Yatırım",
                            help="100,000 TL'lik portföy için önerilen yatırım tutarı",
                            width="medium"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Portföy özeti
                st.markdown('#### 📝 Portföy Özeti')
                st.markdown(f"""
                - 🎯 **Toplam Hisse Sayısı:** {len(active_positions)}
                - 💰 **Ortalama Ağırlık:** {(1/len(active_positions)):.1%}
                - 📊 **En Yüksek Ağırlık:** {active_positions.max():.1%} ({active_positions.idxmax()})
                - 📈 **En Düşük Ağırlık:** {active_positions.min():.1%} ({active_positions.idxmin()})
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ... (diğer grafikler ve analizler için benzer stil iyileştirmeleri) ...
            
    except Exception as e:
        st.error(f"❌ Optimizasyon sırasında bir hata oluştu: {str(e)}")
        st.warning("⚠️ Lütfen farklı parametreler deneyin veya veri setini kontrol edin.")
        logger.error(f"Optimization error: {str(e)}", exc_info=True)

# Eğer weights varsa sonuçları göster
if hasattr(st.session_state, 'weights') and st.session_state.weights is not None:
    weights = st.session_state.weights
    
 