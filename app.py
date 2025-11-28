import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_visual_interface():
    # Configuração da página para layout mais visual
    st.set_page_config(page_title="Robô Trading AI", layout="wide", initial_sidebar_state="expanded")
    
    # CSS personalizado para melhorar a visualização
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
    .positive { color: #28a745; font-weight: bold; }
    .negative { color: #dc3545; font-weight: bold; }
    .warning { color: #ffc107; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown('<h1 class="main-header">🤖 ROBÔ TRADING AI - DASHBOARD</h1>', unsafe_allow_html=True)
    
    # Layout principal com 3 colunas
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Gráfico de preço em tempo real
        st.subheader("📈 Gráfico de Preço em Tempo Real - BTC/USD")
        fig_price = create_realtime_price_chart()
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Tabela de candles
        st.subheader("🕯️ Candles Gerados (Últimos 10)")
        candles_df = create_sample_candles()
        st.dataframe(candles_df.style.format({
            'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}', 'Volume': '{:.1f}'
        }), use_container_width=True)
    
    with col2:
        # Controles e estados
        st.subheader("🎮 Controles e Estados")
        
        # Status do robô
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Status Robô", "🟢 ATIVO", "Executando")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Última decisão AI
        st.markdown("**Última Decisão AI:**")
        decision_col1, decision_col2 = st.columns([1, 2])
        with decision_col1:
            st.markdown("<div style='text-align: center'>🟢</div>", unsafe_allow_html=True)
        with decision_col2:
            st.markdown("**BUY**<br>Confiança: <span class='positive'>87.2%</span>", unsafe_allow_html=True)
        
        # Métricas de performance
        st.markdown("**Métricas de Performance:**")
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Acurácia", "76.5%", "2.1%")
        with col_metric2:
            st.metric("Profit Total", "R$ 245,80", "R$ 45,20")
        
        # Posições ativas
        st.markdown("**Posições Ativas:**")
        positions_df = create_sample_positions()
        st.dataframe(positions_df.style.format({
            'Preço Entrada': '{:.2f}', 'P&L': '{:.2f}'
        }), use_container_width=True)
    
    with col3:
        # Monitoramento AI
        st.subheader("🧠 Monitoramento AI")
        
        # Feature Importance
        st.markdown("**Feature Importance:**")
        fig_features = create_feature_importance_chart()
        st.plotly_chart(fig_features, use_container_width=True)
        
        # Learning Progress
        st.markdown("**Learning Progress:**")
        progress = st.progress(70)
        st.markdown("<div style='text-align: center'>70% Completo</div>", unsafe_allow_html=True)
        
        # Confidence Level
        st.markdown("**Confidence Level:**")
        st.markdown("<h2 class='positive' style='text-align: center'>85.2%</h2>", unsafe_allow_html=True)
        
        # Alertas AI
        st.markdown("**Alertas AI:**")
        st.markdown("""
        <div class="alert-box">
            ⚠️ <strong>Alta volatilidade detectada: 2.35%</strong>
        </div>
        <div class="alert-box">
            🟢 <strong>Trend bullish confirmado</strong>
        </div>
        <div class="alert-box">
            🔄 <strong>Modelo em retreinamento</strong>
        </div>
        """, unsafe_allow_html=True)

    # Seção de logs
    st.markdown("---")
    st.subheader("📋 Logs do Sistema em Tempo Real")
    logs_container = st.container()
    with logs_container:
        logs = create_sample_logs()
        for log in logs:
            if "🟢" in log or "BUY" in log:
                st.markdown(f"<div style='color: #28a745'>{log}</div>", unsafe_allow_html=True)
            elif "🔴" in log or "SELL" in log:
                st.markdown(f"<div style='color: #dc3545'>{log}</div>", unsafe_allow_html=True)
            elif "⚠️" in log:
                st.markdown(f"<div style='color: #ffc107'>{log}</div>", unsafe_allow_html=True)
            else:
                st.text(log)

def create_realtime_price_chart():
    """Cria gráfico de preço em tempo real"""
    # Dados de exemplo
    times = pd.date_range(start='2024-01-01 14:30:00', periods=50, freq='1min')
    prices = 50000 + np.cumsum(np.random.randn(50) * 100)
    
    fig = go.Figure()
    
    # Linha de preço principal
    fig.add_trace(go.Scatter(
        x=times, y=prices,
        mode='lines',
        name='BTC/USD',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Último ponto destacado
    fig.add_trace(go.Scatter(
        x=[times[-1]], y=[prices[-1]],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Preço Atual'
    ))
    
    fig.update_layout(
        title="Preço em Tempo Real - BTC/USD",
        xaxis_title="Tempo",
        yaxis_title="Preço (USD)",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_feature_importance_chart():
    """Cria gráfico de feature importance"""
    features = ['RSI', 'MACD', 'Volatility', 'Price Action', 'Volume', 'Momentum', 'BB Width']
    importance = [0.45, 0.32, 0.28, 0.25, 0.18, 0.15, 0.12]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title="Importância das Features",
        xaxis_title="Importância",
        yaxis_title="Features",
        template="plotly_white",
        height=300,
        showlegend=False
    )
    
    return fig

def create_sample_candles():
    """Cria dados de candles de exemplo"""
    data = {
        'Timestamp': pd.date_range(start='2024-01-01 14:30:00', periods=10, freq='5min'),
        'Open': [51200, 51280, 51350, 51420, 51380, 51400, 51320, 51380, 51450, 51500],
        'High': [51350, 51400, 51500, 51450, 51480, 51450, 51400, 51500, 51550, 51580],
        'Low': [51100, 51200, 51300, 51300, 51300, 51250, 51200, 51300, 51400, 51420],
        'Close': [51280, 51350, 51420, 51380, 51400, 51320, 51380, 51450, 51500, 51520],
        'Volume': [125.5, 118.2, 142.8, 98.7, 110.3, 135.6, 122.1, 145.9, 138.4, 152.7]
    }
    return pd.DataFrame(data)

def create_sample_positions():
    """Cria dados de posições de exemplo"""
    data = {
        'Ativo': ['BTCUSD', 'ETHUSD', 'AAPL'],
        'Direção': ['LONG', 'SHORT', 'LONG'],
        'Tamanho': [1.2, 0.8, 10],
        'Preço Entrada': [51200, 2850, 148.50],
        'Preço Atual': [51520, 2820, 149.20],
        'P&L': [384.00, -24.00, 7.00]
    }
    return pd.DataFrame(data)

def create_sample_logs():
    """Cria logs de exemplo"""
    return [
        "14:45:01 - 🧠 AI: Predição - BUY com 87.2% de confiança",
        "14:45:00 - 📊 Novo candle formado: O=51420 H=51450 L=51300 C=51380",
        "14:44:55 - 🔧 SIMULAÇÃO: BUY BTCUSD (Conf: 0.87)",
        "14:44:50 - ✅ Modelo LSTM treinado - Loss: 0.2345, Accuracy: 0.765",
        "14:44:45 - 📈 Tick recebido: 51380.50 (Volume: 5.2)",
        "14:44:40 - 🧠 Feature Importance atualizada: RSI=0.45, MACD=0.32",
        "14:44:35 - ⚠️ Alta volatilidade detectada: 2.35%",
        "14:44:30 - 🔧 SIMULAÇÃO: SELL BTCUSD (Conf: 0.68)",
        "14:44:25 - 📊 Indicadores técnicos calculados",
        "14:44:20 - 🟢 Trend bullish identificado"
    ]

# Sidebar function
def create_sidebar():
    with st.sidebar:
        st.title("🤖 Robô Trading AI")
        
        # Configurações básicas
        st.subheader("Configurações Básicas")
        symbol = st.selectbox("Par", ["BTCUSD", "ETHUSD", "AAPL", "PETR4"], index=0)
        timeframe = st.number_input("Timeframe (segundos)", min_value=1, value=5)
        tick_sleep = st.number_input("Intervalo ticks (s)", min_value=0.1, value=1.0)
        safe_mode = st.checkbox("Safe Mode", value=True)
        
        st.markdown("---")
        
        # Configurações AI
        st.subheader("🧠 Configurações AI/ML")
        ai_enabled = st.checkbox("Ativar Sistema AI", value=True)
        
        if ai_enabled:
            model_type = st.selectbox(
                "Arquitetura do Modelo",
                ["LSTM_Advanced", "Transformer_Trading", "CNN_LSTM_Hybrid", 
                 "XGBoost_Ensemble", "Deep_Reinforcement_Learning", "RandomForest_Advanced"]
            )
            
            learning_rate = st.number_input("Learning Rate", min_value=0.00001, value=0.001, format="%.5f")
            epochs = st.slider("Épocas Treinamento", 1, 1000, 100)
            batch_size = st.slider("Batch Size", 16, 256, 32)
            
            st.markdown("**Arquitetura Neural**")
            lstm_units = st.slider("Unidades LSTM", 32, 512, 128)
            dense_units = st.slider("Unidades Dense", 16, 256, 64)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            
            st.markdown("**Engenharia de Features**")
            lookback = st.slider("Janela Temporal", 10, 200, 50)
            features = st.multiselect(
                "Features Técnicas",
                ["RSI", "MACD", "BBANDS", "ATR", "VOLATILITY", "MOMENTUM", "VOLUME_PROFILE"],
                default=["RSI", "MACD", "BBANDS", "VOLATILITY"]
            )
        
        st.markdown("---")
        
        # Controles
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Iniciar Robô", use_container_width=True):
                st.success("Robô iniciado!")
        with col2:
            if st.button("⏹️ Parar Robô", use_container_width=True):
                st.warning("Robô parado!")
        
        if st.button("🔧 Treinar Modelo", use_container_width=True):
            st.info("Modelo em treinamento...")
        
        st.markdown("---")
        st.subheader("Atalhos Profit")
        st.text("BUY: F2\nSELL: F3\nCLOSE: F4")

# Executar a aplicação
if __name__ == "__main__":
    create_sidebar()
    create_visual_interface()
