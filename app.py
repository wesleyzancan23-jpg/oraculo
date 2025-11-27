import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# === Carregar CSV ===
try:
    df = pd.read_csv("WINZ25_F_0_5min.csv", encoding="latin1", sep=",", errors="ignore")
except:
    df = pd.read_csv("WINZ25_F_0_5min.csv", encoding="utf-8", sep=";", errors="ignore")

st.write("Pré-visualização dos dados:")
st.write(df.head())

# === Preparar dados para Prophet ===
df_prophet = df.rename(columns={"Datetime": "ds", "Close": "y"})
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

# === Treinar modelo ===
modelo = Prophet()
modelo.fit(df_prophet)

# === Horizonte de previsão ===
periodo = st.slider("Dias de previsão:", 1, 60, 15)
futuro = modelo.make_future_dataframe(periods=periodo, freq="5min")
forecast = modelo.predict(futuro)

st.subheader("📊 Gráfico do Mercado")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Preço WIN"))
st.plotly_chart(fig)

st.subheader("📈 Previsão Prophet")
grafico2 = plot_plotly(modelo, forecast)
st.plotly_chart(grafico2)
