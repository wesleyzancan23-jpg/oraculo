import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# === 1) Carregar CSV ===
df = pd.read_csv("WINZ25_F_0_5min.csv", engine="python", sep=None)

# Mostrar prévia
st.subheader("Pré-visualização dos dados:")
st.write(df.head())

# === 2) Preparar dados ===
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# Preparar para o Prophet
df_prophet = df.rename(columns={
    "datetime": "ds",
    "close": "y"
})

df_prophet = df_prophet[["ds", "y"]]

# === 3) Modelo Prophet ===
modelo = Prophet()
modelo.fit(df_prophet)

# Seleção horizonte
periodos = st.slider("Selecione o horizonte de previsão (em minutos):", 50, 2000, 400)

futuro = modelo.make_future_dataframe(periods=periodos, freq="5min")
forecast = modelo.predict(futuro)

# Mostrar tabela final
st.subheader("Previsões:")
st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# === 4) Gráficos ===
st.subheader("Gráfico de Previsão")
grafico = plot_plotly(modelo, forecast)
st.plotly_chart(grafico)

# === 5) Gráfico do preço real ===
st.subheader("Preço Real (Close)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["close"], name="Fechamento"))
st.plotly_chart(fig)
