import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# Carrega o arquivo CSV enviado
df = pd.read_csv("WINZ25_F_0_5min.csv")

# Garante que a coluna Date exista
df.rename(columns={"time": "Date", "date": "Date"}, inplace=True)

# Converte a coluna de data se necessário
df['Date'] = pd.to_datetime(df['Date'])

# Mostra dados brutos
st.subheader("Dados Brutos")
st.write(df.head())

# Plot de preços
st.subheader("Preço – Abertura e Fechamento")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['open'], name="Abertura"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['close'], name="Fechamento"))
fig.layout.update(title_text="WINZ25 – 5min", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Previsão Prophet
st.subheader("Previsões com IA – Prophet")

df_treino = df[['Date', 'close']].rename(columns={"Date": "ds", "close": "y"})

modelo = Prophet()
modelo.fit(df_treino)

anos = st.slider("Horizonte (anos):", 1, 4)
periodo = anos * 365

futuro = modelo.make_future_dataframe(periods=periodo)
forecast = modelo.predict(futuro)

st.write("Últimas previsões:")
st.write(forecast[['ds', 'yhat']].tail())

st.subheader("Gráfico da Previsão")
grafico = plot_plotly(modelo, forecast)
st.plotly_chart(grafico)
