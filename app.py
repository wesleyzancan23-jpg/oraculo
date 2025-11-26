import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import warnings
warnings.filterwarnings("ignore")

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão de Ativos Financeiros")

# Carregar arquivo enviado
df = pd.read_csv("WINZ25_F_0_5min.csv")

st.write("### Dados Carregados:")
st.dataframe(df.tail())

# Seleção de coluna para previsão
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

coluna_escolhida = st.selectbox("Selecione a coluna para previsão:", colunas_numericas)

# Preparando os dados
df_treino = pd.DataFrame()
df_treino["ds"] = pd.to_datetime(df.iloc[:, 0])  # primeira coluna é data
df_treino["y"] = df[coluna_escolhida]

# Treino
modelo = Prophet()
modelo.fit(df_treino)

# Horizonte
num_dias = st.slider("Horizonte (dias):", 30, 365)
futuro = modelo.make_future_dataframe(periods=num_dias)
forecast = modelo.predict(futuro)

st.subheader("Previsões")
st.write(forecast.tail())

grafico = plot_plotly(modelo, forecast)
st.plotly_chart(grafico)
