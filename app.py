import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date
import warnings
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

# Define a data de início para coleta de dados
INICIO = "2015-01-01"
HOJE = date.today().strftime("%Y-%m-%d")

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")

# 🔥 CORREÇÃO DO ERRO DE ENCODING
df = pd.read_csv(
    "WINZ25_F_0_5min.csv",
    encoding="latin1",        # <- resolve o UnicodeDecodeError
    sep=",",
    engine="python"
)

# Padronizando códigos das ações conforme Yahoo Finance
df['codigo'] = df['codigo'].apply(lambda x: x + ".SA")

empresas = df['codigo']
empresa_selecionada = st.selectbox("Selecione a empresa:", empresas)

@st.cache_data
def carrega_dados(ticker):
    dados = yf.download(ticker, INICIO, HOJE)
    dados.reset_index(inplace=True)
    return dados

st.text("Carregando os dados...")
dados = carrega_dados(empresa_selecionada)
st.text("Dados carregados!")

st.subheader("Visualização dos Dados Brutos")
st.write(dados.tail())

def plot_dados_brutos():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados['Date'], y=dados['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=dados['Date'], y=dados['Close'], name="stock_close"))
    fig.layout.update(
        title_text="Preço de Abertura e Fechamento das Ações",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)

plot_dados_brutos()

st.subheader("Previsões com Machine Learning")

df_treino = dados[['Date', 'Close']]
df_treino = df_treino.rename(columns={"Date": "ds", "Close": "y"})

modelo = Prophet()
modelo.fit(df_treino)

num_anos = st.slider("Horizonte de previsão (anos):", 1, 4)
periodo = num_anos * 365

futuro = modelo.make_future_dataframe(periods=periodo)
forecast = modelo.predict(futuro)

st.subheader("Dados Previstos")
previsao = forecast[['ds', 'yhat']]
st.write(previsao.tail())

st.subheader("Previsão de Preço")
grafico2 = plot_plotly(modelo, forecast)
st.plotly_chart(grafico2)
