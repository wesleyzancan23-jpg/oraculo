import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# --- Carregar CSV ---
try:
    df = pd.read_csv("WINZ25_F_0_5min.csv", engine="python")
except Exception as e:
    st.error("Erro ao carregar o arquivo CSV.")
    st.stop()

# --- Garantir que a coluna datetime existe ---
if "datetime" not in df.columns:
    st.error("O arquivo CSV precisa ter a coluna 'datetime'.")
    st.write("Colunas encontradas:", df.columns.tolist())
    st.stop()

# --- Converter a coluna datetime ---
df["datetime"] = pd.to_datetime(df["datetime"])

st.subheader("Pré-visualização dos dados:")
st.write(df.head())

# --- Gráfico de preço ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["close"], name="Fechamento"))
fig.update_layout(title="Preço de Fechamento (WIN)")
st.plotly_chart(fig)

# --- Preparar dados para Prophet ---
df_prophet = df[["datetime", "close"]]
df_prophet = df_prophet.rename(columns={"datetime": "ds", "close": "y"})

# --- Criar modelo ---
modelo = Prophet()
modelo.fit(df_prophet)

# --- Selecionar horizonte ---
periodos = st.slider("Dias de previsão:", 1, 30, 10)

# --- Gerar datas futuras ---
futuro = modelo.make_future_dataframe(periods=periodos, freq="5min")
previsao = modelo.predict(futuro)

st.subheader("Previsão")
st.write(previsao[["ds", "yhat"]].tail())

# --- Gráfico previsão ---
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=previsao["ds"], y=previsao["yhat"], name="Previsão"))
fig2.add_trace(go.Scatter(x=df["datetime"], y=df["close"], name="Real"))
fig2.update_layout(title="Previsão do WIN")
st.plotly_chart(fig2)
