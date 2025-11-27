import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# ------------------------------
# 1) CARREGAR O ARQUIVO CSV
# ------------------------------
try:
    df = pd.read_csv("WINZ25_F_0_5min.csv", sep=";", engine="python")

    st.success("Arquivo carregado com sucesso!")
    st.write("Pré-visualização dos dados:")
    st.dataframe(df.head())

    st.write("📌 Colunas detectadas:", list(df.columns))

except Exception as e:
    st.error("Erro ao carregar o arquivo CSV.")
    st.stop()

# Verificação das colunas necessárias
colunas_necessarias = ["Data", "Hora", "Fechamento"]
for c in colunas_necessarias:
    if c not in df.columns:
        st.error(f"❌ O arquivo não contém a coluna obrigatória: **{c}**")
        st.stop()

# ------------------------------
# 2) CRIAR COLUNA DATETIME
# ------------------------------
df["datetime"] = pd.to_datetime(df["Data"] + " " + df["Hora"])

# ------------------------------
# 3) PREPARAR PARA O PROPHET
# ------------------------------
df_prophet = df.rename(columns={
    "datetime": "ds",
    "Fechamento": "y"
})

df_prophet = df_prophet[["ds", "y"]]

st.subheader("📌 Dados prontos para o modelo Prophet:")
st.dataframe(df_prophet.head())

# ------------------------------
# 4) MODELAGEM – PROPHET
# ------------------------------
modelo = Prophet()
modelo.fit(df_prophet)

# Slider para horizonte de previsão
periodos = st.slider("Período de previsão (em dias):", 1, 60, 10)

# Criar datas futuras
future = modelo.make_future_dataframe(periods=periodos, freq="5min")

# Prever
forecast = modelo.predict(future)

st.subheader("📈 Previsão dos preços")
st.dataframe(forecast[["ds", "yhat"]].tail())

# ------------------------------
# 5) PLOT DA PREVISÃO
# ------------------------------
st.subheader("📊 Gráfico da Previsão")
grafico = plot_plotly(modelo, forecast)
st.plotly_chart(grafico)

# ------------------------------
# 6) GRÁFICO DOS PREÇOS ORIGINAIS
# ------------------------------
st.subheader("📉 Preço Real – Fechamento")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["Fechamento"], name="Fechamento"))
fig.update_layout(title="Fechamento do WIN (Histórico)", xaxis_title="Data", yaxis_title="Preço")
st.plotly_chart(fig)
