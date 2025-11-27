import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# -----------------------------
# 1. TENTAR CARREGAR O CSV
# -----------------------------
st.write("### Pré-visualização dos dados:")

try:
    df = pd.read_csv(
        "WINZ25_F_0_5min.csv",
        sep=None,               # auto detecta separador
        engine="python",
        encoding="latin1",      # impede UnicodeDecodeError
        on_bad_lines="skip"     # ignora linhas com erro
    )

    st.dataframe(df.head())

except Exception as e:
    st.error("Erro ao carregar o arquivo CSV.")
    st.code(str(e))
    st.stop()

# -----------------------------
# 2. VERIFICAR SE A COLUNA datetime EXISTE
# -----------------------------
colunas = df.columns.tolist()

st.write("📌 Colunas detectadas no arquivo:", colunas)

if "datetime" not in df.columns:
    st.error("❌ O arquivo não contém a coluna 'datetime'.")
    st.stop()

# -----------------------------
# 3. PREPARAR DADOS PARA O PROPHET
# -----------------------------
df["datetime"] = pd.to_datetime(df["datetime"])

df_prophet = df[["datetime", "close"]].rename(columns={
    "datetime": "ds",
    "close": "y"
})

# -----------------------------
# 4. TREINAR MODELO
# -----------------------------
st.write("### 🔮 Previsão com Prophet")

modelo = Prophet()
modelo.fit(df_prophet)

# Previsão de 5 dias (480 candles de 5 min)
periodos = 480
futuro = modelo.make_future_dataframe(periods=periodos, freq="5min")

forecast = modelo.predict(futuro)

st.write("### Últimas previsões")
st.dataframe(forecast[["ds", "yhat"]].tail())

# -----------------------------
# 5. GRÁFICO
# -----------------------------
fig1 = modelo.plot(forecast)
st.pyplot(fig1)
