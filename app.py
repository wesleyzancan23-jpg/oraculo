import streamlit as st
import pandas as pd
import plotly.graph_objs as go

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# ======================
# 1. Leitura do CSV
# ======================
try:
    df = pd.read_csv("WINZ25_F_0_5min.csv", sep=";", encoding="latin1")
except:
    st.error("Erro ao carregar o arquivo CSV.")
    st.stop()

# ======================
# 2. Exibir colunas detectadas
# ======================
st.write("📌 *Colunas detectadas no arquivo:*")
st.write(list(df.columns))

# ======================
# 3. Criar coluna datetime
# ======================
if "Data" in df.columns and "Hora" in df.columns:
    df["datetime"] = pd.to_datetime(df["Data"] + " " + df["Hora"], dayfirst=True)
else:
    st.error("❌ O arquivo precisa ter colunas 'Data' e 'Hora'.")
    st.stop()

# ======================
# 4. Renomear colunas para padrão do Prophet
# ======================
df_prophet = pd.DataFrame()
df_prophet["ds"] = df["datetime"]
df_prophet["y"] = df["Fechamento"].astype(float)

st.subheader("Pré-visualização dos dados:")
st.write(df_prophet.head())

# ======================
# 5. Gráfico de preços (Plotly)
# ======================
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["Fechamento"], name="Fechamento"))
fig.update_layout(title="Preço – WIN", xaxis_title="Tempo", yaxis_title="Preço")
st.plotly_chart(fig)
