import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

# ===============================
# 1) Carregamento seguro do CSV
# ===============================
st.write("### Pré-visualização dos dados:")

try:
    # Lê o CSV exatamente no formato enviado
    df = pd.read_csv("WINZ25_F_0_5min.csv", sep=";", engine="python")

    st.write("📌 Colunas detectadas no arquivo:")
    st.write(list(df.columns))

    # Verificar colunas obrigatórias
    required = ["Data", "Hora", "Fechamento"]
    if not all(col in df.columns for col in required):
        st.error("❌ O arquivo CSV não contém as colunas obrigatórias: Data, Hora, Fechamento.")
        st.stop()

    # Criar datetime corretamente
    df["datetime"] = pd.to_datetime(df["Data"] + " " + df["Hora"])

    # Renomear para Prophet
    df_prophet = df.rename(columns={
        "datetime": "ds",
        "Fechamento": "y"
    })

    # Selecionar somente o necessário
    df_prophet = df_prophet[["ds", "y"]]

    st.write(df_prophet.head())

except Exception as e:
    st.error("Erro ao carregar o arquivo CSV.")
    st.stop()


# ===============================
# 2) Plot dos dados originais
# ===============================
st.subheader("📊 Gráfico de Preço (Fechamento)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], name="Fechamento"))
fig.update_layout(title="Histórico do Mini-Índice (WIN)")
st.plotly_chart(fig)


# ===============================
# 3) Previsões com Prophet
# ===============================
st.subheader("🔮 Previsão do Mini-Índice")

# Horizonte em dias
dias = st.slider("Selecione o horizonte de previsão (dias):", 1, 60, 15)

modelo = Prophet(daily_seasonality=True)
modelo.fit(df_prophet)

futuro = modelo.make_future_dataframe(periods=dias, freq="5min")
previsao = modelo.predict(futuro)

st.write("### Dados Previstos:")
st.dataframe(previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# Gráfico da previsão
st.write("### Gráfico da Previsão")
fig2 = plot_plotly(modelo, previsao)
st.plotly_chart(fig2)
