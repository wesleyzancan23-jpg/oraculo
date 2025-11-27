import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

###########################
# 1) Carregar o CSV
###########################
st.info("Carregando arquivo WINZ25_F_0_5min.csv...")

try:
    df = pd.read_csv("WINZ25_F_0_5min.csv", sep=";", engine="python")

    st.success("Arquivo carregado com sucesso!")

    st.write("### Pré-visualização dos dados:")
    st.write(df.head())

except Exception as e:
    st.error("Erro ao carregar o arquivo CSV.")
    st.stop()

###########################
# 2) Verificar colunas
###########################
colunas_necessarias = ["Data", "Hora", "Fechamento"]

if not all(col in df.columns for col in colunas_necessarias):
    st.error("❌ O arquivo não contém as colunas necessárias: Data, Hora, Fechamento")
    st.write("Colunas encontradas:", df.columns.tolist())
    st.stop()

###########################
# 3) Criar coluna datetime
###########################
df["datetime"] = pd.to_datetime(df["Data"] + " " + df["Hora"])

###########################
# 4) Preparar dados para Prophet
###########################
df_prophet = df.rename(columns={
    "datetime": "ds",
    "Fechamento": "y"
})

df_prophet = df_prophet[["ds", "y"]]

###########################
# 5) Exibir dados
###########################
st.write("### Dados preparados para previsão:")
st.write(df_prophet.tail())

###########################
# 6) Criar gráfico do preço
###########################
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["Fechamento"], name="Fechamento"))
fig.update_layout(title="📊 Preço do Mini-Índice (WIN)", xaxis_title="Data", yaxis_title="Preço")
st.plotly_chart(fig)

###########################
# 7) Criar modelo Prophet
###########################
st.subheader("🧠 Previsão Machine Learning (Prophet)")

periodos = st.slider(
    "Quantos dias para prever?", 
    min_value=5, 
    max_value=60, 
    value=15
)

modelo = Prophet()
modelo.fit(df_prophet)

futuro = modelo.make_future_dataframe(periods=periodos, freq="5min")
previsao = modelo.predict(futuro)

###########################
# 8) Exibir previsão
###########################
st.write("### Previsões:")
st.write(previsao[["ds", "yhat"]].tail())

grafico_previsao = plot_plotly(modelo, previsao)
st.plotly_chart(grafico_previsao)
