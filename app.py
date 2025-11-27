import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(layout="wide")

st.title("📈 Oráculo – Previsões Inteligentes de Mercado")
st.subheader("Dashboard Interativo para Análise e Previsão do WIN (Mini-Índice)")

###########################
# 1) Carregar o CSV usando UPLOADER e TRATAMENTO DE ENCODING
###########################
st.markdown("---")
uploaded_file = st.file_uploader(
    "📤 **Carregue o arquivo WINZ25_F_0_5min.csv** (ou qualquer CSV de mercado com Data, Hora, Fechamento)", 
    type="csv"
)

if uploaded_file is None:
    st.info("Aguardando o upload do arquivo CSV para iniciar a análise e previsão.")
    st.stop()

# Listas de tentativas
possible_separators = [";", ","]
possible_encodings = ["utf-8", "latin-1", "cp1252"] # 'latin-1' ou 'cp1252' é o mais provável para o erro 0xe1
df = None
success = False
used_sep = None
used_enc = None
colunas_necessarias = ["Data", "Hora", "Fechamento"]

# Tenta todas as combinações de separador e encoding
for sep in possible_separators:
    for enc in possible_encodings:
        try:
            # Volta ao início do arquivo para cada nova tentativa de leitura
            uploaded_file.seek(0) 
            df_temp = pd.read_csv(uploaded_file, sep=sep, engine="python", encoding=enc)
            
            # Heurística de sucesso: verifica se as colunas essenciais estão presentes
            if all(col in df_temp.columns for col in colunas_necessarias):
                 df = df_temp # Atribui o DataFrame válido
                 success = True
                 used_sep = sep
                 used_enc = enc
                 break # Sai do loop de encodings
            
        except Exception:
            continue # Tenta o próximo encoding
    
    if success:
        break # Sai do loop de separadores

if success:
    st.success(f"Arquivo carregado com sucesso! (Separador: '{used_sep}', Codificação: '{used_enc}')")
    st.write("### Pré-visualização dos dados brutos:")
    st.dataframe(df.head(), use_container_width=True)

else:
    st.error("❌ Erro fatal ao processar o arquivo CSV. Tentei os separadores ';' e ',' e as codificações 'utf-8', 'latin-1', e 'cp1252', mas não consegui ler o arquivo ou encontrar as colunas essenciais. Verifique o formato do seu CSV.")
    st.stop()

###########################
# 2) O df está garantido a partir daqui, mas verificamos novamente as colunas
###########################
if not all(col in df.columns for col in colunas_necessarias):
    st.error(f"❌ O arquivo não contém as colunas necessárias, apesar das tentativas de carregamento: {', '.join(colunas_necessarias)}")
    st.write("Colunas encontradas:", df.columns.tolist())
    st.stop()

###########################
# 3) Criar coluna datetime e limpeza de dados
###########################
df["datetime"] = pd.to_datetime(df["Data"] + " " + df["Hora"])

# Garantir que 'Fechamento' seja numérico
df["Fechamento"] = pd.to_numeric(df["Fechamento"], errors='coerce')
df.dropna(subset=['Fechamento'], inplace=True)

# Ordenar por data
df.sort_values(by="datetime", inplace=True)

###########################
# 4) Preparar dados para Prophet
###########################
df_prophet = df.rename(columns={
    "datetime": "ds",
    "Fechamento": "y"
})

df_prophet = df_prophet[["ds", "y"]]

###########################
# 5) Exibir dados e gráfico histórico
###########################
st.write("---")
st.write("### Dados Históricos Preparados para o Modelo:")
st.dataframe(df_prophet.tail(), use_container_width=True)

# Criar gráfico do preço histórico
st.subheader("📊 Histórico de Preço do Mini-Índice (WIN)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df["datetime"], y=df["Fechamento"], name="Fechamento", 
                              line=dict(color='#1f77b4')))
fig_hist.update_layout(
    xaxis_title="Data e Hora", 
    yaxis_title="Preço (Pontos)",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig_hist, use_container_width=True)

###########################
# 6) Configuração e Treinamento do Modelo Prophet
###########################
st.subheader("🧠 Previsão Machine Learning (Prophet)")
st.markdown("O modelo Prophet (do Facebook) é otimizado para dados de séries temporais que exibem fortes efeitos sazonais.")

col1, col2 = st.columns(2)

with col1:
    periodos = st.slider(
        "Quantos *períodos de 5 minutos* para prever?", 
        min_value=12, # 1 hora
        max_value=720, # 60 horas (cerca de 15 dias úteis de mercado)
        value=144, # 12 horas / 1 dia
        step=12,
        help="Cada período representa um intervalo de 5 minutos, conforme o seu CSV."
    )

with col2:
    freq_label = f"Previsão para aproximadamente **{round(periodos * 5 / 60, 2)} horas** futuras."
    st.metric("Horizonte de Previsão", value=freq_label)

# Inicializar e treinar o modelo
with st.spinner('Treinando o modelo Prophet e gerando previsões...'):
    modelo = Prophet(
        daily_seasonality=True, 
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    modelo.fit(df_prophet)

    # Criar dataframe futuro
    futuro = modelo.make_future_dataframe(periods=periodos, freq="5min")
    
    # Gerar previsão
    previsao = modelo.predict(futuro)

st.success("Previsão gerada com sucesso!")

###########################
# 7) Exibir previsão
###########################
st.write("---")
st.write("### Tabela de Previsões (Pontos 'yhat'):")
# Exibir as últimas linhas da previsão
st.dataframe(previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10), use_container_width=True)


# Plotar gráfico da previsão
st.subheader("🔮 Gráfico de Previsão Futura")
grafico_previsao = plot_plotly(modelo, previsao)
# Ajustar título
grafico_previsao.update_layout(
    title="Previsão do Mini-Índice (WIN) com Banda de Incerteza", 
    xaxis_title="Data e Hora", 
    yaxis_title="Preço Previsto",
    template="plotly_white"
)
st.plotly_chart(grafico_previsao, use_container_width=True)

###########################
# 8) Análise de Componentes (Opcional, mas útil)
###########################
st.subheader("🛠️ Análise dos Componentes do Modelo")
st.markdown("Esta seção mostra as tendências e sazonalidades detectadas pelo modelo.")
fig_comp = modelo.plot_components(previsao)
st.pyplot(fig_comp, use_container_width=True)
plt.close(fig_comp) 
