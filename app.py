symbol_selected = symbol_choice[1]
CANDLE_TIMEFRAME = st.sidebar.number_input("Timeframe candle (segundos)", min_value=1, value=int(CANDLE_TIMEFRAME))
TICK_SLEEP = st.sidebar.number_input("Intervalo ticks (s)", min_value=0.01, value=float(TICK_SLEEP))
SAFE_MODE = st.sidebar.checkbox("Safe Mode (não enviar ordens)", value=True)
run_button = st.sidebar.button("Iniciar Robô")
stop_button = st.sidebar.button("Parar Robô")

st.sidebar.markdown("Atalhos Profit (configurar no Profit)")
st.sidebar.text(f"BUY: {BUY_HOTKEY}")
st.sidebar.text(f"SELL: {SELL_HOTKEY}")
st.sidebar.text(f"CLOSE ALL: {CLOSE_HOTKEY}")

# Main layout
col1, col2 = st.beta_columns((2,1))

with col1:
    st.subheader("Gráfico de Preço (Últimos ticks)")
    price_chart = st.empty()
    st.subheader("Candles Gerados")
    candles_table = st.empty()

with col2:
    st.subheader("Estados e Controles")
    st.markdown("Última decisão:")
    last_decision_box = st.empty()
    st.markdown("Posições (simulação):")
    positions_box = st.empty()
    st.markdown("Logs Recentes:")
    logs_box = st.empty()

# loop controlador (rodando dentro do Streamlit via while com stop)
running = False
if run_button:
    running = True
    st.session_state['running'] = True

if stop_button:
    st.session_state['running'] = False
    running = False

if 'running' not in st.session_state:
    st.session_state['running'] = False

# main loop (non-blocking-ish using Streamlit)
if st.session_state['running']:
    try:
        # single iteration per rerun; Streamlit will rerun the script frequently
        robot.symbol = symbol_selected
        robot.run_cycle()

        # plot ticks
        df_ticks = pd.DataFrame(robot.ticks, columns=['time','price'])
        if not df_ticks.empty:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(df_ticks['time'], df_ticks['price'])
            ax.set_title(f"Ticks - {robot.symbol} (últimos {len(df_ticks)} pontos)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            price_chart.pyplot(fig)

        # candles
        candles_df = robot.get_candles()
        if not candles_df.empty:
            candles_table.dataframe(candles_df.tail(200))

        # status
        last_decision_box.info(f"{robot.last_decision}")
        positions_box.dataframe(robot.get_positions().tail(20))
        logs_box.text("\n".join(robot.logs[-20:]))

    except Exception as e:
        st.error(f"Erro no loop: {e}")

else:
    st.info("Robô parado. Clique em 'Iniciar Robô' na barra lateral.")

st.markdown("---")
st.markdown("README local do RTDApi que você enviou:")
st.code("/mnt/data/README.md")
