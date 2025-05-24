# %% Imports
import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
from core.stock_prediction import stock_prediction

# %% Constantes
START_DATE = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# %% Funções auxiliares
def carregar_ibov(start_date, end_date):
    ibov_data = yf.download('^BVSP', start=start_date, end=end_date)
    ibov_data = ibov_data.xs('^BVSP', level='Ticker', axis=1)
    ibov_data['Retorno'] = ibov_data['Close'] / ibov_data['Close'].iloc[0]
    return ibov_data

def carregar_dados_acao(ticker):
    try:
        acao = yf.Ticker(ticker)
        info = acao.info
    except:
        ticker = f'{ticker}.SA'
        acao = yf.Ticker(ticker)
        info = acao.info
    return acao, ticker, info

def processar_historico(acao, start_date, end_date):
    historico = acao.history(start=start_date, end=end_date)
    historico.reset_index(inplace=True)
    historico['Date'] = pd.to_datetime(historico['Date']).dt.date
    historico.set_index('Date', inplace=True)
    historico['Retorno'] = historico['Close'] / historico['Close'].iloc[0]
    return historico

def plotar_correlacao(dados_correlacao, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados_correlacao.index, y=dados_correlacao[ticker], name=f"Variação {ticker}"))
    fig.add_trace(go.Scatter(x=dados_correlacao.index, y=dados_correlacao['IBOV'], name="Variação IBOVESPA"))
    fig.update_layout(title='Relação de Variação Percentual', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plotar_previsao(train_data, pred_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data.Close, name="Fechamento Histórico"))
    fig.add_trace(go.Scatter(x=pred_data.index, y=pred_data.Close, name="Fechamento Real"))
    fig.add_trace(go.Scatter(x=pred_data.index, y=pred_data.Predictions, name="Fechamento Previsto"))

    fig.update_layout(
        title='Fechamento Histórico + Previsão',
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )
    st.plotly_chart(fig)

# %% Interface Streamlit
st.title("Análise de Ações")

# Entrada do usuário
selected_stock = st.sidebar.text_input("Selecione o Ticker do Yahoo Finance", 'VALE3.SA')

# Carregamento e processamento
ibov_data = carregar_ibov(START_DATE, TODAY)
try:
    acao, selected_stock, info = carregar_dados_acao(selected_stock)
except:
    st.error("Ticker inválido. Tente novamente.")
    st.stop()
acao_historica = processar_historico(acao, START_DATE, TODAY)

# Exibição de informações da empresa
st.subheader(info.get("longName", "Nome não disponível"))

with st.container():
    col1, col2 = st.columns(2)
    col1.write(f"Setor: {info.get('sector', '-')}")
    col1.write(f"Preço Atual: R${round(info.get('currentPrice', 0), 2)}")
    col1.write(f"Dividend Yield: {round(info.get('dividendYield', 0) * 100, 2)}%")
    col1.write(f"Índice P/L: {round(info.get('trailingPE', 0), 2)}")
    col1.write(f"ROE: {round(info.get('returnOnEquity', 0) * 100, 2)}%")
    col1.write(f"Crescimento da Receita Tri: {round(info.get('revenueGrowth', 0) * 100, 2)}%")

    col2.write(f"Máxima 52 Sem: R${round(info.get('fiftyTwoWeekHigh', 0), 2)}")
    col2.write(f"Mínima 52 Sem: R${round(info.get('fiftyTwoWeekLow', 0), 2)}")
    col2.write(f"Média 52 Sem: R${round(info.get('fiftyDayAverage', 0), 2)}")
    col2.write(f"P/VP: {round(info.get('priceToBook', 0), 2)}")
    col2.write(f"Valor Patrimonial: R${info.get('enterpriseValue', 0)}")

# Correlação de retornos
st.subheader("Relação de Variação Percentual")
dados_correlacao = pd.DataFrame({
    selected_stock: acao_historica["Retorno"],
    "IBOV": ibov_data["Retorno"]
})
plotar_correlacao(dados_correlacao, selected_stock)

# Previsão de preços
erro, treino, previsao = stock_prediction(selected_stock, START_DATE, 7)
plotar_previsao(treino, previsao)
st.write(f"Erro Calculado: {round(erro, 2)}")
