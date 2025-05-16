# %%
import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from stock_prediction import stock_prediction
from plotly import graph_objs as go

#%% Filtering Stock Data by Date
START_DATE = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Page Title
st.title("Analise de Ações")

# Selecting all available Tickers
selected_stock = st.sidebar.text_input("Select Stock for Analysis - Try write the Yahoo Finance Ticker", 'VALE3.SA')

# Dados Ibovespa
ibov_info = yf.download('^BVSP', start=START_DATE, end=TODAY)
ibov_info = ibov_info.xs('^BVSP', level='Ticker', axis=1)
ibov_info['Retorno'] = ibov_info.Close / ibov_info.Close.iloc[0]

# Dados Ação Escolhida
try:
    acao = yf.Ticker(selected_stock)
    info = acao.info
except:
    acao = yf.Ticker(f'{selected_stock}.SA')
    selected_stock = f'{selected_stock}.SA'
    info = acao.info

# Carregando dados históricos
acao_historica = acao.history(start=START_DATE, end=TODAY)
acao_historica.reset_index(inplace=True)
acao_historica['Date'] = pd.to_datetime(acao_historica['Date']).dt.date
acao_historica.set_index('Date', inplace=True)
acao_historica['Retorno'] = acao_historica.Close / acao_historica.Close.iloc[0]

# Nome da Ação
st.subheader(f'{acao.info["longName"]}')

# Container Info
with st.container():
    col1, col2 = st.columns(2)
    # Company Info
    # Setor
    col1.write(f"Setor: {acao.info['sector']}")

    # Company Results
    # Preço Atual
    col1.write(f"Preço Atual: R${round(acao.info['currentPrice'], 2)}")
    # Dividend Yield
    col1.write(f"Dividend Yield: {round(acao.info['dividendYield'] * 100, 2)}%")
    # P/L
    col1.write(f'Indice P/L: {round(acao.info["trailingPE"],2)}')
    # ROE
    col1.write(f'ROE: {round(acao.info["returnOnEquity"] * 100, 2)}%')
    # Resultados 52 Sem
    col2.write(f'Maxima 52 Sem: R${round(acao.info["fiftyTwoWeekHigh"], 2)}')
    col2.write(f'Mínima 52 Sem: R${round(acao.info["fiftyTwoWeekLow"], 2)}')
    col2.write(f'Média 52 Sem: R${round(acao.info["fiftyDayAverage"], 2)}')
    # P/VP
    col2.write(f'P/VP: {round(acao.info["priceToBook"],2)}')
    # Valor Patrimonial
    col2.write(f'Valor Patrimonial: R${acao.info["enterpriseValue"]}')
    # Crescimento de Receita
    col1.write(f'Crescimento da Receita Tri: {round(acao.info["revenueGrowth"] * 100, 2)}%')

# Descrição Ações
# st.dataframe(acao_historica.describe()[['Close', 'Volume', 'Dividends']])


# Plotagem

# Correlação de Retornos
st.subheader('Relação de variação Percentual')
corr = pd.DataFrame()
corr[selected_stock] = acao_historica.Retorno
corr['IBOV'] = ibov_info.Retorno


def plot_corr_data():
    correlation = go.Figure()
    correlation.add_trace(go.Scatter(x=corr.index, y=corr[selected_stock], name=f"Variação {selected_stock}"))
    correlation.add_trace(go.Scatter(x=corr.index, y=corr['IBOV'], name="Variação IBOVESPA"))
    correlation.layout.update(title_text='Relação de variação Percentual',
                              xaxis_rangeslider_visible=True)
    st.plotly_chart(correlation)


plot_corr_data()


# Previsão dos preços
error, train_data, pred_data = stock_prediction(selected_stock, START_DATE, 7)


def plot_pred_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_data.index, y=pred_data.Close, name="Fechamento Real"))
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data.Close, name="Fechamento Histórico"))
    fig.add_trace(go.Scatter(x=pred_data.index, y=pred_data.Predictions, name="Fechamento Previsto"))
    fig.layout.update(title_text='Fechamento Histórico + Previsão',
                      xaxis_rangeslider_visible=True)
    fig.update_xaxes(
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
    st.plotly_chart(fig)


plot_pred_data()
st.write(f'Erro Calculado: {round(error, 2)}')