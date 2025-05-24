# stock_prediction_lstm.py

import pandas as pd
import numpy as np
import yfinance as yf

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.95


def load_stock_data(ticker: str, start_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=datetime.now())
    df = df.xs(ticker, level='Ticker', axis=1)
    return df[['Close']]


def scale_data(data: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def create_sequences(data: np.ndarray, sequence_length: int):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i - sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


def build_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(model, recent_data, forecast_days: int, scaler) -> pd.DataFrame:
    future = []
    data = recent_data.copy()
    for day in range(1, forecast_days + 1):
        seq = data.reshape(1, SEQUENCE_LENGTH, 1)
        pred = model.predict(seq, verbose=0)
        price = scaler.inverse_transform(pred)[0][0]
        future.append({
            'Date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
            'Predictions': price
        })
        data = np.append(data[1:], pred)
    return pd.DataFrame(future).set_index('Date')


def stock_prediction(ticker: str, start_date: str, forecast_days: int):
    data = load_stock_data(ticker, start_date)
    dataset = data.values
    train_len = int(np.ceil(len(dataset) * TRAIN_RATIO))

    scaled_data, scaler = scale_data(data)

    train_data = scaled_data[:train_len]
    x_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH)
    x_train = x_train.reshape(-1, SEQUENCE_LENGTH, 1)

    model = build_model((SEQUENCE_LENGTH, 1))
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    test_data = scaled_data[train_len - SEQUENCE_LENGTH:]
    x_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH)
    x_test = x_test.reshape(-1, SEQUENCE_LENGTH, 1)

    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean((predictions - dataset[train_len:]) ** 2))

    future_df = predict_future(model, scaled_data[-SEQUENCE_LENGTH:], forecast_days, scaler)

    train = data.iloc[:train_len]
    valid = data.iloc[train_len:].copy()
    valid['Predictions'] = predictions
    full_valid = pd.concat([valid, future_df])

    return rmse, train, full_valid
