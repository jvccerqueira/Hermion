import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM

# For reading stock data from yahoopip 
import yfinance as yf

# For time stamps
from datetime import datetime, timedelta


def stock_prediction(stock_ticker, start_date, date_range):
    stock_prices = yf.download(stock_ticker, start=start_date, end=datetime.now())
    stock_prices = stock_prices.xs(stock_ticker, level='Ticker', axis=1)
    data = stock_prices.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95))
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002
    test_data = scaled_data[training_data_len - 60:, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    past_days = scaled_data[-60:, :]
    future_days = pd.DataFrame(columns=['Date', 'Predictions'])

    for day in range(1, date_range + 1):
        # Selecting Past Data
        past_days = np.array(past_days)
        past_days = np.reshape(past_days, (1, past_days.shape[0], 1))
        # Predicting
        next_day = model.predict(past_days)
        # Adding Predicted Value
        past_days = np.delete(past_days, 0)
        past_days = np.append(past_days, next_day)
        # Generating Predicted DataFrame
        pred_day = {
            'Date': (datetime.now() + timedelta(day)).strftime('%Y-%m-%d'),
            'Predictions': scaler.inverse_transform(next_day)[0][0]
        }
        pred_day = pd.Series(pred_day).to_frame().T
        future_days = pd.concat([future_days, pred_day])

    future_days.set_index('Date', inplace=True)

    # Generate Dataframe to plot
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    valid = pd.concat([valid, future_days])

    return rmse, train, valid
