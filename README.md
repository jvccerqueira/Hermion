# 📈 Stock Price Prediction with LSTM

This project is a web application built with **Streamlit** and **Keras LSTM** models for analyzing and forecasting stock prices using historical data from **Yahoo Finance**.

## 🧠 Features

- Interactive web interface with Streamlit
- Fetches stock data using `yfinance`
- Displays key financial indicators and sector information
- Compares stock performance with the Ibovespa index
- Trains a basic LSTM model for time series forecasting
- Visualizes actual, historical, and predicted stock prices
- Predicts future stock closing prices

## 📁 Project Structure

```
app/
│
├── core/
│   └── stock_prediction.py   # Prediction model and preprocessin   
├── main.py                   # Streamlit application
requirements.txt              # Dependencies
README.md                     # Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jvccerqueira/Hermion.git
cd PD
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed.
>[!TIP]
> It is a good practice to create a Virtual Environment to run python projects, you can do this by  
>```python -m venv .venv```  
>```source .venv/Scripts/activate``` To activate the virtual environment on Windows  
>```source .venv/bin/activate``` To activate the virtual environment on MacOS and Linux

Install the project dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app/main.py
```

## 🧪 Example Usage

Search for a stock using its Yahoo Finance ticker, e.g. `VALE3.SA`, to analyze historical trends and predict future prices.

## 📦 Dependencies

- streamlit
- pandas
- numpy
- yfinance
- scikit-learn
- keras
- tensorflow
- plotly

## 📌 Notes

- This app uses a very basic LSTM model for demonstration and learning purposes. For production or financial advice, further model tuning and risk management are necessary.
- The forecast is based only on past closing prices.

## 📄 License

This project is licensed under the MIT License.
