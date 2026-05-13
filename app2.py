# File: comprehensive_stock_dashboard_updated.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from datetime import datetime, timedelta


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance.

    yfinance treats end date as exclusive, so we add 1 day
    to include the selected end date.
    """
    inclusive_end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    data = yf.download(
        ticker,
        start=start_date,
        end=inclusive_end_date,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        return data

    # Convert MultiIndex columns from yfinance to simple columns.
    # Example: ('Close', 'RELIANCE.NS') becomes 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1, drop_level=True)
        else:
            data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.sort_index()

    # Keep rows only up to selected end date
    data = data[data.index.date <= end_date]

    # Add latest intraday quote if end date is today
    data = add_latest_intraday_quote(data, ticker, end_date)

    return data


def add_latest_intraday_quote(data, ticker, end_date):
    """
    If the selected end date is today, update today's row using intraday data.
    This makes the dashboard closer to real time.
    """
    today = datetime.today().date()

    if end_date < today:
        return data

    try:
        intraday = yf.Ticker(ticker).history(period="1d", interval="1m")
    except Exception:
        return data

    if intraday.empty:
        return data

    intraday.index = pd.to_datetime(intraday.index).tz_localize(None)
    intraday = intraday.sort_index()

    latest_day = intraday.index[-1].normalize()

    if latest_day.date() > end_date:
        return data

    try:
        latest_quote = {
            "Open": intraday["Open"].dropna().iloc[0],
            "High": intraday["High"].max(),
            "Low": intraday["Low"].min(),
            "Close": intraday["Close"].dropna().iloc[-1],
            "Volume": intraday["Volume"].sum(),
        }
    except Exception:
        return data

    if "Adj Close" in data.columns:
        latest_quote["Adj Close"] = latest_quote["Close"]

    for column, value in latest_quote.items():
        if column in data.columns:
            data.loc[latest_day, column] = value

    return data.sort_index()


def format_date_axis(ax, start_date, end_date):
    """
    Force the chart x-axis to include the selected end year.

    This fixes the problem where the graph has 2026 data,
    but the x-axis only labels up to 2024.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    total_years = end_ts.year - start_ts.year

    if total_years <= 8:
        step = 1
    elif total_years <= 16:
        step = 2
    elif total_years <= 30:
        step = 4
    else:
        step = 5

    tick_years = list(range(start_ts.year, end_ts.year + 1, step))

    if end_ts.year not in tick_years:
        tick_years.append(end_ts.year)

    tick_dates = [
        pd.Timestamp(year=year, month=1, day=1)
        for year in sorted(set(tick_years))
    ]

    # Add room after the selected date so the final year label does not overlap.
    right_limit = max(end_ts, pd.Timestamp(year=end_ts.year, month=12, day=31))
    ax.set_xlim(start_ts, right_limit)
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_horizontalalignment("right")


def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def calculate_rsi(data, window=14):
    delta = data["Close"].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(data, window=20):
    rolling_mean = data["Close"].rolling(window=window).mean()
    rolling_std = data["Close"].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = data["Close"].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line


def simulate_trades(data, initial_capital=100000):
    trades = []
    capital = initial_capital
    position = None
    quantity = 100

    for i in range(len(data)):
        if data["Signal"].iloc[i] == "Buy" and position is None:
            position = {
                "Buy Date": data.index[i],
                "Buy Price": data["Close"].iloc[i],
            }

        elif data["Signal"].iloc[i] == "Sell" and position is not None:
            sell_price = data["Close"].iloc[i]
            profit = (sell_price - position["Buy Price"]) * quantity
            capital += profit

            trades.append({
                "Buy Date": position["Buy Date"],
                "Sell Date": data.index[i],
                "Buy Price": position["Buy Price"],
                "Sell Price": sell_price,
                "Quantity": quantity,
                "Profit": profit,
            })

            position = None

    trade_log = pd.DataFrame(trades)

    if trade_log.empty:
        return trade_log, 0, 0, 0, capital

    trade_log["Profit"] = pd.to_numeric(trade_log["Profit"], errors="coerce")
    trade_log = trade_log.dropna(subset=["Profit"])

    total_profit = trade_log["Profit"].sum() if not trade_log.empty else 0
    avg_profit = trade_log["Profit"].mean() if not trade_log.empty else 0
    win_rate = (trade_log["Profit"] > 0).mean() * 100 if not trade_log.empty else 0

    return trade_log, total_profit, avg_profit, win_rate, capital


st.title("InvestEdge: Stock Recommendation and Trading Platform")

st.sidebar.header("Stock Parameters")

stock_ticker = st.sidebar.text_input(
    "Stock Ticker (e.g., AAPL, RELIANCE.NS)",
    "RELIANCE.NS",
).strip().upper()

if stock_ticker:
    try:
        data_info = yf.Ticker(stock_ticker).history(period="max")

        if not data_info.empty:
            min_date = data_info.index.min().date()
            max_date = datetime.today().date()
        else:
            min_date = (datetime.today() - timedelta(days=5 * 365)).date()
            max_date = datetime.today().date()

    except Exception:
        min_date = (datetime.today() - timedelta(days=5 * 365)).date()
        max_date = datetime.today().date()
else:
    min_date = (datetime.today() - timedelta(days=5 * 365)).date()
    max_date = datetime.today().date()

start_date = st.sidebar.date_input(
    "Start Date",
    min_date,
    min_value=min_date,
    max_value=max_date,
)

end_date = st.sidebar.date_input(
    "End Date",
    max_date,
    min_value=min_date,
    max_value=max_date,
)

prediction_days = st.sidebar.slider("Prediction Days", 30, 180, 60)

if st.sidebar.button("Fetch Data"):
    st.subheader(f"Fetching data for {stock_ticker}")

    data = fetch_stock_data(stock_ticker, start_date, end_date)

    if data.empty:
        st.error("No data fetched. Please check the stock ticker or date range.")
        st.stop()

    if "Close" not in data.columns:
        st.error("Close column not found in downloaded stock data.")
        st.stop()

    data = data.dropna(subset=["Close"])

    if len(data) <= prediction_days:
        st.error(
            "Not enough data for the selected prediction window. "
            "Please choose an earlier start date or reduce Prediction Days."
        )
        st.stop()

    latest_data_date = data.index[-1].date()

    st.success(f"Latest available data date: {latest_data_date}")

    if latest_data_date < end_date:
        st.info(
            f"The selected end date is {end_date}, but Yahoo Finance latest available "
            f"trading data is {latest_data_date}. This can happen on weekends, holidays, "
            f"or before the latest daily candle is published."
        )

    st.write(data.tail())

    st.subheader("Data Preprocessing")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    x_full = []
    y_full = []

    for i in range(prediction_days, len(scaled_data)):
        x_full.append(scaled_data[i - prediction_days:i, 0])
        y_full.append(scaled_data[i, 0])

    x_full = np.array(x_full)
    y_full = np.array(y_full)

    x_full = np.reshape(x_full, (x_full.shape[0], x_full.shape[1], 1))

    st.subheader("Training LSTM Model")

    model = create_model((x_full.shape[1], 1))
    model.fit(x_full, y_full, epochs=5, batch_size=32, verbose=1)

    st.success("Model trained successfully!")

    st.subheader("Predicting Stock Prices for Entire Dataset")

    predictions = model.predict(x_full)
    predictions = scaler.inverse_transform(predictions)

    full_predictions = np.full(len(data), np.nan)
    full_predictions[prediction_days:] = predictions.flatten()

    data["Predicted"] = full_predictions

    latest_actual_value = data["Close"].iloc[-1]
    latest_predicted = data["Predicted"].iloc[-1]

    percentage_difference = (
        (latest_predicted - latest_actual_value) / latest_actual_value
    ) * 100

    st.subheader("Real-Time Price Metrics")
    st.write(f"*Latest Available Price ({latest_data_date}):* ₹{latest_actual_value:.2f}")
    st.write(f"*Latest Predicted Price:* ₹{latest_predicted:.2f}")
    st.write(f"*Percentage Difference:* {percentage_difference:.2f}%")

    st.subheader("Actual vs Predicted Prices")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["Close"], label="Actual Prices", color="blue")
    ax.plot(
        data.index,
        data["Predicted"],
        label="Predicted Prices",
        color="orange",
        linestyle="--",
    )

    ax.legend()
    ax.set_title(f"{stock_ticker}: Actual vs Predicted Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    format_date_axis(ax, start_date, end_date)
    st.pyplot(fig)

    data["RSI"] = calculate_rsi(data)
    data["MA20"], data["Upper Band"], data["Lower Band"] = calculate_bollinger_bands(data)
    data["MACD"], data["Signal Line"] = calculate_macd(data)

    st.subheader("Moving Averages and Bollinger Bands")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["Close"], label="Actual Price", color="green", alpha=0.7)
    ax.plot(data.index, data["MA20"], label="MA20", color="yellow")
    ax.plot(
        data.index,
        data["Upper Band"],
        label="Upper Bollinger Band",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        data.index,
        data["Lower Band"],
        label="Lower Bollinger Band",
        color="blue",
        linestyle="--",
    )

    ax.legend()
    ax.set_title(f"Moving Averages and Bollinger Bands for {stock_ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    format_date_axis(ax, start_date, end_date)
    st.pyplot(fig)

    st.subheader("RSI (Relative Strength Index)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["RSI"], label="RSI", color="blue")
    ax.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", label="Oversold (30)")

    ax.legend()
    ax.set_title(f"RSI for {stock_ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI Value")
    format_date_axis(ax, start_date, end_date)
    st.pyplot(fig)

    st.subheader("MACD (Moving Average Convergence Divergence)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["MACD"], label="MACD", color="blue")
    ax.plot(
        data.index,
        data["Signal Line"],
        label="Signal Line",
        color="orange",
        linestyle="--",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    ax.legend()
    ax.set_title(f"MACD for {stock_ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("MACD Value")
    format_date_axis(ax, start_date, end_date)
    st.pyplot(fig)

    st.subheader("Buy/Sell Recommendations")

    signal_data = data.dropna(subset=["Predicted", "Close"]).copy()

    signal_data["Signal"] = "Hold"

    signal_data.loc[
        (signal_data["Predicted"] > signal_data["Close"])
        & (signal_data["Close"].diff() < 0),
        "Signal",
    ] = "Buy"

    signal_data.loc[
        (signal_data["Predicted"] < signal_data["Close"])
        & (signal_data["Close"].diff() > 0),
        "Signal",
    ] = "Sell"

    buy_signals = signal_data[signal_data["Signal"] == "Buy"]
    sell_signals = signal_data[signal_data["Signal"] == "Sell"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        signal_data.index,
        signal_data["Close"],
        label="Actual Prices",
        color="green",
        alpha=0.7,
    )

    ax.plot(
        signal_data.index,
        signal_data["Predicted"],
        label="Predicted Prices",
        color="orange",
        linestyle="--",
        alpha=0.7,
    )

    if not buy_signals.empty:
        ax.scatter(
            buy_signals.index,
            buy_signals["Close"],
            label="Buy Signal",
            color="blue",
            marker="^",
            alpha=1,
            edgecolors="black",
        )

    if not sell_signals.empty:
        ax.scatter(
            sell_signals.index,
            sell_signals["Close"],
            label="Sell Signal",
            color="red",
            marker="v",
            alpha=1,
            edgecolors="black",
        )

    ax.set_title(f"Buy/Sell Recommendations for {stock_ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    format_date_axis(ax, start_date, end_date)
    ax.legend()
    st.pyplot(fig)

    trade_log, total_profit, avg_profit, win_rate, capital = simulate_trades(signal_data)
    trade_log.to_csv("trade_log.csv", index=False)

    st.subheader("Predicted Prices for the Next 5 Trading Days")

    last_sequence = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)

    future_predicted_prices = []
    future_dates = []

    current_date = data.index[-1]

    while len(future_predicted_prices) < 5:
        current_date += timedelta(days=1)

        if current_date.weekday() >= 5:
            continue

        predicted_price_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0, 0]

        future_predicted_prices.append(predicted_price)
        future_dates.append(current_date)

        predicted_price_scaled_reshaped = predicted_price_scaled.reshape(1, 1, 1)

        last_sequence = np.append(
            last_sequence[:, 1:, :],
            predicted_price_scaled_reshaped,
            axis=1,
        )

    st.write(f"*Current Date:* {data.index[-1].strftime('%Y-%m-%d')}")
    st.write(f"*Current Price:* ₹{data['Close'].iloc[-1]:.2f}")
    st.write("*Predicted Prices for the Next 5 Trading Days:*")

    for date, price in zip(future_dates, future_predicted_prices):
        st.write(f"{date.strftime('%Y-%m-%d')}: ₹{price:.2f}")

    st.subheader("Model Performance Metrics")

    valid_mask = ~np.isnan(data["Predicted"])

    actual_prices = data["Close"][valid_mask].values.flatten()
    predicted_prices = data["Predicted"][valid_mask].values.flatten()

    mae = np.mean(np.abs(actual_prices - predicted_prices))
    mse = np.mean((actual_prices - predicted_prices) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    denominator = np.sum((actual_prices - np.mean(actual_prices)) ** 2)

    if denominator == 0:
        r_squared = 0
    else:
        r_squared = 1 - (
            np.sum((actual_prices - predicted_prices) ** 2) / denominator
        )

    actual_prices_diff = np.diff(actual_prices)
    predicted_prices_diff = np.diff(predicted_prices)

    if len(actual_prices_diff) == 0:
        direction_accuracy = 0
    else:
        direction_accuracy = (
            np.mean(np.sign(actual_prices_diff) == np.sign(predicted_prices_diff)) * 100
        )

    st.write(f"*Mean Absolute Error (MAE):* {mae:.2f}")
    st.write(f"*Mean Squared Error (MSE):* {mse:.2f}")
    st.write(f"*Root Mean Squared Error (RMSE):* {rmse:.2f}")
    st.write(f"*Mean Absolute Percentage Error (MAPE):* {mape:.2f}%")
    st.write(f"*R-Squared (R²):* {r_squared:.2f}")
    st.write(f"*Directional Accuracy (DA):* {direction_accuracy:.2f}%")

    st.write(
        f"*Latest Closing Price for {stock_ticker}:* ₹{data['Close'].iloc[-1]:.2f}"
    )
