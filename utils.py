import pickle
import pandas as pd
import numpy as np
import requests
import ta

def get_model(model):
    try:
        # Load the model from the file
        with open(model, 'rb') as file:
            loaded_model = pickle.load(file)
    except Exception as e:
        print("couldn't load model")
    return loaded_model


def fetch_data(asset, trading_session, timeframe, start_date, end_date):
    """
    Fetches data from the backtester API based on the provided parameters.
    
    Parameters:
    asset (str): The asset symbol (e.g., 'QQQ').
    trading_session (str): The trading session (e.g., 'RTH').
    timeframe (str): The timeframe for data (e.g., '30m').
    start_date (str): The start date for fetching data (YYYY-MM-DD).
    end_date (str): The end date for fetching data (YYYY-MM-DD).
    
    Returns:
    dict: The JSON response from the API if the request was successful.
    None: If there was an error with the request.
    """
    
    # Define the base URL for the API
    base_url = "http://51.81.60.92:7003/backtester/data-management/"
    
    # Define query parameters
    params = {
        'asset': asset,
        'trading_session': trading_session,
        'timeframe': timeframe,
        'start_date': start_date,
        'end_date': end_date
    }
    
    try:
        # print("Fetching data...")
        # Make the request
        response = requests.get(base_url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            # print("Data fetched successfully.")
            return data
        else:
            print(f"Failed to fetch data. Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def data_preprocessing(data):
    rename_dict = {
          'timestamp': 'Date',
          'open': 'Open',
          'high': 'High',
          'low': 'Low',
          'close': 'Close',
          'volume': 'Volume'  # You can leave this if you want to keep the 'Volume' column
      }

    # Use the rename method to rename the columns
    data = data.rename(columns=rename_dict)
    data['Date'] = pd.to_datetime(data['Date'])
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
    data['Returns'] = data['Close'].pct_change()
    return data

# Feature Engineering Function (Including Fibonacci, MACD, Bollinger Bands, and ATR)
def feature_engineering(df):

    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_90'] = df['Close'].rolling(window=90).mean()
    df['SMA_180'] = df['Close'].rolling(window=180).mean()
    df['SMA_400'] = df['Close'].rolling(window=400).mean()
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    # Candle-related features
    df['high_low_ratio'] = df['High'] / df['Low']
    df['open_adjclose_ratio'] = df['Close'] / df['Open']
    df['candle_to_wick_ratio'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    upper_wick_size = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_wick_size = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['upper_to_lower_wick_ratio'] = upper_wick_size / lower_wick_size

    

    # Lag features
    for lag in range(1, 6):
        df[f'lag{lag}'] = df['Close'].shift(lag)

    df['close_to_lag1_ratio'] = df['Close'] / df['lag1']
    df['close_to_lag2_ratio'] = df['Close'] / df['lag2']

    # Moving averages
    df['ema5'] = ta.trend.ema_indicator(df['Close'], window=5)
    df['sma10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['close_ema5_ratio'] = df['Close'] / df['ema5']
    df['close_sma10_ratio'] = df['Close'] / df['sma10']
    df['ema5_sma10_ratio'] = df['ema5'] / df['sma10']

    # # Volume-related features
    df['volume_sma5'] = ta.trend.sma_indicator(df['Volume'], window=5)
    df['volume_sma10'] = ta.trend.sma_indicator(df['Volume'], window=10)
    df['volume_shock'] = (df['Volume'] - df['volume_sma10']) / df['volume_sma10']
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_price_trend'] = df['volume_change'] / df['Close'].pct_change()

    # Volatility features
    df['10_days_volatility'] = df['Close'].pct_change().rolling(window=10).std()
    df['20_days_volatility'] = df['Close'].pct_change().rolling(window=20).std()
    df['9_to_20_day_vol_ratio'] = df['10_days_volatility'] / df['20_days_volatility']

    # Momentum features
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['rsi_overbought'] = (df['rsi'] >= 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] <= 30).astype(int)
    df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=10, constant=0.015)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=10).adx()
    df['ADI'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()

    # Weekly returns
    df['weekly_returns'] = np.round(((df['Close'] - df['Open']) / df['Open']) * 100, 2)

    # Fibonacci Retracement
    df['High_Max'] = df['High'].rolling(window=14).max()
    df['Low_Min'] = df['Low'].rolling(window=14).min()
    df['Fib_0.0'] = df['High_Max']
    df['Fib_0.236'] = df['High_Max'] - 0.236 * (df['High_Max'] - df['Low_Min'])
    df['Fib_0.382'] = df['High_Max'] - 0.382 * (df['High_Max'] - df['Low_Min'])
    df['Fib_0.5'] = df['High_Max'] - 0.5 * (df['High_Max'] - df['Low_Min'])
    df['Fib_0.618'] = df['High_Max'] - 0.618 * (df['High_Max'] - df['Low_Min'])
    df['Fib_0.764'] = df['High_Max'] - 0.764 * (df['High_Max'] - df['Low_Min'])
    df['Fib_1.0'] = df['Low_Min']
    df.drop(['High_Max', 'Low_Min'], axis=1, inplace=True)

    # MACD
    ema_short = df['Close'].ewm(span=12, adjust=False).mean()
    ema_long = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_short - ema_long
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    middle_band = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Middle'] = middle_band
    df['Bollinger_Upper'] = middle_band + (rolling_std * 2)
    df['Bollinger_Lower'] = middle_band - (rolling_std * 2)

    # ATR
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'Prev_Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Prev_Close']), abs(x['Low'] - x['Prev_Close'])), axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.drop(['Prev_Close', 'TR'], axis=1, inplace=True)

    features =['Date', 'Open', 'High', 'Low', 'Close', 'SMA_7', 'SMA_30',
                'EMA_8', 'EMA_20', 'Fib_0.0',
                'high_low_ratio',
                'open_adjclose_ratio',
                'candle_to_wick_ratio',
                'upper_to_lower_wick_ratio',
                'lag1', 'lag2', 'lag3', 'lag4', 'lag5',
                'close_to_lag1_ratio', 'close_to_lag2_ratio',
                'ema5', 'sma10',
                'volume_shock', 
                'weekly_returns', "Returns"
            ]
    
    df = df[features]
    # Remove infinities and NaNs
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)

    return df

def calculate_returns(data, model):
    """
    Calculate strategy returns based on model predictions.
    """
    features =['Open', 'High', 'Low', 'Close', 'SMA_7', 'SMA_30', 
                'EMA_8', 'EMA_20', 'Fib_0.0',
                'high_low_ratio',
                'open_adjclose_ratio',
                'candle_to_wick_ratio',
                'upper_to_lower_wick_ratio',
                'lag1', 'lag2', 'lag3', 'lag4', 'lag5',
                'close_to_lag1_ratio', 'close_to_lag2_ratio',
                'ema5', 'sma10',
                'volume_shock', 
                'weekly_returns'
            ]
    
    input_features = data[features]
    data['Predicted_Signal'] = model.predict(input_features)
    data['Strategy_Returns'] = data['Returns'] * data['Predicted_Signal'].shift(1)
    data['Buy_and_Hold_Returns'] = data['Returns']
    return data

def filter_data_by_date(data, start_date, end_date):
    """
    Filter data based on the specified date range.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

def calculate_cumulative_returns(data):
    """
    Calculate cumulative returns for strategy and buy-and-hold.
    """
    data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod() - 1
    data['Buy_and_Hold_Cumulative_Returns'] = (1 + data['Buy_and_Hold_Returns']).cumprod() - 1
    return data


def adjust_signals_before_first_buy(data):
    """
    Replace all signals with 'Hold' until the first 'Buy' signal occurs.
    """
    # Set a flag to indicate when the first 'Buy' has been found
    first_buy_found = False
    data = data.reset_index(drop=True)
    # Iterate through the DataFrame and update the Signal column
    for i in range(len(data)):
        if not first_buy_found:
            if data.loc[i, 'Trades'] == 1.0:
                first_buy_found = True
            else:
                data.loc[i, 'Trades'] = 0.0
        else:
            break  # Stop processing after the first Buy has been encountered

    return data

def evaluate_strategy(data):
    """
    Evaluate the trading strategy using various metrics, including average trade duration.
    """
    
    data["Trades"] = data['Predicted_Signal'].diff().fillna(0)
    # Adjust the signals before the first Buy
    data = adjust_signals_before_first_buy(data)
    data['Signal'] = data["Trades"].copy()
    data['Signal'] = data['Signal'].replace({
        1.0: "Buy",
        0.0: "Hold",
        -1.0: "Sell"
    })

    signal = data[["Date", "Signal"]].tail(1)

    trades = data["Trades"]
    entries = trades[trades == 1].index
    exits = trades[trades == -1].index

    if len(exits) > len(entries):
        exits = exits[:len(entries)]

    trade_pairs = zip(entries, exits)
    trade_profits = []
    trade_durations = []  # To store trade durations
    trade_table = pd.DataFrame()

    for entry, exit in trade_pairs:
        entry_time = data.loc[entry, 'Date']
        exit_time = data.loc[exit, 'Date']
        entry_price = data.loc[entry, 'Open']
        exit_price = data.loc[exit, 'Close']
        PandL = ((entry_price - exit_price) / entry_price) * 100
        trade_duration = exit_time - entry_time  # Calculate the duration of the trade
        trade_durations.append(trade_duration.total_seconds() / 3600)  # Store duration in hours

        new_trade = pd.DataFrame([{
            'Entry_Time': entry_time,
            'Exit_Time': exit_time,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'P/L': PandL,
            'Average': PandL - 0.05/100,
            'Trade_Duration_Hours': trade_duration.total_seconds() / 3600  # Store duration in hours
        }])
        trade_table = pd.concat([trade_table, new_trade], ignore_index=True)
        trade_profits.append((entry_price - exit_price) / entry_price)

    # Calculate metrics
    metrics = {
        "win_rate": (sum(1 for profit in trade_profits if profit > 0) / len(trade_profits)) if len(trade_profits) > 0 else 0,
        "total_return": sum(trade_profits) if len(trade_profits) > 0 else 0,
        "average_upside": np.mean([profit for profit in trade_profits if profit > 0]) if any(profit > 0 for profit in trade_profits) else 0,
        "average_downside": np.mean([profit for profit in trade_profits if profit < 0]) if any(profit < 0 for profit in trade_profits) else 0,
        "expected_pl_per_trade": np.mean(trade_profits) if len(trade_profits) > 0 else 0,
        "max_loss": np.min(trade_profits) if len(trade_profits) > 0 else 0,
        "max_profit": np.max(trade_profits) if len(trade_profits) > 0 else 0,
        "max_drawdown": -(data['Strategy_Cumulative_Returns'].cummax() - data['Strategy_Cumulative_Returns']).max() if len(data['Strategy_Cumulative_Returns']) > 0 else 0,
        "max_runup": (data['Strategy_Cumulative_Returns'] - data['Strategy_Cumulative_Returns'].cummin()).max() if len(data['Strategy_Cumulative_Returns']) > 0 else 0,
        "number_of_trades": len(trade_profits),
        "profit_factor": (sum(profit for profit in trade_profits if profit > 0) / 
                        abs(sum(loss for loss in trade_profits if loss < 0))) if len(trade_profits) > 0 and any(loss < 0 for loss in trade_profits) else 0,
        "average_trade_duration_hours": np.mean(trade_durations) if trade_durations else 0  # Average trade duration in hours
    }

    if trade_table.empty:
        print("No trades were executed.")
        return None, signal, metrics

    return trade_table.tail(1), signal, metrics

def print_metrics(metrics):
    """
    Print trading strategy metrics.
    """
    if metrics:
        print("\n")
        print("Trading Strategy Metrics:")
        print("Number of Trades: {}".format(metrics['number_of_trades']))
        print("Win Rate: {:.2f}%".format(metrics['win_rate'] * 100))
        print("Total Return: {:.2f}%".format(metrics['total_return'] * 100))
        print("Average Upside/Winning Trade: {:.2f}%".format(metrics['average_upside'] * 100))
        print("Average Downside/Losing Trade: {:.2f}%".format(metrics['average_downside'] * 100))
        print("Expected P/L Per Trade: {:.2f}%".format(metrics['expected_pl_per_trade'] * 100))
        print("Largest Winning Trade: {:.2f}%".format(metrics['max_profit'] * 100))
        print("Largest Losing Trade: {:.2f}%".format(metrics['max_loss'] * 100))
        print("Maximum Drawdown: {:.2f}%".format(metrics['max_drawdown'] * 100))
        print("Maximum Run-up: {:.2f}%".format(metrics['max_runup'] * 100))
        print("Profit Factor: {:.2f}".format(metrics['profit_factor']))

def print_buy_and_hold_metrics(data):
    """
    Print buy-and-hold metrics.
    """
    buy_and_hold_value = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
    buy_and_hold_return = data['Buy_and_Hold_Cumulative_Returns'].iloc[-1]
    buy_and_hold_max_drawdown = -(data['Buy_and_Hold_Cumulative_Returns'].cummax() - data['Buy_and_Hold_Cumulative_Returns']).max()
    buy_and_hold_max_runup = (data['Buy_and_Hold_Cumulative_Returns'] - data['Buy_and_Hold_Cumulative_Returns'].cummin()).max()

    print("\nBuy-and-Hold Metrics:")
    print("Total Return (Cumulative): {:.2f}%".format(buy_and_hold_return * 100))
    print("Buy and Hold Value: {:.2f}%".format(buy_and_hold_value * 100))
    print("Maximum Drawdown: {:.2f}%".format(buy_and_hold_max_drawdown * 100))
    print("Maximum Run-up: {:.2f}%".format(buy_and_hold_max_runup * 100))
    
    print("---------------------------------------------")
    print("\n\n")

def metrics_to_dict(metrics):
    """
    Convert trading strategy metrics into a dictionary.
    """
    metrics_dict = None
    if metrics:
        metrics_dict = {
            "Total Return (%)": round(metrics['total_return'] * 100, 4),
            "Number of Trades": metrics['number_of_trades'],
            "Win Rate (%)": round(metrics['win_rate'] * 100, 4),
            "Average Upside/Winning Trade (%)": round(metrics['average_upside'] * 100, 4),
            "Average Downside/Losing Trade (%)": round(metrics['average_downside'] * 100, 4),
            "Expected P/L Per Trade (%)": round(metrics['expected_pl_per_trade'] * 100, 4),
            "Largest Winning Trade (%)": round(metrics['max_profit'] * 100, 4),
            "Largest Losing Trade (%)": round(metrics['max_loss'] * 100, 4),
            "Max Drawdown (%)": round(metrics['max_drawdown'] * 100, 4),
            "Max Run-up (%)": round(metrics['max_runup'] * 100, 4),
            "Profit Factor": round(metrics['profit_factor'], 4),
            "Average Trade Duration (Hours)": round(metrics['average_trade_duration_hours'], 4)
        }
    
    return metrics_dict

def buy_and_hold_metrics_to_dict(data):
    """
    Convert buy-and-hold metrics into a dictionary.
    """
    buy_and_hold_value = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / 100
    buy_and_hold_return = data['Buy_and_Hold_Cumulative_Returns'].iloc[-1]
    buy_and_hold_max_drawdown = -(data['Buy_and_Hold_Cumulative_Returns'].cummax() - data['Buy_and_Hold_Cumulative_Returns']).max()
    buy_and_hold_max_runup = (data['Buy_and_Hold_Cumulative_Returns'] - data['Buy_and_Hold_Cumulative_Returns'].cummin()).max()

    metrics_dict = {
        "Buy and Hold Return (Cumulative) (%)": round(buy_and_hold_return * 100, 2),
        "Buy and Hold Value (%)": round(buy_and_hold_value * 100, 2),
        "Buy and Hold Max Drawdown (%)": round(buy_and_hold_max_drawdown * 100, 2),
        "Buy and Hold Max Run-up (%)": round(buy_and_hold_max_runup * 100, 2)
    }
    
    return metrics_dict
