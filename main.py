import requests
import time
from utils import *
import pandas as pd
import json
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Define the base URL for the API
base_url = "http://51.81.60.92:7003/backtester/data-management/"

def main():
    print("starting...")
    # Define query parameters with a delay of 2 seconds between rows
    params_forward = {
        'asset': 'QQQ',
        'trading_session': 'RTH',
        'timeframe': '5m',
        'start_date': '2024-09-03 00:00:00',
        'end_date': '2024-09-10 15:30:00',
        'raw_data': 'true',
        'delay': 1  # Delay of 5 seconds between data drops
    }

    params = {
        'asset': 'QQQ',
        'trading_session': 'RTH',
        'timeframe': '5m',
        'start_date': '2019-07-01 09:30:00',
        'end_date': '2024-09-03 00:00:00',
    }

    # Make the request and stream the response
    with requests.get(base_url, params=params_forward, stream=True) as response:
        if response.status_code == 200:
            iter_lines = response.iter_lines()  # Create an iterator over the response lines
            trades_in_session = pd.DataFrame()
            while True:
                
                try:
                    # Fetch the next line (data point)
                    line = next(iter_lines)

                    if line:
                        data = line.decode('utf-8')
                        # print("new trade point: ", data[5:])  # Output the current data point
                        historical_data = fetch_data(params['asset'], params['trading_session'], 
                                            params['timeframe'], params['start_date'], params['end_date'])
                        data_dict = json.loads(data[5:].replace("'", '"'))
                        print("-------------------------------------")
                        print(data_dict)
                        new_trade_point = pd.DataFrame([data_dict])
                        trades_in_session = pd.concat([trades_in_session, new_trade_point], ignore_index=True)
                        data = pd.concat([historical_data, trades_in_session], ignore_index=True)
                        # print("concatenation", data.tail(5))
                        data = data_preprocessing(data)
                        # print("after preprocessing", data.tail(5))
                        data = feature_engineering(data)
                        # print("after feature engineering", data.tail(5))
                        model = get_model('model.pkl')
                        data = calculate_returns(data, model)
                        data = filter_data_by_date(data, params_forward['start_date'], params_forward['end_date'])
                        # print("calculating returns", data.tail(5))
                        data = calculate_cumulative_returns(data)
                        # print("after cumulative", data.tail(5))
                        trade_table, signal, metrics = evaluate_strategy(data)
                        print_metrics(metrics)
                        print_buy_and_hold_metrics(data)
                                            

                    # Add a delay (if you need client-side delay in addition to server-side)
                    time.sleep(params_forward['delay'])

                except StopIteration:
                    print("End of stream")
                    break  # Stop when no more data is available
        else:
            print(f"Error: {response.status_code}")

if __name__ == "__main__":
    main()
