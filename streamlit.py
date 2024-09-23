import streamlit as st
from utils import *
import requests
import time
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt

# Suppress all warnings
warnings.filterwarnings('ignore')

# Define the base URL for the API
base_url = "http://51.81.60.92:7003/backtester/data-management/"

def main():
    st.set_page_config(layout="wide")  # Set Streamlit app to wide mode

    st.title("Real-Time Trading Data Streamer")

    # Create layout with 2 columns for upper part and 2 columns for lower part
    col_input, col_graph = st.columns(2)  # Upper part
    col_table, col_trades  = st.columns(2)  # Lower part

    # Left side (upper) for user input fields
    with col_input:
        st.header("Input Parameters")
        asset = st.text_input("Asset", "QQQ")
        trading_session = st.selectbox("Trading Session", ["RTH", "ETH"], index=0)
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)
        start_date = st.text_input("Start Date", "2024-09-03 00:00:00")
        end_date = st.text_input("End Date", "2024-09-10 15:30:00")
        delay = st.slider("Delay between data drops (seconds)", 1, 10, 2)
        tick_counter_placeholder = st.empty()
        update_placeholder = st.empty()

        params_forward = {
            'asset': asset,
            'trading_session': trading_session,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'raw_data': 'true',
            'delay': delay  # Delay between data drops
        }

    # Placeholders for dynamic data in the rest of the layout
    with col_graph:
        st.header("Cumulative Returns")
        plot_placeholder = st.empty()  # Plot of Cumulative Returns
    
    with col_trades:
        st.header("Trade Table")
        trades_placeholder = st.empty()  # Data Table of Trades
    
    with col_table:
        st.header("Return Metrics")
        table_placeholder = st.empty()  # Data Table of Returns

    params = {
        'asset': 'QQQ',
        'trading_session': 'RTH',
        'timeframe': '5m',
        'start_date': '2019-07-01 09:30:00',
        'end_date': '2024-09-03 00:00:00',
    }

    if st.button("Start Streaming"):
        # Display loading message
        st.write("Starting the stream...")

        # Create an empty DataFrame for trade points
        table = pd.DataFrame()
        complete_trade = pd.DataFrame()
        trades_in_session = pd.DataFrame()

        tick_count = 0
        # Make the request and stream the response
        with requests.get(base_url, params=params_forward, stream=True) as response:
            if response.status_code == 200:
                iter_lines = response.iter_lines()  # Create an iterator over the response lines

                while True:
                    try:
                        # Fetch the next line (data point)
                        line = next(iter_lines)

                        if line:
                            tick_count += 1
                            data = line.decode('utf-8')
                            # Simulate fetching historical data (use your actual function)
                            historical_data = fetch_data(params['asset'], params['trading_session'], 
                                                params['timeframe'], params['start_date'], params['end_date'])
                            
                            # Parse new trade point
                            data_dict = json.loads(data[5:].replace("'", '"'))
                            tick_counter_placeholder.write(f"Tick: {tick_count}")
                            update_placeholder.write(f"Latest Tick: {data_dict}")

                            new_trade_point = pd.DataFrame([data_dict])
                            trades_in_session = pd.concat([trades_in_session, new_trade_point], ignore_index=True)
                            data = pd.concat([historical_data, trades_in_session], ignore_index=True)

                            data = data_preprocessing(data)
                            data = feature_engineering(data)
                            model = get_model('model.pkl')
                            data = calculate_returns(data, model)
                            data = filter_data_by_date(data, params_forward['start_date'], params_forward['end_date'])
                            data = calculate_cumulative_returns(data)
                            trade_table, signal, metrics = evaluate_strategy(data)
                            metrics = metrics_to_dict(metrics)
                            buy_hold = buy_and_hold_metrics_to_dict(data)

                            # Combine both metrics dictionaries
                            if metrics is None:
                                metrics = {}
                            combined_metrics = {**metrics, **buy_hold}

                            for key, value in combined_metrics.items():
                                try:
                                    signal[key] = value
                                except Exception as e:
                                    pass

                            for key, value in combined_metrics.items():
                                try:
                                    trade_table[key] = value
                                except Exception as e:
                                    pass

                            table = pd.concat([table, signal], ignore_index=True)
                            complete_trade = pd.concat([complete_trade, trade_table], ignore_index=True)
                            if len(complete_trade) > 0:
                                complete_trade = complete_trade.groupby('Entry_Time', as_index=False).first()
                                        
                            if not table.empty:
                                # Plot cumulative returns
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(table['Date'], table['Buy and Hold Return (Cumulative) (%)'], label='Buy and Hold', color='blue')
                                ax.plot(table['Date'], table['Total Return (%)'], label='Strategy', color='green')
                                ax.set_xlabel("Date")
                                ax.set_ylabel("Cumulative Returns")
                                ax.legend()

                                # Update the placeholders
                                plot_placeholder.pyplot(fig)
                                table_placeholder.dataframe(table, use_container_width=True)
                                trades_placeholder.dataframe(complete_trade, use_container_width=True)

                            # Delay for next data point
                            time.sleep(params_forward['delay'])

                    except StopIteration:
                        st.write("End of data stream.")
                        break

            else:
                st.error(f"Error: {response.status_code}")

if __name__ == "__main__":
    main()
