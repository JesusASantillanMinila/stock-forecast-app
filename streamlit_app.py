import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import combinations
import plotly.graph_objects as go
import datetime
import time

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Forecast Pro", layout="wide")

st.title("Stock Forecasting Engine")

# --- placeholder for company info ---
company_info_placeholder = st.empty()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Configuration")
    
    var_ticker_input = st.text_input("Stock Ticker", value="").upper()
    var_past_horizon_mo = st.number_input("History Lookback (Months)", min_value=12, value=48, step=6)
    var_future_fcst_mo = st.number_input("Future Forecast (Months)", min_value=1, value=2, step=1)
    
    algo_choice = st.selectbox(
        "Forecasting Algorithm", 
        ("Facebook Prophet", "LSTM (Deep Learning)", "ARIMA (Coming Soon)")
    )
    
    run_button = st.button("Run Forecast", type="primary")

# --- DATA LOADING FUNCTION ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker, months):
    """
    Fetches Stock, VIX, Dividends, Earnings data AND Company Metadata.
    """
    try:
        time.sleep(1)
        var_ticker_class = yf.Ticker(ticker)
        
        # 0. Fetch Company Metadata
        try:
            time.sleep(1)
            t_info = var_ticker_class.info
            metadata = {
                "longName": t_info.get("longName", ticker),
                "longBusinessSummary": t_info.get("longBusinessSummary", "No summary available.")
            }
        except Exception as e:
            metadata = {
                "longName": ticker,
                "longBusinessSummary": "Company Summary Currently Unavailable" 
            }
        
        # 1. Download Stock Price
        df_stock_price = yf.download(ticker, period=f'{months}mo', progress=False)
        
        if isinstance(df_stock_price.columns, pd.MultiIndex):
            df_stock_price.columns = df_stock_price.columns.droplevel(1)
            
        df_stock_price.index.name = None
        df_stock_price["Date"] = pd.to_datetime(df_stock_price.index, errors='coerce')
        df_stock_price["Date"] = df_stock_price["Date"].dt.date
        df_stock_price = df_stock_price.reset_index(drop=True)

        if df_stock_price.empty:
            return None, None, "No price data found for ticker."

        # 2. Get Dividends
        try:
            df_div_splits = var_ticker_class.actions
            df_div_splits.index.name = None
            df_div_splits["Date"] = pd.to_datetime(df_div_splits.index, errors='coerce')
            df_div_splits = df_div_splits.reset_index(drop=True)
            df_div_splits["Date"] = df_div_splits["Date"].dt.date
            df_div_splits = df_div_splits[df_div_splits['Date'] >= df_stock_price['Date'].min()]
        except:
            df_div_splits = pd.DataFrame(columns=['Date', 'Dividends'])

        # Merge Dividends
        df_fcst_input = pd.merge(
            df_stock_price,
            df_div_splits[['Date', 'Dividends']] if 'Dividends' in df_div_splits.columns else df_div_splits,
            how='left',
            on="Date"
        )

        # 3. Get Earnings (EPS)
        try:
            df_eps = var_ticker_class.earnings_dates
            if df_eps is not None and not df_eps.empty:
                df_eps.index.name = None
                df_eps["Date"] = pd.to_datetime(df_eps.index, errors='coerce')
                df_eps = df_eps.reset_index(drop=True)
                df_eps["Date"] = df_eps["Date"].dt.date
                df_eps = df_eps[df_eps['Date'] >= df_stock_price['Date'].min()]
            else:
                df_eps = pd.DataFrame(columns=['Date', 'Reported EPS'])
        except:
            df_eps = pd.DataFrame(columns=['Date', 'Reported EPS'])

        if 'Reported EPS' in df_eps.columns:
            df_fcst_input = pd.merge(df_fcst_input, df_eps[['Date', 'Reported EPS']], how='left', on="Date")
        else:
            df_fcst_input['Reported EPS'] = np.nan

        # 4. Get VIX Data
        var_ticker_class_vix = yf.Ticker("^VIX")
        df_vix = var_ticker_class_vix.history(period=f'{months}mo')
        df_vix.index.name = None
        df_vix["Date"] = pd.to_datetime(df_vix.index, errors='coerce')
        df_vix["Date"] = df_vix["Date"].dt.date
        df_vix = df_vix.reset_index(drop=True)
        df_vix = df_vix.add_prefix('Volatility Index ')

        # Merge VIX
        df_fcst_input = pd.merge(
            df_fcst_input,
            df_vix,
            how='left',
            left_on="Date",
            right_on="Volatility Index Date"
        )
        
        # Final Prep
        df_prophet = df_fcst_input.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Fill NaNs
        if 'Dividends' in df_prophet.columns: df_prophet['Dividends'] = df_prophet['Dividends'].fillna(0)
        if 'Reported EPS' in df_prophet.columns: df_prophet['Reported EPS'] = df_prophet['Reported EPS'].fillna(0)
        if 'Volatility Index Close' in df_prophet.columns: df_prophet['Volatility Index Close'] = df_prophet['Volatility Index Close'].ffill().bfill()
        if 'Volume' in df_prophet.columns: df_prophet['Volume'] = df_prophet['Volume'].ffill().bfill()
        
        return df_prophet, metadata, None
        
    except Exception as e:
        return None, None, str(e)

# --- PROPHET MODEL FUNCTION ---
def run_prophet_competition(df, history_months):
    """
    Runs the regressor competition loop for Prophet.
    """
    potential_regressors = ['Volume', 'Reported EPS', 'Dividends', 'Volatility Index Close']
    available_regressors = [r for r in potential_regressors if r in df.columns]
    
    best_rmse = float('inf')
    best_model = None
    best_regressor_combo = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combos = sum(1 for r in range(1, len(available_regressors) + 1) 
                       for _ in combinations(available_regressors, r))
    current_iter = 0
    results_log = []

    for r in range(1, len(available_regressors) + 1):
        for combo in combinations(available_regressors, r):
            regressor_list = list(combo)
            status_text.text(f"Testing features: {', '.join(regressor_list)}...")
            
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            for reg in regressor_list:
                m.add_regressor(reg)
            
            m.fit(df)
            
            days_history = history_months * 30
            initial_days = f"{int(days_history * 0.5)} days"
            
            try:
                df_cv = cross_validation(
                    m, 
                    initial=initial_days, 
                    period='90 days', 
                    horizon='30 days', 
                    parallel="processes"
                )
                df_p = performance_metrics(df_cv)
                current_rmse = df_p['rmse'].mean()
                
                results_log.append({
                    "Features": ", ".join(regressor_list),
                    "RMSE": current_rmse
                })

                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_model = m
                    best_regressor_combo = regressor_list
                    
            except Exception as e:
                pass
            
            current_iter += 1
            progress_bar.progress(current_iter / total_combos)

    progress_bar.empty()
    status_text.empty()
    
    return best_model, best_regressor_combo, best_rmse, pd.DataFrame(results_log)

# --- LSTM MODEL FUNCTION ---
def run_lstm_forecast(df, future_months):
    """
    Runs an LSTM forecast and returns a dataframe in Prophet format (ds, yhat, yhat_lower, yhat_upper).
    """
    # 1. Prepare Data
    data = df.filter(['y'])
    dataset = data.values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # 2. Split Data (Keep most for training)
    # For a forecast app, we ideally want to train on as much as possible, 
    # but we hold back a small bit to calculate error/uncertainty bands
    training_data_len = int(np.ceil(len(dataset) * .95))
    
    train_data = scaled_data[0:int(training_data_len), :]
    
    # Create the training data set
    # Window size = 60 days
    window_size = 60
    x_train = []
    y_train = []
    
    if len(train_data) <= window_size:
        st.error("Not enough data to run LSTM. Need more history.")
        return None

    for i in range(window_size, len(train_data)):
        x_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # 3. Build LSTM Model
    # Simple architecture to run fast on Streamlit Cloud CPU
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    with st.spinner('Training LSTM Neural Network...'):
        model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
        
    # 4. Calculate Uncertainty (RMSE) on the withheld 5% data
    test_data = scaled_data[training_data_len - window_size:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    
    for i in range(window_size, len(test_data)):
        x_test.append(test_data[i-window_size:i, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    
    # 5. Recursive Forecasting for Future
    future_days = future_months * 30
    
    # Start with the last 60 days of the ENTIRE dataset
    last_60_days = scaled_data[-window_size:]
    current_batch = last_60_days.reshape((1, window_size, 1))
    
    future_preds = []
    
    for i in range(future_days):
        # Predict 1 step ahead
        current_pred = model.predict(current_batch, verbose=0)[0]
        future_preds.append(current_pred)
        
        # Update batch: append prediction, remove first item
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
    # Inverse scale
    future_preds = scaler.inverse_transform(future_preds)
    
    # 6. Create Output DataFrame
    last_date = df['ds'].iloc[-1]
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.date()
        
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, future_days + 1)]
    
    # Construct Prophet-like Dataframe
    df_future = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_preds.flatten(),
        'yhat_lower': future_preds.flatten() - (1.96 * rmse), # Approx 95% CI
        'yhat_upper': future_preds.flatten() + (1.96 * rmse)
    })
    
    # Ensure datetimes match
    df_future['ds'] = pd.to_datetime(df_future['ds'])
    
    return df_future

# --- MAIN APP LOGIC ---

if run_button:
    with st.spinner('Downloading Data and Preprocessing...'):
        df_data, meta_data, error = get_stock_data(var_ticker_input, var_past_horizon_mo)

    if error:
        st.error(f"Error: {error}")
    else:
        # --- DISPLAY COMPANY INFO ---
        with company_info_placeholder.container():
            st.markdown(f"## {meta_data['longName']}")
            with st.expander("Show Company Summary", expanded=False):
                st.write(meta_data['longBusinessSummary'])
            st.divider()

        # --- FORECAST LOGIC SWITCH ---
        forecast_df = None
        
        # 1. FACEBOOK PROPHET PATH
        if algo_choice == "Facebook Prophet":
            st.subheader("Model Optimization (Prophet)")
            if len(df_data) < 180:
                st.warning("Data history is very short. Forecast quality may be low.")
            
            best_model, best_combo, best_rmse, results_df = run_prophet_competition(df_data, var_past_horizon_mo)

            if best_model is None:
                st.error("Could not find a valid model.")
            else:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.success("Optimization Complete!")
                    st.write("**Best Regressors:**")
                    for feature in best_combo:
                        st.code(feature)
                
                # Make Future DF
                future = best_model.make_future_dataframe(periods=var_future_fcst_mo*30)
                
                # Fill Regressors
                for reg in best_combo:
                    last_known_value = df_data[reg].iloc[-1]
                    future[reg] = df_data[reg] # Fill historical
                    future[reg] = future[reg].fillna(last_known_value) # Fill future

                # Predict
                forecast_raw = best_model.predict(future)
                
                # Filter to only future dates for consistency with LSTM logic
                forecast_df = forecast_raw[forecast_raw['ds'] > pd.Timestamp(df_data['ds'].max())]

        # 2. LSTM PATH
        elif algo_choice == "LSTM (Deep Learning)":
            st.subheader("Model Training (LSTM)")
            if len(df_data) < 180:
                st.error("LSTM requires at least 180 days of history to create sequences.")
            else:
                try:
                    forecast_df = run_lstm_forecast(df_data, var_future_fcst_mo)
                    st.success("LSTM Network Trained & Forecast Generated!")
                except Exception as e:
                    st.error(f"LSTM Training Failed: {e}")

        # --- PLOTTING ---
        if forecast_df is not None:
            col_chart = st.container()
            
            with col_chart:
                st.subheader(f"Forecast: {var_ticker_input}")
                
                fig = go.Figure()

                # Historical Data (Actuals)
                fig.add_trace(go.Scatter(
                    x=df_data['ds'], 
                    y=df_data['y'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='deepskyblue')
                ))

                # Forecast Data
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orangered')
                ))

                # Uncertainty Intervals
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 69, 0, 0.2)',
                    line=dict(width=0),
                    name='Uncertainty Interval'
                ))

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified",
                    template="plotly_white",
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
