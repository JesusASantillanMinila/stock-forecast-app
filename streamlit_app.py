import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
import time

# --- PROPHET ---
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import combinations

# --- LSTM ---
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Forecast Pro", layout="wide")

st.title("Stock Forecasting Engine")

# --- placeholder for company info ---
company_info_placeholder = st.empty()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Configuration")
    
    var_ticker_input = st.text_input("Stock Ticker", value="").upper()
    var_past_horizon_mo = st.number_input("History Lookback (Months)", min_value=12, value=12, step=6)
    var_future_fcst_mo = st.number_input("Future Forecast (Months)", min_value=1, value=1, step=1)
    
    algo_choice = st.selectbox(
        "Forecasting Algorithm", 
        ("Facebook Prophet", "LSTM", "ARIMA (Coming Soon)")
    )
    
    run_button = st.button("Run Forecast", type="primary")

# --- DATA LOADING FUNCTION ---
@st.cache_data(ttl=3600) 
def get_stock_data(ticker, months):
    """
    Fetches Stock, VIX, Dividends, Earnings data AND Company Metadata.
    """
    try:
        
        var_ticker_class = yf.Ticker(ticker)
        time.sleep(1)
        
        # 0. Fetch Company Metadata
        try:
            
            t_info = var_ticker_class.info
            time.sleep(1)
            var_long_name = t_info.get("longName", ticker)
            time.sleep(1)
            var_business_summary = t_info.get("longBusinessSummary", "No summary available. Too Many API  Requests. Rate limited. Try after a while.")
            
            metadata = {
                "longName": var_long_name,
                "longBusinessSummary": var_business_summary
            }
        except Exception as e:
            metadata = {
                "longName": ticker,
                "longBusinessSummary": "No summary available. Too Many API  Requests. Rate limited. Try after a while."
            }
        
        # 1. Download Stock Price
        # Fetch an extra 12 months to calculate the 12-month moving average correctly
        fetch_months = months + 12
        df_stock_price = yf.download(ticker, period=f'{fetch_months}mo', progress=False)
        
        if isinstance(df_stock_price.columns, pd.MultiIndex):
            df_stock_price.columns = df_stock_price.columns.droplevel(1)
            
        df_stock_price.index.name = None
        df_stock_price["Date"] = pd.to_datetime(df_stock_price.index, errors='coerce')
        df_stock_price["Date"] = df_stock_price["Date"].dt.date
        df_stock_price = df_stock_price.reset_index(drop=True)

        if df_stock_price.empty:
            return None, None, "No price data found for ticker."

        # --- moving averages ---
        df_stock_price['Moving Average 50 Days'] = df_stock_price['Close'].rolling(window=50).mean()
        df_stock_price['Moving Average 100 Days'] = df_stock_price['Close'].rolling(window=100).mean()
        df_stock_price['Moving Average 200 Days'] = df_stock_price['Close'].rolling(window=200).mean()

        # --- slice data ---
        max_date = pd.to_datetime(df_stock_price['Date']).max()
        cutoff_date = (max_date - pd.DateOffset(months=months)).date()
        
        # Filter for the requested period
        df_stock_price = df_stock_price[df_stock_price['Date'] > cutoff_date].reset_index(drop=True)

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

        # Merge EPS
        if 'Reported EPS' in df_eps.columns:
            df_fcst_input = pd.merge(df_fcst_input, df_eps[['Date', 'Reported EPS']], how='left', on="Date")
        else:
            df_fcst_input['Reported EPS'] = np.nan

        # 4. Get VIX Data
        # We fetch the extended period for VIX too, then merge will handle the filtering automatically
        var_ticker_class_vix = yf.Ticker("^VIX")
        df_vix = var_ticker_class_vix.history(period=f'{fetch_months}mo')
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
        
        # Final Prep for Prophet
        df_prophet = df_fcst_input.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Fill NaNs for regressors
        if 'Dividends' in df_prophet.columns: df_prophet['Dividends'] = df_prophet['Dividends'].fillna(0)
        if 'Reported EPS' in df_prophet.columns: df_prophet['Reported EPS'] = df_prophet['Reported EPS'].fillna(0)
        
        # Fill Forward/Back for continuous variables
        cols_to_fill = ['Volatility Index Close', 'Volume', 'Moving Average 50 Days', 'Moving Average 100 Days', 'Moving Average 200 Days']
        for col in cols_to_fill:
            if col in df_prophet.columns:
                df_prophet[col] = df_prophet[col].ffill().bfill()
        
        return df_prophet, metadata, None
        
    except Exception as e:
        return None, None, str(e)

# --- PROPHET TRAINING FUNCTION ---
def run_prophet_competition(df, history_months):
    """
    Runs the regressor competition loop.
    """
    
    potential_regressors = [
        'Volume'
        , 'Reported EPS'
        , 'Dividends'
        # , 'Volatility Index Close'
        , 'Moving Average 50 Days'
        # , 'Moving Average 100 Days'
        , 'Moving Average 200 Days'
    ]
    
    # Filter only columns that actually exist in the dataframe
    available_regressors = [r for r in potential_regressors if r in df.columns]
    
    best_rmse = float('inf')
    best_model = None
    best_regressor_combo = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total iterations for progress bar
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
            
            # Dynamic Cross Validation Params based on history length
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

# --- LSTM TRAINING FUNCTION ---
def run_lstm_model(df, forecast_months):
    """
    Runs a Univariate LSTM on Log Returns to ensure stationarity.
    Fixes 'straight line' forecasts by predicting volatility instead of price levels.
    """
    
    # 1. Preprocessing: Convert Price to Log Returns
    # This makes the data stationary (oscillating around 0) rather than trending
    df['log_ret'] = np.log(df['y'] / df['y'].shift(1))
    
    # Drop the NaN created by the shift
    df_model = df.dropna().copy()
    
    # We use ONLY Log Returns. 
    # (Removing Moving Averages prevents the 'smoothing' effect that causes flat lines)
    data = df_model['log_ret'].values.reshape(-1, 1)
    
    # 2. Scale Data
    # MinMax on returns usually lands between 0.4 and 0.6, keeping gradients stable
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Create Sequences
    look_back = 60
    X, y = [], []
    
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 4. Build Model with Gradient Clipping
    model = Sequential()
    # return_sequences=True feeds the hidden state to the next layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # OPTIMIZER FIX: clipvalue=1.0 prevents the "Exploding Gradient" numerical error
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train
    model.fit(X, y, batch_size=32, epochs=25, verbose=0)
    
    # 5. Forecasting Loop
    future_days = forecast_months * 30
    
    # Start sequence with the last 'look_back' days of known returns
    curr_sequence = scaled_data[-look_back:]
    curr_sequence = curr_sequence.reshape(1, look_back, 1)
    
    future_log_rets = []
    
    # We need the last actual price to reconstruct the chain later
    last_actual_price = df['y'].iloc[-1]
    
    for _ in range(future_days):
        # Predict next Step's Return
        next_pred_scaled = model.predict(curr_sequence, verbose=0)
        
        # Store prediction
        future_log_rets.append(next_pred_scaled[0, 0])
        
        # Update Sequence: Remove oldest return, add new predicted return
        # This recursive structure allows the model to generate its own volatility
        curr_sequence = np.append(curr_sequence[:, 1:, :], [next_pred_scaled], axis=1)
        
    # 6. Reconstruct Price Path
    # Inverse scale the predictions to get "Real" Log Returns
    future_log_rets = np.array(future_log_rets).reshape(-1, 1)
    future_log_rets_unscaled = scaler.inverse_transform(future_log_rets)
    
    # Convert Log Returns back to Price: Price_t = Price_{t-1} * e^(return)
    # Cumulative sum of log returns = Total Log Growth
    future_cumulative_growth = np.exp(np.cumsum(future_log_rets_unscaled))
    
    # Apply growth to the last known price
    future_prices = last_actual_price * future_cumulative_growth
    
    # 7. Create Future DataFrame
    last_date = pd.to_datetime(df['ds'].max())
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, future_days + 1)]
    
    # Dynamic Uncertainty Intervals
    # We calculate volatility from the model's own history
    hist_volatility = df_model['log_ret'].std()
    # Cone expands over time (sqrt of time rule for random walks)
    uncertainty_cone = np.array([hist_volatility * last_actual_price * np.sqrt(t) for t in range(1, future_days + 1)])
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_prices.flatten(),
        'yhat_upper': future_prices.flatten() + (uncertainty_cone * 1.96), # 95% Confidence rough proxy
        'yhat_lower': future_prices.flatten() - (uncertainty_cone * 1.96)
    })
    
    return forecast_df# --- MAIN APP LOGIC ---

if run_button:
    if algo_choice == "ARIMA (Coming Soon)":
        st.warning(f"{algo_choice} is not yet implemented.")
        st.stop()

    with st.spinner('Downloading Data and Preprocessing...'):
        # Unpack the 3 return values
        df_data, meta_data, error = get_stock_data(var_ticker_input, var_past_horizon_mo)

    if error:
        st.error(f"Error: {error}")
    else:
        # --- DISPLAY COMPANY INFO ---
        with company_info_placeholder.container():
            st.markdown(f"## {meta_data['longName']}")
            with st.expander("Show Business Summary", expanded=False):
                st.write(meta_data['longBusinessSummary'])
            st.divider()

        # Run Competition / Model
        st.subheader("Model Optimization")
        
        # Check if we have enough data for the requested horizon
        if len(df_data) < 180:
            st.warning("Data history is very short. Forecast quality may be low.")
            
        forecast_results = None
        
        # --- BRANCHING LOGIC ---
        if algo_choice == "Facebook Prophet":
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
                        
                # Make Prophet Forecast
                future = best_model.make_future_dataframe(periods=var_future_fcst_mo*30) 
                for reg in best_combo:
                    last_known_value = df_data[reg].iloc[-1]
                    future[reg] = df_data[reg] # Fill historical
                    future[reg] = future[reg].fillna(last_known_value) # Fill future

                full_forecast = best_model.predict(future)
                # Filter only for future part for plotting consistency with LSTM logic
                forecast_results = full_forecast[full_forecast['ds'] > pd.Timestamp(df_data['ds'].max())]

        elif algo_choice == "LSTM":
            # Run LSTM
            with st.spinner("Training Multivariate LSTM Neural Network"):
                forecast_results = run_lstm_model(df_data, var_future_fcst_mo)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.success("LSTM Network Trained!")
                
                # Formal, objective explanation
                st.write(
                    "Long Short-Term Memory (LSTM) is a recurrent neural network architecture specifically "
                    "engineered for time-series forecasting. By employing specialized gating mechanisms to "
                    "regulate information flow, the model is able to distinguish between significant long-term "
                    "trends and short-term fluctuations, allowing it to effectively retain relevant historical "
                    "dependencies throughout the sequence."
                )

        # --- SHARED PLOTTING LOGIC ---
        if forecast_results is not None:
            with col2:
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
                    x=forecast_results['ds'],
                    y=forecast_results['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orangered')
                ))

                # Uncertainty Intervals (Upper/Lower bounds)
                fig.add_trace(go.Scatter(
                    x=forecast_results['ds'],
                    y=forecast_results['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_results['ds'],
                    y=forecast_results['yhat_lower'],
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
