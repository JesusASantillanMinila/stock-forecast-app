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

# --- NEW IMPORTS FOR LSTM ---
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

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
    
    # Updated Algo Choice
    algo_choice = st.selectbox(
        "Forecasting Algorithm", 
        ("Facebook Prophet", "LSTM (Deep Learning)", "ARIMA (Coming Soon)")
    )
    
    run_button = st.button("Run Forecast", type="primary")

# --- DATA LOADING FUNCTION ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker, months):
    """
    Fetches Stock, VIX, Dividends, Earnings data AND Company Metadata.
    """
    try:
        time.sleep(1)
        var_ticker_class = yf.Ticker(ticker)
        
        # 0. Fetch Company Metadata (Name & Summary)
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
                "longBusinessSummary": str(e)
            }
        
        # 1. Download Stock Price
        df_stock_price = yf.download(ticker, period=f'{months}mo', progress=False)
        
        # Handle yfinance multi-index columns if present
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

        # Merge EPS
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
        
        # Final Prep for Prophet
        df_prophet = df_fcst_input.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Fill NaNs for regressors
        if 'Dividends' in df_prophet.columns: df_prophet['Dividends'] = df_prophet['Dividends'].fillna(0)
        if 'Reported EPS' in df_prophet.columns: df_prophet['Reported EPS'] = df_prophet['Reported EPS'].fillna(0)
        if 'Volatility Index Close' in df_prophet.columns: df_prophet['Volatility Index Close'] = df_prophet['Volatility Index Close'].ffill().bfill()
        if 'Volume' in df_prophet.columns: df_prophet['Volume'] = df_prophet['Volume'].ffill().bfill()
        
        return df_prophet, metadata, None
        
    except Exception as e:
        return None, None, str(e)

# --- PROPHET TRAINING FUNCTION ---
def run_prophet_competition(df, history_months):
    """
    Runs the regressor competition loop.
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

# --- LSTM TRAINING FUNCTION (NEW) ---
def run_lstm_forecast(df, future_months):
    """
    Runs a Univariate LSTM model with gradient clipping.
    Returns a dataframe formatted exactly like Prophet's 'forecast' df.
    """
    # 1. Prepare Data
    data = df[['ds', 'y']].sort_values('ds')
    dataset = data['y'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Lookback window (steps in the past to look at)
    look_back = 60 
    
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # 2. Build LSTM Model
    # Designed to be lightweight for Free Tier
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    
    # --- CRITICAL: GRADIENT EXPLOSION CONTROL ---
    # Using clipnorm=1.0 to ensure gradients do not exceed 1.0
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train (Verbose=0 to keep logs clean)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # 3. Forecast Future
    # Start with the last 'look_back' days from the data
    current_batch = scaled_data[-look_back:].reshape((1, look_back, 1))
    predicted_prices = []
    
    future_days = future_months * 30
    
    for i in range(future_days):
        # Predict next step
        next_pred = model.predict(current_batch, verbose=0)[0]
        predicted_prices.append(next_pred)
        
        # Update batch: remove first item, add new prediction
        next_pred_reshaped = next_pred.reshape((1, 1, 1))
        current_batch = np.append(current_batch[:, 1:, :], next_pred_reshaped, axis=1)
        
    # Inverse transform predictions
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    
    # 4. Estimate Uncertainty (Mocking Prophet's intervals)
    # We calculate RMSE on the training set to create a simple confidence band
    train_predict = model.predict(X_train, verbose=0)
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    
    rmse = np.sqrt(np.mean(((train_predict - y_train_inv.T) ** 2)))
    
    # 5. Construct DataFrame matching Prophet format
    last_date = data['ds'].max()
    future_dates = [last_date + datetime.timedelta(days=x) for x in range(1, future_days + 1)]
    
    # Create the future dataframe
    df_future = pd.DataFrame({
        'ds': pd.to_datetime(future_dates),
        'yhat': predicted_prices.flatten()
    })
    
    # Add simple uncertainty intervals (approx 95% CI assuming normal errors)
    df_future['yhat_lower'] = df_future['yhat'] - (1.96 * rmse)
    df_future['yhat_upper'] = df_future['yhat'] + (1.96 * rmse)
    
    # We also need to attach historical data to this dataframe to match Prophet's 'forecast' object
    # Prophet returns the Whole history + future.
    df_history = data.copy()
    df_history['yhat'] = df_history['y'] # For history, prediction is actual (simplification for viz)
    df_history['yhat_lower'] = df_history['y']
    df_history['yhat_upper'] = df_history['y']
    
    # Concatenate
    df_final = pd.concat([df_history, df_future], ignore_index=True)
    df_final['ds'] = pd.to_datetime(df_final['ds'])
    
    return df_final, rmse

# --- MAIN APP LOGIC ---

if run_button:
    if algo_choice == "ARIMA (Coming Soon)":
         st.warning("ARIMA is not yet implemented.")
         st.stop()

    with st.spinner('Downloading Data and Preprocessing...'):
        # Unpack the 3 return values
        df_data, meta_data, error = get_stock_data(var_ticker_input, var_past_horizon_mo)

    if error:
        st.error(f"Error: {error}")
    else:
        # --- DISPLAY COMPANY INFO ---
        # We populate the placeholder we created at the top
        with company_info_placeholder.container():
            st.markdown(f"## {meta_data['longName']}")
            with st.expander("Show Business Summary", expanded=False):
                st.write(meta_data['longBusinessSummary'])
            st.divider()

        # Check if we have enough data for the requested horizon
        if len(df_data) < 180:
            st.warning("Data history is very short. Forecast quality may be low.")
            
        forecast = None
        best_combo = []

        # --- BRANCH: FACEBOOK PROPHET ---
        if algo_choice == "Facebook Prophet":
            st.subheader("Model Optimization (Prophet)")
            best_model, best_combo, best_rmse, results_df = run_prophet_competition(df_data, var_past_horizon_mo)

            if best_model is None:
                st.error("Could not find a valid model. Try increasing the history lookback period.")
            else:
                 # Success Message
                st.success(f"Optimization Complete! Best RMSE: {best_rmse:.2f}")
                st.write("**Best Regressors for this stock:**")
                st.write(", ".join(best_combo) if best_combo else "None (Univariate)")

                # FORECASTING FUTURE
                future = best_model.make_future_dataframe(periods=var_future_fcst_mo*30) 
                
                # Forward fill future regressors 
                for reg in best_combo:
                    last_known_value = df_data[reg].iloc[-1]
                    future[reg] = df_data[reg] # Fill historical
                    future[reg] = future[reg].fillna(last_known_value) # Fill future

                forecast = best_model.predict(future)

        # --- BRANCH: LSTM ---
        elif algo_choice == "LSTM (Deep Learning)":
            st.subheader("Model Training (LSTM)")
            status_text = st.empty()
            status_text.text("Initializing TensorFlow & scaling data...")
            progress_bar = st.progress(0)
            
            # Run LSTM
            try:
                progress_bar.progress(30)
                status_text.text("Training LSTM Network (this may take a moment)...")
                
                forecast, train_rmse = run_lstm_forecast(df_data, var_future_fcst_mo)
                
                progress_bar.progress(100)
                status_text.text("Training Complete.")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"LSTM Training Complete! Training RMSE: {train_rmse:.2f}")
                st.info("Note: Uncertainty intervals for LSTM are approximated using training error.")
                best_combo = ["LSTM (Univariate)", "Gradient Clipping: True"]
                
            except Exception as e:
                st.error(f"LSTM Training Failed: {e}")

        # --- RESULTS SECTION (Unified for both models) ---
        if forecast is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # If it's prophet, we already showed the regressors. 
                # If LSTM, we show model details.
                if algo_choice == "LSTM (Deep Learning)":
                    st.write("**Model Parameters:**")
                    for feat in best_combo:
                        st.code(feat)

            # --- INTERACTIVE PLOTLY CHART ---
            with col2:
                st.subheader(f"Forecast: {var_ticker_input}")
                
                fig = go.Figure()

                # Historical Data (Actuals) - We use df_data for the pure actuals
                fig.add_trace(go.Scatter(
                    x=df_data['ds'], 
                    y=df_data['y'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='deepskyblue')
                ))

                # Forecast Data
                # Filter forecast to only show the future part + slightly overlapping 
                future_forecast = forecast[forecast['ds'] > pd.Timestamp(df_data['ds'].max())]
                
                fig.add_trace(go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orangered')
                ))

                # Uncertainty Intervals (Upper/Lower bounds)
                fig.add_trace(go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat_lower'],
                    mode='lines',
                    fill='tonexty', # Fill area between upper and lower
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
