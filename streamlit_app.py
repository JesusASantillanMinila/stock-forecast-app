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
    
    # Updated Algo Choice
    algo_choice = st.selectbox(
        "Forecasting Algorithm", 
        ("Facebook Prophet", "LSTM (Deep Learning)")
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
        # We use a try/except block specifically for metadata so it doesn't fail the whole forecast if missing
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
                "longBusinessSummary": f"Company summary could not be retrieved. ({str(e)})"
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
        # Rename column to ensure consistency
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
        
        # Fill NaNs for regressors (Prophet crashes with NaNs in regressors)
        # We fill with 0 for Dividends/EPS (event based) and ffill for VIX/Volume
        if 'Dividends' in df_prophet.columns: df_prophet['Dividends'] = df_prophet['Dividends'].fillna(0)
        if 'Reported EPS' in df_prophet.columns: df_prophet['Reported EPS'] = df_prophet['Reported EPS'].fillna(0)
        if 'Volatility Index Close' in df_prophet.columns: df_prophet['Volatility Index Close'] = df_prophet['Volatility Index Close'].ffill().bfill()
        if 'Volume' in df_prophet.columns: df_prophet['Volume'] = df_prophet['Volume'].ffill().bfill()
        
        return df_prophet, metadata, None
        
    except Exception as e:
        return None, None, str(e)

# --- PROPHET MODEL TRAINING FUNCTION ---
def run_prophet_competition(df, history_months):
    """
    Runs the regressor competition loop.
    """
    potential_regressors = ['Volume', 'Reported EPS', 'Dividends', 'Volatility Index Close']
    
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
            # If history is short, we reduce the initial training period
            days_history = history_months * 30
            initial_days = f"{int(days_history * 0.5)} days" # Use 50% for training
            
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
                # Often fails if data is too short for CV
                pass
            
            current_iter += 1
            progress_bar.progress(current_iter / total_combos)

    progress_bar.empty()
    status_text.empty()
    
    return best_model, best_regressor_combo, best_rmse, pd.DataFrame(results_log)

# --- NEW LSTM TRAINING FUNCTION ---
def run_lstm_forecast(df, months_forecast):
    """
    Runs LSTM model with available regressors.
    """
    # 1. Setup Data & Regressors
    potential_regressors = ['Volume', 'Reported EPS', 'Dividends', 'Volatility Index Close']
    available_regressors = [r for r in potential_regressors if r in df.columns]
    
    # We use all available regressors for LSTM to maximize information
    feature_cols = ['y'] + available_regressors
    data = df[feature_cols].values
    
    # 2. Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 3. Create Sequences
    look_back = 60 # Days to look back
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0]) # Predict 'y' (index 0)
        
    X, y = np.array(X), np.array(y)
    
    # 4. Build Model
    # Simple architecture to ensure it runs on Streamlit Free Tier (RAM constraints)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Progress indication
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Training Neural Network (LSTM)...")
    
    # Train
    model.fit(X, y, batch_size=32, epochs=15, verbose=0)
    progress_bar.progress(100)
    
    # Calculate Training RMSE for "Uncertainty" bands estimation
    train_predict = model.predict(X)
    # Inverse transform is tricky because scaler expects all features. 
    # We create a dummy array to inverse transform just the price column.
    def inverse_price(pred_arr):
        # Create a matrix of zeros with shape (len(pred), n_features)
        dummy = np.zeros((len(pred_arr), len(feature_cols)))
        # Put predictions in the first column (where 'y' is)
        dummy[:, 0] = pred_arr.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    y_true_inv = inverse_price(y.reshape(-1, 1))
    y_pred_inv = inverse_price(train_predict)
    rmse = np.sqrt(np.mean((y_true_inv - y_pred_inv) ** 2))
    
    # 5. Future Forecasting (Recursive)
    future_days = months_forecast * 30
    
    # Start with the last 'look_back' days of data
    curr_seq = scaled_data[-look_back:] 
    future_predictions = []
    future_dates = []
    
    last_date = df['ds'].iloc[-1]
    
    status_text.text("Generating Future Forecast...")
    
    # Get last known values for regressors (Naive approach: forward fill)
    last_known_regressors = scaled_data[-1, 1:] # All columns except 'y'
    
    for i in range(future_days):
        # Predict next price (scaled)
        # Reshape input to (1, look_back, features)
        curr_seq_reshaped = curr_seq.reshape(1, look_back, len(feature_cols))
        next_pred_scaled = model.predict(curr_seq_reshaped, verbose=0)[0, 0]
        
        # Create next input row: [Predicted Price, Regressor1, Regressor2...]
        # We assume regressors stay constant (ffill) logic similar to the Prophet implementation
        next_row = np.hstack(([next_pred_scaled], last_known_regressors))
        
        # Update sequence: remove first day, add new day
        curr_seq = np.vstack((curr_seq[1:], next_row))
        
        # Store result
        future_predictions.append(next_pred_scaled)
        
        # Increment Date
        last_date += datetime.timedelta(days=1)
        # Skip weekends (simple logic)
        while last_date.weekday() > 4:
            last_date += datetime.timedelta(days=1)
        future_dates.append(last_date)
        
    # Inverse Transform Future Predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_prices = inverse_price(future_predictions)
    
    # 6. Format Output for Plotting (Mimic Prophet Structure)
    df_future = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_prices,
        # Create uncertainty bands using training RMSE
        'yhat_lower': future_prices - rmse,
        'yhat_upper': future_prices + rmse
    })
    
    # We must properly format 'ds' to datetime
    df_future['ds'] = pd.to_datetime(df_future['ds'])
    
    # Combine history with future for the plot (Prophet `make_future_dataframe` style logic)
    # The plotting code expects the full dataframe + prediction
    # So we return the dataframe structure that matches Prophet's output
    
    status_text.empty()
    progress_bar.empty()
    
    return model, available_regressors, rmse, df_future

# --- MAIN APP LOGIC ---

if run_button:
    if algo_choice == "ARIMA (Coming Soon)":
        st.warning("ARIMA is not yet implemented.")

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

        # Run Selected Model
        st.subheader("Model Optimization")
        
        # Check if we have enough data for the requested horizon
        if len(df_data) < 180:
            st.warning("Data history is very short. Forecast quality may be low.")
            
        future_forecast = pd.DataFrame() # Initialize
        
        if algo_choice == "Facebook Prophet":
            best_model, best_combo, best_rmse, results_df = run_prophet_competition(df_data, var_past_horizon_mo)
            
            if best_model is None:
                st.error("Could not find a valid model.")
            else:
                 # Generate Future Dataframe
                future = best_model.make_future_dataframe(periods=var_future_fcst_mo*30) 
                # Forward fill future regressors
                for reg in best_combo:
                    last_known_value = df_data[reg].iloc[-1]
                    future[reg] = df_data[reg] # Fill historical
                    future[reg] = future[reg].fillna(last_known_value) # Fill future

                forecast = best_model.predict(future)
                future_forecast = forecast # Standard Prophet Output
                
        elif algo_choice == "LSTM (Deep Learning)":
            best_model, best_combo, best_rmse, forecast_df = run_lstm_forecast(df_data, var_future_fcst_mo)
            
            # For plotting consistency, we append the historical data to the forecast dataframe
            # But the plotting logic below filters `future_forecast` by date > max(history)
            # So we just need to ensure `future_forecast` contains the future rows with yhat/ds
            future_forecast = forecast_df

        # --- RESULTS SECTION ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success("Optimization Complete!")
            
            st.write(f"**Model:** {algo_choice}")
            st.write(f"**Training RMSE:** {best_rmse:.4f}")
            
            st.write("**Features Used:**")
            for feature in best_combo:
                st.code(feature)

        # --- INTERACTIVE PLOTLY CHART ---
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
            # Filter forecast to only show the future part
            # Note: For LSTM, future_forecast already only contains future data, 
            # but for Prophet it contains history+future. The filter below handles both cases safely.
            plot_forecast = future_forecast[future_forecast['ds'] > pd.Timestamp(df_data['ds'].max())]
            
            fig.add_trace(go.Scatter(
                x=plot_forecast['ds'],
                y=plot_forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='orangered')
            ))

            # Uncertainty Intervals (Upper/Lower bounds)
            fig.add_trace(go.Scatter(
                x=plot_forecast['ds'],
                y=plot_forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=plot_forecast['ds'],
                y=plot_forecast['yhat_lower'],
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
