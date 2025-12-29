import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import combinations
import plotly.graph_objects as go
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Forecast Pro", layout="wide")

st.title("📈 Stock Forecasting Engine")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Configuration")
    
    var_ticker_input = st.text_input("Stock Ticker", value="AAPL").upper()
    var_past_horizon_mo = st.number_input("History Lookback (Months)", min_value=12, value=48, step=6)
    var_future_fcst_mo = st.number_input("Future Forecast (Months)", min_value=1, value=2, step=1)
    
    # Placeholder for future algorithms
    algo_choice = st.selectbox(
        "Forecasting Algorithm", 
        ("Facebook Prophet", "ARIMA (Coming Soon)", "LSTM (Coming Soon)")
    )
    
    run_button = st.button("Run Forecast", type="primary")

# --- DATA LOADING FUNCTION ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker, months):
    """
    Fetches Stock, VIX, Dividends, and Earnings data based on user input.
    """
    try:
        var_ticker_class = yf.Ticker(ticker)
        
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
            return None, "No price data found for ticker."

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
        df_vix = df_vix.add_prefix('VIX_')

        # Merge VIX
        df_fcst_input = pd.merge(
            df_fcst_input,
            df_vix,
            how='left',
            left_on="Date",
            right_on="VIX_Date"
        )
        
        # Final Prep for Prophet
        df_prophet = df_fcst_input.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Fill NaNs for regressors (Prophet crashes with NaNs in regressors)
        # We fill with 0 for Dividends/EPS (event based) and ffill for VIX/Volume
        if 'Dividends' in df_prophet.columns: df_prophet['Dividends'] = df_prophet['Dividends'].fillna(0)
        if 'Reported EPS' in df_prophet.columns: df_prophet['Reported EPS'] = df_prophet['Reported EPS'].fillna(0)
        if 'VIX_Close' in df_prophet.columns: df_prophet['VIX_Close'] = df_prophet['VIX_Close'].ffill().bfill()
        if 'Volume' in df_prophet.columns: df_prophet['Volume'] = df_prophet['Volume'].ffill().bfill()
        
        return df_prophet, None
        
    except Exception as e:
        return None, str(e)

# --- MODEL TRAINING FUNCTION ---
def run_prophet_competition(df, history_months):
    """
    Runs the regressor competition loop.
    """
    potential_regressors = ['Volume', 'Reported EPS', 'Dividends', 'VIX_Close']
    
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

# --- MAIN APP LOGIC ---

if run_button:
    if algo_choice != "Facebook Prophet":
        st.warning(f"{algo_choice} is not yet implemented. Using Prophet logic as placeholder.")

    with st.spinner('Downloading Data and Preprocessing...'):
        df_data, error = get_stock_data(var_ticker_input, var_past_horizon_mo)

    if error:
        st.error(f"Error: {error}")
    else:
        # Run Competition
        st.subheader("Model Optimization")
        
        # Check if we have enough data for the requested horizon
        if len(df_data) < 180:
            st.warning("Data history is very short. Forecast quality may be low.")
            
        best_model, best_combo, best_rmse, results_df = run_prophet_competition(df_data, var_past_horizon_mo)

        if best_model is None:
            st.error("Could not find a valid model. Try increasing the history lookback period.")
        else:
            # --- RESULTS SECTION ---
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.success("Optimization Complete!")
                st.metric(label="Lowest RMSE (Error)", value=f"{best_rmse:.4f}")
                st.write("**Best Regressors for this stock:**")
                for feature in best_combo:
                    st.code(feature)

            # --- FORECASTING FUTURE ---
            future = best_model.make_future_dataframe(periods=var_future_fcst_mo*30) ## Bring in forecasted 
            
            # Forward fill future regressors (Naive approach as per original code)
            for reg in best_combo:
                last_known_value = df_data[reg].iloc[-1]
                future[reg] = df_data[reg] # Fill historical
                future[reg] = future[reg].fillna(last_known_value) # Fill future

            forecast = best_model.predict(future)

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
                # Filter forecast to only show the future part + slightly overlapping for visual continuity
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
                    margin=dict(l=0, r=0, t=0, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)
