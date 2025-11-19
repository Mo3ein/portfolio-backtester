import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Pro Portfolio Backtester", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stPlotlyChart {
        background-color: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #333;
    }
    h1, h2, h3 {
        color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Pro Portfolio Backtester")
st.markdown("Advanced analytics, Monte Carlo simulations, Walk Forward Analysis, and Optimization.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
    
    st.subheader("Risk Management")
    commission_rate = st.number_input("Commission Rate (%)", value=0.06, step=0.01, help="Per trade side")
    slippage_rate = st.number_input("Slippage (%)", value=0.02, step=0.01)
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=2.0, step=0.1, help="For Sharpe Ratio") / 100.0
    
    st.divider()
    st.info("Upload strategy CSVs to unlock analytics.")

# --- Data Loading ---
uploaded_files = st.file_uploader("Upload Backtest CSVs", type=['csv'], accept_multiple_files=True)

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        # Flexible column handling
        df.columns = [c.strip() for c in df.columns]
        
        if 'time' not in df.columns:
            # Try to detect other time columns
            if 'Date' in df.columns:
                df['time'] = pd.to_datetime(df['Date']).astype(int) / 10**9
            else:
                return None, f"Missing 'time' column in {file.name}"
            
        # Convert time
        # Handle if time is already datetime string or int
        if df['time'].dtype == object:
             df['datetime'] = pd.to_datetime(df['time'])
        else:
             # Assume timestamp
             df['datetime'] = pd.to_datetime(df['time'], unit='s')
             
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df, None
    except Exception as e:
        return None, str(e)

def calculate_metrics(daily_returns, start_cap=10000, trades_df=None):
    if len(daily_returns) < 2:
        return {}
    
    # Geometric mean return
    total_ret = (1 + daily_returns).prod() - 1
    # Handle division by zero for short periods
    years = len(daily_returns) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility if volatility != 0 else 0
    
    downside = daily_returns[daily_returns < 0]
    sortino = (cagr - risk_free_rate) / (downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() != 0 else 0
    
    # Max Drawdown
    cum_ret = (1 + daily_returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    metrics = {
        "Total Return": total_ret,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd != 0 else 0
    }
    
    if trades_df is not None and not trades_df.empty:
        wins = trades_df[trades_df['Net PnL'] > 0]
        losses = trades_df[trades_df['Net PnL'] <= 0]
        
        metrics['Win Rate'] = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        metrics['Profit Factor'] = abs(wins['Net PnL'].sum() / losses['Net PnL'].sum()) if losses['Net PnL'].sum() != 0 else 0
        metrics['Trades'] = len(trades_df)
    
    return metrics

if uploaded_files:
    data_objects = {}
    valid_files = []
    
    for f in uploaded_files:
        df, err = load_data(f)
        if df is not None:
            name = f.name.split(',')[0].replace('BINANCE_', '').replace('COINBASE_', '').replace('KRAKEN_', '')
            data_objects[name] = df
            valid_files.append(name)
        else:
            st.error(f"Error loading {f.name}: {err}")

    if data_objects:
        # --- Tab Structure ---
        tabs = st.tabs([
            "📊 Dashboard", 
            "🔄 Walk Forward",
            "🎲 Monte Carlo", 
            "🔍 Deep Dive", 
            "🧠 Optimizer", 
            "📝 Raw Data"
        ])
        
        # Pre-process Data
        all_daily_returns = pd.DataFrame()
        all_trades_list = []
        buy_hold_curves = pd.DataFrame()
        
        for name, df in data_objects.items():
            # 1. Extract Trades
            if 'Trade PnL %' in df.columns:
                trades = df[df['Trade PnL %'].notna()].copy()
                trades['Net PnL %'] = trades['Trade PnL %'] - commission_rate - slippage_rate
                trades['Strategy'] = name
                all_trades_list.append(trades)
                
                # Daily Returns
                daily_pnl = trades.resample('D')['Net PnL %'].sum() / 100.0
                all_daily_returns[name] = daily_pnl
            
            # 2. Buy & Hold
            if 'close' in df.columns:
                price = df['close'].resample('D').last().ffill()
                bh_ret = price.pct_change().fillna(0)
                buy_hold_curves[name] = (1 + bh_ret).cumprod()
                
        # Fill NaN with 0
        all_daily_returns.fillna(0, inplace=True)
        
        # --- SIDEBAR WEIGHTS ---
        with st.sidebar:
            st.subheader("Portfolio Weights")
            weights = {}
            if len(valid_files) > 0:
                total_w = 0
                default_w = 1.0 / len(valid_files)
                for name in valid_files:
                    w = st.slider(f"{name}", 0.0, 1.0, default_w, 0.05)
                    weights[name] = w
                    total_w += w
                
                # Normalize
                if total_w > 0:
                    for k in weights:
                        weights[k] /= total_w

        # Calculate Static Portfolio
        if not all_daily_returns.empty:
            port_daily_ret = all_daily_returns.mul(pd.Series(weights)).sum(axis=1)
            port_equity = initial_capital * (1 + port_daily_ret).cumprod()
            
            # Full Trades DF
            if all_trades_list:
                full_trades_df = pd.concat(all_trades_list)
                full_trades_df['Net PnL'] = initial_capital * (full_trades_df['Net PnL %']/100.0)
            else:
                full_trades_df = pd.DataFrame()

            # --- TAB 1: DASHBOARD ---
            with tabs[0]:
                metrics = calculate_metrics(port_daily_ret, initial_capital, full_trades_df)
                
                # KPI Row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Return", f"{metrics.get('Total Return', 0)*100:.1f}%", f"${port_equity.iloc[-1] - initial_capital:,.0f}")
                c2.metric("Sharpe Ratio", f"{metrics.get('Sharpe', 0):.2f}")
                c3.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0)*100:.1f}%")
                c4.metric("CAGR", f"{metrics.get('CAGR', 0)*100:.1f}%")
                
                # Main Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=port_equity.index, y=port_equity, name='Portfolio', line=dict(color='#00FF00', width=2)))
                
                # Buy & Hold Benchmark
                if not buy_hold_curves.empty and len(weights) > 0:
                    bh_aligned = buy_hold_curves.reindex(port_equity.index).ffill()
                    # Weighted sum of normalized curves * initial capital
                    bh_port = bh_aligned.mul(pd.Series(weights)).sum(axis=1) * initial_capital
                    fig.add_trace(go.Scatter(x=bh_port.index, y=bh_port, name='Buy & Hold', line=dict(color='orange', dash='dash')))
                
                fig.update_layout(title="Equity Curve", height=500, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly Table
                st.subheader("📅 Monthly Performance")
                monthly_ret = (1 + port_daily_ret).resample('M').prod() - 1
                if not monthly_ret.empty:
                    m_df = pd.DataFrame({'Return': monthly_ret.values}, index=monthly_ret.index)
                    m_df['Year'] = m_df.index.year
                    m_df['Month'] = m_df.index.strftime('%b')
                    
                    m_pivot = m_df.pivot(index='Year', columns='Month', values='Return')
                    # Reorder months
                    months_ord = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    m_pivot = m_pivot.reindex(columns=[m for m in months_ord if m in m_pivot.columns])
                    
                    # Color map
                    st.dataframe(m_pivot.style.format("{:.2%}")
                                 .background_gradient(cmap='RdYlGn', vmin=-0.1, vmax=0.1), 
                                 use_container_width=True)

            # --- TAB 2: WALK FORWARD ---
            with tabs[1]:
                st.subheader("🔄 Walk Forward Optimization (Rolling Portfolio)")
                st.markdown("""
                Simulate a rebalancing strategy where you periodically select the best performing strategies.
                """)
                
                col_wf1, col_wf2 = st.columns(2)
                with col_wf1:
                    lookback_months = st.slider("Lookback Period (Months)", 1, 12, 3)
                with col_wf2:
                    rebalance_freq = st.selectbox("Rebalance Frequency", ["Monthly", "Quarterly"])
                
                if st.button("Run Walk Forward Analysis"):
                    # Resample returns to monthly
                    monthly_rets = (1 + all_daily_returns).resample('M').prod() - 1
                    
                    # Simulation
                    wf_equity = [initial_capital]
                    wf_dates = [monthly_rets.index[0]]
                    current_capital = initial_capital
                    
                    # Weights history
                    weight_history = []
                    
                    # Start after lookback
                    freq_step = 1 if rebalance_freq == "Monthly" else 3
                    
                    for i in range(lookback_months, len(monthly_rets), freq_step):
                        # Lookback window
                        window = monthly_rets.iloc[i-lookback_months:i]
                        
                        # Selection Logic: Rank by Sharpe (simplified as Mean/Std)
                        # Avoid div by zero
                        stds = window.std()
                        means = window.mean()
                        sharpes = means / stds.replace(0, np.inf)
                        
                        # Select Top 1 or Weighted? Let's do Inverse Volatility or Top 1
                        # Let's pick Best Performer
                        if not sharpes.empty and sharpes.max() > -np.inf:
                            best_strat = sharpes.idxmax()
                        else:
                            # Fallback if no data or all NaN
                             best_strat = monthly_rets.columns[0] 
                        
                        # Determine Return for next period (i to i+freq_step)
                        next_period = monthly_rets.iloc[i:min(i+freq_step, len(monthly_rets))]
                        
                        if not next_period.empty and best_strat in next_period.columns:
                            # Apply return of selected strategy
                            # (Using the best_strat column)
                            period_ret = (1 + next_period[best_strat]).prod() - 1
                            current_capital *= (1 + period_ret)
                            
                            wf_equity.append(current_capital)
                            wf_dates.append(next_period.index[-1])
                            
                            weight_history.append({'Date': next_period.index[0], 'Strategy': best_strat})
                    
                    # Plot WFA vs Static
                    wf_series = pd.Series(wf_equity, index=wf_dates)
                    
                    fig_wfa = go.Figure()
                    fig_wfa.add_trace(go.Scatter(x=wf_series.index, y=wf_series, name='Walk Forward (Momentum)', line=dict(color='cyan')))
                    fig_wfa.add_trace(go.Scatter(x=port_equity.index, y=port_equity, name='Static Allocation', line=dict(color='grey', dash='dot')))
                    
                    fig_wfa.update_layout(title="Walk Forward Performance", template="plotly_dark")
                    st.plotly_chart(fig_wfa, use_container_width=True)
                    
                    # Show Allocations
                    st.subheader("Rebalancing Log")
                    st.dataframe(pd.DataFrame(weight_history), use_container_width=True)

            # --- TAB 3: MONTE CARLO ---
            with tabs[2]:
                st.subheader("🎲 Detailed Monte Carlo Simulation")
                
                mc_col1, mc_col2 = st.columns(2)
                with mc_col1:
                    mc_runs = st.number_input("Simulations", 100, 5000, 1000)
                with mc_col2:
                    mc_days = st.number_input("Forecast Days", 30, 756, 252)
                
                mu = port_daily_ret.mean()
                sigma = port_daily_ret.std()
                
                if sigma > 0:
                    # Simulation
                    daily_sims = np.random.normal(mu, sigma, (mc_days, mc_runs))
                    cum_sims = (1 + daily_sims).cumprod(axis=0) * port_equity.iloc[-1]
                    
                    # Stats
                    final_values = cum_sims[-1, :]
                    var_95 = np.percentile(final_values, 5)
                    cvar_95 = final_values[final_values <= var_95].mean()
                    
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Median Forecast", f"${np.median(final_values):,.0f}")
                    m2.metric("95% VaR (Year End)", f"${var_95:,.0f}")
                    m3.metric("95% CVaR", f"${cvar_95:,.0f}")
                    
                    # Plot
                    fig_mc = go.Figure()
                    # Sample paths
                    for i in range(min(100, mc_runs)):
                        fig_mc.add_trace(go.Scatter(y=cum_sims[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=1), showlegend=False))
                    
                    # Fan Chart (Percentiles)
                    x_ax = np.arange(mc_days)
                    p95 = np.percentile(cum_sims, 95, axis=1)
                    p05 = np.percentile(cum_sims, 5, axis=1)
                    p50 = np.percentile(cum_sims, 50, axis=1)
                    
                    fig_mc.add_trace(go.Scatter(x=x_ax, y=p95, line=dict(color='green'), name='95th %'))
                    fig_mc.add_trace(go.Scatter(x=x_ax, y=p50, line=dict(color='white'), name='Median'))
                    fig_mc.add_trace(go.Scatter(x=x_ax, y=p05, line=dict(color='red'), name='5th %'))
                    
                    fig_mc.update_layout(title="Monte Carlo Forecast", template="plotly_dark", xaxis_title="Days Ahead", yaxis_title="Equity")
                    st.plotly_chart(fig_mc, use_container_width=True)

            # --- TAB 4: DEEP DIVE ---
            with tabs[3]:
                st.subheader("🔍 Deep Dive Analytics")
                
                # Correlations
                if len(valid_files) > 1:
                    st.markdown("### Correlation Matrix")
                    corr = all_daily_returns.corr()
                    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                    fig_corr.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Drawdown Analysis
                st.markdown("### Drawdown Analysis")
                peak = port_equity.cummax()
                dd = (port_equity - peak) / peak
                
                fig_dd = px.area(dd, title="Underwater Plot", labels={'value':'Drawdown'})
                fig_dd.update_traces(line_color='red', fillcolor='rgba(255,0,0,0.2)')
                fig_dd.update_layout(template="plotly_dark")
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Win/Loss Dist
                if not full_trades_df.empty:
                    st.markdown("### PnL Distribution")
                    fig_dist = px.histogram(full_trades_df, x="Net PnL %", color="Strategy", nbins=50, marginal="box")
                    fig_dist.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_dist, use_container_width=True)

            # --- TAB 5: OPTIMIZER ---
            with tabs[4]:
                st.subheader("🧠 Mean-Variance Optimizer")
                if len(valid_files) < 2:
                    st.warning("Need 2+ strategies.")
                else:
                    if st.button("Run Optimization"):
                        n_sims = 2000
                        # Vectorized calc
                        mean_ret = all_daily_returns.mean() * 252
                        cov = all_daily_returns.cov() * 252
                        
                        # Random weights
                        w = np.random.random((n_sims, len(valid_files)))
                        w = w / w.sum(axis=1)[:, None]
                        
                        # Metrics
                        port_ret = np.dot(w, mean_ret)
                        port_vol = np.sqrt(np.einsum('ij,jk,ik->i', w, cov, w))
                        port_sharpe = (port_ret - risk_free_rate) / port_vol
                        
                        # Max Sharpe
                        idx_max = np.argmax(port_sharpe)
                        
                        # Plot
                        fig_opt = go.Figure()
                        fig_opt.add_trace(go.Scatter(
                            x=port_vol, y=port_ret, mode='markers',
                            marker=dict(color=port_sharpe, colorscale='Viridis', showscale=True),
                            text=[f"Sharpe: {s:.2f}" for s in port_sharpe],
                            name='Random'
                        ))
                        fig_opt.add_trace(go.Scatter(
                            x=[port_vol[idx_max]], y=[port_ret[idx_max]],
                            mode='markers', marker=dict(color='red', size=15, symbol='star'),
                            name='Max Sharpe'
                        ))
                        fig_opt.update_layout(template="plotly_dark", xaxis_title="Volatility", yaxis_title="Return")
                        st.plotly_chart(fig_opt, use_container_width=True)
                        
                        st.write("### Optimal Weights")
                        opt_df = pd.DataFrame({'Strategy': valid_files, 'Weight': w[idx_max]})
                        fig_pie = px.pie(opt_df, values='Weight', names='Strategy', hole=0.4)
                        st.plotly_chart(fig_pie, use_container_width=True)

            # --- TAB 6: RAW DATA ---
            with tabs[5]:
                st.dataframe(full_trades_df, use_container_width=True)

        else:
            st.warning("No valid data found in CSVs.")
    else:
        st.info("Upload CSVs to begin.")
