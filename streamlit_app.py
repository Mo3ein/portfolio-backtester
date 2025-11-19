import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots

st.set_page_config(page_title="Pro Portfolio Backtester", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for "Shock" value ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stPlotlyChart {
        background-color: #0e1117;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Pro Portfolio Backtester")
st.markdown("Advanced analytics, Monte Carlo simulations, and Portfolio Optimization.")

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
            return None, f"Missing 'time' column in {file.name}"
            
        # Convert time
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df, None
    except Exception as e:
        return None, str(e)

def calculate_advanced_metrics(daily_returns, trades_df=None, start_cap=10000):
    if len(daily_returns) < 2:
        return {}
    
    # Geometric mean return
    total_ret = (1 + daily_returns).prod() - 1
    cagr = (1 + total_ret) ** (252 / len(daily_returns)) - 1
    
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
        
        metrics['Win Rate'] = len(wins) / len(trades_df)
        metrics['Avg Win'] = wins['Net PnL'].mean() if not wins.empty else 0
        metrics['Avg Loss'] = losses['Net PnL'].mean() if not losses.empty else 0
        metrics['Profit Factor'] = abs(wins['Net PnL'].sum() / losses['Net PnL'].sum()) if losses['Net PnL'].sum() != 0 else 0
        metrics['Trades'] = len(trades_df)
        
        # Streak Analysis
        trades_df['Win'] = trades_df['Net PnL'] > 0
        trades_df['group'] = (trades_df['Win'] != trades_df['Win'].shift()).cumsum()
        streaks = trades_df.groupby('group')['Win'].agg(['first', 'count'])
        metrics['Max Win Streak'] = streaks[streaks['first'] == True]['count'].max() if not streaks.empty else 0
        metrics['Max Loss Streak'] = streaks[streaks['first'] == False]['count'].max() if not streaks.empty else 0

    return metrics

if uploaded_files:
    data_objects = {}
    valid_files = []
    
    for f in uploaded_files:
        df, err = load_data(f)
        if df is not None:
            # Extract Strategy Name from filename (simplified)
            name = f.name.split(',')[0].replace('BINANCE_', '').replace('COINBASE_', '').replace('KRAKEN_', '')
            data_objects[name] = df
            valid_files.append(name)
        else:
            st.error(f"Error loading {f.name}: {err}")

    if data_objects:
        # --- Tab Structure ---
        tab_overview, tab_analysis, tab_mc, tab_opt, tab_raw = st.tabs([
            "📊 Dashboard", "🔍 Deep Dive", "🎲 Monte Carlo", "🧠 Optimizer", "📝 Raw Data"
        ])
        
        # Pre-process Data
        all_daily_returns = pd.DataFrame()
        all_trades_list = []
        buy_hold_curves = pd.DataFrame()
        
        for name, df in data_objects.items():
            # 1. Extract Trades
            if 'Trade PnL %' in df.columns:
                trades = df[df['Trade PnL %'].notna()].copy()
                
                # Calculate Net PnL
                trades['Net PnL %'] = trades['Trade PnL %'] - commission_rate - slippage_rate
                trades['Strategy'] = name
                
                all_trades_list.append(trades)
                
                # Create Daily Returns Series from Trades
                daily_pnl = trades.resample('D')['Net PnL %'].sum() / 100.0
                all_daily_returns[name] = daily_pnl
            
            # 2. Buy & Hold (Close Price)
            if 'close' in df.columns:
                price = df['close'].resample('D').last().ffill()
                bh_ret = price.pct_change().fillna(0)
                # Normalize start to 1.0
                buy_hold_curves[name] = (1 + bh_ret).cumprod()
                
        # Fill NaN in daily returns with 0 (days with no trades)
        all_daily_returns.fillna(0, inplace=True)
        
        # --- Global Settings for Portfolio ---
        with st.sidebar:
            st.subheader("Portfolio Weights")
            weights = {}
            total_w = 0
            if len(valid_files) > 0:
                default_w = 1.0 / len(valid_files)
                for name in valid_files:
                    w = st.slider(f"{name} Weight", 0.0, 1.0, default_w, 0.05)
                    weights[name] = w
                    total_w += w
            
                if abs(total_w - 1.0) > 0.01:
                    st.warning(f"Total Weight: {total_w:.2f} (Should be 1.0)")
                    # Normalize for calculation
                    norm_factor = 1.0 / total_w if total_w > 0 else 0
                    for k in weights:
                        weights[k] *= norm_factor
            
        # Calculate Portfolio Curve
        if not all_daily_returns.empty:
            # Weighted sum of daily returns
            port_daily_ret = all_daily_returns.mul(pd.Series(weights)).sum(axis=1)
            port_equity = initial_capital * (1 + port_daily_ret).cumprod()
            
            # Portfolio Trades DF
            if all_trades_list:
                full_trades_df = pd.concat(all_trades_list)
                # Add approximate $ PnL column based on initial capital for simple stats
                # Real PnL depends on dynamic equity which is complex to back-calc on the fly for individual trades with weights
                # So we use simple % * initial_capital for the 'Deep Dive' histogram
                full_trades_df['Net PnL'] = initial_capital * (full_trades_df['Net PnL %']/100.0)
            else:
                full_trades_df = pd.DataFrame()

            # --- DASHBOARD TAB ---
            with tab_overview:
                # Top Level Metrics
                metrics = calculate_advanced_metrics(port_daily_ret, full_trades_df)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{metrics.get('Total Return', 0)*100:.2f}%", f"${port_equity.iloc[-1] - initial_capital:,.0f}")
                col2.metric("Sharpe Ratio", f"{metrics.get('Sharpe', 0):.2f}")
                col3.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0)*100:.2f}%")
                col4.metric("CAGR", f"{metrics.get('CAGR', 0)*100:.2f}%")
                
                # Main Chart
                st.subheader("Equity Curve vs Buy & Hold")
                fig = go.Figure()
                
                # Portfolio
                fig.add_trace(go.Scatter(x=port_equity.index, y=port_equity, mode='lines', name='Active Portfolio', 
                                       line=dict(color='#00FF00', width=3)))
                
                # Buy & Hold Average (Benchmark)
                if not buy_hold_curves.empty:
                    # Create an Equal Weight Buy & Hold Benchmark of the assets in the portfolio
                    # We use the same weights as selected in sidebar for fair comparison?
                    # Or just equal weight? Let's use sidebar weights.
                    
                    # Align B&H curves index with portfolio
                    # Reindex to ensure match
                    bh_aligned = buy_hold_curves.reindex(port_equity.index).ffill()
                    
                    # Weighted Benchmark
                    bh_benchmark_ret = pd.DataFrame()
                    for name, w in weights.items():
                        if name in bh_aligned.columns:
                            # Convert price curve back to returns? Or just weighted sum of normalized curves?
                            # Weighted sum of normalized curves is a "rebalanced" portfolio of assets? No.
                            # Weighted sum of curves = Fixed fractional allocation without rebalancing? 
                            # Let's keep it simple: Weighted sum of the normalized curves * Initial Capital
                            pass
                            
                    # Simpler: Just plot the weighted average of the B&H curves
                    if len(weights) > 0:
                         # bh_aligned has columns [Strat1, Strat2...]
                         # We want sum(Strat1 * w1, Strat2 * w2)
                         bh_portfolio = bh_aligned.mul(pd.Series(weights)).sum(axis=1)
                         # Scale to Initial Capital
                         bh_portfolio = bh_portfolio * initial_capital
                         
                         fig.add_trace(go.Scatter(x=bh_portfolio.index, y=bh_portfolio, mode='lines', name='B&H Benchmark (Weighted)', 
                                           line=dict(color='orange', dash='dash')))

                fig.update_layout(height=500, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown Chart
                st.subheader("Underwater Plot")
                peak = port_equity.cummax()
                dd = (port_equity - peak) / peak
                fig_dd = px.area(dd, title="Portfolio Drawdown", labels={'value': 'Drawdown', 'datetime': 'Date'})
                fig_dd.update_traces(line_color='red')
                fig_dd.update_layout(height=300, template="plotly_dark", yaxis_tickformat='.1%')
                st.plotly_chart(fig_dd, use_container_width=True)

            # --- DEEP DIVE TAB ---
            with tab_analysis:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("Strategy Correlations")
                    if len(valid_files) > 1:
                        corr = all_daily_returns.corr()
                        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                        fig_corr.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Need multiple strategies for correlation analysis.")
                        
                with col_b:
                    st.subheader("Monthly Returns Heatmap")
                    monthly_ret_pct = ((1 + port_daily_ret).resample('M').prod() - 1) * 100
                    
                    if not monthly_ret_pct.empty:
                        monthly_df = pd.DataFrame({
                            'Year': monthly_ret_pct.index.year,
                            'Month': monthly_ret_pct.index.strftime('%b'),
                            'Return': monthly_ret_pct.values
                        })
                        
                        # Pivot
                        pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
                        # Sort months
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        pivot_table = pivot_table.reindex(columns=[m for m in months if m in pivot_table.columns])
                        
                        fig_heat = px.imshow(pivot_table, text_auto='.2f', color_continuous_scale='RdYlGn', aspect='auto')
                        fig_heat.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_heat, use_container_width=True)
                
                # Trade Analysis
                st.divider()
                st.subheader("Trade Statistics")
                if not full_trades_df.empty:
                    c1, c2 = st.columns(2)
                    with c1:
                        # Histogram of Returns
                        fig_hist = px.histogram(full_trades_df, x="Net PnL %", color="Strategy", nbins=50, 
                                              title="Distribution of Trade Returns")
                        fig_hist.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                    with c2:
                        # Day of Week
                        full_trades_df['Day'] = full_trades_df.index.day_name()
                        day_agg = full_trades_df.groupby('Day')['Net PnL %'].mean().reindex(
                            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        )
                        fig_day = px.bar(day_agg, title="Average Return by Day of Week", color=day_agg.values, color_continuous_scale='RdYlGn')
                        fig_day.update_layout(template="plotly_dark", showlegend=False)
                        st.plotly_chart(fig_day, use_container_width=True)

            # --- MONTE CARLO TAB ---
            with tab_mc:
                st.subheader("Monte Carlo Simulation")
                st.markdown("Simulating 1,000 random future paths based on historical daily returns.")
                
                mc_runs = 1000
                days_to_sim = 252 # 1 year
                
                mu = port_daily_ret.mean()
                sigma = port_daily_ret.std()
                
                if sigma > 0:
                    # Generate random paths
                    daily_sims = np.random.normal(mu, sigma, (days_to_sim, mc_runs))
                    cum_sims = (1 + daily_sims).cumprod(axis=0) * port_equity.iloc[-1]
                    
                    # Plot
                    fig_mc = go.Figure()
                    
                    # First 50 traces
                    for i in range(min(50, mc_runs)):
                        fig_mc.add_trace(go.Scatter(y=cum_sims[:, i], mode='lines', line=dict(color='grey', width=1), opacity=0.1, showlegend=False))
                    
                    # Percentiles
                    p95 = np.percentile(cum_sims, 95, axis=1)
                    p50 = np.percentile(cum_sims, 50, axis=1)
                    p05 = np.percentile(cum_sims, 5, axis=1)
                    
                    x_axis = list(range(days_to_sim))
                    
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', name='95th Percentile', line=dict(color='green')))
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=p50, mode='lines', name='Median', line=dict(color='white')))
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', name='5th Percentile', line=dict(color='red')))
                    
                    fig_mc.update_layout(title=f"Projected Equity (Next {days_to_sim} Days)", template="plotly_dark", xaxis_title="Trading Days")
                    st.plotly_chart(fig_mc, use_container_width=True)
                else:
                    st.warning("Not enough data/volatility for Monte Carlo.")

            # --- OPTIMIZER TAB ---
            with tab_opt:
                st.subheader("Mean-Variance Optimization")
                st.markdown("Simulating 2,000 random portfolios to find the Efficient Frontier.")
                
                if len(valid_files) < 2:
                    st.warning("Need at least 2 strategies to optimize.")
                else:
                    if st.button("Run Optimizer"):
                        n_portfolios = 2000
                        results = np.zeros((3, n_portfolios)) # Ret, Vol, Sharpe
                        weights_record = []
                        
                        mean_ret = all_daily_returns.mean() * 252
                        cov_mat = all_daily_returns.cov() * 252
                        
                        progress_bar = st.progress(0)
                        
                        # Vectorized simulation (faster)
                        rand_weights = np.random.random((n_portfolios, len(valid_files)))
                        rand_weights = rand_weights / rand_weights.sum(axis=1)[:, None]
                        
                        # Returns
                        p_rets = np.dot(rand_weights, mean_ret)
                        
                        # Volatility
                        # Diag(w @ cov @ w.T)
                        # To avoid loop:
                        # (N, S) @ (S, S) = (N, S)
                        # (N, S) * (N, S) -> sum axis 1
                        temp = np.dot(rand_weights, cov_mat)
                        p_vols = np.sqrt(np.sum(temp * rand_weights, axis=1))
                        
                        p_sharpes = (p_rets - risk_free_rate) / p_vols
                        
                        progress_bar.progress(100)
                        
                        # Plot
                        max_sharpe_idx = np.argmax(p_sharpes)
                        sd_p, ret_p = p_vols[max_sharpe_idx], p_rets[max_sharpe_idx]
                        
                        fig_opt = go.Figure()
                        fig_opt.add_trace(go.Scatter(
                            x=p_vols, y=p_rets, mode='markers',
                            marker=dict(color=p_sharpes, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
                            text=[f"Sharpe: {s:.2f}" for s in p_sharpes],
                            name='Random Portfolios'
                        ))
                        
                        # Highlight Max Sharpe
                        fig_opt.add_trace(go.Scatter(
                            x=[sd_p], y=[ret_p], mode='markers',
                            marker=dict(color='red', size=20, symbol='star'),
                            name='Max Sharpe'
                        ))
                        
                        fig_opt.update_layout(title="Efficient Frontier", xaxis_title="Annualized Volatility", yaxis_title="Annualized Return", template="plotly_dark")
                        st.plotly_chart(fig_opt, use_container_width=True)
                        
                        # Show Optimal Weights
                        st.subheader("Optimal Weights (Max Sharpe)")
                        opt_w = rand_weights[max_sharpe_idx]
                        opt_df = pd.DataFrame({'Strategy': valid_files, 'Weight': opt_w})
                        fig_pie = px.pie(opt_df, values='Weight', names='Strategy', title="Optimal Allocation", hole=0.3)
                        st.plotly_chart(fig_pie, use_container_width=True)

            # --- RAW DATA TAB ---
            with tab_raw:
                st.dataframe(full_trades_df.sort_index(ascending=False), use_container_width=True)
        else:
            st.warning("No data could be processed.")
    else:
        st.info("Awaiting CSV files...")
