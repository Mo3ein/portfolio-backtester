import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Portfolio Backtester", layout="wide")

st.title("Portfolio Backtester")
st.write("Upload your backtest CSV files to analyze performance.")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    initial_capital = st.number_input("Initial Capital ($)", min_value=100.0, value=10000.0, step=100.0)
    
    st.subheader("Allocation")
    allocation_type = st.radio("Capital Allocation", ["Equal Weight", "Fixed Amount per Strategy"])
    
    st.info("Note: 'Equal Weight' divides Initial Capital by the number of strategies. 'Fixed Amount' applies Initial Capital to EACH strategy.")

# File uploader
uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

def load_and_process_data(file):
    try:
        df = pd.read_csv(file)
        
        # Basic validation
        required_columns = ['time', 'Trade PnL %']
        if not all(col in df.columns for col in required_columns):
            st.error(f"File {file.name} is missing required columns: {required_columns}")
            return None

        # Convert time
        # Assuming unix timestamp in seconds based on inspection
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Filter for trades
        trades = df[df['Trade PnL %'].notna()].copy()
        
        return trades
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def calculate_metrics(equity_curve):
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Final Equity": equity_curve[-1]
    }

if uploaded_files:
    processed_data = {}
    
    for uploaded_file in uploaded_files:
        trades = load_and_process_data(uploaded_file)
        if trades is not None:
            processed_data[uploaded_file.name] = trades

    if processed_data:
        st.divider()
        st.subheader("Performance Analysis")
        
        # Determine capital per strategy
        num_strategies = len(processed_data)
        if allocation_type == "Equal Weight":
            capital_per_strategy = initial_capital / num_strategies
        else:
            capital_per_strategy = initial_capital

        # Calculate Equity Curves
        equity_curves = pd.DataFrame()
        
        # We need a common date index for the portfolio
        # Get min and max dates across all files
        all_dates = []
        for name, df in processed_data.items():
            all_dates.extend(df.index.tolist())
        
        if not all_dates:
            st.warning("No trades found in uploaded files.")
            st.stop()
            
        full_index = sorted(list(set(all_dates)))
        
        portfolio_equity = pd.Series(0.0, index=full_index)
        # Initialize portfolio with total starting capital
        # But simpler: track cumulative PnL $ and add to initial capital
        
        # Actually, to do it correctly with "Trade PnL %":
        # Each strategy starts with 'capital_per_strategy'.
        # When a trade happens, balance updates.
        
        summary_stats = []

        fig = go.Figure()

        for name, trades in processed_data.items():
            # Create a series aligned to the full index, filling with 0 PnL where no trade
            # But wait, we can just iterate the trades in order.
            
            # Calculate equity curve for this strategy
            # Start with capital_per_strategy
            # Apply returns
            
            strategy_balance = [capital_per_strategy]
            strategy_dates = [trades.index[0]] # Start from first trade? Or we should probably have a start date. 
            # Let's just map the equity after each trade.
            
            current_balance = capital_per_strategy
            equity_series = pd.Series(index=trades.index, dtype=float)
            
            for date, row in trades.iterrows():
                pnl_pct = row['Trade PnL %']
                # pnl_pct is e.g. 3.5 for 3.5%
                pnl_amount = current_balance * (pnl_pct / 100.0)
                current_balance += pnl_amount
                equity_series[date] = current_balance
            
            # Add to plot
            fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series, mode='lines', name=name))
            
            # Calculate metrics
            metrics = calculate_metrics(pd.Series([capital_per_strategy] + equity_series.tolist()))
            stats = {
                "Strategy": name,
                "Total Return": f"{metrics['Total Return']*100:.2f}%",
                "Max Drawdown": f"{metrics['Max Drawdown']*100:.2f}%",
                "Final Equity": f"${metrics['Final Equity']:,.2f}",
                "Trade Count": len(trades)
            }
            summary_stats.append(stats)
            
            # For portfolio combination
            # We need to align these equity values to the common time index
            # Forward fill the equity values (your equity stays same until next trade)
            # Reindex to full_index
            aligned_equity = equity_series.reindex(full_index).ffill().fillna(capital_per_strategy)
            
            # Add this strategy's equity contribution to the portfolio
            portfolio_equity += aligned_equity

        # Plot Portfolio
        if len(processed_data) > 1:
            fig.add_trace(go.Scatter(x=portfolio_equity.index, y=portfolio_equity, mode='lines', name='PORTFOLIO', line=dict(width=4, color='white')))
            
            # Portfolio Stats
            port_metrics = calculate_metrics(portfolio_equity)
            summary_stats.append({
                "Strategy": "PORTFOLIO",
                "Total Return": f"{port_metrics['Total Return']*100:.2f}%",
                "Max Drawdown": f"{port_metrics['Max Drawdown']*100:.2f}%",
                "Final Equity": f"${port_metrics['Final Equity']:,.2f}",
                "Trade Count": sum([s['Trade Count'] for s in summary_stats])
            })

        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Statistics")
        st.dataframe(pd.DataFrame(summary_stats))
        
else:
    st.info("Please upload one or more CSV files to begin.")
