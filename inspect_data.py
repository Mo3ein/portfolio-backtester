import pandas as pd
import os

files = [f for f in os.listdir('/Users/fractalyst/CascadeProjects/Portfolio-backtester') if f.endswith('.csv')]
if not files:
    print("No CSV files found")
    exit()

file_path = os.path.join('/Users/fractalyst/CascadeProjects/Portfolio-backtester', files[0])
print(f"Reading {file_path}")

# Read csv, skip bad lines if any
df = pd.read_csv(file_path)
print("Columns:", df.columns.tolist())

# Check 'Trade PnL %'
if 'Trade PnL %' in df.columns:
    non_null_pnl = df[df['Trade PnL %'].notna()]['Trade PnL %']
    print("\nFirst 5 non-null Trade PnL % values:")
    print(non_null_pnl.head())
    
    print("\nStats:")
    print(non_null_pnl.describe())
else:
    print("'Trade PnL %' column not found")
