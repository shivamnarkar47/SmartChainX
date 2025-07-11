import streamlit as st
import pandas as pd

# Load all forecasts
df = pd.read_csv("../01_Demand_Forecasting/all_forecast.csv", parse_dates=['date'])

st.set_page_config(page_title="SmartChainX Full Forecast", layout="wide")
st.title("📊 SmartChainX - Full Demand Forecast Dashboard")

# Store & Item Select
store_list = sorted(df['store'].unique())
item_list = sorted(df['item'].unique())

store = st.selectbox("🏪 Select Store", store_list)
item = st.selectbox("🧾 Select Item", item_list)

filtered = df[(df['store'] == store) & (df['item'] == item)].copy()
filtered.set_index('date', inplace=True)

# Plot
st.line_chart(filtered[['sales', 'predicted_sales']])

# Table
st.dataframe(filtered[['sales', 'predicted_sales']])

# Download
csv = filtered.to_csv().encode('utf-8')
st.download_button("⬇️ Download Forecast CSV", data=csv, file_name=f"forecast_store{store}_item{item}.csv", mime='text/csv')
