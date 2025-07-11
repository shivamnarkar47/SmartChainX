import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load full forecast CSV
df = pd.read_csv("../01_Demand_Forecasting/forecast_gb_all.csv")
df['date'] = pd.to_datetime(df['date'])

st.set_page_config(page_title="SmartChainX Forecast", layout="wide")
st.title("📦 SmartChainX - Demand Forecast Dashboard")

# Store & Item Dropdowns
store_list = sorted(df['store'].unique())
item_list = sorted(df['item'].unique())

col1, col2 = st.columns(2)
with col1:
    selected_store = st.selectbox("🏪 Select Store", store_list)
with col2:
    selected_item = st.selectbox("🧾 Select Item", item_list)

# Filter data
filtered = df[(df['store'] == selected_store) & (df['item'] == selected_item)].copy()
filtered.set_index('date', inplace=True)

# Plot
st.subheader(f"📈 Forecast for Store {selected_store}, Item {selected_item}")
st.line_chart(filtered[['sales', 'predicted_sales']])

# Table
st.subheader("🔍 Forecast Data")
st.dataframe(filtered[['sales', 'predicted_sales']])

# Download Button
csv = filtered.reset_index().to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv,
    file_name=f"forecast_store{selected_store}_item{selected_item}.csv",
    mime='text/csv'
)
