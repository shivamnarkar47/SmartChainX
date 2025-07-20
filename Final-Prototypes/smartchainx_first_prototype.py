import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from networkx.algorithms.approximation import traveling_salesman_problem
import folium

st.set_page_config(page_title="SmartChainX", layout="wide")
st.title("🚛📦 SmartChainX – Forecast + Route Optimizer")

# ----------- 📈 DEMAND FORECASTING -----------
st.header("📈 Demand Forecasting")
demand_file = st.file_uploader("Upload `train.csv` (demand data)", type=['csv'], key="demand")

if demand_file:
    df = pd.read_csv(demand_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    stores = sorted(df['store'].unique())
    items = sorted(df['item'].unique())

    store = st.selectbox("Select Store", stores)
    item = st.selectbox("Select Item", items)

    # Filter & feature engineer
    df_filtered = df[(df['store'] == store) & (df['item'] == item)].copy()
    for lag in [1, 7, 14]:
        df_filtered[f'sales_lag_{lag}'] = df_filtered['sales'].shift(lag)
    df_filtered['day'] = df_filtered.index.day
    df_filtered['month'] = df_filtered.index.month
    df_filtered['dayofweek'] = df_filtered.index.dayofweek
    df_filtered.dropna(inplace=True)

    train = df_filtered.iloc[:-30]
    test = df_filtered.iloc[-30:]
    X_train = train.drop(['sales', 'store', 'item'], axis=1)
    y_train = train['sales']
    X_test = test.drop(['sales', 'store', 'item'], axis=1)
    y_test = test['sales']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Display forecast
    test['predicted_sales'] = preds
    st.line_chart(test[['sales', 'predicted_sales']])
    st.dataframe(test[['sales', 'predicted_sales']])

    forecast_csv = test[['sales', 'predicted_sales']].to_csv().encode('utf-8')
    st.download_button("⬇️ Download Forecast CSV", forecast_csv, "forecast_output.csv", "text/csv")

# ----------- 🚚 ROUTE OPTIMIZATION -----------
st.header("🚚 Route Optimization")
loc_file = st.file_uploader("Upload `locations.csv`", type=['csv'], key="route2")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

if loc_file:
    df_loc = pd.read_csv(loc_file)
    G = nx.complete_graph(len(df_loc))

    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                dist = haversine(df_loc.iloc[i]['latitude'], df_loc.iloc[i]['longitude'],
                                 df_loc.iloc[j]['latitude'], df_loc.iloc[j]['longitude'])
                G[i][j]['weight'] = dist

    route = traveling_salesman_problem(G, cycle=True)
    route_names = df_loc.loc[route]['location_name'].tolist()

    st.subheader("📍 Optimized Route")
    for idx, name in enumerate(route_names):
        st.write(f"{idx+1}. {name}")

    # Show Map
    m = folium.Map(location=[df_loc['latitude'].mean(), df_loc['longitude'].mean()], zoom_start=12)
    for i in route:
        loc = df_loc.iloc[i]
        folium.Marker(
            location=[loc['latitude'], loc['longitude']],
            popup=loc['location_name'],
            icon=folium.Icon(color='blue')
        ).add_to(m)

    for i in range(len(route)-1):
        loc1 = df_loc.iloc[route[i]]
        loc2 = df_loc.iloc[route[i+1]]
        folium.PolyLine(
            locations=[[loc1['latitude'], loc1['longitude']],
                       [loc2['latitude'], loc2['longitude']]],
            color="green", weight=3
        ).add_to(m)

    st.subheader("🗺️ Route Map")
    st.components.v1.html(m._repr_html_(), height=500)


# ----------- 🚚 Shortest Path Tree (SPT) + Map -----------
st.header("🚚 Shortest Path Tree (SPT) Delivery")
loc_file = st.file_uploader("Upload `locations.csv`", type=['csv'], key="route")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

if loc_file:
    df_loc = pd.read_csv(loc_file)
    G = nx.Graph()

    for i in range(len(df_loc)):
        G.add_node(i, name=df_loc.loc[i, 'location_name'])

    for i in range(len(df_loc)):
        for j in range(len(df_loc)):
            if i != j:
                dist = haversine(
                    df_loc.loc[i, 'latitude'], df_loc.loc[i, 'longitude'],
                    df_loc.loc[j, 'latitude'], df_loc.loc[j, 'longitude']
                )
                G.add_edge(i, j, weight=dist)

    # SPT from Warehouse (node 0)
    source = 0
    paths = nx.single_source_dijkstra_path(G, source=source, weight='weight')
    distances = nx.single_source_dijkstra_path_length(G, source=source, weight='weight')

    st.subheader("📍 Shortest Paths from Warehouse")
    for target in range(len(df_loc)):
        if target != source:
            path = paths[target]
            names = df_loc.loc[path]['location_name'].tolist()
            st.markdown(f"**To {df_loc.loc[target, 'location_name']}**")
            st.write(f"➡️ Path: {' → '.join(names)}")
            st.write(f"🛣️ Distance: {distances[target]:.2f} km")

    # 🗺️ Folium Map for SPT
    import folium
    m = folium.Map(location=[df_loc['latitude'].mean(), df_loc['longitude'].mean()], zoom_start=12)

    # Add markers
    for i, row in df_loc.iterrows():
        popup = f"{row['location_name']} (ID: {row['location_id']})"
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            icon=folium.Icon(color='blue' if i == source else 'green')
        ).add_to(m)

    # Draw path lines from warehouse to each point
    for target in range(len(df_loc)):
        if target != source:
            path = paths[target]
            coords = [[df_loc.loc[i, 'latitude'], df_loc.loc[i, 'longitude']] for i in path]
            folium.PolyLine(coords, color="red", weight=3, opacity=0.8).add_to(m)

    st.subheader("🗺️ Shortest Path Tree Map")
    st.components.v1.html(m._repr_html_(), height=500)
