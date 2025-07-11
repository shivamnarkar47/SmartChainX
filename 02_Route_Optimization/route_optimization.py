import streamlit as st
import pandas as pd
import networkx as nx
import folium
from math import radians, sin, cos, sqrt, atan2
from networkx.algorithms.approximation import traveling_salesman_problem

st.set_page_config(page_title="🚛 Mumbai Route Optimizer", layout="wide")
st.title("📍 SmartChainX – Mumbai Route Optimizer")

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Upload location CSV
uploaded_file = st.file_uploader("📤 Upload Mumbai Locations CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # Build Graph
    G = nx.complete_graph(len(df))
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                dist = haversine(df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                                 df.iloc[j]['latitude'], df.iloc[j]['longitude'])
                G[i][j]['weight'] = dist

    # Solve TSP
    route = traveling_salesman_problem(G, cycle=True)
    route_names = df.loc[route]['location_name'].tolist()

    st.subheader("📈 Optimized Route Order")
    for idx, name in enumerate(route_names):
        st.write(f"{idx+1}. {name}")

    # Draw Map
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    for i in route:
        loc = df.iloc[i]
        folium.Marker(
            location=[loc['latitude'], loc['longitude']],
            popup=loc['location_name'],
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    for i in range(len(route) - 1):
        loc1 = df.iloc[route[i]]
        loc2 = df.iloc[route[i+1]]
        folium.PolyLine(
            locations=[[loc1['latitude'], loc1['longitude']],
                       [loc2['latitude'], loc2['longitude']]],
            color="green", weight=3, opacity=0.8
        ).add_to(m)

    # Save & Show Map
    m.save("mumbai_optimized_route.html")
    st.subheader("🗺️ Optimized Route Map:")
    st.components.v1.html(m._repr_html_(), height=500)

    # Download link
    with open("mumbai_optimized_route.html", "rb") as f:
        st.download_button("⬇️ Download Map as HTML", data=f, file_name="mumbai_optimized_route.html", mime="text/html")
