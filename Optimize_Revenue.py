import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Optimize Revenue",
    page_icon="⚙️",
    layout="wide"
)

st.subheader("💰 **Revenue Optimization Tools**")

# Create tabs for different optimization strategies
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Price Optimization",
    "📊 Demand Forecasting",
    "🏨 Room Mix Optimization",
    "📈 Performance Analytics"
])

# ------------------------------
# TAB 1 - Price Optimization
# ------------------------------
with tab1:
    st.subheader("🎯 **Dynamic Price Optimization**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Optimization Parameters:**")
        
        # Input parameters
        occupancy_rate = st.slider("Expected Occupancy Rate (%)", 10, 100, 70, 5)
        room_nights = st.number_input("Number of Room Nights", min_value=1, value=30)
        price_range = st.slider("Price Range (EGP)", 500, 5000, (1000, 3000), 100)
        
        # Market segment selection
        market_type = st.selectbox("Market Segment", ["All", "Local", "International"])
        
        # Seasonality factor
        season_factors_dict = {
            "Low Season": 0.8,
            "Regular Season": 1.0,
            "High Season": 1.2,
            "Peak Season": 1.5
        }
        selected_season_key = st.selectbox("Season Factor", list(season_factors_dict.keys()))
        season_factor = season_factors_dict[selected_season_key]
        
        # Competition factor
        competition_factor = st.slider("Competition Factor", 0.5, 1.5, 1.0, 0.1)
    
    with col2:
        st.write("**Price Optimization Analysis:**")
        
        # Calculate optimal pricing
        prices = np.arange(price_range[0], price_range[1] + 100, 100)
        demand = occupancy_rate / 100 * room_nights * season_factor * competition_factor
        
        revenues = prices * demand * np.exp(-0.0005 * (prices - price_range[0]))
        
        df_prices = pd.DataFrame({
            "Price (EGP)": prices,
            "Estimated Revenue": revenues
        })
        
        optimal_price = df_prices.loc[df_prices["Estimated Revenue"].idxmax()]
        
        fig_price = px.line(df_prices, x="Price (EGP)", y="Estimated Revenue",
                            title="Revenue vs Price")
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.success(f"💡 **Optimal Price:** {optimal_price['Price (EGP)']:.2f} EGP")
        st.info(f"Expected Revenue: {optimal_price['Estimated Revenue']:.2f} EGP")

# ------------------------------
# TAB 2 - Demand Forecasting
# ------------------------------
with tab2:
    st.subheader("📊 **Demand Forecasting**")
    
    st.write("Upload historical booking data (CSV) to forecast demand.")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        df_forecast = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df_forecast.head())
        
        if 'Date' in df_forecast.columns and 'Bookings' in df_forecast.columns:
            df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
            df_forecast = df_forecast.sort_values('Date')
            
            fig_forecast = px.line(df_forecast, x='Date', y='Bookings',
                                   title="Historical Bookings")
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.write("**(Placeholder)** Demand forecasting model can be integrated here (e.g., Prophet).")
        else:
            st.error("Uploaded CSV must have 'Date' and 'Bookings' columns.")

# ------------------------------
# TAB 3 - Room Mix Optimization
# ------------------------------
with tab3:
    st.subheader("🏨 **Room Mix Optimization**")
    
    st.write("Adjust room types to maximize total revenue.")
    
    room_types = ["Standard", "Deluxe", "Suite"]
    room_counts = {}
    prices_per_type = {}
    
    total_rooms = st.number_input("Total Available Rooms", 10, 500, 100)
    
    for r in room_types:
        room_counts[r] = st.slider(f"Number of {r} Rooms", 0, total_rooms, total_rooms // len(room_types))
        prices_per_type[r] = st.number_input(f"Price for {r} (EGP)", 500, 10000, 1500)
    
    total_revenue = sum([room_counts[r] * prices_per_type[r] for r in room_types])
    st.success(f"Estimated Revenue for Selected Mix: **{total_revenue} EGP**")

# ------------------------------
# TAB 4 - Performance Analytics
# ------------------------------
with tab4:
    st.subheader("📈 **Performance Analytics**")
    
    st.write("Upload performance data to analyze KPIs.")
    perf_file = st.file_uploader("Upload Performance Data (CSV)", type="csv", key="perf")
    
    if perf_file:
        df_perf = pd.read_csv(perf_file)
        st.write("### Preview of Performance Data")
        st.dataframe(df_perf.head())
        
        if 'Date' in df_perf.columns and 'Revenue' in df_perf.columns:
            df_perf['Date'] = pd.to_datetime(df_perf['Date'])
            df_perf = df_perf.sort_values('Date')
            
            fig_perf = px.line(df_perf, x='Date', y='Revenue', title="Revenue Performance Over Time")
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.error("Performance data must have 'Date' and 'Revenue' columns.")
