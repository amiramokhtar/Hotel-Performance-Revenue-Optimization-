import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Hotel Performance Analysis",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 **Hotel Performance Analysis**")
st.markdown("### Comprehensive analysis of hotel key performance indicators and trends.")

# Load data from session state or file
if 'df' in st.session_state:
    df = st.session_state.df
    df_filtered = st.session_state.df_filtered
else:
    DATA_FILE = "Cleaned_Hotel_Booking.csv"
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Convert date columns if available
        if 'Arrival date' in df.columns:
            df['Arrival date'] = pd.to_datetime(df['Arrival date'], errors='coerce')
            df['Year'] = df['Arrival date'].dt.year
            df['Month'] = df['Arrival date'].dt.month
            df['Month_Year'] = df['Arrival date'].dt.to_period('M').astype(str)
        df_filtered = df.copy()
    else:
        st.error("Please upload data from the main page first.")
        st.stop()

# ==========================================
# KPIs Section
# ==========================================
st.subheader("📌 **Key Performance Indicators (KPIs)**")
total_revenue = df_filtered['total rate net'].sum() if 'total rate net' in df_filtered.columns else 0
total_nights = df_filtered['Room night'].sum() if 'Room night' in df_filtered.columns else 0
avg_adr = df_filtered['ADR'].mean() if 'ADR' in df_filtered.columns else 0
revpar = total_revenue / total_nights if total_nights > 0 else 0
total_bookings = len(df_filtered)

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
kpi_col1.metric("💰 Total Revenue (EGP)", f"{total_revenue:,.0f}")
kpi_col2.metric("🛏 Total Room Nights", f"{total_nights:,}")
kpi_col3.metric("🏷 Average ADR (EGP)", f"{avg_adr:,.2f}")
kpi_col4.metric("📈 RevPAR (EGP)", f"{revpar:,.2f}")
kpi_col5.metric("📊 Total Bookings", f"{total_bookings:,}")

# ==========================================
# Charts & Insights
# ==========================================
st.markdown("---")
st.subheader("📊 **Performance Charts & Insights**")

# Create tabs for different analysis sections
tab1, tab2, tab3, tab4 = st.tabs(["📍 Geographic Analysis", "📅 Temporal Analysis", "🏨 Room & Agent Analysis", "💹 Revenue Analysis"])

with tab1:
    st.subheader("🌍 Geographic Performance Analysis")
    
    # Top Countries by Bookings
    if "Country" in df_filtered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            top_countries = df_filtered['Country'].value_counts().head(10)
            top_countries_df = top_countries.reset_index()
            top_countries_df.columns = ['Country', 'Reservations']
            fig_country = px.bar(
                top_countries_df,
                x='Country',
                y='Reservations',
                title='Top 10 Countries by Reservations',
                color='Reservations',
                color_continuous_scale='teal'
            )
            st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            # Revenue by Country
            country_revenue = df_filtered.groupby("Country")["total rate net"].sum().sort_values(ascending=False).head(10).reset_index()
            fig_country_rev = px.bar(
                country_revenue, 
                x='Country', 
                y='total rate net',
                title='Top 10 Countries by Total Revenue',
                color='total rate net',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_country_rev, use_container_width=True)
    
    # Market Share (Local vs International)
    if "Country" in df_filtered.columns:
        df_filtered['Market'] = df_filtered['Country'].apply(
            lambda x: 'Local' if str(x).strip().lower() == 'egypt' else 'International'
        )
        market_rev = df_filtered.groupby('Market')['total rate net'].sum().reset_index()
        fig_market = px.pie(
            market_rev,
            names='Market',
            values='total rate net',
            hole=0.5,
            title='🎯 Local vs International Market Share'
        )
        st.plotly_chart(fig_market, use_container_width=True)

with tab2:
    st.subheader("📅 Temporal Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Reservations Trend
        if "Month" in df_filtered.columns:
            monthly_counts = df_filtered.groupby('Month')['Arrival date'].count().reindex(range(1, 13), fill_value=0)
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            monthly_df = pd.DataFrame({'Month': month_names, 'Reservations': monthly_counts.values})
            fig_monthly = px.bar(
                monthly_df,
                x='Month',
                y='Reservations',
                title="📅 Number of Reservations Per Month",
                color='Reservations',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Total Revenue by Month
        if "total rate net" in df_filtered.columns:
            monthly_revenue = df_filtered.groupby('Month')['total rate net'].sum().reindex(range(1, 13)).reset_index()
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            monthly_revenue['Month Name'] = month_names
            fig_monthly_rev = px.bar(
                monthly_revenue,
                x='Month Name',
                y='total rate net',
                text='total rate net',
                title='💰 Total Revenue by Month',
                color='total rate net',
                color_continuous_scale='Viridis'
            )
            fig_monthly_rev.update_traces(textposition='outside')
            st.plotly_chart(fig_monthly_rev, use_container_width=True)
    
    # Average ADR over Time
    if "ADR" in df_filtered.columns:
        adr_by_date = df_filtered.groupby('Arrival date')['ADR'].mean().reset_index()
        fig_adr = px.line(
            adr_by_date,
            x='Arrival date',
            y='ADR',
            title="💹 Average ADR Over Time",
            markers=True
        )
        st.plotly_chart(fig_adr, use_container_width=True)
    
    # Year-over-Year Comparison
    if "Year" in df_filtered.columns and len(df_filtered['Year'].unique()) > 1:
        yearly_comparison = df_filtered.groupby('Year').agg({
            'total rate net': 'sum',
            'Room night': 'sum',
            'ADR': 'mean'
        }).reset_index()
        
        fig_yearly = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total Revenue by Year', 'Total Room Nights by Year', 'Average ADR by Year'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_yearly.add_trace(
            go.Bar(x=yearly_comparison['Year'], y=yearly_comparison['total rate net'], name='Revenue'),
            row=1, col=1
        )
        fig_yearly.add_trace(
            go.Bar(x=yearly_comparison['Year'], y=yearly_comparison['Room night'], name='Room Nights'),
            row=1, col=2
        )
        fig_yearly.add_trace(
            go.Bar(x=yearly_comparison['Year'], y=yearly_comparison['ADR'], name='ADR'),
            row=1, col=3
        )
        
        fig_yearly.update_layout(height=400, showlegend=False, title_text="Year-over-Year Performance Comparison")
        st.plotly_chart(fig_yearly, use_container_width=True)

with tab3:
    st.subheader("🏨 Room Types & Travel Agents Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Room Type Analysis
        if "Room Type" in df_filtered.columns:
            room_count = df_filtered['Room Type'].value_counts().reset_index()
            room_count.columns = ['Room Type', 'Count']
            fig_room = px.bar(
                room_count,
                x='Room Type',
                y='Count',
                text='Count',
                title="🏨 Most Booked Room Types",
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_room.update_traces(textposition='outside')
            st.plotly_chart(fig_room, use_container_width=True)
    
    with col2:
        # Top Travel Agents Revenue
        if "Travel Agent" in df_filtered.columns:
            travel_agent_rev = df_filtered.groupby('Travel Agent')['total rate net'].sum().nlargest(10).reset_index()
            fig_agents = px.bar(
                travel_agent_rev,
                x='Travel Agent',
                y='total rate net',
                title="🏆 Top 10 Travel Agents by Revenue",
                color='total rate net',
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_agents, use_container_width=True)
    
    # Room Type Performance Table
    if "Room Type" in df_filtered.columns:
        st.subheader("📊 Room Type Performance Summary")
        room_kpi = df_filtered.groupby('Room Type').agg({
            'total rate net': 'sum',
            'Room night': 'sum',
            'ADR': 'mean'
        }).sort_values(by='total rate net', ascending=False).round(2)
        room_kpi.columns = ['Total Revenue', 'Total Room Nights', 'Average ADR']
        st.dataframe(room_kpi, use_container_width=True)

with tab4:
    st.subheader("💹 Revenue Analysis & Trends")
    
    # Monthly Revenue Trend with Growth
    if 'Arrival date' in df_filtered.columns:
        df_filtered['YearMonth'] = df_filtered['Arrival date'].dt.to_period('M').astype(str)
        monthly_trend = df_filtered.groupby('YearMonth')['total rate net'].sum().reset_index()
        monthly_trend['MoM Growth %'] = monthly_trend['total rate net'].pct_change() * 100
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Scatter(x=monthly_trend['YearMonth'], y=monthly_trend['total rate net'], 
                      name='Total Revenue', mode='lines+markers'),
            secondary_y=False,
        )
        
        fig_trend.add_trace(
            go.Scatter(x=monthly_trend['YearMonth'], y=monthly_trend['MoM Growth %'], 
                      name='MoM Growth %', mode='lines+markers', line=dict(dash='dash')),
            secondary_y=True,
        )
        
        fig_trend.update_xaxes(title_text="Month")
        fig_trend.update_yaxes(title_text="Total Revenue", secondary_y=False)
        fig_trend.update_yaxes(title_text="MoM Growth %", secondary_y=True)
        fig_trend.update_layout(title_text="📊 Monthly Revenue Trend with Growth Rate")
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Revenue Distribution by Market Segments
    if 'Market' in df_filtered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly revenue by market type
            market_monthly = df_filtered.groupby(['YearMonth', 'Market'])['total rate net'].sum().reset_index()
            fig_market_trend = px.line(
                market_monthly,
                x='YearMonth',
                y='total rate net',
                color='Market',
                title='📈 Monthly Revenue by Market Type',
                markers=True
            )
            st.plotly_chart(fig_market_trend, use_container_width=True)
        
        with col2:
            # Market performance comparison
            market_performance = df_filtered.groupby('Market').agg({
                'total rate net': 'sum',
                'Room night': 'sum',
                'ADR': 'mean'
            }).reset_index()
            
            fig_market_perf = px.bar(
                market_performance,
                x='Market',
                y='total rate net',
                title='💰 Total Revenue by Market Type',
                color='total rate net',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_market_perf, use_container_width=True)

# ==========================================
# Performance Summary Tables
# ==========================================
st.markdown("---")
st.subheader("📋 **Performance Summary Tables**")

col1, col2, col3 = st.columns(3)

with col1:
    if "Country" in df_filtered.columns:
        st.write("**Top Countries Performance**")
        country_kpi = df_filtered.groupby('Country').agg({
            'total rate net': 'sum',
            'Room night': 'sum',
            'ADR': 'mean'
        }).sort_values(by='total rate net', ascending=False).head(10).round(2)
        country_kpi.columns = ['Revenue', 'Room Nights', 'Avg ADR']
        st.dataframe(country_kpi)

with col2:
    if "Travel Agent" in df_filtered.columns:
        st.write("**Top Travel Agents Performance**")
        agent_kpi = df_filtered.groupby('Travel Agent').agg({
            'total rate net': 'sum',
            'Room night': 'sum',
            'ADR': 'mean'
        }).sort_values(by='total rate net', ascending=False).head(10).round(2)
        agent_kpi.columns = ['Revenue', 'Room Nights', 'Avg ADR']
        st.dataframe(agent_kpi)

with col3:
    if "Month" in df_filtered.columns:
        st.write("**Monthly Performance Summary**")
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly_kpi = df_filtered.groupby('Month').agg({
            'total rate net': 'sum',
            'Room night': 'sum',
            'ADR': 'mean'
        }).round(2)
        monthly_kpi.index = [month_names[i-1] for i in monthly_kpi.index]
        monthly_kpi.columns = ['Revenue', 'Room Nights', 'Avg ADR']
        st.dataframe(monthly_kpi)

