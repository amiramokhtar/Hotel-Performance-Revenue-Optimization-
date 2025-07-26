import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")
import os

# Helper function for safe float conversion
def safe_float(value):
    """Safely convert value to float, return 0.0 if conversion fails"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Hotel Performance & Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Main Title and Description
# ----------------------------
st.title("🏨 **Hotel Performance & Analytics Dashboard**")
st.markdown("### Comprehensive hotel analytics with machine learning forecasting and revenue optimization")

# ----------------------------
# Data Upload and Loading
# ----------------------------
st.sidebar.header("📁 **Data Management**")

DATA_FILE = "Cleaned_Hotel_Booking.csv"
df = None

# Check if data exists in session state
if 'df' not in st.session_state:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.sidebar.success(f"✅ Loaded data from {DATA_FILE}")
    else:
        st.sidebar.warning(f"⚠️ {DATA_FILE} not found. Please upload your dataset below.")
        uploaded_file = st.sidebar.file_uploader("Upload Hotel Booking CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Data loaded from uploaded file.")
        else:
            st.warning("📌 **Please upload hotel data to start using the dashboard.**")
            st.info("The dashboard requires a CSV file with hotel booking data including columns like 'Arrival date', 'ADR', 'Room night', 'total rate net', etc.")
            st.stop()
    
    # Store data in session state
    st.session_state.df = df
else:
    df = st.session_state.df
    st.sidebar.success("✅ Data loaded from session")

# Convert date columns if available
if 'Arrival date' in df.columns:
    df['Arrival date'] = pd.to_datetime(df['Arrival date'], errors='coerce')
    df['Year'] = df['Arrival date'].dt.year
    df['Month'] = df['Arrival date'].dt.month
    df['Month_Year'] = df['Arrival date'].dt.to_period('M').astype(str)
else:
    st.error("❌ **'Arrival date' column is missing in the dataset.**")
    st.stop()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 **Global Filters**")
st.sidebar.markdown("*These filters apply to all pages*")

# Get unique values for filters
years = sorted(df['Year'].dropna().unique()) if 'Year' in df.columns else []
agents = sorted(df['Travel Agent'].dropna().unique()) if 'Travel Agent' in df.columns else []
countries = sorted(df['Country'].dropna().unique()) if 'Country' in df.columns else []
rooms = sorted(df['Room Type'].dropna().unique()) if 'Room Type' in df.columns else []

# Filter controls with "Select All" functionality
col1, col2 = st.sidebar.columns(2)

with col1:
    select_all_years = st.checkbox("Select All Years", value=True)
with col2:
    if select_all_years:
        selected_year = years
    else:
        selected_year = st.multiselect("Select Year", years, default=[])

col1, col2 = st.sidebar.columns(2)
with col1:
    select_all_agents = st.checkbox("Select All Agents", value=True)
with col2:
    if select_all_agents:
        selected_agent = agents
    else:
        selected_agent = st.multiselect("Select Travel Agent", agents, default=[])

col1, col2 = st.sidebar.columns(2)
with col1:
    select_all_countries = st.checkbox("Select All Countries", value=True)
with col2:
    if select_all_countries:
        selected_country = countries
    else:
        selected_country = st.multiselect("Select Country", countries, default=[])

col1, col2 = st.sidebar.columns(2)
with col1:
    select_all_rooms = st.checkbox("Select All Room Types", value=True)
with col2:
    if select_all_rooms:
        selected_room = rooms
    else:
        selected_room = st.multiselect("Select Room Type", rooms, default=[])

# Apply filters
df_filtered = df.copy()
if selected_year:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_year)]
if selected_agent:
    df_filtered = df_filtered[df_filtered['Travel Agent'].isin(selected_agent)]
if selected_country:
    df_filtered = df_filtered[df_filtered['Country'].isin(selected_country)]
if selected_room:
    df_filtered = df_filtered[df_filtered['Room Type'].isin(selected_room)]

# Store filtered data in session state
st.session_state.df_filtered = df_filtered

# ----------------------------
# Data Overview
# ----------------------------
st.subheader("📊 **Data Overview**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df_filtered):,}")
with col2:
    date_range = f"{df_filtered['Arrival date'].min().strftime('%Y-%m-%d')} to {df_filtered['Arrival date'].max().strftime('%Y-%m-%d')}"
    st.metric("Date Range", date_range)
with col3:
    st.metric("Countries", f"{df_filtered['Country'].nunique()}")
with col4:
    st.metric("Room Types", f"{df_filtered['Room Type'].nunique()}")

# ----------------------------
# Quick Stats
# ----------------------------
st.subheader("📈 **Quick Performance Stats**")

col1, col2, col3, col4, col5 = st.columns(5)

total_revenue = df_filtered['total rate net'].sum() if 'total rate net' in df_filtered.columns else 0
total_nights = df_filtered['Room night'].sum() if 'Room night' in df_filtered.columns else 0
avg_adr = df_filtered['ADR'].mean() if 'ADR' in df_filtered.columns else 0
revpar = total_revenue / total_nights if total_nights > 0 else 0
total_bookings = len(df_filtered)

with col1:
    st.metric("💰 Total Revenue", f"{total_revenue:,.0f} EGP")
with col2:
    st.metric("🛏 Room Nights", f"{total_nights:,}")
with col3:
    st.metric("🏷 Avg ADR", f"{avg_adr:,.2f} EGP")
with col4:
    st.metric("📈 RevPAR", f"{revpar:,.2f} EGP")
with col5:
    st.metric("📊 Bookings", f"{total_bookings:,}")

# ----------------------------
# Navigation Instructions
# ----------------------------
st.markdown("---")
st.subheader("🧭 **Navigation Guide**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🏨 **Hotel Performance**
    - Key performance indicators
    - Geographic and temporal analysis
    - Room types and agent performance
    - Revenue trends and insights
    """)

with col2:
    st.markdown("""
    ### 🔮 **Forecast and Prediction**
    - Machine learning models
    - ADR prediction and forecasting
    - Prophet time series analysis
    - Model performance comparison
    """)

with col3:
    st.markdown("""
    ### ⚙️ **Optimize Revenue**
    - Dynamic pricing optimization
    - Demand forecasting tools
    - Room mix optimization
    - Performance benchmarking
    """)

# ----------------------------
# Quick Data Preview
# ----------------------------
with st.expander("📋 **Data Preview**", expanded=False):
    st.write("**Filtered Dataset Preview:**")
    st.dataframe(df_filtered.head(10), use_container_width=True)
    
    st.write("**Dataset Info:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Column Names:**")
        st.write(list(df_filtered.columns))
    
    with col2:
        st.write("**Data Types:**")
        st.write(df_filtered.dtypes.to_dict())

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏨 Hotel Performance & Analytics Dashboard | Built with Streamlit and Machine Learning</p>
    <p>Use the sidebar to navigate between different analysis pages and apply filters</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Instructions for using the app
# ----------------------------
st.info("""
**💡 How to use this dashboard:**

1. **Upload Data**: Use the sidebar to upload your hotel booking CSV file
2. **Apply Filters**: Use the global filters in the sidebar to focus on specific data segments
3. **Navigate Pages**: Use the page navigation in the sidebar to explore different analysis sections:
   - **Hotel Performance**: Comprehensive performance analysis and KPIs
   - **Forecast and Prediction**: Machine learning models and forecasting
   - **Optimize Revenue**: Revenue optimization tools and strategies
4. **Interactive Charts**: All charts are interactive - hover, zoom, and click to explore
5. **Export Data**: Use the data preview section to examine your filtered dataset
""")

# ----------------------------
# Dynamic Page Content
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Hotel Performance", "Forecast and Prediction", "Optimize Revenue"])

if page == "Hotel Performance":
    st.title("🏨 **Hotel Performance Analysis**")
    st.markdown("### Comprehensive analysis of hotel key performance indicators and trends.")

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
                # Fix: Use update_layout instead of update_xaxis
                fig_room.update_layout(xaxis={'tickangle': 45})
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
                # Fix: Use update_layout instead of update_xaxis
                fig_agents.update_layout(xaxis={'tickangle': 45})
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

elif page == "Forecast and Prediction":
    st.title("🔮 **Forecast and Prediction (ML Machine Learning)**")
    st.markdown("### Advanced forecasting and predictive modeling for hotel performance.")

    # ==========================================
    # Machine Learning Section
    # ==========================================
    st.subheader("🤖 **Machine Learning Models for ADR Prediction**")

    # Prepare data for ML
    ml_df = df_filtered.copy()
    required_cols = ['ADR', 'Room night', 'total rate net', 'Year', 'Month']

    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in ml_df.columns]
    if missing_cols:
        st.error(f"Missing required columns for ML: {missing_cols}")
        st.stop()

    # Drop rows with NaN in critical columns for ML
    ml_df = ml_df.dropna(subset=['ADR', 'Room night', 'total rate net'])

    if len(ml_df) < 10:
        st.error("Not enough data for machine learning analysis. Need at least 10 records.")
        st.stop()

    # Feature Engineering
    ml_df['day_of_week'] = ml_df['Arrival date'].dt.dayofweek if 'Arrival date' in ml_df.columns else 0
    ml_df['day_of_year'] = ml_df['Arrival date'].dt.dayofyear if 'Arrival date' in ml_df.columns else 1
    ml_df['arrival_month_year'] = ml_df['Arrival date'].dt.to_period('M').astype(str) if 'Arrival date' in ml_df.columns else '2023-01'

    # Encode categorical features
    categorical_cols = ['Country', 'Travel Agent', 'Room Type']
    label_encoders = {}

    for col in categorical_cols:
        if col in ml_df.columns:
            ml_df[col] = ml_df[col].astype(str)
            le = LabelEncoder()
            ml_df[col] = le.fit_transform(ml_df[col])
            label_encoders[col] = le

    # Define features (X) and target (y)
    feature_cols = ['Room night', 'total rate net', 'Year', 'Month', 'day_of_week', 'day_of_year']
    if 'Country' in ml_df.columns:
        feature_cols.append('Country')
    if 'Travel Agent' in ml_df.columns:
        feature_cols.append('Travel Agent')
    if 'Room Type' in ml_df.columns:
        feature_cols.append('Room Type')

    X = ml_df[feature_cols]
    y = ml_df['ADR']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # Model Training and Comparison
    # ==========================================
    st.subheader("📊 **Model Training and Performance Comparison**")

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    model_results = {}
    best_model = None
    best_score = float('inf')

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Performance Metrics:**")
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'pipeline': pipeline,
                'predictions': y_pred
            }
            
            if rmse < best_score:
                best_score = rmse
                best_model = name
            
            st.write(f"**{name}:**")
            st.write(f"- RMSE: {rmse:.2f}")
            st.write(f"- MAE: {mae:.2f}")
            st.write(f"- R²: {r2:.3f}")
            st.write("---")

    with col2:
        # Model comparison chart
        metrics_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'RMSE': [model_results[m]['RMSE'] for m in model_results.keys()],
            'MAE': [model_results[m]['MAE'] for m in model_results.keys()],
            'R2': [model_results[m]['R2'] for m in model_results.keys()]
        })
        
        fig_metrics = px.bar(
            metrics_df,
            x='Model',
            y='RMSE',
            title='Model Performance Comparison (RMSE)',
            color='RMSE',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    st.success(f"🏆 **Best Model: {best_model}** (Lowest RMSE: {best_score:.2f})")

    # ==========================================
    # Prediction Example
    # ==========================================
    st.subheader("📊 **Prediction Example**")

    best_pipeline = model_results[best_model]['pipeline']

    if not X_test.empty:
        sample = X_test.iloc[[0]]
        predicted_ADR = best_pipeline.predict(sample)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sample Input:**")
            sample_display = sample.copy()
            for col in sample_display.columns:
                if col in label_encoders:
                    # Decode categorical variables for display
                    sample_display[col] = label_encoders[col].inverse_transform(sample_display[col])
            st.dataframe(sample_display)
        
        with col2:
            st.write("**Prediction Result:**")
            st.metric("Predicted ADR", f"{predicted_ADR[0]:.2f} EGP")
            actual_adr = y_test.iloc[0]
            st.metric("Actual ADR", f"{actual_adr:.2f} EGP")
            error = abs(predicted_ADR[0] - actual_adr)
            st.metric("Prediction Error", f"{error:.2f} EGP")

    # ==========================================
    # Model Performance Visualization
    # ==========================================
    st.subheader("📈 **Model Performance Visualization**")

    if not X_test.empty:
        y_pred = best_pipeline.predict(X_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot for Actual vs Predicted
            fig_scatter = px.scatter(
                x=y_test, 
                y=y_pred,
                labels={'x': 'Actual ADR', 'y': 'Predicted ADR'},
                title="Actual vs Predicted ADR"
            )
            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Residuals distribution
            residuals = y_test - y_pred
            fig_residuals = px.histogram(
                x=residuals,
                nbins=20,
                title="Residuals Distribution",
                labels={'x': 'Residuals', 'y': 'Frequency'}
            )
            fig_residuals.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)

    # ==========================================
    # Forecast Visualization by arrival_month_year
    # ==========================================
    st.subheader("🗓️ **Forecast Visualization by Month**")

    if 'arrival_month_year' in ml_df.columns and not X_test.empty:
        # Create forecast DataFrame
        forecast_df = X_test.copy()
        forecast_df["Actual_ADR"] = y_test.values
        forecast_df["Predicted_ADR"] = best_pipeline.predict(X_test)
        
        # Add arrival_month_year back to forecast_df
        test_indices = X_test.index
        forecast_df['arrival_month_year'] = ml_df.loc[test_indices, 'arrival_month_year'].values
        
        # Sort by arrival_month_year
        forecast_df = forecast_df.sort_values(by='arrival_month_year')
        
        # Aggregate by month for cleaner visualization
        monthly_forecast = forecast_df.groupby('arrival_month_year').agg({
            'Actual_ADR': 'mean',
            'Predicted_ADR': 'mean'
        }).reset_index()
        
        # Plot using Plotly
        fig_forecast = px.line(
            monthly_forecast, 
            x='arrival_month_year', 
            y=['Actual_ADR', 'Predicted_ADR'],
            labels={"value": "ADR", "variable": "Legend", "arrival_month_year": "Month"},
            title="Monthly ADR Forecast vs Actual"
        )
        fig_forecast.update_layout(template="plotly_white", legend_title="ADR Type")
        st.plotly_chart(fig_forecast, use_container_width=True)

    # ==========================================
    # Prophet Forecasting
    # ==========================================
    st.subheader("📈 **Prophet Time Series Forecasting**")

    if 'Arrival date' in df_filtered.columns:
        # Prepare data for Prophet
        prophet_data = df_filtered.groupby('Arrival date')['ADR'].mean().reset_index()
        prophet_data.columns = ['ds', 'y']
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) > 10:
            col1, col2 = st.columns(2)
            
            with col1:
                forecast_periods = st.slider("Forecast Periods (days)", 30, 365, 90)
            
            with col2:
                if st.button("Generate Prophet Forecast"):
                    with st.spinner("Training Prophet model..."):
                        # Train Prophet model
                        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                        model.fit(prophet_data)
                        
                        # Make future dataframe
                        future = model.make_future_dataframe(periods=forecast_periods)
                        forecast = model.predict(future)
                        
                        # Plot forecast
                        fig_prophet = go.Figure()
                        
                        # Historical data
                        fig_prophet.add_trace(go.Scatter(
                            x=prophet_data['ds'],
                            y=prophet_data['y'],
                            mode='markers',
                            name='Historical ADR',
                            marker=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig_prophet.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Confidence intervals
                        fig_prophet.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_upper'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig_prophet.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_lower'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='Confidence Interval',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
                        
                        fig_prophet.update_layout(
                            title='ADR Forecast using Prophet',
                            xaxis_title='Date',
                            yaxis_title='ADR',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_prophet, use_container_width=True)
                        
                        # Show forecast statistics
                        future_forecast = forecast[forecast['ds'] > prophet_data['ds'].max()]
                        st.write("**Forecast Summary:**")
                        st.write(f"- Average Predicted ADR: {future_forecast['yhat'].mean():.2f} EGP")
                        st.write(f"- Minimum Predicted ADR: {future_forecast['yhat'].min():.2f} EGP")
                        st.write(f"- Maximum Predicted ADR: {future_forecast['yhat'].max():.2f} EGP")
        else:
            st.warning("Not enough historical data for Prophet forecasting. Need at least 10 data points.")

    # ==========================================
    # Monthly Model Comparison
    # ==========================================
    st.subheader("📊 **Monthly Model Comparison**")

    if 'arrival_month_year' in ml_df.columns:
        # Create monthly comparison
        monthly_data = ml_df.groupby('arrival_month_year').agg({
            'total rate net': 'sum',
            'ADR': 'mean',
            'Room night': 'sum'
        }).reset_index()
        
        if len(monthly_data) > 3:
            # Split monthly data for training
            monthly_X = monthly_data[['ADR', 'Room night']].fillna(0)
            monthly_y = monthly_data['total rate net']
            
            # Train models on monthly data
            monthly_models = {
                'Total Rate Model': RandomForestRegressor(n_estimators=50, random_state=42),
                'RevPAR Model': GradientBoostingRegressor(n_estimators=50, random_state=42)
            }
            
            monthly_comparison = monthly_data.copy()
            
            for name, model in monthly_models.items():
                model.fit(monthly_X, monthly_y)
                predictions = model.predict(monthly_X)
                monthly_comparison[f'Predicted_{name.split()[0]}'] = predictions
                rmse = np.sqrt(mean_squared_error(monthly_y, predictions))
                monthly_comparison[f'{name.split()[0]}_RMSE'] = rmse
            
            # Determine best model for each month
            monthly_comparison['Best_Model'] = monthly_comparison.apply(
                lambda row: 'Total Rate' if row['Total_RMSE'] < row['RevPAR_RMSE'] else 'RevPAR', axis=1
            )
            
            # Visualization
            fig_comparison = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=False,
                vertical_spacing=0.1,
                subplot_titles=("Monthly Actual vs Predicted Revenue", "Model Performance Metrics")
            )
            
            # Line chart
            fig_comparison.add_trace(
                go.Scatter(
                    x=monthly_comparison['arrival_month_year'],
                    y=monthly_comparison['total rate net'],
                    mode='lines+markers',
                    name='Actual Total Revenue',
                    line=dict(color='black', width=3)
                ),
                row=1, col=1
            )
            
            fig_comparison.add_trace(
                go.Scatter(
                    x=monthly_comparison['arrival_month_year'],
                    y=monthly_comparison['Predicted_Total'],
                    mode='lines+markers',
                    name='Predicted Total Rate',
                    line=dict(color='blue', dash='dot')
                ),
                row=1, col=1
            )
            
            fig_comparison.add_trace(
                go.Scatter(
                    x=monthly_comparison['arrival_month_year'],
                    y=monthly_comparison['Predicted_RevPAR'],
                    mode='lines+markers',
                    name='Predicted RevPAR',
                    line=dict(color='green', dash='dot')
                ),
                row=1, col=1
            )
            
            # Bar chart for RMSE
            fig_comparison.add_trace(
                go.Bar(
                    x=monthly_comparison['arrival_month_year'],
                    y=monthly_comparison['Total_RMSE'],
                    name='Total Rate RMSE',
                    marker_color='orange'
                ),
                row=2, col=1
            )
            
            fig_comparison.add_trace(
                go.Bar(
                    x=monthly_comparison['arrival_month_year'],
                    y=monthly_comparison['RevPAR_RMSE'],
                    name='RevPAR RMSE',
                    marker_color='green'
                ),
                row=2, col=1
            )
            
            fig_comparison.update_layout(
                height=800,
                title_text="Monthly Revenue Forecast Comparison",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_total_rmse = monthly_comparison['Total_RMSE'].mean()
                st.metric("Avg Total Rate RMSE", f"{avg_total_rmse:.2f}")
            
            with col2:
                avg_revpar_rmse = monthly_comparison['RevPAR_RMSE'].mean()
                st.metric("Avg RevPAR RMSE", f"{avg_revpar_rmse:.2f}")
            
            with col3:
                best_model_overall = monthly_comparison['Best_Model'].mode()[0]
                st.metric("Overall Best Model", best_model_overall)
            
            # Display comparison table
            st.subheader("📋 **Monthly Model Comparison Table**")
            display_cols = ['arrival_month_year', 'total rate net', 'Predicted_Total', 'Predicted_RevPAR', 'Total_RMSE', 'RevPAR_RMSE', 'Best_Model']
            comparison_display = monthly_comparison[display_cols].round(2)
            comparison_display.columns = ['Month', 'Actual Total', 'Predicted Total', 'Predicted RevPAR', 'Total RMSE', 'RevPAR RMSE', 'Best Model']
            st.dataframe(comparison_display, use_container_width=True)

elif page == "Optimize Revenue":
    st.title("⚙️ **Optimize Revenue**")
    st.markdown("### Advanced tools and insights for optimizing hotel revenue strategies.")

    # ==========================================
    # Revenue Optimization Tools
    # ==========================================
    st.subheader("💰 **Revenue Optimization Tools**")

    # Create tabs for different optimization strategies
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Price Optimization", "📊 Demand Forecasting", "🏨 Room Mix Optimization", "📈 Performance Analytics"])

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
            
            # Apply market and seasonal adjustments
            adjusted_occupancy = safe_float(occupancy_rate) * safe_float(season_factor) * safe_float(competition_factor) / 100
            adjusted_occupancy = np.clip(adjusted_occupancy, 0.1, 1.0)  # Keep within realistic bounds
            
            # Calculate revenues and other metrics
            revenues = prices * adjusted_occupancy * room_nights
            revpars = revenues / room_nights
            
            # Calculate demand elasticity (simplified model)
            demand_elasticity = 1 - (prices - price_range[0]) / (price_range[1] - price_range[0]) * 0.5
            adjusted_demand = demand_elasticity * adjusted_occupancy
            
            df_opt = pd.DataFrame({
                "Price": prices, 
                "Revenue": revenues, 
                "RevPAR": revpars,
                "Demand": adjusted_demand * 100,
                "Occupancy": adjusted_occupancy * 100
            })
            
            # Find optimal price
            optimal_row = df_opt.loc[df_opt['Revenue'].idxmax()]
            
            # Display optimal results
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Optimal Price", f"{optimal_row['Price']:.0f} EGP")
            with col2_2:
                st.metric("Max Revenue", f"{optimal_row['Revenue']:,.0f} EGP")
            with col2_3:
                st.metric("Optimal RevPAR", f"{optimal_row['RevPAR']:.2f} EGP")
            
            # Price vs Revenue chart
            fig_price = px.line(df_opt, x="Price", y="Revenue", markers=True, 
                               title="Price vs Revenue Optimization")
            fig_price.add_vline(x=optimal_row['Price'], line_dash="dash", line_color="red",
                               annotation_text=f"Optimal: {optimal_row['Price']:.0f} EGP")
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Multi-metric analysis
            fig_multi = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price vs Revenue', 'Price vs RevPAR', 'Price vs Demand', 'Price vs Occupancy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_multi.add_trace(go.Scatter(x=df_opt['Price'], y=df_opt['Revenue'], mode='lines+markers', name='Revenue'), row=1, col=1)
            fig_multi.add_trace(go.Scatter(x=df_opt['Price'], y=df_opt['RevPAR'], mode='lines+markers', name='RevPAR'), row=1, col=2)
            fig_multi.add_trace(go.Scatter(x=df_opt['Price'], y=df_opt['Demand'], mode='lines+markers', name='Demand'), row=2, col=1)
            fig_multi.add_trace(go.Scatter(x=df_opt['Price'], y=df_opt['Occupancy'], mode='lines+markers', name='Occupancy'), row=2, col=2)
            
            fig_multi.update_layout(height=600, showlegend=False, title_text="Comprehensive Price Analysis")
            st.plotly_chart(fig_multi, use_container_width=True)

    with tab2:
        st.subheader("📊 **Demand Forecasting & Capacity Planning**")
        
        if 'Arrival date' in df_filtered.columns:
            # Historical demand analysis
            daily_demand = df_filtered.groupby('Arrival date').agg({
                'Room night': 'sum',
                'total rate net': 'sum',
                'ADR': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily demand trend
                fig_demand = px.line(daily_demand, x='Arrival date', y='Room night',
                                   title='Historical Daily Demand (Room Nights)')
                st.plotly_chart(fig_demand, use_container_width=True)
            
            with col2:
                # Revenue trend
                fig_revenue_trend = px.line(daily_demand, x='Arrival date', y='total rate net',
                                          title='Historical Daily Revenue')
                st.plotly_chart(fig_revenue_trend, use_container_width=True)
            
            # Seasonal demand patterns
            df_filtered['DayOfWeek'] = df_filtered['Arrival date'].dt.day_name()
            df_filtered['WeekOfYear'] = df_filtered['Arrival date'].dt.isocalendar().week
            
            # Day of week analysis
            dow_demand = df_filtered.groupby('DayOfWeek')['Room night'].sum().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig_dow = px.bar(x=dow_demand.index, y=dow_demand.values,
                            title='Demand by Day of Week',
                            labels={'x': 'Day of Week', 'y': 'Total Room Nights'})
            st.plotly_chart(fig_dow, use_container_width=True)
            
            # Monthly seasonality
            monthly_demand = df_filtered.groupby('Month').agg({
                'Room night': 'sum',
                'total rate net': 'sum',
                'ADR': 'mean'
            }).reset_index()
            
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            monthly_demand['Month_Name'] = [month_names[i-1] for i in monthly_demand['Month']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_monthly_demand = px.bar(monthly_demand, x='Month_Name', y='Room night',
                                          title='Monthly Demand Seasonality')
                st.plotly_chart(fig_monthly_demand, use_container_width=True)
            
            with col2:
                fig_monthly_adr = px.bar(monthly_demand, x='Month_Name', y='ADR',
                                       title='Monthly ADR Seasonality')
                st.plotly_chart(fig_monthly_adr, use_container_width=True)
            
            # Capacity utilization analysis
            st.subheader("🏨 **Capacity Utilization Analysis**")
            
            total_rooms = st.number_input("Total Hotel Rooms", min_value=1, value=100)
            
            # Calculate occupancy rate
            daily_demand['Occupancy_Rate'] = (daily_demand['Room night'] / total_rooms) * 100
            daily_demand['Occupancy_Rate'] = daily_demand['Occupancy_Rate'].clip(0, 100)
            
            # Occupancy distribution
            fig_occupancy = px.histogram(daily_demand, x='Occupancy_Rate', nbins=20,
                                       title='Occupancy Rate Distribution')
            fig_occupancy.add_vline(x=daily_demand['Occupancy_Rate'].mean(), 
                                  line_dash="dash", line_color="red",
                                  annotation_text=f"Avg: {daily_demand['Occupancy_Rate'].mean():.1f}%")
            st.plotly_chart(fig_occupancy, use_container_width=True)
            
            # Capacity metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_occupancy = daily_demand['Occupancy_Rate'].mean()
                st.metric("Average Occupancy", f"{avg_occupancy:.1f}%")
            
            with col2:
                max_occupancy = daily_demand['Occupancy_Rate'].max()
                st.metric("Peak Occupancy", f"{max_occupancy:.1f}%")
            
            with col3:
                low_occupancy_days = (daily_demand['Occupancy_Rate'] < 50).sum()
                st.metric("Low Occupancy Days (<50%)", low_occupancy_days)
            
            with col4:
                high_occupancy_days = (daily_demand['Occupancy_Rate'] > 80).sum()
                st.metric("High Occupancy Days (>80%)", high_occupancy_days)

    with tab3:
        st.subheader("🏨 **Room Mix Optimization**")
        
        if 'Room Type' in df_filtered.columns:
            # Room type performance analysis
            room_performance = df_filtered.groupby('Room Type').agg({
                'total rate net': 'sum',
                'Room night': 'sum',
                'ADR': 'mean'
            }).reset_index()
            
            room_performance['RevPAR'] = room_performance['total rate net'] / room_performance['Room night']
            room_performance['Revenue_Share'] = (room_performance['total rate net'] / room_performance['total rate net'].sum()) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue by room type
                fig_room_revenue = px.pie(room_performance, values='total rate net', names='Room Type',
                                        title='Revenue Distribution by Room Type')
                st.plotly_chart(fig_room_revenue, use_container_width=True)
            
            with col2:
                # ADR by room type
                fig_room_adr = px.bar(room_performance, x='Room Type', y='ADR',
                                    title='Average ADR by Room Type',
                                    color='ADR', color_continuous_scale='viridis')
                # Fix: Use update_layout instead of update_xaxis
                fig_room_adr.update_layout(xaxis={'tickangle': 45})
                st.plotly_chart(fig_room_adr, use_container_width=True)
            
            # Room type performance table
            st.subheader("📊 **Room Type Performance Summary**")
            room_display = room_performance.round(2)
            room_display.columns = ['Room Type', 'Total Revenue', 'Room Nights', 'Avg ADR', 'RevPAR', 'Revenue Share %']
            st.dataframe(room_display, use_container_width=True)
            
            # Room mix optimization recommendations
            st.subheader("💡 **Room Mix Optimization Recommendations**")
            
            # Find best and worst performing room types
            best_revpar = room_performance.loc[room_performance['RevPAR'].idxmax()]
            worst_revpar = room_performance.loc[room_performance['RevPAR'].idxmin()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Best Performing Room Type:**\n\n"
                          f"🏆 **{best_revpar['Room Type']}**\n"
                          f"- RevPAR: {best_revpar['RevPAR']:.2f} EGP\n"
                          f"- ADR: {best_revpar['ADR']:.2f} EGP\n"
                          f"- Revenue Share: {best_revpar['Revenue_Share']:.1f}%")
            
            with col2:
                st.warning(f"**Needs Improvement:**\n\n"
                          f"⚠️ **{worst_revpar['Room Type']}**\n"
                          f"- RevPAR: {worst_revpar['RevPAR']:.2f} EGP\n"
                          f"- ADR: {worst_revpar['ADR']:.2f} EGP\n"
                          f"- Revenue Share: {worst_revpar['Revenue_Share']:.1f}%")
            
            # Room type trends over time
            if 'Month' in df_filtered.columns:
                monthly_room_trends = df_filtered.groupby(['Month', 'Room Type'])['total rate net'].sum().reset_index()
                
                fig_room_trends = px.line(monthly_room_trends, x='Month', y='total rate net', 
                                        color='Room Type', title='Monthly Revenue Trends by Room Type',
                                        markers=True)
                st.plotly_chart(fig_room_trends, use_container_width=True)

    with tab4:
        st.subheader("📈 **Performance Analytics & Benchmarking**")
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue = df_filtered['total rate net'].sum()
        total_nights = df_filtered['Room night'].sum()
        avg_adr = df_filtered['ADR'].mean()
        revpar = total_revenue / total_nights if total_nights > 0 else 0
        
        with col1:
            st.metric("Total Revenue", f"{total_revenue:,.0f} EGP")
        with col2:
            st.metric("Total Room Nights", f"{total_nights:,}")
        with col3:
            st.metric("Average ADR", f"{avg_adr:.2f} EGP")
        with col4:
            st.metric("RevPAR", f"{revpar:.2f} EGP")
        
        # Performance trends
        if 'Arrival date' in df_filtered.columns:
            # Monthly performance trends
            monthly_performance = df_filtered.groupby(df_filtered['Arrival date'].dt.to_period('M')).agg({
                'total rate net': 'sum',
                'Room night': 'sum',
                'ADR': 'mean'
            }).reset_index()
            
            monthly_performance['RevPAR'] = monthly_performance['total rate net'] / monthly_performance['Room night']
            monthly_performance['Arrival date'] = monthly_performance['Arrival date'].dt.to_timestamp()
            
            # Multi-metric performance chart
            fig_performance = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Revenue', 'Monthly Room Nights', 'Monthly ADR', 'Monthly RevPAR'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_performance.add_trace(
                go.Scatter(x=monthly_performance['Arrival date'], y=monthly_performance['total rate net'], 
                          mode='lines+markers', name='Revenue'),
                row=1, col=1
            )
            fig_performance.add_trace(
                go.Scatter(x=monthly_performance['Arrival date'], y=monthly_performance['Room night'], 
                          mode='lines+markers', name='Room Nights'),
                row=1, col=2
            )
            fig_performance.add_trace(
                go.Scatter(x=monthly_performance['Arrival date'], y=monthly_performance['ADR'], 
                          mode='lines+markers', name='ADR'),
                row=2, col=1
            )
            fig_performance.add_trace(
                go.Scatter(x=monthly_performance['Arrival date'], y=monthly_performance['RevPAR'], 
                          mode='lines+markers', name='RevPAR'),
                row=2, col=2
            )
            
            fig_performance.update_layout(height=600, showlegend=False, title_text="Monthly Performance Trends")
            st.plotly_chart(fig_performance, use_container_width=True)
        
        # Market segment analysis
        if 'Country' in df_filtered.columns:
            df_filtered['Market'] = df_filtered['Country'].apply(
                lambda x: 'Local' if str(x).strip().lower() == 'egypt' else 'International'
            )
            
            market_performance = df_filtered.groupby('Market').agg({
                'total rate net': 'sum',
                'Room night': 'sum',
                'ADR': 'mean'
            }).reset_index()
            
            market_performance['RevPAR'] = market_performance['total rate net'] / market_performance['Room night']
            
            st.subheader("🎯 **Market Segment Performance**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_market_revenue = px.bar(market_performance, x='Market', y='total rate net',
                                          title='Revenue by Market Segment',
                                          color='total rate net', color_continuous_scale='blues')
                st.plotly_chart(fig_market_revenue, use_container_width=True)
            
            with col2:
                fig_market_revpar = px.bar(market_performance, x='Market', y='RevPAR',
                                         title='RevPAR by Market Segment',
                                         color='RevPAR', color_continuous_scale='greens')
                st.plotly_chart(fig_market_revpar, use_container_width=True)
        
        # Benchmarking
        st.subheader("🎯 **Performance Benchmarking**")
        
        # Calculate benchmarks
        if 'Country' in df_filtered.columns:
            country_benchmarks = df_filtered.groupby('Country').agg({
                'ADR': 'mean',
                'total rate net': 'sum',
                'Room night': 'sum'
            }).reset_index()
            
            country_benchmarks['RevPAR'] = country_benchmarks['total rate net'] / country_benchmarks['Room night']
            country_benchmarks = country_benchmarks.sort_values('RevPAR', ascending=False).head(10)
            
            fig_benchmark = px.bar(country_benchmarks, x='Country', y='RevPAR',
                                 title='RevPAR Benchmarking by Country (Top 10)')
            # Fix: Use update_layout instead of update_xaxis
            fig_benchmark.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_benchmark, use_container_width=True)
        
        # Performance insights
        st.subheader("💡 **Performance Insights**")
        
        insights = []
        
        # Revenue insights
        if total_revenue > 0:
            insights.append(f"• Total revenue generated: **{total_revenue:,.0f} EGP**")
        
        # ADR insights
        if avg_adr > 0:
            insights.append(f"• Average daily rate: **{avg_adr:.2f} EGP**")
        
        # RevPAR insights
        if revpar > 0:
            insights.append(f"• Revenue per available room: **{revpar:.2f} EGP**")
        
        # Market insights
        if 'Country' in df_filtered.columns:
            top_country = df_filtered['Country'].value_counts().index[0]
            insights.append(f"• Top performing market: **{top_country}**")
        
        # Room type insights
        if 'Room Type' in df_filtered.columns:
            top_room = df_filtered.groupby('Room Type')['total rate net'].sum().idxmax()
            insights.append(f"• Highest revenue room type: **{top_room}**")
        
        for insight in insights:
            st.write(insight)

