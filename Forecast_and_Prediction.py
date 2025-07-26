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
warnings.filterwarnings('ignore')
import os

st.set_page_config(
    page_title="Forecast and Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 **Forecast and Prediction (ML Machine Learning)**")
st.markdown("### Advanced forecasting and predictive modeling for hotel performance.")

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
        
        # RMSE comparison
        fig_comparison.add_trace(
            go.Bar(
                x=monthly_comparison['arrival_month_year'],
                y=monthly_comparison['Total_RMSE'],
                name='Total Rate RMSE',
                marker_color='blue'
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

# ==========================================
# Feature Importance
# ==========================================
st.subheader("🎯 **Feature Importance Analysis**")

if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
    importances = best_pipeline.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for ADR Prediction',
        color='Importance',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.write("**Top 3 Most Important Features:**")
    for i, row in feature_importance_df.head(3).iterrows():
        st.write(f"{i+1}. **{row['Feature']}**: {row['Importance']:.3f}")

