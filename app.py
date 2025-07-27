import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from collections import Counter
from predictor import load_and_train_model, predict_price

# Additional ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# Load the trained model
model = load_and_train_model("SA_Aqar.csv")

# Page Config
st.set_page_config(
    page_title="Saudi Lands Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    try:
        with open('styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("styles.css not found. Using default styling.")

load_css()

# Load Data
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["city"] = df["city"].str.strip()
    return df

df_original = load_data("SA_Aqar.csv")

# Sidebar
st.sidebar.header("📊 Dashboard Filters")
cities = sorted(df_original["city"].dropna().unique())
selected_city = st.sidebar.selectbox("🏙️ Select City", ["All"] + cities)

min_price, max_price = int(df_original["price"].min()), int(df_original["price"].max())
price_range = st.sidebar.slider(
    "💰 Price Range (SAR)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=1000
)

# Preprocessing (same as your original code)
df_with_outliers = df_original.drop_duplicates()
if 'details' in df_with_outliers.columns:
    df_with_outliers = df_with_outliers.drop(columns='details')
df_with_outliers = df_with_outliers[df_with_outliers['price'] >= 30000]
df_with_outliers['city'] = df_with_outliers['city'].str.strip()
df_with_outliers['city'] = df_with_outliers['city'].replace({
    'الرياض': 'Riyadh', 'جدة': 'Jeddah', 'الدمام': 'Dammam', 'الخبر': 'Khobar'
})

if 'front' in df_with_outliers.columns:
    df_with_outliers['front'] = df_with_outliers['front'].replace({
        'شمال': 'North', 'جنوب': 'South', 'شرق': 'East', 'غرب': 'West',
        'شمال غربي': 'Northwest', 'شمال شرقي': 'Northeast',
        'جنوب شرقي': 'Southeast', 'جنوب غربي': 'Southwest',
        '3 شوارع': 'Three Streets', '4 شوارع': 'Four Streets'
    })

# Outlier Detection
Q1_price = df_with_outliers['price'].quantile(0.25)
Q3_price = df_with_outliers['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

Q1_size = df_with_outliers['size'].quantile(0.25)
Q3_size = df_with_outliers['size'].quantile(0.75)
IQR_size = Q3_size - Q1_size
lower_size = Q1_size - 1.5 * IQR_size
upper_size = Q3_size + 1.5 * IQR_size

outlier_mask = (
    (df_with_outliers['price'] < lower_price) | (df_with_outliers['price'] > upper_price) |
    (df_with_outliers['size'] < lower_size) | (df_with_outliers['size'] > upper_size)
)

# Outlier separation
df_outliers = df_with_outliers[outlier_mask]
df_no_outliers = df_with_outliers[~outlier_mask]

def get_city_english(city):
    if city == "الرياض":
        return "Riyadh"
    elif city == "جدة":
        return "Jeddah"
    elif city == "الدمام":
        return "Dammam"
    elif city == "الخبر":
        return "Khobar"
    else:
        return city

# Filter logic
if selected_city != "All":
    no_outliers_filter_mask = (
        df_no_outliers["city"].eq(get_city_english(selected_city)) &
        df_no_outliers["price"].between(price_range[0], price_range[1])
    )
    outliers_filter_mask = (
        df_with_outliers["city"].eq(get_city_english(selected_city)) &
        df_with_outliers["price"].between(price_range[0], price_range[1])
    )
else:
    no_outliers_filter_mask = df_no_outliers["price"].between(price_range[0], price_range[1])
    outliers_filter_mask = df_with_outliers["price"].between(price_range[0], price_range[1])

df_no_outliers_filtered = df_no_outliers[no_outliers_filter_mask].copy()
df_outliers_filtered = df_outliers[outliers_filter_mask].copy()
df_with_outliers_filtered = df_with_outliers[outliers_filter_mask].copy()

# Main Title
st.title("🏠 Saudi Arabia Real Estate Analytics Dashboard")

# Add data preview
st.sidebar.subheader("📊 Data Preview")
with st.expander("Show Data Sample", expanded=False):
    st.dataframe(df_original.head(10), use_container_width=True)


# TOP SECTION: Outlier Toggle with Key Statistics
st.markdown("### 📊 Data Overview")

# Outlier toggle selector at the top
include_outliers = st.radio(
    "📈 Data Analysis Mode:",
    ["Without Outliers", "With Outliers"],
    horizontal=True,
    help="Choose whether to include outliers in your analysis"
)

# Select the appropriate dataset based on the toggle
if include_outliers == "With Outliers":
    current_df = df_with_outliers_filtered
    outlier_info = f"📊 Total dataset including {df_outliers.shape[0]:,} outliers"
else:
    current_df = df_no_outliers_filtered
    outlier_info = f"🔍 Clean dataset with {df_outliers.shape[0]:,} outliers removed"

# Display outlier information
st.info(outlier_info)

# Key Statistics for the selected dataset
cols = st.columns(5)
with cols[0]:
    st.metric("Total Listings", f"{current_df.shape[0]:,}")
with cols[1]:
    st.metric("Average Price", f"{current_df['price'].mean():,.0f} SAR")
with cols[2]:
    st.metric("Median Price", f"{current_df['price'].median():,.0f} SAR")
with cols[3]:
    st.metric("Price Std Dev", f"{current_df['price'].std():,.0f}")
with cols[4]:
    st.metric("Price Variance", f"{current_df['price'].var():,.0f}")

st.markdown("---")

# NAVIGATION BAR: Analysis Type Buttons
st.markdown("### 🔍 Analysis Navigation")

# Create navigation buttons
nav_col0, nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(6)

with nav_col0:
    if st.button("📊 Overview", use_container_width=True):
        st.session_state.analysis_type = "Overview"
with nav_col1:
    if st.button("📈 Price Distribution", use_container_width=True):
        st.session_state.analysis_type = "Price Distribution"
        
with nav_col2:
    if st.button("🌍 Geographic Analysis", use_container_width=True):
        st.session_state.analysis_type = "Geographic Analysis"
        
with nav_col3:
    if st.button("🏠 Property Features", use_container_width=True):
        st.session_state.analysis_type = "Property Features"
        
with nav_col4:
    if st.button("🔬 Advanced Analytics", use_container_width=True):
        st.session_state.analysis_type = "Advanced Analytics"
        
with nav_col5:
    if st.button("🤖 ML Models", use_container_width=True):
        st.session_state.analysis_type = "Machine Learning Models"

# Initialize session state if not exists
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = "Overview"

# Display current analysis type
analysis_type = st.session_state.analysis_type
st.markdown(f"**Current View:** {analysis_type}")

st.markdown("---")

# ML Model Comparison Function
@st.cache_data
def train_and_compare_models(df):
    """Train multiple ML models and return comparison results"""
    # Prepare the dataset
    df_ml = df.copy()
    
    # Drop unnecessary columns
    columns_to_drop = ["id", "details"]
    df_ml = df_ml.drop(columns=[col for col in columns_to_drop if col in df_ml.columns])
    
    # Handle categorical columns
    categorical_cols = []
    for col in ["district", "front", "city"]:
        if col in df_ml.columns:
            categorical_cols.append(col)
    
    # Add age_group if property_age exists
    if 'property_age' in df_ml.columns:
        df_ml['age_group'] = pd.cut(
            df_ml['property_age'],
            bins=[0, 5, 10, 15, 20, 50],
            labels=['0-5', '6-10', '11-15', '16-20', '21+'],
            include_lowest=True
        )
        categorical_cols.append('age_group')
    
    # One-hot encoding
    if categorical_cols:
        df_ml = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)
    
    # Define features and target
    X = df_ml.drop(columns=["price"])
    y = df_ml["price"]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor()
    }
    
    if xgb_available:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    
    # Store results
    results = []
    model_predictions = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        model_predictions[name] = {
            'y_test': y_test.values,
            'y_pred': y_pred,
            'model': model
        }
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        accuracy = 100 - mape
        
        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R²": round(r2, 3),
            "MAPE (%)": round(mape, 2),
            "Accuracy (%)": round(accuracy, 2)
        })
    
    results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
    
    return results_df, model_predictions, X.columns.tolist()

# ANALYSIS SECTIONS (Using current_df which respects the outlier toggle)

# Price Distribution Analysis 

if analysis_type == "Overview":
    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    x_clean = df_original['price'].tolist()
    
    stats_data = {
        'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Variance', 'Min', 'Max', 'Range'],
        'Value': [
            len(x_clean),
            f"{np.mean(x_clean):,.0f}",
            f"{np.median(x_clean):,.0f}",
            f"{np.std(x_clean):,.0f}",
            f"{np.var(x_clean):,.0f}",
            f"{min(x_clean):,.0f}",
            f"{max(x_clean):,.0f}",
            f"{max(x_clean) - min(x_clean):,.0f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)
elif analysis_type == "Price Distribution":
    st.subheader("📈 Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Distribution vs Normal Curve 
        prices = current_df['price'].tolist()
        mean = np.mean(prices)
        median = np.median(prices)
        std_dev = np.std(prices)

        # Create histogram with normal curve
        x_range = np.linspace(min(prices), min(200000, max(prices)), 1000)
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=prices,
            histnorm='probability density',
            name='Actual Price Distribution',
            opacity=0.6,
            marker=dict(color='lightblue')
        ))
        
        # Normal distribution curve
        fig.add_trace(go.Scatter(
            x=x_range,
            y=norm.pdf(x_range, mean, std_dev),
            mode='lines',
            name='Normal Distribution',
            line=dict(color='green', width=3)
        ))
        
        # Mean and median lines
        fig.add_vline(x=mean, line_dash="dash", line_color="red", 
                     annotation_text="Mean", annotation_position="top right")
        fig.add_vline(x=median, line_dash="dot", line_color="blue", 
                     annotation_text="Median", annotation_position="top left")
        
        fig.update_layout(
            title=f'Price Distribution vs Normal Curve ({include_outliers})',
            title_x=0,
            xaxis_title='Price (SAR)',
            yaxis_title='Density',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mean vs Median Bar Chart 
        prices_clean = current_df['price'].dropna()
        mean_val = prices_clean.mean()
        median_val = prices_clean.median()
        
        fig_mean_median = go.Figure(data=[
            go.Bar(
                name='Mean', 
                x=['Mean'], 
                y=[mean_val], 
                marker_color='red',
                text=[f'{mean_val:,.0f}'],
                textposition='auto'
            ),
            go.Bar(
                name='Median', 
                x=['Median'], 
                y=[median_val], 
                marker_color='blue',
                text=[f'{median_val:,.0f}'],
                textposition='auto'
            )
        ])
        
        fig_mean_median.update_layout(
            title='Mean vs Median of Rental Prices',
            xaxis_title='',
            yaxis_title='Price (SAR)',
            barmode='group',
            legend_title='Metric',
            showlegend=True
        )
        
        st.plotly_chart(fig_mean_median, use_container_width=True)

# Geographic Analysis 
elif analysis_type == "Geographic Analysis":
    st.subheader("🌍 Geographic Analysis")
    
    # Average Price by City 
    avg_price_per_city = current_df.groupby('city')['price'].mean().reset_index()
    avg_price_per_city = avg_price_per_city.sort_values(by='price', ascending=False)
    
    fig_city = px.bar(
        avg_price_per_city, 
        x='city', 
        y='price',
        title=f'Average Rental Price per City ({include_outliers})',
        labels={'price': 'Average Price (SAR)', 'city': 'City'},
        color='city',
        text_auto='.2s',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_city.update_layout(showlegend=False, title_x=0)
    st.plotly_chart(fig_city, use_container_width=True)
    
    # Top 5 Most Expensive Districts 
    st.subheader(f"🏙️ Top 5 Most Expensive Districts in {get_city_english(selected_city) if selected_city != 'All' else 'All Cities'}")
    
    if 'district' in current_df.columns:
        top5_expensive_districts = (
            current_df.groupby('district')['price']
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .astype(int)
                .to_frame(name='Average Price')
        )
        
        fig_districts = px.bar(
            top5_expensive_districts.reset_index(),
            x='district',
            y='Average Price',
            color='district',
            text='Average Price',
            title='Top 5 Most Expensive Districts',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_districts.update_layout(showlegend=False, title_x=0)
        st.plotly_chart(fig_districts, use_container_width=True)
    else:
        st.info("District information not available in the dataset")
    
    # Direction Analysis (if front column exists)
    if 'front' in current_df.columns:
        def simplify_direction(direction):
            direction = str(direction).strip()
            if 'North' in direction:
                return 'North'
            elif 'South' in direction:
                return 'South'
            elif 'East' in direction:
                return 'East'
            elif 'West' in direction:
                return 'West'
            else:
                return 'Other'
        
        current_df_copy = current_df.copy()
        current_df_copy['simple_front'] = current_df_copy['front'].apply(simplify_direction)
        current_df_copy['city_front_cleaned'] = current_df_copy['city'] + ' - ' + current_df_copy['simple_front']
        
        fig_direction = px.box(
            current_df_copy,
            x='city_front_cleaned',
            y='price',
            title='Price by City and Direction',
            labels={'city_front_cleaned': 'City - Direction', 'price': 'Price (SAR)'},
            color='city',
            points='outliers'
        )
        fig_direction.update_layout(title_x=0, xaxis_tickangle=-45)
        # st.plotly_chart(fig_direction, use_container_width=True)

# Property Features Analysis 
elif analysis_type == "Property Features":
    st.subheader("🏠 Property Features Analysis")
    
    # First Row: Size Distribution and Size-Price Relationship
    st.markdown("#### 📐 Property Size Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Property Size Distribution
        current_df_copy = current_df.copy()
        bins = [0, 100, 200, 300, 400, 500, 1000]
        labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '501+']
        current_df_copy['size_group'] = pd.cut(current_df_copy['size'], bins=bins, labels=labels, right=False)
        size_counts = current_df_copy['size_group'].value_counts().sort_index().reset_index()
        size_counts.columns = ['size_group', 'count']
        
        fig_size_dist = px.bar(
            size_counts,
            x='size_group',
            y='count',
            text='count',
            title='Number of Properties by Size Group',
            labels={'size_group': 'Property Size Group (m²)', 'count': 'Number of Properties'},
            color='size_group',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_size_dist.update_layout(
            title_x=0,
            showlegend=False,
            xaxis_tickfont=dict(size=12),
            yaxis=dict(title='Number of Properties')
        )
        fig_size_dist.update_traces(textposition='outside')
        st.plotly_chart(fig_size_dist, use_container_width=True)
    
    with col2:
        # Size Analysis: Count + Average Price
        size_counts_filtered = size_counts[size_counts['count'] > 10]
        avg_prices = current_df_copy.groupby('size_group')['price'].mean().reset_index()
        avg_prices.columns = ['size_group', 'avg_price']
        merged = pd.merge(size_counts_filtered, avg_prices, on='size_group')
        
        fig_combined = go.Figure()
        
        fig_combined.add_trace(go.Bar(
            x=merged['size_group'],
            y=merged['count'],
            name='Number of Properties',
            marker_color='indianred',
            text=merged['count'],
            textposition='auto',
            yaxis='y1'
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=merged['size_group'],
            y=merged['avg_price'],
            name='Average Price',
            mode='lines+markers',
            marker=dict(color='blue', size=8),
            line=dict(width=3),
            yaxis='y2'
        ))
        
        fig_combined.update_layout(
            title='Number of Properties and Average Price by Size Group',
            title_x=0,
            xaxis=dict(title='Property Size Group (m²)'),
            yaxis=dict(title='Number of Properties', side='left'),
            yaxis2=dict(
                title='Average Price (SAR)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(x=0.5, xanchor='center', y=-0.15, orientation='h')
        )
        
        # st.plotly_chart(fig_combined, use_container_width=True)
    
    # Display additional property features if available
    feature_cols = st.columns(3)
    
    features_to_analyze = ['elevator', 'basement', 'garage', 'property_age']
    available_features = [col for col in features_to_analyze if col in current_df.columns]
    
    for i, feature in enumerate(available_features[:3]):
        with feature_cols[i]:
            if feature in ['elevator', 'basement', 'garage']:
                # Binary feature analysis
                feature_avg = current_df.groupby(feature)['price'].mean().reset_index()
                feature_avg[feature] = feature_avg[feature].replace({0: f'No {feature.title()}', 1: f'Has {feature.title()}'})
                
                fig_feature = px.bar(
                    feature_avg,
                    x=feature,
                    y='price',
                    title=f'Impact of {feature.title()} on Price',
                    labels={feature: feature.title(), 'price': 'Average Price (SAR)'},
                    text='price',
                    color=feature,
                    color_discrete_sequence=['#8da0cb', '#fc8d62']
                )
                fig_feature.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig_feature.update_layout(showlegend=False, title_x=0)
                st.plotly_chart(fig_feature, use_container_width=True)
            
            elif feature == 'property_age':
                # Age group analysis
                current_df_copy = current_df.copy()
                current_df_copy['age_group'] = pd.cut(
                    current_df_copy['property_age'],
                    bins=[0, 5, 10, 15, 20, 50],
                    labels=['0–5', '6–10', '11–15', '16–20', '21+'],
                    include_lowest=True
                )
                
                fig_age = px.box(
                    current_df_copy,
                    x='age_group',
                    y='price',
                    color='age_group',
                    title='Price Distribution by Property Age',
                    labels={'price': 'Price (SAR)', 'age_group': 'Property Age (Years)'},
                    category_orders={'age_group': ['0–5', '6–10', '11–15', '16–20', '21+']}
                )
                fig_age.update_layout(showlegend=False, title_x=0)
                st.plotly_chart(fig_age, use_container_width=True)

# Advanced Analytics
elif analysis_type == "Advanced Analytics":
    st.subheader("🔬 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Size Scatter Plot (Simple)
        x = current_df['size'].values.reshape(-1, 1)
        y = current_df['price'].values

        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)

        # Calculate R-squared
        r_squared = model.score(x, y)

        # Create scatter plot
        fig = go.Figure()


        fig.add_trace(go.Scatter(x=current_df['size'], y=current_df['price'],
                                mode='markers',
                                name='Actual Data',
                                marker=dict(color='rgba(99, 110, 250, 0.6)', size=6)))

        # Regression line
        fig.add_trace(go.Scatter(x=current_df['size'], y=y_pred,
                                mode='lines',
                                name='Regression Line',
                                line=dict(color='red')))

        fig.update_layout(
            title='Relationship between Size and Price',
            xaxis_title='Size (m²)',
            yaxis_title='Price (SAR)',
            annotations=[
                dict(
                    x=0.05,
                    y=0.95,
                    xref='paper',
                    yref='paper',
                    text=f"R² = {r_squared:.2f}",
                    showarrow=False,
                    font=dict(size=14, color='black')
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.text(f"R² = {r_squared:.2f} (Coefficient of Determination)")

        # Correlation Matrix (if multiple numerical columns exist)
        numerical_cols = ['price', 'size']
        if 'bedrooms' in current_df.columns:
            numerical_cols.append('bedrooms')
        if 'bathrooms' in current_df.columns:
            numerical_cols.append('bathrooms')
        if 'property_age' in current_df.columns:
            numerical_cols.append('property_age')
        
        if len(numerical_cols) > 2:
            # Price Distribution Analysis
            st.subheader("📊 Price Distribution Insights")
            
            # Price quartiles
            q1 = current_df['price'].quantile(0.25)
            q2 = current_df['price'].quantile(0.50)  # median
            q3 = current_df['price'].quantile(0.75)
            
            st.write(f"**Price Quartiles:**")
            st.write(f"• Q1 (25th percentile): {q1:,.0f} SAR")
            st.write(f"• Q2 (Median): {q2:,.0f} SAR") 
            st.write(f"• Q3 (75th percentile): {q3:,.0f} SAR")
            
            # IQR
            iqr = q3 - q1
            st.write(f"• Interquartile Range (IQR): {iqr:,.0f} SAR")
            # st.subheader("🔗 Feature Correlations")
            # corr_matrix = current_df[numerical_cols].corr()
            
            # fig_corr = px.imshow(
            #     corr_matrix,
            #     text_auto=True,
            #     aspect="auto",
            #     title="Correlation Matrix",
            #     color_continuous_scale='RdBu'
            # )
            # fig_corr.update_layout(title_x=0)
            # st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # High-value Properties Analysis
        high_price_threshold = st.slider("High Price Threshold (SAR)", 100000, 500000, 200000, 10000)
        
        high_price_props = current_df[current_df['price'] > high_price_threshold]
        
        st.metric("High-Value Properties", f"{len(high_price_props):,}")
        
        if len(high_price_props) > 0:
            st.subheader(f"🏆 Top High-Value Properties ({include_outliers})")
            columns_to_show = ['price', 'size', 'city']
            if 'district' in high_price_props.columns:
                columns_to_show.append('district')
            if 'bedrooms' in high_price_props.columns:
                columns_to_show.append('bedrooms')
            if 'bathrooms' in high_price_props.columns:
                columns_to_show.append('bathrooms')
            
            top_properties = high_price_props[columns_to_show].sort_values(by='price', ascending=False).head(10)
            st.dataframe(top_properties, use_container_width=True)
            
            # Distribution of high-value properties by city
            high_value_city_dist = high_price_props['city'].value_counts().reset_index()
            high_value_city_dist.columns = ['city', 'count']
            
            fig_high_value = px.pie(
                high_value_city_dist,
                values='count',
                names='city',
                title=f'Distribution of Properties > {high_price_threshold:,} SAR by City ({include_outliers})'
            )
            fig_high_value.update_layout(title_x=0)
            st.plotly_chart(fig_high_value, use_container_width=True)

# Machine Learning Models
elif analysis_type == "Machine Learning Models":
    st.subheader("🤖 Machine Learning Models & Predictions")
    
    # Create tabs for different ML functionalities
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["🔮 Price Prediction", "📊 Model Comparison", "📈 Model Performance"])
    
    with ml_tab1:
        st.markdown("### 🏠 Property Price Prediction")
        st.markdown("Enter property details to get an estimated rental price using our trained ML model.")
        
        # Create input form for prediction
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Basic property details
                pred_city = st.selectbox("City", ["Riyadh", "Jeddah", "Dammam", "Khobar"])
                
                # Get districts for the selected city (if available)
                if 'district' in current_df.columns:
                    city_districts = current_df[current_df['city'] == pred_city]['district'].dropna().unique()
                    if len(city_districts) > 0:
                        pred_district = st.selectbox("District", sorted(city_districts))
                    else:
                        pred_district = "Al Malqa"  # Default
                else:
                    pred_district = "Al Malqa"  # Default
                
                pred_size = st.number_input("Property Size (m²)", min_value=50, max_value=1000, value=200, step=10)
                pred_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
            
            with col2:
                pred_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2, step=1)
                pred_kitchen = st.number_input("Number of Kitchens", min_value=1, max_value=3, value=1, step=1)
                pred_livingrooms = st.number_input("Number of Living Rooms", min_value=1, max_value=5, value=1, step=1)
                pred_property_age = st.number_input("Property Age (years)", min_value=0, max_value=50, value=5, step=1)
            
            with col3:
                pred_garage = st.selectbox("Garage", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                
                # Additional features (if available in dataset)
                if 'front' in current_df.columns:
                    pred_front = st.selectbox("Front Direction", ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"])
                else:
                    pred_front = "North"
                
                if 'elevator' in current_df.columns:
                    pred_elevator = st.selectbox("Elevator", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                else:
                    pred_elevator = 0
                
                if 'furnished' in current_df.columns:
                    pred_furnished = st.selectbox("Furnished", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                else:
                    pred_furnished = 0
            
            # Prediction button
            predict_button = st.form_submit_button("🔮 Predict Price", use_container_width=True)
            
            if predict_button:
                try:
                    # Make prediction using the trained model
                    predicted_price = predict_price(
                        model=model,
                        city=pred_city,
                        district=pred_district,
                        bedrooms=pred_bedrooms,
                        bathrooms=pred_bathrooms,
                        kitchen=pred_kitchen,
                        livingrooms=pred_livingrooms,
                        garage=pred_garage,
                        size=pred_size,
                        property_age=pred_property_age,
                        front=pred_front,
                        elevator=pred_elevator,
                        furnished=pred_furnished
                    )
                    
                    # Display prediction result
                    st.success("🎉 Prediction Complete!")
                    
                    # Create a nice display for the prediction
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            label="Predicted Rental Price",
                            value=f"{predicted_price:,.0f} SAR",
                            help="This is an estimated price based on the property features you provided"
                        )
                    
                    # Price per square meter
                    price_per_sqm = predicted_price / pred_size
                    st.info(f"📏 **Price per m²:** {price_per_sqm:.0f} SAR/m²")
                    
                    # Compare with market averages
                    city_avg = current_df[current_df['city'] == pred_city]['price'].mean()
                    price_diff = predicted_price - city_avg
                    price_diff_pct = (price_diff / city_avg) * 100
                    
                    if price_diff > 0:
                        st.warning(f"📈 This property is **{price_diff:,.0f} SAR ({price_diff_pct:.1f}%)** above the average for {pred_city}")
                    else:
                        st.success(f"📉 This property is **{abs(price_diff):,.0f} SAR ({abs(price_diff_pct):.1f}%)** below the average for {pred_city}")
                    
                    # Property summary
                    with st.expander("📋 Property Summary"):
                        st.write(f"**Location:** {pred_city}, {pred_district}")
                        st.write(f"**Size:** {pred_size} m²")
                        st.write(f"**Bedrooms:** {pred_bedrooms} | **Bathrooms:** {pred_bathrooms}")
                        st.write(f"**Living Rooms:** {pred_livingrooms} | **Kitchens:** {pred_kitchen}")
                        st.write(f"**Property Age:** {pred_property_age} years")
                        st.write(f"**Garage:** {'Yes' if pred_garage else 'No'}")
                        if 'front' in current_df.columns:
                            st.write(f"**Front Direction:** {pred_front}")
                        if 'elevator' in current_df.columns:
                            st.write(f"**Elevator:** {'Yes' if pred_elevator else 'No'}")
                        if 'furnished' in current_df.columns:
                            st.write(f"**Furnished:** {'Yes' if pred_furnished else 'No'}")
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please check your inputs and try again.")
    
    with ml_tab2:
        st.markdown("### 📊 ML Model Comparison")
        st.markdown("Compare different machine learning algorithms on the current dataset.")
        
        if st.button("🚀 Train and Compare Models", use_container_width=True):
            with st.spinner("Training multiple models... This may take a few moments."):
                try:
                    results_df, model_predictions, features_used = train_and_compare_models(current_df)
                    
                    st.success("✅ Model training completed!")
                    
                    # Display results table
                    st.subheader("🏆 Model Performance Comparison")
                    
                    # Color-code the best performing model
                    def highlight_best(s):
                        if s.name in ['R²', 'Accuracy (%)']:
                            is_max = s == s.max()
                        else:  # MAE, RMSE, MAPE - lower is better
                            is_max = s == s.min()
                        return ['background-color: #90EE90' if v else '' for v in is_max]
                    
                    styled_results = results_df.style.apply(highlight_best)
                    st.dataframe(styled_results, use_container_width=True)
                    
                    # Best model insights
                    best_model = results_df.iloc[0]
                    st.info(f"🥇 **Best Model:** {best_model['Model']} with R² score of {best_model['R²']}")
                    
                    # Model performance visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # R² Score Comparison
                        fig_r2 = px.bar(
                            results_df,
                            x='Model',
                            y='R²',
                            title='R² Score Comparison',
                            color='Model',
                            text='R²'
                        )
                        fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig_r2.update_layout(showlegend=False, title_x=0)
                        st.plotly_chart(fig_r2, use_container_width=True)
                    
                    with col2:
                        # MAE Comparison
                        fig_mae = px.bar(
                            results_df,
                            x='Model',
                            y='MAE',
                            title='Mean Absolute Error (Lower is Better)',
                            color='Model',
                            text='MAE'
                        )
                        fig_mae.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                        fig_mae.update_layout(showlegend=False, title_x=0)
                        st.plotly_chart(fig_mae, use_container_width=True)
                    
                    # Features used
                    st.subheader("🔧 Features Used in Training")
                    st.write(f"**Number of features:** {len(features_used)}")
                    
                    # Display features in a nice format
                    feature_cols = st.columns(3)
                    for i, feature in enumerate(features_used):
                        with feature_cols[i % 3]:
                            st.write(f"• {feature}")
                    
                except Exception as e:
                    st.error(f"❌ Model training failed: {str(e)}")
                    st.info("This might be due to insufficient data or missing features. Please try with a different dataset filter.")
        
        # Information about models
        with st.expander("ℹ️ About the Models"):
            st.markdown("""
            **Linear Regression:** Simple linear relationship between features and price
            
            **Random Forest:** Ensemble method using multiple decision trees
            
            **K-Nearest Neighbors (KNN):** Prediction based on similar properties
            
            **XGBoost:** Advanced gradient boosting algorithm (if available)
            
            **Metrics Explained:**
            - **MAE (Mean Absolute Error):** Average difference between predicted and actual prices
            - **RMSE (Root Mean Square Error):** Penalizes large errors more heavily
            - **R² Score:** Proportion of variance explained by the model (higher is better)
            - **MAPE:** Mean Absolute Percentage Error
            - **Accuracy:** 100% - MAPE
            """)
    
    with ml_tab3:
        st.markdown("### 📈 Model Performance Analysis")
        
        # Feature importance analysis (if Random Forest is available)
        if st.button("🔍 Analyze Feature Importance", use_container_width=True):
            with st.spinner("Analyzing feature importance..."):
                try:
                    # Prepare data for feature importance
                    df_importance = current_df.copy()
                    
                    # Drop unnecessary columns
                    columns_to_drop = ["id", "details"]
                    df_importance = df_importance.drop(columns=[col for col in columns_to_drop if col in df_importance.columns])
                    
                    # Handle categorical columns
                    categorical_cols = []
                    for col in ["district", "front", "city"]:
                        if col in df_importance.columns:
                            categorical_cols.append(col)
                    
                    # One-hot encoding
                    if categorical_cols:
                        df_importance = pd.get_dummies(df_importance, columns=categorical_cols, drop_first=True)
                    
                    # Define features and target
                    X_importance = df_importance.drop(columns=["price"])
                    y_importance = df_importance["price"]
                    
                    # Train Random Forest for feature importance
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_importance, y_importance)
                    
                    # Get feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': X_importance.columns,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    # Plot feature importance
                    fig_importance = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features for Price Prediction',
                        labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
                    )
                    fig_importance.update_layout(
                        title_x=0,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Feature importance insights
                    top_feature = feature_importance.iloc[0]
                    st.success(f"🎯 **Most Important Feature:** {top_feature['Feature']} ({top_feature['Importance']:.3f})")
                    
                    # Display top 10 features
                    st.subheader("🔝 Top 10 Most Important Features")
                    importance_display = feature_importance.head(10)[['Feature', 'Importance']]
                    importance_display['Importance'] = importance_display['Importance'].round(4)
                    st.dataframe(importance_display, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Feature importance analysis failed: {str(e)}")
        
        # Model accuracy by city (if we have city data)
        if 'city' in current_df.columns:
            st.subheader("🌍 Model Performance by City")
            
            city_performance = []
            for city in current_df['city'].unique():
                city_data = current_df[current_df['city'] == city]
                if len(city_data) > 50:  # Only analyze cities with sufficient data
                    
                    # Simple train-test split for this city
                    X_city = city_data[['size', 'bedrooms']].fillna(city_data[['size', 'bedrooms']].median())
                    y_city = city_data['price']
                    
                    if len(X_city) > 10:
                        X_train_city, X_test_city, y_train_city, y_test_city = train_test_split(
                            X_city, y_city, test_size=0.3, random_state=42
                        )
                        
                        # Train a simple model
                        city_model = RandomForestRegressor(n_estimators=50, random_state=42)
                        city_model.fit(X_train_city, y_train_city)
                        
                        # Predict and calculate metrics
                        y_pred_city = city_model.predict(X_test_city)
                        r2_city = r2_score(y_test_city, y_pred_city)
                        mae_city = mean_absolute_error(y_test_city, y_pred_city)
                        
                        city_performance.append({
                            'City': city,
                            'R² Score': round(r2_city, 3),
                            'MAE': round(mae_city, 0),
                            'Data Points': len(city_data)
                        })
            
            if city_performance:
                city_perf_df = pd.DataFrame(city_performance)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² by city
                    fig_city_r2 = px.bar(
                        city_perf_df,
                        x='City',
                        y='R² Score',
                        title='Model R² Score by City',
                        text='R² Score',
                        color='City'
                    )
                    fig_city_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig_city_r2.update_layout(showlegend=False, title_x=0)
                    st.plotly_chart(fig_city_r2, use_container_width=True)
                
                with col2:
                    # Data points by city
                    fig_city_data = px.bar(
                        city_perf_df,
                        x='City',
                        y='Data Points',
                        title='Available Data Points by City',
                        text='Data Points',
                        color='City'
                    )
                    fig_city_data.update_traces(texttemplate='%{text}', textposition='outside')
                    fig_city_data.update_layout(showlegend=False, title_x=0)
                    st.plotly_chart(fig_city_data, use_container_width=True)
                
                # Display city performance table
                st.dataframe(city_perf_df, use_container_width=True)
        
        # Model limitations and recommendations
        with st.expander("⚠️ Model Limitations & Recommendations"):
            st.markdown("""
            **Model Limitations:**
            - Predictions are based on historical data and may not reflect current market conditions
            - Model performance varies by city and property type
            - Unusual or luxury properties may not be predicted accurately
            - External factors (economy, regulations) are not considered
            
            **Recommendations:**
            - Use predictions as a starting point, not absolute values
            - Consider local market expertise for final decisions
            - Regularly retrain models with fresh data
            - Validate predictions against recent comparable sales
            
            **Data Quality Impact:**
            - More data generally leads to better predictions
            - Feature completeness affects model accuracy
            - Outlier handling significantly impacts results
            """)

