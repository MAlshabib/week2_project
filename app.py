import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from collections import Counter
from predictor import load_and_train_model

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
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

# Add analysis type selector
analysis_type = st.sidebar.selectbox(
    "📈 Analysis Type",
    ["Overview", "Price Distribution", "Geographic Analysis", "Property Features", "Advanced Analytics"]
)

# Preprocessing (same as your original code)
df_with_outliers = df_original.drop_duplicates()
df_with_outliers = df_with_outliers.drop(columns='details')
df_with_outliers = df_with_outliers[df_with_outliers['price'] >= 30000]
df_with_outliers['city'] = df_with_outliers['city'].str.strip()
df_with_outliers['city'] = df_with_outliers['city'].replace({
    'الرياض': 'Riyadh', 'جدة': 'Jeddah', 'الدمام': 'Dammam', 'الخبر': 'Khobar'
})
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
st.markdown("---")

# Overview Section (ONLY section with outlier toggle)
if analysis_type == "Overview":
    # Key Statistics with custom styling
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Key Statistics")
    
    with_outliers_tab, without_outliers_tab = st.tabs(["With Outliers", "Without Outliers"])
    
    with with_outliers_tab:
        cols = st.columns(5)
        with cols[0]:
            st.metric("Total Listings", f"{df_with_outliers_filtered.shape[0]:,}")
        with cols[1]:
            st.metric("Average Price", f"{df_with_outliers_filtered['price'].mean():,.0f} SAR")
        with cols[2]:
            st.metric("Median Price", f"{df_with_outliers_filtered['price'].median():,.0f} SAR")
        with cols[3]:
            st.metric("Price Std Dev", f"{df_with_outliers_filtered['price'].std():,.0f}")
        with cols[4]:
            st.metric("Price Variance", f"{df_with_outliers_filtered['price'].var():,.0f}")
    
    with without_outliers_tab:
        st.info(f"🔍 Detected: {df_outliers.shape[0]:,} outliers")
        cols = st.columns(5)
        with cols[0]:
            st.metric("Total Listings", f"{df_no_outliers_filtered.shape[0]:,}")
        with cols[1]:
            st.metric("Average Price", f"{df_no_outliers_filtered['price'].mean():,.0f} SAR")
        with cols[2]:
            st.metric("Median Price", f"{df_no_outliers_filtered['price'].median():,.0f} SAR")
        with cols[3]:
            st.metric("Price Std Dev", f"{df_no_outliers_filtered['price'].std():,.0f}")
        with cols[4]:
            st.metric("Price Variance", f"{df_no_outliers_filtered['price'].var():,.0f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Price Distribution Analysis 
elif analysis_type == "Price Distribution":
    st.subheader("📈 Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Original Price Distribution vs Normal Curve 
        prices = df_no_outliers_filtered['price'].tolist()
        mean = np.mean(prices)
        median = np.median(prices)
        std_dev = np.std(prices)

        # Create histogram with normal curve (limit to 200k)
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
        
        # Normal distribution curve (limited to 200k)
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
            title='Price Distribution vs Normal Curve (Without Outliers)',
            xaxis_title='Price (SAR)',
            yaxis_title='Density',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mean vs Median Bar Chart 
        prices_clean = df_no_outliers_filtered['price'].dropna()
        mean_val = prices_clean.mean()
        median_val = prices_clean.median()
        
        fig_mean_median = go.Figure(data=[
            go.Bar(
                name='Mean', 
                x=['Mean'], 
                y=[mean_val], 
                marker_color='red',  # Match the red dashed line from left chart
                text=[f'{mean_val:,.0f}'],
                textposition='auto'
            ),
            go.Bar(
                name='Median', 
                x=['Median'], 
                y=[median_val], 
                marker_color='blue',  # Match the blue dotted line from left chart
                text=[f'{median_val:,.0f}'],
                textposition='auto'
            )
        ])
        
        fig_mean_median.update_layout(
            title='Mean vs Median of Rental Prices ',
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
    avg_price_per_city = df_no_outliers_filtered.groupby('city')['price'].mean().reset_index()
    avg_price_per_city = avg_price_per_city.sort_values(by='price', ascending=False)
    
    fig_city = px.bar(
        avg_price_per_city, 
        x='city', 
        y='price',
        title='Average Rental Price per City ',
        labels={'price': 'Average Price (SAR)', 'city': 'City'},
        color='city',
        text_auto='.2s',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_city.update_layout(showlegend=False, title_x=0.5)
    st.plotly_chart(fig_city, use_container_width=True)
    
    # Top 5 Most Expensive Districts 
    st.subheader(f"🏙️ Top 5 Most Expensive Districts in {get_city_english(selected_city) if selected_city != 'All' else 'All Cities'}")
    
    filtered_df = df_no_outliers_filtered if selected_city == "All" else df_no_outliers_filtered
    top5_expensive_districts = (
        filtered_df.groupby('district')['price']
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
        title=f'Top 5 Most Expensive Districts ',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_districts.update_layout(showlegend=False, title_x=0.5)
    st.plotly_chart(fig_districts, use_container_width=True)
    
    # Direction Analysis 
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
    
    df_no_outliers_filtered['simple_front'] = df_no_outliers_filtered['front'].apply(simplify_direction)
    df_no_outliers_filtered['city_front_cleaned'] = df_no_outliers_filtered['city'] + ' - ' + df_no_outliers_filtered['simple_front']
    
    fig_direction = px.box(
        df_no_outliers_filtered,
        x='city_front_cleaned',
        y='price',
        title='Price by City and Direction ',
        labels={'city_front_cleaned': 'City - Direction', 'price': 'Price (SAR)'},
        color='city',
        points='outliers'
    )
    fig_direction.update_layout(title_x=0.5, xaxis_tickangle=-45)
    st.plotly_chart(fig_direction, use_container_width=True)

# Property Features Analysis 
elif analysis_type == "Property Features":
    st.subheader("🏠 Property Features Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Elevator Impact 
        fig_elevator = px.box(
            df_no_outliers_filtered, 
            x='elevator', 
            y='price',
            title='Price Impact of Elevator Availability ',
            labels={'elevator': 'Elevator', 'price': 'Price (SAR)'}
        )
        fig_elevator.update_layout(title_x=0.5)
        st.plotly_chart(fig_elevator, use_container_width=True)
        
        # Basement Impact 
        basement_avg = df_no_outliers_filtered.groupby('basement')['price'].mean().reset_index()
        basement_avg['basement'] = basement_avg['basement'].replace({0: 'No Basement', 1: 'Has Basement'})
        basement_avg = basement_avg.sort_values(by='price', ascending=False)
        
        fig_basement = px.bar(
            basement_avg,
            x='basement',
            y='price',
            title='Impact of Basement on Average Rental Price ',
            labels={'basement': 'Basement', 'price': 'Average Price (SAR)'},
            text='price',
            color='basement',
            color_discrete_sequence=['#8da0cb', '#fc8d62']
        )
        fig_basement.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_basement.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_basement, use_container_width=True)
    
    with col2:
        # Property Age Analysis
        df_no_outliers_filtered['age_group'] = pd.cut(
            df_no_outliers_filtered['property_age'],
            bins=[0, 5, 10, 15, 20, 50],
            labels=['0–5', '6–10', '11–15', '16–20', '21+'],
            include_lowest=True
        )
        
        fig_age = px.box(
            df_no_outliers_filtered,
            x='age_group',
            y='price',
            color='age_group',
            title='Price Distribution by Property Age ',
            labels={'price': 'Price (SAR)', 'age_group': 'Property Age (Years)'},
            category_orders={'age_group': ['0–5', '6–10', '11–15', '16–20', '21+']}
        )
        fig_age.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Average Price by Front Direction
        avg_price_by_front = (
            df_no_outliers_filtered
                .groupby('front')['price']
                .mean()
                .sort_values(ascending=False)
                .astype(int)
                .to_frame(name='Average Price')
        )
        
        fig_front = px.bar(
            avg_price_by_front.reset_index(),
            x='front',
            y='Average Price',
            color='front',
            title='Average Rental Price by House Front ',
            text='Average Price',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_front.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_front, use_container_width=True)

# Advanced Analytics (WITHOUT OUTLIERS)
elif analysis_type == "Advanced Analytics":
    st.subheader("🔬 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Size vs Price Relationship with Linear Regression (WITHOUT OUTLIERS)
        X = df_no_outliers_filtered['size'].values.reshape(-1, 1)
        y = df_no_outliers_filtered['price'].values
        
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        y_pred = model_lr.predict(X)
        r_squared = model_lr.score(X, y)
        
        fig_regression = px.scatter(
            df_no_outliers_filtered, 
            x='size', 
            y='price',
            title='Property Size vs Rental Price Relationship ',
            labels={'size': 'Property Size (m²)', 'price': 'Rental Price (SAR)'},
            opacity=0.6
        )
        
        fig_regression.add_trace(go.Scatter(
            x=df_no_outliers_filtered['size'],
            y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2)
        ))
        
        fig_regression.add_annotation(
            x=max(df_no_outliers_filtered['size']),
            y=max(y_pred),
            text=f"R² = {r_squared:.3f}",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="right"
        )
        
        fig_regression.update_layout(title_x=0.5)
        st.plotly_chart(fig_regression, use_container_width=True)
        
        # Statistical Summary (WITHOUT OUTLIERS)
        st.subheader("📊 Statistical Summary ")
        x_clean = df_no_outliers_filtered['price'].tolist()
        
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
    
    with col2:
        # High-value Properties Analysis (WITHOUT OUTLIERS)
        high_price_threshold = st.slider("High Price Threshold (SAR)", 100000, 500000, 200000, 10000)
        high_price_props = df_no_outliers_filtered[df_no_outliers_filtered['price'] > high_price_threshold]
        
        st.metric("High-Value Properties", f"{len(high_price_props):,}")
        
        if len(high_price_props) > 0:
            st.subheader("🏆 Top High-Value Properties (Without Outliers)")
            top_properties = high_price_props[['price', 'size', 'city', 'district', 'bedrooms', 'bathrooms']].sort_values(by='price', ascending=False).head(10)
            st.dataframe(top_properties, use_container_width=True)
            
            # Distribution of high-value properties by city (WITHOUT OUTLIERS)
            high_value_city_dist = high_price_props['city'].value_counts().reset_index()
            high_value_city_dist.columns = ['city', 'count']
            
            fig_high_value = px.pie(
                high_value_city_dist,
                values='count',
                names='city',
                title=f'Distribution of Properties > {high_price_threshold:,} SAR by City (Without Outliers)'
            )
            fig_high_value.update_layout(title_x=0.5)
            st.plotly_chart(fig_high_value, use_container_width=True)

# Price Prediction Section
st.markdown("---")
st.subheader("🎯 Price Prediction Tool")

# Create expandable section for advanced features
with st.expander("🔧 Advanced Property Features", expanded=False):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        pred_furnished = st.selectbox("Furnished", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        pred_ac = st.selectbox("Air Conditioning", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=1)
    
    with adv_col2:
        pred_elevator = st.selectbox("Elevator", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        pred_pool = st.selectbox("Swimming Pool", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with adv_col3:
        pred_basement = st.selectbox("Basement", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        pred_front = st.selectbox("Front Direction", ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"])

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    pred_city = st.selectbox("City", ["Riyadh", "Jeddah", "Dammam", "Khobar"])
    pred_bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    pred_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

with pred_col2:
    pred_district = st.text_input("District", "Al Malqa")
    pred_kitchen = st.number_input("Kitchens", min_value=0, max_value=5, value=1)
    pred_livingrooms = st.number_input("Living Rooms", min_value=0, max_value=5, value=1)

with pred_col3:
    pred_garage = st.number_input("Garage", min_value=0, max_value=5, value=0)
    pred_size = st.number_input("Size (m²)", min_value=50, max_value=1000, value=200)
    pred_age = st.number_input("Property Age", min_value=0, max_value=50, value=5)

if st.button("🔮 Predict Price", type="primary"):
    try:
        # Import the predict_price function
        from predictor import predict_price
        
        # Make prediction with additional default features
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
            property_age=pred_age,
            # Add user-selected advanced features
            furnished=pred_furnished,
            ac=pred_ac,
            elevator=pred_elevator,
            pool=pred_pool,
            basement=pred_basement,
            front=pred_front
        )
        
        st.success(f"🏠 Predicted Price: **{predicted_price:,.0f} SAR**")
        
        # Show comparison with average (using data WITHOUT outliers)
        if len(df_no_outliers_filtered) > 0:
            if pred_city in df_no_outliers_filtered['city'].values:
                avg_city_price = df_no_outliers_filtered[df_no_outliers_filtered['city'] == pred_city]['price'].mean()
            else:
                avg_city_price = df_no_outliers_filtered['price'].mean()
            
            difference = predicted_price - avg_city_price
            percentage_diff = (difference / avg_city_price) * 100
            
            if difference > 0:
                st.info(f"📈 This property is {difference:,.0f} SAR ({percentage_diff:.1f}%) above the average for {pred_city}")
            else:
                st.info(f"📉 This property is {abs(difference):,.0f} SAR ({abs(percentage_diff):.1f}%) below the average for {pred_city}")
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.info("💡 This might be due to missing features in the dataset. The model will be retrained to handle this.")

# Data Preview Section (WITHOUT OUTLIERS)
with st.expander("🔍 View Raw Data (Without Outliers)"):
    st.dataframe(df_no_outliers_filtered.head(100), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #00C2FF;'>
        <h5>✅ Built by Meran, Sahar, Naif, Maram, Yazeed, Mohammad</h5>
        <p>Saudi Arabia Real Estate Analytics Dashboard</p>
    </div>
    """, 
    unsafe_allow_html=True
)