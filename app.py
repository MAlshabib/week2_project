import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from predictor import load_and_train_model

model = load_and_train_model("SA_Aqar.csv")

# Page Config
st.set_page_config(
    page_title="Saudi Lands Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["city"] = df["city"].str.strip()
    return df

df_original = load_data("SA_Aqar.csv")

# Sidebar
st.sidebar.header("Dashboard Filters")
cities = sorted(df_original["city"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", ["All"] + cities)

min_price, max_price = int(df_original["price"].min()), int(df_original["price"].max())
price_range = st.sidebar.slider(
    "Price Range (SAR)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=1000
)

# Preprocessing
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

# FIXED filter logic
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

def stat_card(label, value, unit="SAR"):
    if pd.isna(value):
        st.markdown(f"""
        <div class="card">
            <h5>{label}</h5>
            <h3>🚫 No Data</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            <h5>{label}</h5>
            <h3>{int(value):,} {unit}</h3>
        </div>
        """, unsafe_allow_html=True)

with_outliers_tab, without_outliers_tab = st.tabs(["With Outliers", "Without Outliers"])

with with_outliers_tab:
    st.subheader("📊 Key Statistics (With Outliers)")
    cols = st.columns(3)
    with cols[0]: stat_card("Total Listings", df_with_outliers_filtered.shape[0], unit="")
    with cols[1]: stat_card("Average Price", df_with_outliers_filtered["price"].mean())
    with cols[2]: stat_card("Price Variance", df_with_outliers_filtered["price"].var())

    cols2 = st.columns(3)
    with cols2[0]: stat_card("Price Std Dev", df_with_outliers_filtered["price"].std())
    with cols2[1]: stat_card("Median Price", df_with_outliers_filtered["price"].median())

with without_outliers_tab:
    st.subheader("📊 Key Statistics (Without Outliers)")
    st.text("Detected: " + str(df_outliers.shape[0]) + " outliers")
    cols = st.columns(3)
    with cols[0]: stat_card("Total Listings", df_no_outliers_filtered.shape[0], unit="")
    with cols[1]: stat_card("Average Price", df_no_outliers_filtered["price"].mean())
    with cols[2]: stat_card("Price Variance", df_no_outliers_filtered["price"].var())

    cols2 = st.columns(3)
    with cols2[0]: stat_card("Price Std Dev", df_no_outliers_filtered["price"].std())
    with cols2[1]: stat_card("Median Price", df_no_outliers_filtered["price"].median())


st.markdown("---")

# Price Distribution by City
fig = px.box(
    df_no_outliers,
    x='city',
    y='price',
    color='city',
    title='Price Distribution by City'
)

fig.update_layout(
    title_x=0.5,
    title_y=0.87,
    xaxis_title='City',
    yaxis_title='Price (SAR)',
    showlegend=False,
    xaxis_tickfont=dict(size=14)
)

st.plotly_chart(fig, use_container_width=True)

# Top 5 Most Expensive Districts in City

st.subheader(f"🏙️ Top 5 Most Expensive Districts in {get_city_english(selected_city)}")
filtered_df = df_no_outliers[df_no_outliers['city'] == get_city_english(selected_city)] if selected_city != "All" else df_no_outliers
top5_expensive_districts_city = (
    filtered_df.groupby('district')['price']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .astype(int)
        .to_frame(name='Average Price')
)
fig = px.bar(
    top5_expensive_districts_city.reset_index(),
    x='district',
    y='Average Price',
    color='district',
    text='Average Price',
    title=f'Top 5 Most Expensive Districts in {get_city_english(selected_city)} by Average Rental Price',
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig.update_layout(
    title_x=0.5,
    title_y=0.87,
    xaxis_title='District',
    yaxis_title='Average Price (SAR)',
    showlegend=False,
    xaxis_tickfont=dict(size=14)
)

st.plotly_chart(fig, use_container_width=True)


# Average Price by Front
avg_price_by_front = (
    df_no_outliers
        .groupby('front')['price']
        .mean()
        .sort_values(ascending=False)
        .astype(int)
        .to_frame(name='Average Price')
)

fig = px.bar(
    avg_price_by_front.reset_index(),
    x='front',
    y='Average Price',
    color='front',
    orientation='v',
    title='Average Rental Price by House Front',
    text='Average Price',
    color_discrete_sequence=px.colors.qualitative.Pastel
)

fig.update_layout(
    title_x=0.5,
    xaxis_title='House Front',
    yaxis_title='Average Price (SAR)',
    title_y=0.87,
    showlegend=False,
    xaxis_tickfont=dict(size=14)
)
st.plotly_chart(fig, use_container_width=True)

# Price Distribution by Age
df_no_outliers['age_group'] = pd.cut(
    df_no_outliers['property_age'],
    bins=[0, 5, 10, 15, 20, 50],
    labels=['0–5', '6–10', '11–15', '16–20', '21+'],
    include_lowest=True
)

fig = px.box(
    df_no_outliers,
    x='age_group',
    y='price',
    color='age_group',
    title='Price Distribution by Property Age',
    labels={'price': 'Price (SAR)', 'age_group': 'Property Age (Years)'},
    category_orders={'age_group': ['0–5', '6–10', '11–15', '16–20', '21+']}
)

fig.update_layout(
    title_x=0.5,
    showlegend=False,
    title_y=0.87,
    xaxis_tickfont=dict(size=14)
)
st.plotly_chart(fig, use_container_width=True)

# 

# --- Heatmap by Listings (More Points) ---
# Coordinates per city (central point)
# city_coordinates = {
#     "الرياض": (24.7136, 46.6753),
#     "جدة": (21.4858, 39.1925),
#     "الدمام": (26.4207, 50.0888),
#     "الخبر": (26.2172, 50.1971)
# }

# Assign lat/lon with small random jitter to simulate many listings
# filtered_df["lat"] = filtered_df["city"].map(lambda c: city_coordinates.get(c, (None, None))[0]) + np.random.uniform(-0.03, 0.03, len(filtered_df))
# filtered_df["lon"] = filtered_df["city"].map(lambda c: city_coordinates.get(c, (None, None))[1]) + np.random.uniform(-0.03, 0.03, len(filtered_df))
# filtered_df = filtered_df.dropna(subset=["lat", "lon"])


# Properties with an elevator tend to be more expensive
fig = px.box(df_no_outliers_filtered, x='elevator', y='price',
       title='Price by Elevator Availability',
       labels={
           'elevator': 'Elevator',
           'price': 'Price (SAR)'
        }
)

fig.update_layout(
    title_x=0.5,
    title_y=0.87,
    xaxis_title='Elevator Availability',
    yaxis_title='Price (SAR)',
    showlegend=False,
    xaxis_tickfont=dict(size=14)
)
st.plotly_chart(fig, use_container_width=True)

# Group by city and get average price, then sort descending
avg_price_per_city = df_no_outliers_filtered.groupby('city')['price'].mean().reset_index()
avg_price_per_city = avg_price_per_city.sort_values(by='price', ascending=False)

# Plot bar chart
fig = px.bar(avg_price_per_city, x='city', y='price',
             title='Average Rental Price per City ',
             labels={'price': 'Average Price (SAR)', 'city': 'City'},
             color='city',
             text_auto='.2s',
             color_discrete_sequence=px.colors.qualitative.Set2)

fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

def simplify_direction(direction):
    direction = direction.strip()
    if 'North' in direction:
        return 'North'
    elif 'South' in direction:
        return 'South'
    elif 'East' in direction:
        return 'East'
    elif 'West' in direction:
        return 'West'
    else:
        return 'Other'  # For things like "3 شوارع" or invalid entries

# Apply to 'front' column
df_no_outliers_filtered['simple_front'] = df_no_outliers_filtered['front'].apply(simplify_direction)

# Combine city + simplified direction
df_no_outliers_filtered['city_front_cleaned'] = df_no_outliers_filtered['city'] + ' - ' + df_no_outliers_filtered['simple_front']

fig = px.box(
    df_no_outliers_filtered,
    x='city_front_cleaned',
    y='price',
    title='Price by City and Simplified Direction',
    labels={'city_front_cleaned': 'City - Direction', 'price': 'Price (SAR)'},
    color='city',
    points='outliers'
)

# fig.update_layout(xaxis_tickangle=-45)
fig.update_layout(
    title_x=0.5,
    title_y=0.87,
    xaxis_title='City - Direction',
    yaxis_title='Price (SAR)',
    showlegend=False,
    xaxis_tickfont=dict(size=14)
)

st.plotly_chart(fig, use_container_width=True)

# Preview data
# st.subheader("🔍 Data Preview")
# st.dataframe(df_no_outliers_filtered.head(10), use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("✅ Built by Meran, Sahar, Naif, Maram, Yazeed, Mohammad")
