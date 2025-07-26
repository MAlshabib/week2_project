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

# FIXED filter logic
if selected_city != "All":
    no_outliers_filter_mask = (
        df_no_outliers["city"].eq(selected_city) &
        df_no_outliers["price"].between(price_range[0], price_range[1])
    )
    outliers_filter_mask = (
        df_with_outliers["city"].eq(selected_city) &
        df_with_outliers["price"].between(price_range[0], price_range[1])
    )
else:
    no_outliers_filter_mask = df_no_outliers["price"].between(price_range[0], price_range[1])
    outliers_filter_mask = df_with_outliers["price"].between(price_range[0], price_range[1])

# Apply filters
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
st.markdown("\u2705 Built by Meran, Sahar, Naif, Maram, Yazeed, Mohammad")