import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Saudi Lands Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["city"] = df["city"].str.strip()
    return df

df = load_data("SA_Aqar.csv")

# --- Sidebar ---
st.sidebar.header("Dashboard Filters")

cities = sorted(df["city"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", ["All"] + cities)

min_price, max_price = int(df["price"].min()), int(df["price"].max())
price_range = st.sidebar.slider(
    "Price Range (SAR)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=1000
)

# --- Filter Data ---
mask = (
    df["price"].between(price_range[0], price_range[1]) &
    (df["city"] == selected_city if selected_city != "All" else True)
)
filtered_df = df[mask].copy()


# --- Title ---
st.title("🏙️ Saudi Lands Dashboard")

# --- KPI Cards ---
st.subheader("📊 Key Statistics")
cols = st.columns(3)

def stat_card(label, value, unit="SAR"):
    st.markdown(f"""
    <div class="card">
        <h5>{label}</h5>
        <h3>{int(value):,} {unit}</h3>
    </div>
    """, unsafe_allow_html=True)

with cols[0]: stat_card("Total Listings", filtered_df.shape[0], unit="")
with cols[1]: stat_card("Average Price", filtered_df["price"].mean())
with cols[2]: stat_card("Max Price", filtered_df["price"].max())

cols2 = st.columns(3)
with cols2[0]: stat_card("Min Price", filtered_df["price"].min())
with cols2[1]: stat_card("Median Price", filtered_df["price"].median())
with cols2[2]: stat_card("Price Std Dev", filtered_df["price"].std())

st.markdown("---")

# --- Scatter Plot: Size vs Price ---
fig = px.scatter(df, x='size', y='price',
                 title='Size vs Price (Detecting Outliers)',
                 labels={'size': 'Size (m²)', 'price': 'Price (SAR)'},
                 width=800, height=600)

st.plotly_chart(fig, use_container_width=True)

# --- Heatmap by Listings (More Points) ---
# Coordinates per city (central point)
city_coordinates = {
    "الرياض": (24.7136, 46.6753),
    "جدة": (21.4858, 39.1925),
    "الدمام": (26.4207, 50.0888),
    "الخبر": (26.2172, 50.1971)
}

# Assign lat/lon with small random jitter to simulate many listings
filtered_df["lat"] = filtered_df["city"].map(lambda c: city_coordinates.get(c, (None, None))[0]) + np.random.uniform(-0.03, 0.03, len(filtered_df))
filtered_df["lon"] = filtered_df["city"].map(lambda c: city_coordinates.get(c, (None, None))[1]) + np.random.uniform(-0.03, 0.03, len(filtered_df))
filtered_df = filtered_df.dropna(subset=["lat", "lon"])

# --- Heatmap Plot ---
st.subheader("🔥 Land Listings Heatmap (More Points)")

fig = px.density_mapbox(
    filtered_df,
    lat="lat",
    lon="lon",
    z="price",  # Could be None for count-based heat
    radius=30,
    center={"lat": 24.7, "lon": 46.7},
    zoom=5.3,
    mapbox_style="carto-darkmatter",
    opacity=0.6,
    height=550,
    hover_name="city",
    hover_data=["price", "size"]
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

fig = px.box(filtered_df_cleaned, x='city', y='price', color='city',
             title='Rental Price Distribution by City',
             labels={'price': 'Price (SAR)', 'city': 'City'},
             points='outliers',  # Show outliers
             color_discrete_sequence=px.colors.qualitative.Set2)

fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# --- Data Preview ---
with st.expander("🔍 Preview Filtered Data"):
    st.dataframe(filtered_df.head(10))

# --- Footer ---
st.markdown("---")
st.markdown("✅ Built by Meran, Mohammad, Maram, Naif, Yazeed")
