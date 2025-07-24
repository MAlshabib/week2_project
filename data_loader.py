import pandas as pd

@st.cache_data
def load_data(filepath="/SA_Aqar.csv"):
    return pd.read_csv(filepath)
