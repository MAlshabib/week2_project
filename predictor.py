import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and clean dataset
def load_and_train_model(csv_path="SA_Aqar.csv"):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.drop_duplicates().drop(columns='details')
    df = df[df['price'] >= 30000]
    df['city'] = df['city'].str.strip()

    # Translate
    df['city'] = df['city'].replace({
        'الرياض': 'Riyadh', 'جدة': 'Jeddah', 'الدمام': 'Dammam', 'الخبر': 'Khobar'
    })

    df['front'] = df['front'].replace({
        'شمال': 'North', 'جنوب': 'South', 'شرق': 'East', 'غرب': 'West',
        'شمال غربي': 'Northwest', 'شمال شرقي': 'Northeast',
        'جنوب شرقي': 'Southeast', 'جنوب غربي': 'Southwest',
        '3 شوارع': 'Three Streets', '4 شوارع': 'Four Streets'
    })

    # Remove outliers
    Q1_p, Q3_p = df['price'].quantile([0.25, 0.75])
    Q1_s, Q3_s = df['size'].quantile([0.25, 0.75])
    IQR_p = Q3_p - Q1_p
    IQR_s = Q3_s - Q1_s

    df = df[
        (df['price'] >= Q1_p - 1.5 * IQR_p) & (df['price'] <= Q3_p + 1.5 * IQR_p) &
        (df['size'] >= Q1_s - 1.5 * IQR_s) & (df['size'] <= Q3_s + 1.5 * IQR_s)
    ]

    # Features and target
    features = ['city', 'district', 'bedrooms', 'bathrooms', 'kitchen', 'livingrooms', 'garage', 'size', 'property_age']
    target = 'price'
    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    # Preprocessing + Model
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['city', 'district'])
    ], remainder='passthrough')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model
