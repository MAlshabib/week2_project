import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Global variable to store trained features
TRAINED_FEATURES = []

def load_and_train_model(csv_path="SA_Aqar.csv"):
    """
    Load dataset, preprocess, and train a Random Forest model for price prediction.
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        Pipeline: Trained scikit-learn pipeline
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        
        # Initial data cleaning
        df = df.drop_duplicates()
        
        # Drop details column if it exists
        if 'details' in df.columns:
            df = df.drop(columns='details')
        
        # Filter out extremely low prices (likely data errors)
        df = df[df['price'] >= 30000]
        
        # Clean city names
        df['city'] = df['city'].str.strip()

        # Translate Arabic city names to English
        city_translation = {
            'الرياض': 'Riyadh', 
            'جدة': 'Jeddah', 
            'الدمام': 'Dammam', 
            'الخبر': 'Khobar'
        }
        df['city'] = df['city'].replace(city_translation)

        # Translate Arabic directions to English
        direction_translation = {
            'شمال': 'North', 
            'جنوب': 'South', 
            'شرق': 'East', 
            'غرب': 'West',
            'شمال غربي': 'Northwest', 
            'شمال شرقي': 'Northeast',
            'جنوب شرقي': 'Southeast', 
            'جنوب غربي': 'Southwest',
            '3 شوارع': 'Three Streets', 
            '4 شوارع': 'Four Streets'
        }
        
        if 'front' in df.columns:
            df['front'] = df['front'].replace(direction_translation)

        # Remove outliers using IQR method for price and size
        def remove_outliers(df, columns):
            df_clean = df.copy()
            for col in columns:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[
                        (df_clean[col] >= lower_bound) & 
                        (df_clean[col] <= upper_bound)
                    ]
            return df_clean

        # Remove outliers from price and size
        outlier_columns = ['price']
        if 'size' in df.columns:
            outlier_columns.append('size')
        
        df = remove_outliers(df, outlier_columns)

        # Define feature columns based on what's available in the dataset
        potential_features = [
            'city', 'district', 'bedrooms', 'bathrooms', 'kitchen', 
            'livingrooms', 'garage', 'size', 'property_age', 'front',
            'furnished', 'ac', 'elevator', 'pool', 'basement'
        ]
        
        # Select only features that exist in the dataset
        available_features = [col for col in potential_features if col in df.columns]
        
        # Store the final feature list for later use in predictions
        global TRAINED_FEATURES
        TRAINED_FEATURES = available_features.copy()
        
        # Ensure we have the target variable
        if 'price' not in df.columns:
            raise ValueError("Price column not found in dataset")
        
        # Create feature set
        target = 'price'
        features = available_features
        
        # Remove target from features if accidentally included
        if target in features:
            features.remove(target)
        
        # Filter dataframe to include only selected features and target
        df_model = df[features + [target]].copy()
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
        if target in numerical_cols:
            numerical_cols.remove(target)
        
        for col in numerical_cols:
            df_model[col] = df_model[col].fillna(df_model[col].median())
        
        # For categorical columns, fill with mode or 'Unknown'
        categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            mode_value = df_model[col].mode()
            if len(mode_value) > 0:
                df_model[col] = df_model[col].fillna(mode_value[0])
            else:
                df_model[col] = df_model[col].fillna('Unknown')
        
        # Final cleanup - remove any remaining NaN values
        df_model = df_model.dropna()
        
        if len(df_model) == 0:
            raise ValueError("No data remaining after preprocessing")
        
        # Separate features and target
        X = df_model[features]
        y = df_model[target]
        
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Create the complete pipeline
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Split the data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        model_pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Training Complete!")
        print(f"Dataset size after preprocessing: {len(df_model):,} samples")
        print(f"Features used: {len(features)}")
        print(f"Mean Absolute Error: {mae:,.0f} SAR")
        print(f"R² Score: {r2:.3f}")
        print(f"Features: {features}")
        
        return model_pipeline
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        # Return a simple backup model if main training fails
        return create_backup_model(csv_path)

def create_backup_model(csv_path):
    """
    Create a simplified backup model with minimal features.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df.dropna(subset=['price', 'size', 'bedrooms'])
        
        # Simple feature set
        X = df[['size', 'bedrooms']].fillna(df[['size', 'bedrooms']].median())
        y = df['price']
        
        # Simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        print("Backup model created with basic features (size, bedrooms)")
        return model
        
    except Exception as e:
        print(f"Error creating backup model: {str(e)}")
        return None

def predict_price(model, city, district, bedrooms, bathrooms, kitchen, livingrooms, garage, size, property_age, **kwargs):
    """
    Make a price prediction using the trained model.
    
    Args:
        model: Trained model pipeline
        ... (feature values)
        **kwargs: Additional features that might be required
    
    Returns:
        float: Predicted price
    """
    try:
        # Get the features that were used during training
        global TRAINED_FEATURES
        
        # Create base prediction data with provided parameters
        pred_data = {
            'city': city,
            'district': district,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'kitchen': kitchen,
            'livingrooms': livingrooms,
            'garage': garage,
            'size': size,
            'property_age': property_age
        }
        
        # Add any additional features with default values
        feature_defaults = {
            'furnished': 0,
            'ac': 1,
            'elevator': 0,
            'pool': 0,
            'front': 'North',
            'basement': 0
        }
        
        # Add missing features with default values
        for feature in TRAINED_FEATURES:
            if feature not in pred_data:
                if feature in feature_defaults:
                    pred_data[feature] = feature_defaults[feature]
                elif feature in kwargs:
                    pred_data[feature] = kwargs[feature]
                else:
                    # Set reasonable defaults based on feature type
                    if feature in ['furnished', 'ac', 'elevator', 'pool', 'basement']:
                        pred_data[feature] = 0
                    elif feature == 'front':
                        pred_data[feature] = 'North'
                    else:
                        pred_data[feature] = 0
        
        # Create DataFrame with only the features used during training
        pred_df = pd.DataFrame([pred_data])
        
        # Ensure we only use the features that were used during training
        pred_df = pred_df[[col for col in TRAINED_FEATURES if col in pred_df.columns]]
        
        # Make prediction
        predicted_price = model.predict(pred_df)[0]
        
        return max(0, predicted_price)  # Ensure non-negative price
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        # Simple fallback calculation
        base_price = size * 200  # 200 SAR per m²
        bedroom_premium = bedrooms * 10000
        return base_price + bedroom_premium

if __name__ == "__main__":
    # Test the model
    model = load_and_train_model("SA_Aqar.csv")
    
    if model:
        # Test prediction
        test_price = predict_price(
            model, 
            city="Riyadh", 
            district="Al Malqa", 
            bedrooms=3, 
            bathrooms=2, 
            kitchen=1, 
            livingrooms=1, 
            garage=1, 
            size=200, 
            property_age=5
        )
        print(f"Test prediction: {test_price:,.0f} SAR")