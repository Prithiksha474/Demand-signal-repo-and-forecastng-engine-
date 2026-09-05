import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set up paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

CURATED_FILE = os.path.join(
    PROJECT_ROOT, "data", "curated", "demand_features.csv"
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_forecasting_model():
    print("⏳ Loading curated features for ML model training...")

    if not os.path.exists(CURATED_FILE):
        raise FileNotFoundError(f"Missing input file: {CURATED_FILE}")

    df = pd.read_csv(CURATED_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # Define features and target variable
    feature_cols = [
        "promotion_active",
        "discount_percentage",
        "temperature_c",
        "day_of_week",
        "month",
        "is_weekend",
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "rolling_mean_7",
        "rolling_std_7",
    ]
    target_col = "units_sold"

    # Train-test split based on time (last 30 days for testing)
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=30)

    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print("🤖 Training Random Forest Demand Forecasting Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions & Evaluation using np.sqrt for compatible RMSE calculation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(
        f"📊 Model Performance -> MAE: {mae:.2f} units | RMSE: {rmse:.2f} units"
    )

    # Predict future demand across all records
    df["predicted_demand"] = model.predict(df[feature_cols])

    # Save output dataset with predictions
    output_path = os.path.join(CURATED_FILE)
    df.to_csv(output_path, index=False)

    # Save trained model artifact
    model_path = os.path.join(MODEL_DIR, "demand_forecast_rf.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Trained model saved to: {model_path}")
    print(f"✅ Predictions merged into: {output_path}")


if __name__ == "__main__":
    train_forecasting_model()