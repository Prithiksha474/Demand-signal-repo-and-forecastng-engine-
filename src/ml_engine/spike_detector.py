import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

CURATED_FILE = os.path.join(
    PROJECT_ROOT, "data", "curated", "demand_features.csv"
)


def detect_demand_signals():
    print("⏳ Detecting seasonal patterns and demand spikes...")

    if not os.path.exists(CURATED_FILE):
        raise FileNotFoundError(f"Missing input file: {CURATED_FILE}")

    df = pd.read_csv(CURATED_FILE)

    # Classification Logic for Demand Type
    def classify_demand_type(row):
        actual = row["units_sold"]
        rolling_avg = row["rolling_mean_7"]
        rolling_std = row["rolling_std_7"]
        is_promo = row["promotion_active"]
        month = row["month"]

        # Calculate Z-score to measure standard deviations from recent mean
        z_score = (
            (actual - rolling_avg) / rolling_std if rolling_std > 0 else 0
        )

        # 1. Seasonal Demand: High sales occurring during Q4 / holiday season
        if month in [11, 12] and actual > rolling_avg:
            return "Seasonal Demand"
        # 2. Temporary Demand Spike: Sudden statistical outlier (Z-score > 2) or promo-driven
        elif z_score > 2.0 or (is_promo == 1 and actual > (rolling_avg * 1.5)):
            return "Temporary Demand Spike"
        # 3. Regular Demand: Normal baseline sales pattern
        else:
            return "Regular Demand"

    df["demand_type"] = df.apply(classify_demand_type, axis=1)

    # Save updated dataset
    df.to_csv(CURATED_FILE, index=False)

    print("📊 Demand Signal Type Breakdown:")
    print(df["demand_type"].value_counts())
    print(f"✅ Demand signal types saved to: {CURATED_FILE}")


if __name__ == "__main__":
    detect_demand_signals()