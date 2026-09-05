import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

CURATED_FILE = os.path.join(
    PROJECT_ROOT, "data", "curated", "demand_features.csv"
)


def evaluate_inventory_health():
    print("⏳ Evaluating inventory health and risk levels...")

    if not os.path.exists(CURATED_FILE):
        raise FileNotFoundError(f"Missing input file: {CURATED_FILE}")

    df = pd.read_csv(CURATED_FILE)

    # Classification Logic
    def classify_sku(row):
        stock = row["current_stock"]
        forecast = row["predicted_demand"]
        reorder_pt = row["reorder_point"]

        # 1. Understock Risk: Current stock is below reorder point or forecasted demand
        if stock < reorder_pt or stock < forecast:
            return "Understock Risk"
        # 2. Overstock Risk: Current stock exceeds 3x the forecasted demand
        elif stock > (forecast * 3) and forecast > 5:
            return "Overstock Risk"
        # 3. Slow-Moving Inventory: Extremely low predicted demand relative to stock
        elif forecast <= 2 and stock > 50:
            return "Slow-Moving Inventory"
        # 4. Healthy Stock: Stock levels align well with demand forecast
        else:
            return "Healthy Stock"

    df["inventory_status"] = df.apply(classify_sku, axis=1)

    # Save updated dataset
    df.to_csv(CURATED_FILE, index=False)

    print("📊 Inventory Health Breakdown:")
    print(df["inventory_status"].value_counts())
    print(f"✅ Updated inventory health statuses saved to: {CURATED_FILE}")


if __name__ == "__main__":
    evaluate_inventory_health()