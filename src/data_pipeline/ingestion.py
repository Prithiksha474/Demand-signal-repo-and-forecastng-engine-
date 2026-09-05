import os
import pandas as pd

# Dynamically set paths relative to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_and_clean_data():
    print("⏳ Loading raw data sources...")

    sales_path = os.path.join(RAW_DIR, "sales_data.csv")
    inventory_path = os.path.join(RAW_DIR, "inventory_data.csv")
    external_path = os.path.join(RAW_DIR, "external_signals.csv")
    kaggle_path = os.path.join(RAW_DIR, "retail_store_inventory_2.csv")

    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"Missing required file: {sales_path}")

    # 1. Load Sales Data
    df_sales = pd.read_csv(sales_path)
    df_sales["date"] = pd.to_datetime(df_sales["date"])

    # 2. Load Inventory Master
    df_inventory = pd.read_csv(inventory_path)

    # 3. Load External Signals
    df_external = pd.read_csv(external_path)
    df_external["date"] = pd.to_datetime(df_external["date"])

    # 4. Load Kaggle Retail Data (if present)
    if os.path.exists(kaggle_path):
        df_kaggle = pd.read_csv(kaggle_path)
        df_kaggle.columns = (
            df_kaggle.columns.str.strip().str.lower().str.replace(" ", "_")
        )
        df_kaggle["date"] = pd.to_datetime(df_kaggle["date"])

    print("🔄 Standardizing and merging datasets...")

    # Merge Primary Synthetic Sources
    df_merged = pd.merge(df_sales, df_external, on="date", how="left")
    df_merged = pd.merge(df_merged, df_inventory, on="sku_id", how="left")

    # Clean numerical anomalies
    df_merged["units_sold"] = df_merged["units_sold"].clip(lower=0)
    df_merged["current_stock"] = df_merged["current_stock"].clip(lower=0)

    # Save to processed zone
    output_path = os.path.join(PROCESSED_DIR, "unified_demand_repository.csv")
    df_merged.to_csv(output_path, index=False)

    print(
        f"✅ Unified Demand Signal Repository created successfully at: {output_path}"
    )


if __name__ == "__main__":
    load_and_clean_data()