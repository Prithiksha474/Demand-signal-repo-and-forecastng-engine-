import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Define paths
RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Configuration Parameters
NUM_SKUS = 20
DAYS = 365
START_DATE = datetime.now() - timedelta(days=DAYS)

print("🚀 Starting synthetic data generation...")

# 1. Product & Inventory Data
skus = [f"SKU_{i:03d}" for i in range(1, NUM_SKUS + 1)]
categories = ["Electronics", "Apparel", "Home & Kitchen", "Groceries"]

inventory_data = []
for sku in skus:
    category = random.choice(categories)
    unit_cost = round(random.uniform(5.0, 150.0), 2)
    selling_price = round(unit_cost * random.uniform(1.3, 2.0), 2)
    reorder_point = random.randint(20, 100)
    current_stock = random.randint(10, 300)
    lead_time_days = random.randint(2, 14)

    inventory_data.append(
        {
            "sku_id": sku,
            "category": category,
            "unit_cost": unit_cost,
            "selling_price": selling_price,
            "current_stock": current_stock,
            "reorder_point": reorder_point,
            "lead_time_days": lead_time_days,
        }
    )

df_inventory = pd.DataFrame(inventory_data)
df_inventory.to_csv(
    os.path.join(RAW_DATA_DIR, "inventory_data.csv"), index=False
)
print("✅ Saved: data/raw/inventory_data.csv")

# 2. Historical Sales Data (with seasonality and spikes)
sales_records = []
dates = [START_DATE + timedelta(days=i) for i in range(DAYS)]

for date in dates:
    is_weekend = date.weekday() >= 5
    is_holiday_season = date.month in [11, 12]  # Q4 demand spike

    for sku in skus:
        # Base demand + seasonal/weekend boosters + random noise
        base_demand = random.randint(5, 30)
        multiplier = 1.0

        if is_weekend:
            multiplier *= 1.3
        if is_holiday_season:
            multiplier *= 1.8

        # Random temporary anomaly spike (5% chance)
        if random.random() < 0.05:
            multiplier *= random.uniform(2.5, 4.0)

        units_sold = int(base_demand * multiplier + np.random.normal(0, 3))
        units_sold = max(0, units_sold)  # No negative sales

        sales_records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "sku_id": sku,
                "units_sold": units_sold,
                "store_id": random.choice(["STORE_01", "STORE_02", "STORE_03"]),
            }
        )

df_sales = pd.DataFrame(sales_records)
df_sales.to_csv(os.path.join(RAW_DATA_DIR, "sales_data.csv"), index=False)
print("✅ Saved: data/raw/sales_data.csv")

# 3. External Demand Signals (Promotions & Weather)
external_signals = []
weather_conditions = ["Sunny", "Rainy", "Stormy", "Clear"]

for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    is_promo = random.random() < 0.15  # 15% chance of promo day

    external_signals.append(
        {
            "date": date_str,
            "promotion_active": int(is_promo),
            "discount_percentage": (
                random.choice([10, 20, 30, 50]) if is_promo else 0
            ),
            "weather_condition": random.choice(weather_conditions),
            "temperature_c": round(random.uniform(15.0, 35.0), 1),
        }
    )

df_external = pd.DataFrame(external_signals)
df_external.to_csv(
    os.path.join(RAW_DATA_DIR, "external_signals.csv"), index=False
)
print("✅ Saved: data/raw/external_signals.csv")

print(
    "\n🎉 All raw datasets generated successfully in the 'data/raw/' directory!"
)