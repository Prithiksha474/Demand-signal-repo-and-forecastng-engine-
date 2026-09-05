import os
import chromadb
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (.env)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
CURATED_FILE = os.path.join(PROJECT_ROOT, "data", "curated", "demand_features.csv")

def generate_forecast_explanation(sku_id: str) -> str:
    """Generates an explainable breakdown for a specific SKU's demand forecast."""
    if not os.path.exists(CURATED_FILE):
        return "Error: Curated dataset not found."
    
    df = pd.read_csv(CURATED_FILE)
    sku_data = df[df["sku_id"] == sku_id]
    
    if sku_data.empty:
        return f"No records found for {sku_id}."
    
    latest = sku_data.sort_values("date").iloc[-1]
    
    # Extract operational context
    category = latest.get("category", "N/A")
    actual = latest.get("units_sold", 0)
    forecast = latest.get("predicted_demand", 0)
    stock = latest.get("current_stock", 0)
    status = latest.get("inventory_status", "N/A")
    demand_type = latest.get("demand_type", "N/A")
    promo = latest.get("promotion_active", 0)
    discount = latest.get("discount_percentage", 0)

    # Local template-based fallback explanation
    explanation = (
        f"--- Demand Explanation for {sku_id} ({category}) ---\n"
        f"• Forecasted Demand: {forecast:.1f} units vs Actual Sold: {actual} units.\n"
        f"• Current Inventory Status: '{status}' (Stock on Hand: {stock} units).\n"
        f"• Identified Demand Pattern: '{demand_type}'.\n"
        f"• Promotional Context: Active={bool(promo)}, Discount={discount}%.\n\n"
        f"💡 Actionable Advice: "
    )

    if status == "Understock Risk":
        explanation += f"Reorder recommended immediately! Current stock ({stock}) is insufficient for forecasted demand ({forecast:.1f})."
    elif status == "Overstock Risk":
        explanation += f"Hold further procurement. Stock level ({stock}) significantly exceeds demand forecast."
    else:
        explanation += "Maintain current replenishment schedule. Inventory levels are well-balanced."

    return explanation

if __name__ == "__main__":
    print("⏳ Testing Explainability Engine on SKU_001...\n")
    sample_explanation = generate_forecast_explanation("SKU_001")
    print(sample_explanation)