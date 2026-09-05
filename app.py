import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.genai_engine.chains import hitl_procurement_approval, query_copilot
from src.genai_engine.explainability import generate_forecast_explanation

# Load environment configuration
load_dotenv()

st.set_page_config(
    page_title="Demand Signal & Forecasting Engine",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Demand Signal Repository & Forecasting Engine")
st.markdown("### Agentic Supply Chain Copilot & Inventory Analytics")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CURATED_FILE = os.path.join(PROJECT_ROOT, "data", "curated", "demand_features.csv")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Inventory & Demand Dashboard",
        "AI Copilot & Explainability",
        "Human-in-the-Loop Reorder Desk",
    ],
)

if not os.path.exists(CURATED_FILE):
    st.error(
        f"Curated dataset missing at {CURATED_FILE}. Please run ETL and pipeline scripts first."
    )
    st.stop()

df = pd.read_csv(CURATED_FILE)

# Calculate Inventory Health Metrics
total_records = len(df)
understock_count = len(df[df["inventory_status"] == "Understock Risk"])
overstock_count = len(df[df["inventory_status"] == "Overstock Risk"])
healthy_count = total_records - (understock_count + overstock_count)

health_pct = (healthy_count / total_records * 100) if total_records > 0 else 0.0

if health_pct >= 80:
    health_status = "Healthy 🟢"
elif health_pct >= 50:
    health_status = "Needs Attention ⚠️"
else:
    health_status = "Critical Risk 🔴"

# Page 1: Analytics & Metrics Dashboard
if page == "Inventory & Demand Dashboard":
    st.subheader("📊 Operational Demand Signals & Inventory Health")

    # Adjusted column weights ([1.6, 1, 1, 1.2, 1.2]) to prevent label truncation
    col1, col2, col3, col4, col5 = st.columns([1.6, 1, 1, 1.2, 1.2])
    
    col1.metric("Overall Health Status", health_status)
    col2.metric("Inventory Health", f"{health_pct:.1f}%")
    col3.metric("Total Active SKUs", len(df["sku_id"].unique()))
    col4.metric("Understock Risk SKUs", understock_count)
    col5.metric("Overstock Risk SKUs", overstock_count)

    st.progress(health_pct / 100)

    st.markdown("---")
    st.subheader("Curated Demand Repository")
    st.dataframe(df, use_container_width=True)

# Page 2: Explainability & Conversational AI Copilot
elif page == "AI Copilot & Explainability":
    st.subheader("🤖 AI Supply Chain Copilot & Forecast Reasoning")

    tab1, tab2 = st.tabs(["AI Copilot Assistant", "SKU Demand Explainer"])

    with tab1:
        st.markdown("##### Query Your Repository Context")
        user_input = st.text_input(
            "Ask Copilot a question:",
            placeholder="Which SKUs are facing understock risk?",
        )
        if st.button("Query Copilot", type="primary"):
            if user_input:
                with st.spinner("Analyzing vector DB signals..."):
                    response = query_copilot(user_input)
                    st.markdown(response)
            else:
                st.warning("Please enter a query.")

    with tab2:
        st.markdown("##### Detailed SKU Forecast Explanation")
        selected_sku = st.selectbox(
            "Select SKU ID", sorted(df["sku_id"].unique())
        )
        if st.button("Generate Explanation"):
            explanation = generate_forecast_explanation(selected_sku)
            st.info(explanation)

# Page 3: HITL Procurement Action Center
elif page == "Human-in-the-Loop Reorder Desk":
    st.subheader("⚡ Human-in-the-Loop (HITL) Procurement Action Center")
    st.markdown(
        "Review AI-driven reorder recommendations and issue final procurement approvals."
    )

    high_risk_skus = df[df["inventory_status"] == "Understock Risk"]

    if high_risk_skus.empty:
        st.success("No understock risk SKUs detected.")
    else:
        st.markdown("##### SKUs Requiring Reorder Attention")
        st.dataframe(
            high_risk_skus[
                [
                    "sku_id",
                    "category",
                    "units_sold",
                    "predicted_demand",
                    "current_stock",
                    "inventory_status",
                ]
            ],
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("##### Execute Procurement Decision")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sku_to_act = st.selectbox(
            "Select SKU for Reorder", sorted(df["sku_id"].unique())
        )
    with col_b:
        reorder_qty = st.number_input(
            "Recommended Quantity", min_value=1, value=50, step=5
        )
    with col_c:
        action = st.selectbox(
            "Decision Action", ["APPROVED", "REJECTED", "MODIFIED"]
        )

    if st.button("Submit Decision", type="primary"):
        audit_log = hitl_procurement_approval(sku_to_act, reorder_qty, action)
        st.success("Decision Logged Successfully!")
        st.json(audit_log)