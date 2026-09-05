import os
import re
import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
CURATED_CSV = os.path.join(PROJECT_ROOT, "data", "curated", "demand_features.csv")


def get_dataset_metadata() -> dict:
    """Dynamically extracts all unique categories, statuses, and demand types from the curated CSV."""
    metadata = {
        "categories": set(),
        "inventory_statuses": set(),
        "demand_types": set(),
    }
    if os.path.exists(CURATED_CSV):
        try:
            df = pd.read_csv(CURATED_CSV)
            if "category" in df.columns:
                metadata["categories"] = set(df["category"].dropna().str.lower().unique())
            if "inventory_status" in df.columns:
                metadata["inventory_statuses"] = set(df["inventory_status"].dropna().str.lower().unique())
            if "demand_type" in df.columns:
                metadata["demand_types"] = set(df["demand_type"].dropna().str.lower().unique())
        except Exception:
            pass

    if not metadata["categories"]:
        metadata["categories"] = {"electronics", "groceries", "apparel", "dairy", "household"}
    if not metadata["inventory_statuses"]:
        metadata["inventory_statuses"] = {"understock risk", "overstock risk", "optimal", "healthy"}
    if not metadata["demand_types"]:
        metadata["demand_types"] = {"regular demand", "seasonal demand", "promo demand", "temporary demand"}

    return metadata


def get_chroma_collection():
    """Connects to the persistent ChromaDB collection cleanly."""
    if not os.path.exists(VECTOR_DB_DIR):
        return None
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        return client.get_collection(name="demand_signals")
    except Exception:
        return None


def get_azure_llm():
    """Initializes Azure OpenAI Chat Model if valid environment credentials exist."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key or not endpoint or not deployment:
        return None

    try:
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=0.1,
        )
    except Exception:
        return None


def normalize_text(text: str) -> str:
    """Normalizes query text to catch spaced or alternative keyword variations."""
    t = text.lower()
    t = re.sub(r'\bunder\s+stock\b', 'understock', t)
    t = re.sub(r'\bover\s+stock\b', 'overstock', t)
    t = re.sub(r'\bout\s+of\s+stock\b', 'understock', t)
    return t


def extract_query_intent(user_query: str, metadata: dict) -> dict:
    """Extracts search constraints after normalizing text."""
    q = normalize_text(user_query)

    detected_categories = [cat for cat in metadata["categories"] if cat in q]

    detected_statuses = []
    if "understock" in q:
        detected_statuses.append("understock risk")
    if "overstock" in q:
        detected_statuses.append("overstock risk")
    if "optimal" in q or "healthy" in q:
        detected_statuses.append("optimal")

    detected_demand_types = []
    if "seasonal" in q:
        detected_demand_types.append("seasonal demand")
    if "temporary" in q or "promo" in q:
        detected_demand_types.append("temporary demand")
    if "regular" in q:
        detected_demand_types.append("regular demand")

    wants_promo = any(w in q for w in ["promotion", "promo", "discount", "discounted"])
    sku_matches = re.findall(r'sku[_\s]?(\d+|\w+)', q)

    return {
        "categories": detected_categories,
        "statuses": detected_statuses,
        "demand_types": detected_demand_types,
        "wants_promo": wants_promo,
        "skus": sku_matches,
    }


def query_copilot(user_query: str) -> str:
    """Queries vector repository with strict multi-attribute filtering."""
    collection = get_chroma_collection()

    if collection is None:
        return (
            "⚠️ Vector DB repository directory not found or uninitialized. "
            "Please run `python src/genai_engine/vector_db.py` first."
        )

    metadata = get_dataset_metadata()
    intent = extract_query_intent(user_query, metadata)

    try:
        results = collection.query(query_texts=[user_query], n_results=30)
        retrieved_docs = results.get("documents", [[]])[0]
    except Exception as e:
        return f"Error querying vector repository: {str(e)}"

    if not retrieved_docs:
        return f"No records found in the demand repository for query: '{user_query}'."

    exact_matches = []

    for doc in retrieved_docs:
        doc_lower = doc.lower()

        # SKU Filter
        if intent["skus"]:
            if any(f"sku_{sku}" in doc_lower or f"sku {sku}" in doc_lower for sku in intent["skus"]):
                exact_matches.append(doc)
                continue

        # Strict Category Constraint
        category_pass = not intent["categories"] or any(cat in doc_lower for cat in intent["categories"])

        # Strict Status Constraint
        status_pass = not intent["statuses"] or any(stat in doc_lower for stat in intent["statuses"])

        # Strict Demand Type Constraint
        demand_pass = not intent["demand_types"] or any(dt in doc_lower for dt in intent["demand_types"])

        # Strict Promo Constraint
        promo_pass = True
        if intent["wants_promo"]:
            promo_pass = ("promotion active: 1" in doc_lower) or ("discount:" in doc_lower and "discount: 0%" not in doc_lower)

        if category_pass and status_pass and demand_pass and promo_pass:
            exact_matches.append(doc)

    # Format Output Based Only on Valid Matches
    if exact_matches:
        selected_docs = exact_matches[:5]
        match_type_str = "Exact Filter Match"
        formatted_context = "\n".join([f"- {doc}" for doc in selected_docs])
    else:
        missing_parts = []
        if intent["categories"]:
            missing_parts.append(f"Category: {', '.join(intent['categories']).title()}")
        if intent["statuses"]:
            missing_parts.append(f"Status: {', '.join(intent['statuses']).title()}")
        if intent["demand_types"]:
            missing_parts.append(f"Demand Type: {', '.join(intent['demand_types']).title()}")

        criteria_desc = " | ".join(missing_parts) if missing_parts else user_query
        match_type_str = "No Exact Matches Found"
        formatted_context = f"- No records in the active dataset matched all specified criteria ({criteria_desc})."

    # LLM Synthesis
    llm = get_azure_llm()
    if llm:
        prompt = (
            f"You are an AI Supply Chain Copilot.\n"
            f"User Query: {user_query}\n\n"
            f"Retrieved Context ({match_type_str}):\n{formatted_context}\n\n"
            f"Instructions:\n"
            f"1. Answer strictly based on the retrieved context.\n"
            f"2. If no exact matches were found, explicitly state that no matching records exist.\n"
            f"3. Provide procurement advice based ONLY on actual returned data."
        )
        try:
            response = llm.invoke(prompt)
            return f"🤖 **Azure OpenAI Copilot Insights:**\n\n{response.content}"
        except Exception:
            pass

    # Deterministic Recommendation Logic
    if not exact_matches:
        recommendation = "No matching records detected. Check the dashboard or refine search terms."
    else:
        docs_text = " ".join(selected_docs).lower()
        if "understock risk" in docs_text and "overstock risk" in docs_text:
            recommendation = "Mixed Risks Detected: Reorder understock SKUs and freeze purchase orders for overstock SKUs."
        elif "understock risk" in docs_text:
            recommendation = "High Priority: Initiate procurement reorders immediately for flagged understock SKUs."
        elif "overstock risk" in docs_text:
            recommendation = "Capital Optimization: Freeze new purchase orders for overstock SKUs to prevent holding cost inflation."
        else:
            recommendation = "Review current inventory metrics and execute targeted balancing actions."

    return (
        f"🤖 **AI Supply Chain Copilot Insights:**\n\n"
        f"**Search Query:** '*{user_query}*'\n"
        f"**Match Diagnostics:** {match_type_str}\n\n"
        f"{formatted_context}\n\n"
        f"💡 **System Recommendation:** {recommendation}"
    )


def hitl_procurement_approval(sku_id: str, reorder_qty: int, action: str) -> dict:
    """Human-in-the-Loop approval workflow."""
    action_upper = action.upper()
    return {
        "sku_id": sku_id,
        "recommended_reorder_qty": reorder_qty,
        "human_decision": action_upper,
        "status": "Finalized" if action_upper != "REJECTED" else "Cancelled",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
if __name__ == "__main__":
    print("⏳ Testing Universal Conversational Query Engine...\n")
    print(query_copilot("Find promotional understock items in groceries"))

#  source venv/Scripts/activate
#  rm -rf data/chroma_db
#  python src/genai_engine/vector_db.py