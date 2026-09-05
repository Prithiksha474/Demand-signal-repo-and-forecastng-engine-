import os
import chromadb
from chromadb.config import Settings
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

CURATED_FILE = os.path.join(
    PROJECT_ROOT, "data", "curated", "demand_features.csv"
)
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")


def build_vector_store():
    print("⏳ Reading curated data for Vector DB indexing...")

    if not os.path.exists(CURATED_FILE):
        raise FileNotFoundError(f"Missing input file: {CURATED_FILE}")

    df = pd.read_csv(CURATED_FILE)

    # Initialize persistent ChromaDB client
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Reset or create collection
    collection_name = "demand_signals"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    print("🔄 Formatting SKU demand summaries into embeddings...")

    # Focus on the most recent record per SKU for interactive query indexing
    latest_records = (
        df.sort_values("date").groupby("sku_id").tail(1).reset_index(drop=True)
    )

    for idx, row in latest_records.iterrows():
        doc_text = (
            f"SKU {row['sku_id']} in category {row['category']} on date {row['date']}: "
            f"Actual Units Sold: {row['units_sold']}, Predicted Demand: {row['predicted_demand']:.1f}, "
            f"Current Stock: {row['current_stock']}, Inventory Status: {row['inventory_status']}, "
            f"Demand Type: {row['demand_type']}, Promotion Active: {row['promotion_active']}, "
            f"Discount: {row['discount_percentage']}%."
        )

        documents.append(doc_text)
        metadatas.append(
            {
                "sku_id": str(row["sku_id"]),
                "category": str(row["category"]),
                "inventory_status": str(row["inventory_status"]),
                "demand_type": str(row["demand_type"]),
            }
        )
        ids.append(f"doc_{row['sku_id']}")

    # Upsert into ChromaDB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(
        f"✅ Indexed {len(documents)} SKU demand profiles into ChromaDB at: {VECTOR_DB_DIR}"
    )


if __name__ == "__main__":
    build_vector_store()