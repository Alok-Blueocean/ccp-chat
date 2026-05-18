"""
from datetime import datetime
import json
import pandas as pd
from airflow.sdk import dag, task

@dag(
    dag_id="ccp_indexing_pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)
def indexing_pipeline():

    @task
    def read_csv() -> list[int]:
        # Using a fixed path for local dev to ensure consistency
        path = "/mnt/c/Users/ARL/Videos/CCP/Posts-Export-2026-January-17-0453.csv"
        data = pd.read_csv(path)
        
        # Filter rows
        rows = [
            {"url": row["Permalink"], "title": row["Title"]}
            for index, row in data.iterrows()
            if index >= 1490
        ]
        
        # Limit to 10 for testing as per your return logic
        limited_rows = rows[:10]
        
        # Save to a stable local path
        with open("/tmp/ccp_rows.json", "w") as f:
            json.dump(limited_rows, f)
            
        # Return the indices to map over
        return list(range(len(limited_rows)))

    @task
    def extract_page(row_index: int) -> dict:
        from indexing_pipeline.extract_pages import extract_page_content
        
        with open("/tmp/ccp_rows.json", "r") as f:
            rows = json.load(f)
        
        # Safety check to prevent KeyError
        if row_index >= len(rows):
            raise IndexError(f"Index {row_index} out of bounds for rows list.")
            
        row = rows[row_index]
        return extract_page_content(row["url"], row["title"])

    @task
    def create_document(raw: dict) -> dict:
        from indexing_pipeline.index_algolia import AlgoliaIndexManager
        manager = AlgoliaIndexManager()
        doc = manager.create_document(raw)
        return doc.model_dump(by_alias=True)

    @task
    def push_algolia(doc_dict: dict):
        from indexing_pipeline.index_algolia import AlgoliaIndexManager
        from indexing_pipeline.schema import LectureDocument
        AlgoliaIndexManager().push_document(LectureDocument(**doc_dict))

    @task
    def index_qdrant(doc_dict: dict):
        from indexing_pipeline.index_qdrant import QdrantClintManager
        from indexing_pipeline.schema import LectureDocument
        qdrant = QdrantClintManager()
        qdrant.ensure_collections()
        qdrant.index_document(LectureDocument(**doc_dict))

    # Execution Flow
    row_indices = read_csv()
    
    # Map extract_page over the list of indices
    raw_docs = extract_page.expand(row_index=row_indices)
    
    # Map downstream tasks
    doc_dicts = create_document.expand(raw=raw_docs)
    
    push_algolia.expand(doc_dict=doc_dicts)
    index_qdrant.expand(doc_dict=doc_dicts)

dag = indexing_pipeline()

"""