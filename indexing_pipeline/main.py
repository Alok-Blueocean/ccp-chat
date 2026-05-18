from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="ccp_indexing_pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)
def indexing_pipeline():

    @task
    def read_csv() -> list[dict]:
        import pandas as pd
        path = "/mnt/c/Users/ARL/Videos/CCP/Posts-Export-2026-January-17-0453.csv"
        data = pd.read_csv(path)
        data = data.tail(20)
        return [
            {"url": row["Permalink"], "title": row["Title"]}
            for index, row in data.iterrows()
        ]
        

    @task
    def extract_page(row: dict) -> dict:
        from indexing_pipeline.extract_pages import extract_page_content
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

    rows = read_csv()
    raw_docs = extract_page.expand(row=rows)
    doc_dicts = create_document.expand(raw=raw_docs)

    push_algolia.expand(doc_dict=doc_dicts)
    index_qdrant.expand(doc_dict=doc_dicts)


dag = indexing_pipeline()
