from datetime import datetime
from airflow.decorators import dag, task
import pandas as pd
import redis
import os

from indexing_pipeline.extract_pages import extract_page_content
from indexing_pipeline.index_algolia import AlgoliaIndexManager
from indexing_pipeline.index_qdrant import QdrantClintManager
from indexing_pipeline.schema import LectureDocument


# =========================
# 🔌 REDIS CLOUD (UPSTASH TLS)
# =========================
REDIS_URL = "rediss://default:gQAAAAAAAc7UAAIgcDIwMGE2Y2MwN2E2MDc0Y2MyOWQ4ZDFjOTRmZjNmNGU4Nw@profound-toucan-118484.upstash.io:6379"

r = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    health_check_interval=30,
)


RAW_STREAM = "raw_pages"
DOC_STREAM = "processed_docs"


# =========================
# 🚀 AIRFLOW DAG (PRODUCER ONLY)
# =========================
@dag(
    dag_id="ccp_redis_stream_pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)
def pipeline():

    @task
    def publish_to_redis():
        path = "/mnt/c/Users/ARL/Videos/CCP/Posts-Export-2026-January-17-0453.csv"

        df = pd.read_csv(path).tail(20)

        for _, row in df.iterrows():
            r.xadd(RAW_STREAM, {
                "url": row["Permalink"],
                "title": row["Title"]
            })

        return f"Published {len(df)} rows to Redis stream"

    publish_to_redis()


dag = pipeline()