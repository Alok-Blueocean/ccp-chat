from airflow.decorators import dag, task
from datetime import datetime
import redis


# =========================
# 🔌 REDIS CLOUD
# =========================
REDIS_URL = "rediss://default:gQAAAAAAAc7UAAIgcDIwMGE2Y2MwN2E2MDc0Y2MyOWQ4ZDFjOTRmZjNmNGU4Nw@profound-toucan-118484.upstash.io:6379"
# If port 6379 is blocked by corporate firewall, use port 443:
# REDIS_URL = "rediss://default:gQAAAAAAAc7UAAIgcDIwMGE2Y2MwN2E2MDc0Y2MyOWQ4ZDFjOTRmZjNmNGU4Nw@profound-toucan-118484.upstash.io:443"

def get_redis():
    return redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
    )

RAW_STREAM = "raw_pages"


# =========================
# 🚀 DAG
# =========================
@dag(
    dag_id="ccp_stream_observable_pipeline",
    schedule="*/2 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
)
def pipeline():

    # =========================
    # 1. FETCH FROM REDIS
    # =========================
    @task
    def fetch_batch():

        group = "airflow_group"
        consumer = "airflow_consumer"

        r = get_redis()

        # Create consumer group once
        try:
            r.xgroup_create(
                name=RAW_STREAM,
                groupname=group,
                id="0",
                mkstream=True
            )
            print("Consumer group created")

        except redis.exceptions.ResponseError as e:
            print("Consumer group already exists:", e)

        # First try pending (delivered but unACKed from failed previous runs)
        messages = r.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={RAW_STREAM: "0"},
            count=5,
        )

        # If no pending, read new messages
        if not messages or not messages[0][1]:
            messages = r.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={RAW_STREAM: ">"},
                count=5,
                block=2000,
            )

        print("RAW REDIS RESPONSE:", messages)

        docs = []

        if not messages:
            print("NO NEW MESSAGES FOUND")
            return docs

        for stream_name, msgs in messages:

            for msg_id, data in msgs:

                print("FOUND MESSAGE:", data)

                docs.append({
                    "redis_id": msg_id,
                    "url": data["url"],
                    "title": data["title"]
                })

        print(f"RETURNING {len(docs)} DOCS")

        return docs


    # =========================
    # 2. EXTRACT DOCUMENT
    # =========================
    @task
    def extract_document(doc):
        from indexing_pipeline.extract_pages import extract_page_content

        print(f"EXTRACTING: {doc['url']}")

        try:
            raw = extract_page_content(doc["url"], doc["title"])
        except Exception as e:
            print(f"SKIPPING {doc['url']}: {e}")
            get_redis().xack(RAW_STREAM, "airflow_group", doc["redis_id"])
            return None

        return {
            "redis_id": doc["redis_id"],
            "url": raw["url"],
            "title": raw["title"],
            "transcript": raw["transcript"],
            "soundcloud_links": raw["soundcloud_links"],
            "youtube_links": raw["youtube_links"],
        }


    # =========================
    # 3. INDEX DOCUMENT
    # =========================
    @task(trigger_rule="all_done", max_active_tis_per_dag=1, retries=0)
    def index_document(doc):
        from indexing_pipeline.index_algolia import AlgoliaIndexManager
        from indexing_pipeline.index_qdrant import QdrantClintManager

        if doc is None:
            return

        print(f"INDEXING: {doc['url']}")

        algolia = AlgoliaIndexManager()
        lecture_doc = algolia.create_document(doc)

        qdrant = QdrantClintManager()
        qdrant.ensure_collections()

        algolia.push_document(lecture_doc)

        qdrant.index_document(lecture_doc)

        # ACK ONLY AFTER SUCCESS
        get_redis().xack(
            RAW_STREAM,
            "airflow_group",
            doc["redis_id"]
        )

        print(f"ACKED: {doc['redis_id']}")

        return f"indexed {doc['url']}"


    # =========================
    # PIPELINE
    # =========================
    batch = fetch_batch()

    extracted = extract_document.expand(
        doc=batch
    )

    indexed = index_document.expand(
        doc=extracted
    )

    batch >> extracted >> indexed


dag = pipeline()