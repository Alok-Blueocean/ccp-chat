import logging
import traceback
from time import sleep
from datetime import datetime

import pandas as pd

from indexing_pipeline.extract_pages import extract_page_content
from indexing_pipeline.index_algolia import AlgoliaIndexManager
from indexing_pipeline.index_qdrant import QdrantClintManager


# =========================================================
# LOGGING CONFIG
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("indexing_pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "C:/Users/arl/Videos/CCP/Posts-Export-2026-January-17-0453.csv"

SLEEP_BETWEEN_DOCS = 1
SLEEP_ON_ERROR = 5


# =========================================================
# LOAD CSV
# =========================================================
try:
    logger.info("Loading CSV...")

    data = pd.read_csv(CSV_PATH)

    # optional
    # data = data.tail(20)

    rows = [
        {
            "url": row["Permalink"],
            "title": row["Title"],
        }
        for _, row in data.iterrows()
    ]

    logger.info(f"Loaded {len(rows)} rows")

except Exception as e:
    logger.exception("Failed to load CSV")
    raise


# =========================================================
# INITIALIZE MANAGERS
# =========================================================
try:
    logger.info("Initializing Algolia manager...")
    algolia = AlgoliaIndexManager()

    logger.info("Initializing Qdrant manager...")
    qdrant = QdrantClintManager()

    logger.info("Ensuring Qdrant collections...")
    qdrant.ensure_collections()

except Exception as e:
    logger.exception("Failed during manager initialization")
    raise


# =========================================================
# STATS
# =========================================================
success_count = 0
failure_count = 0
failed_urls = []


# =========================================================
# MAIN LOOP
# =========================================================
for index, row in enumerate(rows, start=1):
    if index<80:
        continue
    url = row.get("url")
    title = row.get("title")

    logger.info("=" * 80)
    logger.info(f"STARTING DOCUMENT {index}/{len(rows)}")
    logger.info(f"TITLE: {title}")
    logger.info(f"URL: {url}")

    start_time = datetime.now()

    try:

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------
        if not url:
            raise ValueError("Missing URL")

        if not title:
            logger.warning("Title missing")

        # -------------------------------------------------
        # STEP 1 - EXTRACT PAGE
        # -------------------------------------------------
        logger.info("STEP 1: Extracting page content...")

        raw = extract_page_content(url, title)

        if not raw:
            raise ValueError("extract_page_content returned empty result")

        logger.info("Page extraction completed")

        # -------------------------------------------------
        # STEP 2 - CREATE DOCUMENT
        # -------------------------------------------------
        logger.info("STEP 2: Creating document...")

        doc = algolia.create_document(raw)

        if not doc:
            raise ValueError("create_document returned empty document")

        logger.info("Document creation completed")

        # -------------------------------------------------
        # STEP 3 - PUSH ALGOLIA
        # -------------------------------------------------
        logger.info("STEP 3: Pushing to Algolia...")

        algolia.push_document(doc)

        logger.info("Algolia push successful")

        # -------------------------------------------------
        # STEP 4 - INDEX QDRANT
        # -------------------------------------------------
        logger.info("STEP 4: Indexing into Qdrant...")

        qdrant.index_document(doc)

        logger.info("Qdrant indexing successful")

        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------
        success_count += 1

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"DOCUMENT SUCCESS | "
            f"index={index} | "
            f"time={elapsed:.2f}s"
        )

    except Exception as e:

        failure_count += 1
        failed_urls.append(url)

        logger.error("=" * 80)
        logger.error(f"FAILED DOCUMENT {index}")
        logger.error(f"TITLE: {title}")
        logger.error(f"URL: {url}")
        logger.error(f"ERROR TYPE: {type(e).__name__}")
        logger.error(f"ERROR MESSAGE: {str(e)}")

        logger.error("FULL TRACEBACK:")
        logger.error(traceback.format_exc())

        logger.error("=" * 80)

        # avoid hammering services continuously on failures
        sleep(SLEEP_ON_ERROR)

        # continue processing next document
        continue

    finally:

        logger.info(
            f"Sleeping {SLEEP_BETWEEN_DOCS} seconds before next document..."
        )

        sleep(SLEEP_BETWEEN_DOCS)


# =========================================================
# FINAL SUMMARY
# =========================================================
logger.info("\n")
logger.info("=" * 80)
logger.info("PIPELINE FINISHED")
logger.info("=" * 80)

logger.info(f"TOTAL DOCUMENTS : {len(rows)}")
logger.info(f"SUCCESS COUNT  : {success_count}")
logger.info(f"FAILURE COUNT  : {failure_count}")

if failed_urls:
    logger.warning("FAILED URLS:")
    for failed_url in failed_urls:
        logger.warning(failed_url)

logger.info("=" * 80)