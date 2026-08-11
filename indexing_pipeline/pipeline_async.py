import asyncio
import logging
import traceback
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
        logging.FileHandler("indexing_pipeline_4603_onward.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "C:/Users/arl/Videos/CCP/Posts-Export-2026-January-17-0453.csv"

# how many documents can process concurrently
MAX_CONCURRENT_DOCS = 40

# qdrant indexing concurrency
# keep this low
MAX_QDRANT_CONCURRENCY = 10

# optional throttling
SLEEP_BETWEEN_DOCS = 1
SLEEP_ON_ERROR = 1


# =========================================================
# GLOBAL STATS
# =========================================================
success_count = 0
failure_count = 0
failed_urls = []

stats_lock = asyncio.Lock()


# =========================================================
# SEMAPHORES
# =========================================================
doc_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOCS)

qdrant_semaphore = asyncio.Semaphore(
    MAX_QDRANT_CONCURRENCY
)


# =========================================================
# LOAD CSV
# =========================================================
def load_rows():

    logger.info("Loading CSV...")

    data = pd.read_csv(CSV_PATH)

    rows = [
        {
            "url": row["Permalink"],
            "title": row["Title"],
        }
        for _, row in data.iterrows()
    ]

    logger.info(f"Loaded {len(rows)} rows")

    return rows


# =========================================================
# PROCESS SINGLE DOCUMENT
# =========================================================
async def process_document(
    index: int,
    total: int,
    row: dict,
    algolia: AlgoliaIndexManager,
    qdrant: QdrantClintManager,
):

    global success_count
    global failure_count

    async with doc_semaphore:

        url = row.get("url")
        title = row.get("title")

        logger.info("=" * 80)
        logger.info(f"STARTING DOCUMENT {index}/{total}")
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

            raw = await asyncio.to_thread(
                extract_page_content,
                url,
                title
            )

            if not raw:
                raise ValueError(
                    "extract_page_content returned empty result"
                )

            logger.info("Page extraction completed")

            # -------------------------------------------------
            # STEP 2 - CREATE DOCUMENT
            # -------------------------------------------------
            logger.info("STEP 2: Creating document...")

            doc = await asyncio.to_thread(
                algolia.create_document,
                raw
            )

            if not doc:
                raise ValueError(
                    "create_document returned empty document"
                )

            logger.info("Document creation completed")

            # -------------------------------------------------
            # STEP 3 - PUSH ALGOLIA
            # -------------------------------------------------
            # logger.info("STEP 3: Pushing to Algolia...")

            # await asyncio.to_thread(
            #     algolia.push_document,
            #     doc
            # )

            # logger.info("Algolia push successful")

            # -------------------------------------------------
            # STEP 4 - INDEX QDRANT
            # -------------------------------------------------
            logger.info("STEP 4: Indexing into Qdrant...")

            # limit qdrant concurrency separately
            async with qdrant_semaphore:

                await asyncio.to_thread(
                    qdrant.index_document,
                    doc
                )

            logger.info("Qdrant indexing successful")

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------
            async with stats_lock:
                success_count += 1

            elapsed = (
                datetime.now() - start_time
            ).total_seconds()

            logger.info(
                f"DOCUMENT SUCCESS | "
                f"index={index} | "
                f"time={elapsed:.2f}s"
            )

        except Exception as e:

            async with stats_lock:
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

            await asyncio.sleep(SLEEP_ON_ERROR)

        finally:

            logger.info(
                f"Sleeping {SLEEP_BETWEEN_DOCS} seconds..."
            )

            # await asyncio.sleep(SLEEP_BETWEEN_DOCS)


# =========================================================
# MAIN
# =========================================================
async def main():

    # -----------------------------------------------------
    # LOAD CSV
    # -----------------------------------------------------
    try:
        rows = load_rows()

    except Exception:
        logger.exception("Failed to load CSV")
        raise

    # -----------------------------------------------------
    # INITIALIZE MANAGERS
    # -----------------------------------------------------
    try:

        logger.info("Initializing Algolia manager...")
        algolia = AlgoliaIndexManager()

        logger.info("Initializing Qdrant manager...")
        qdrant = QdrantClintManager()

        logger.info("Ensuring Qdrant collections...")

        await asyncio.to_thread(
            qdrant.ensure_collections
        )

    except Exception:
        logger.exception(
            "Failed during manager initialization"
        )
        raise

    # -----------------------------------------------------
    # CREATE TASKS
    # -----------------------------------------------------
    tasks = []

    for index, row in enumerate(rows, start=1):

        # skip first 79 docs
        if index < 6745:#4603
            continue

        task = asyncio.create_task(
            process_document(
                index=index,
                total=len(rows),
                row=row,
                algolia=algolia,
                qdrant=qdrant,
            )
        )

        tasks.append(task)

    # -----------------------------------------------------
    # WAIT FOR ALL TASKS
    # -----------------------------------------------------
    await asyncio.gather(*tasks)

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------
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


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    asyncio.run(main())