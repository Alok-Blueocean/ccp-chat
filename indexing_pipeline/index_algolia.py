from dotenv import load_dotenv
import hashlib

from algoliasearch.search.client import SearchClientSync
from indexing_pipeline.schema import LectureDocument
from indexing_pipeline import CONFIG


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()


# =========================================================
# ALGOLIA MANAGER
# =========================================================

class AlgoliaIndexManager:

    APP_ID = CONFIG["algolia"]["app_id"]
    WRITE_API = CONFIG["algolia"]["write_key"]
    READ_API = CONFIG["algolia"]["search_key"]
    INDEX_NAME = CONFIG["algolia"]["index_name"]

    def __init__(self):

        # Sync clients
        self.write_client = SearchClientSync(
            self.APP_ID,
            self.WRITE_API
        )

        self.search_client = SearchClientSync(
            self.APP_ID,
            self.READ_API
        )

    # =====================================================
    # CREATE DOCUMENT
    # =====================================================

    def create_document(
        self,
        data: dict
    ) -> LectureDocument:
        """
        Convert raw extracted dictionary
        into validated LectureDocument
        """

        # Deterministic unique ID
        doc_id = hashlib.md5(
            data["url"].encode()
        ).hexdigest()

        video_links = data.get(
            "youtube_links",
            []
        )

        audio_links = data.get(
            "soundcloud_links",
            []
        )

        document = LectureDocument(
            objectID=doc_id,
            title=data.get(
                "title",
                "Untitled"
            ),
            transcript=data.get(
                "transcript",
                ""
            ),
            url=data.get("url"),
            has_audio=len(audio_links) > 0,
            has_video=len(video_links) > 0,
            video_links=video_links,
            audio_links=audio_links,
            source="thespiritualscientist"
        )

        return document

    # =====================================================
    # PUSH DOCUMENT
    # =====================================================

    def push_document(
        self,
        document: LectureDocument
    ):

        try:

            # Convert Pydantic -> dict
            record = document.model_dump(
                by_alias=True
            )

            # Push to Algolia
            response = self.write_client.save_object(
                index_name=self.INDEX_NAME,
                body=record
            )

            # Wait for indexing task
            self.write_client.wait_for_task(
                index_name=self.INDEX_NAME,
                task_id=response.task_id
            )

            print(
    f"✅ Document indexed successfully: "
    f"{document.id}"
)

            return response

        except Exception as e:

            print(
                f"❌ Failed to push document: {e}"
            )

            raise

    # =====================================================
    # SEARCH DOCUMENTS
    # =====================================================

    def search_documents(
        self,
        query: str,
        limit: int = 5
    ):

        try:

            response = self.search_client.search_single_index(
                index_name=self.INDEX_NAME,
                search_params={
                    "query": query,
                    "hitsPerPage": limit
                }
            )

            return response.results

        except Exception as e:

            print(
                f"❌ Search failed: {e}"
            )

            raise


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    manager = AlgoliaIndexManager()

    raw_data = {
        "url": "https://www.thespiritualscientist.com/how-can-humility-go-along-with-self-respect/",
        "title": "Humility and Self Respect",
        "transcript": "Answer: Let’s look at it this way...",
        "soundcloud_links": [
            "https://soundcloud.com/chaitanya-charan/how-can-humility-go-along-with-self-respect"
        ],
        "youtube_links": []
    }

    # Create validated document
    document = manager.create_document(
        raw_data
    )

    # Push to Algolia
    manager.push_document(document)

    # Optional search test
    results = manager.search_documents(
        query="humility",
        limit=3
    )

    print("\n🔍 Search Results:")
    print(results)