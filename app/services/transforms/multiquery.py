import json
import os
from litellm import completion
from langfuse.decorators import observe
from app.core.configs import get_settings
from app.services.transforms.base import BaseQueryTransform
from app.services.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)
os.environ['OPENAI_API_KEY'] = settings.openai_api_key

MULTIQUERY_PROMPT = """
You are an expert spiritual retrieval query generator.

Generate exactly 3 semantically similar queries.

Rules:
- preserve meaning
- use different wording
- avoid repeated keywords
- concise
- spiritually meaningful
- no explanations

Return valid JSON:
{
  "queries": ["q1", "q2", "q3"]
}
"""


class MultiQueryTransform(BaseQueryTransform):

    def __init__(self, model: str):
        self.model = model

    @observe(name="multiquery_transform")
    def transform(self, query: str):

        logger.info("Generating multi queries")

        response = completion(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": MULTIQUERY_PROMPT,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        raw_content = response.choices[0].message.content

        try:
            data = json.loads(raw_content)
            queries = data.get("queries", [])
        except Exception:
            logger.exception("Failed to parse multiquery output")
            queries = []

        queries.append(query)

        queries = list(dict.fromkeys(queries))

        logger.info(f"Generated {len(queries)} queries")

        return queries