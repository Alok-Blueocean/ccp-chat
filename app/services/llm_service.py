from openai import OpenAI

from app.core.configs import get_settings

settings = get_settings()

CHAT_SYSTEM = """You are a helpful chat assistant answering from articles of chaitanya charan das.

Rules:
- Use only information supported by the provided sources, labeled [1], [2], etc.
- for conversation questions, use the history to understand the user's intent and context.
- After statements that rely on a source, add the bracket citation with the matching number, e.g. [1].
- When audio or video URLs listed for a source are relevant to the user's question, include them in your answer as markdown links (e.g. [listen](url) or [watch](url)).
- Do not invent URLs or facts beyond the sources.
- If the sources do not answer the question, say so briefly.
- Answer in detail, addressing all the aspects of the question.
- Always addrees the user's concern
- Always answer in the same language as the question.
- Always answer in a friendly and engaging tone.
- Always answer in encouraging and uplifting manner and make user feel that you understand them and you are there for them.
"""


class LLMService:

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate_chat_answer(
        self,
        query: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM}]
        if history:
            messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": f"Sources:\n\n{context}\n\n---\n\nQuestion: {query}",
            },
        )
        print("messages passed to LLM")
        print(messages)
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()

    def generate_answer(self, query: str, context: list[str]) -> str:
        joined = "\n\n".join(context)
        simple_system = (
            "You are a helpful assistant. Answer using the provided context when it is relevant."
        )
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": simple_system},
                {"role": "user", "content": f"Context:\n\n{joined}\n\nQuestion: {query}"},
            ],
            temperature=0.3,
        )
        return (completion.choices[0].message.content or "").strip()


llm_service = LLMService()
