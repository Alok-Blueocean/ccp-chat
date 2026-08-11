"""Stub for langchain_community.chat_models.vertexai.

Newer langchain-community releases removed this module — VertexAI support
moved to the separate langchain-google-vertexai partner package — but
ragas/llms/base.py still unconditionally imports ChatVertexAI from this exact
path for internal isinstance()-based provider dispatch. This project never
uses VertexAI (everything here is OpenAI-only), so a no-op stub is sufficient:
it only needs to exist and be importable, never to actually work.
"""


class ChatVertexAI:
    pass
