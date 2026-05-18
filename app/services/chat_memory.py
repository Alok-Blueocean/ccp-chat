from __future__ import annotations

import logging
from uuid import UUID

from postgres.client import get_pool

logger = logging.getLogger(__name__)

# Max messages to load (alternating user/assistant); each turn = 2 messages.
_DEFAULT_MESSAGE_LIMIT = 40


class ChatMemoryStore:
    def fetch_recent_messages(
        self,
        session_id: UUID,
        limit: int = _DEFAULT_MESSAGE_LIMIT,
    ) -> list[dict[str, str]]:
        """Return OpenAI-style messages oldest-first for this session."""
        pool = get_pool()
        if pool is None:
            return []
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (str(session_id), limit),
                )
                rows = cur.fetchall()
        rows = list(reversed(rows))
        return [{"role": row[0], "content": row[1]} for row in rows]

    def append_turn(self, session_id: UUID, user_text: str, assistant_text: str) -> None:
        pool = get_pool()
        if pool is None:
            return
        sid = str(session_id)
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES (%s, 'user', %s), (%s, 'assistant', %s)
                    """,
                    (sid, user_text, sid, assistant_text),
                )
            print("messages pushed ")
            print(user_text)
            conn.commit()


chat_memory_store = ChatMemoryStore()
