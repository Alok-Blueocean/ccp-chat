from __future__ import annotations

import logging
from uuid import UUID

from postgres.client import get_pool

logger = logging.getLogger(__name__)

# Max messages to load (alternating user/assistant); each turn = 2 messages.
_DEFAULT_MESSAGE_LIMIT = 40

# Fail fast instead of waiting on the pool's 30s default if Postgres went away.
_ACQUIRE_TIMEOUT_SECONDS = 3.0


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
        try:
            with pool.connection(timeout=_ACQUIRE_TIMEOUT_SECONDS) as conn:
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
        except Exception as exc:
            # No memory is better than a failed chat turn.
            logger.warning("Could not load chat history: %s", exc)
            return []
        rows = list(reversed(rows))
        return [{"role": row[0], "content": row[1]} for row in rows]

    def append_turn(self, session_id: UUID, user_text: str, assistant_text: str) -> None:
        pool = get_pool()
        if pool is None:
            return
        sid = str(session_id)
        try:
            with pool.connection(timeout=_ACQUIRE_TIMEOUT_SECONDS) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_messages (session_id, role, content)
                        VALUES (%s, 'user', %s), (%s, 'assistant', %s)
                        """,
                        (sid, user_text, sid, assistant_text),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Could not persist chat turn: %s", exc)


chat_memory_store = ChatMemoryStore()
