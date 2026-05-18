from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeWithScore


def _normalize_links(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    return []


def ordered_parent_slots(
    nodes: list[NodeWithScore],
) -> list[tuple[str | None, NodeWithScore]]:
    """First occurrence wins. Duplicate parent_ids share one slot. Missing parent_id uses child node id."""
    seen: set[str] = set()
    slots: list[tuple[str | None, NodeWithScore]] = []
    for nws in nodes:
        meta = nws.node.metadata or {}
        raw_pid = meta.get("parent_id")
        pid = str(raw_pid) if raw_pid else None
        key = pid if pid else f"__child__:{nws.node.node_id}"
        if key in seen:
            continue
        seen.add(key)
        slots.append((pid, nws))
    return slots


def build_numbered_chat_context(
    nodes: list[NodeWithScore],
    parent_payloads: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build LLM context blocks [1]..[n] and parallel reference metadata dicts
    (keys: index, title, url, node_id, audio_links, video_links, source).
    """
    slots = ordered_parent_slots(nodes)
    blocks: list[str] = []
    references: list[dict[str, Any]] = []

    for idx, (parent_id, nws) in enumerate(slots, start=1):
        meta = nws.node.metadata or {}
        payload = parent_payloads.get(parent_id) if parent_id else None

        if parent_id and payload:
            title = (payload.get("title") or "") or (meta.get("title") or "")
            url = payload.get("url") or meta.get("url")
            text = payload.get("text") or ""
            node_id = str(payload.get("node_id") or parent_id)
            audio_links = _normalize_links(payload.get("audio_links"))
            video_links = _normalize_links(payload.get("video_links"))
            source = payload.get("source") or meta.get("source")
        else:
            title = meta.get("title") or ""
            url = meta.get("url")
            text = nws.node.text or ""
            node_id = nws.node.node_id
            audio_links = _normalize_links(meta.get("audio_links"))
            video_links = _normalize_links(meta.get("video_links"))
            source = meta.get("source")

        lines = [f"[{idx}] Title: {title}"]
        if url:
            lines.append(f"URL: {url}")
        lines.append("Text:")
        lines.append(text)
        if audio_links:
            lines.append("Audio links: " + "; ".join(audio_links))
        else:
            lines.append("Audio links: (none)")
        if video_links:
            lines.append("Video links: " + "; ".join(video_links))
        else:
            lines.append("Video links: (none)")
        if source:
            lines.append(f"Source: {source}")

        blocks.append("\n".join(lines))
        references.append(
            {
                "index": idx,
                "title": title or None,
                "url": url,
                "node_id": node_id,
                "audio_links": audio_links,
                "video_links": video_links,
                "source": source,
            }
        )

    if not blocks:
        return "", []

    return "\n\n---\n\n".join(blocks), references
