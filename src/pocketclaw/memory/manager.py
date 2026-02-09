# Memory manager - high-level interface for memory operations.
# Created: 2026-02-02
# Updated: 2026-02-04 - Added Mem0 backend support
# Part of Nanobot Pattern Adoption - Memory System

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pocketclaw.memory.file_store import FileMemoryStore
from pocketclaw.memory.protocol import MemoryEntry, MemoryStoreProtocol, MemoryType

logger = logging.getLogger(__name__)


def create_memory_store(
    backend: str = "file",
    base_path: Path | None = None,
    user_id: str = "default",
    use_inference: bool = True,
) -> MemoryStoreProtocol:
    """
    Factory function to create the appropriate memory store.

    Args:
        backend: Backend type - 'file' or 'mem0'
        base_path: Base path for storage
        user_id: User ID for mem0 scoping
        use_inference: Whether to use LLM inference (mem0 only)

    Returns:
        MemoryStoreProtocol implementation
    """
    if backend == "mem0":
        try:
            # Check if mem0 is actually available before creating store
            import importlib.util

            if importlib.util.find_spec("mem0") is None:
                raise ImportError("mem0ai not installed")

            from pocketclaw.memory.mem0_store import Mem0MemoryStore

            logger.info("Using Mem0 memory backend (semantic search enabled)")
            return Mem0MemoryStore(
                user_id=user_id,
                data_path=base_path,
                use_inference=use_inference,
            )
        except ImportError:
            logger.warning(
                "mem0ai not installed, falling back to file backend. Install with: pip install mem0ai"
            )
            return FileMemoryStore(base_path)
    else:
        logger.info("Using file-based memory backend")
        return FileMemoryStore(base_path)


class MemoryManager:
    """
    High-level memory management facade.

    Provides convenient methods for common memory operations
    while delegating to the underlying store.

    Usage:
        memory = MemoryManager()

        # Remember something long-term
        await memory.remember("User prefers dark mode", tags=["preferences", "ui"])

        # Add daily note
        await memory.note("Had meeting about project X")

        # Get context for agent
        context = await memory.get_context_for_agent()
    """

    def __init__(
        self,
        store: MemoryStoreProtocol | None = None,
        base_path: Path | None = None,
        backend: str = "file",
        user_id: str = "default",
        use_inference: bool = True,
    ):
        """
        Initialize memory manager.

        Args:
            store: Custom store implementation. If None, creates based on backend.
            base_path: Base path for storage.
            backend: Backend type - 'file' or 'mem0'.
            user_id: User ID for mem0 scoping.
            use_inference: Whether to use LLM inference (mem0 only).
        """
        if store:
            self._store = store
        else:
            self._store = create_memory_store(
                backend=backend,
                base_path=base_path,
                user_id=user_id,
                use_inference=use_inference,
            )

    # =========================================================================
    # High-Level Operations
    # =========================================================================

    async def remember(
        self,
        content: str,
        tags: list[str] | None = None,
        header: str | None = None,
    ) -> str:
        """
        Store a long-term memory.

        Args:
            content: The content to remember.
            tags: Optional tags for categorization.
            header: Optional header/title for the memory.

        Returns:
            The memory entry ID.
        """
        entry = MemoryEntry(
            id="",
            type=MemoryType.LONG_TERM,
            content=content,
            tags=tags or [],
            metadata={"header": header or "Memory"},
        )
        return await self._store.save(entry)

    async def note(
        self,
        content: str,
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a daily note.

        Args:
            content: The note content.
            tags: Optional tags.

        Returns:
            The note entry ID.
        """
        entry = MemoryEntry(
            id="",
            type=MemoryType.DAILY,
            content=content,
            tags=tags or [],
            metadata={"header": datetime.now().strftime("%H:%M")},
        )
        return await self._store.save(entry)

    async def add_to_session(
        self,
        session_key: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a message to session history.

        Args:
            session_key: The session identifier.
            role: Message role (user, assistant, system).
            content: Message content.
            metadata: Optional metadata.

        Returns:
            The entry ID.
        """
        entry = MemoryEntry(
            id="",
            type=MemoryType.SESSION,
            content=content,
            role=role,
            session_key=session_key,
            metadata=metadata or {},
        )
        return await self._store.save(entry)

    async def get_session_history(
        self,
        session_key: str,
        limit: int = 50,
    ) -> list[dict[str, str]]:
        """
        Get session history in LLM message format.

        Returns:
            List of {"role": "...", "content": "..."} dicts.
        """
        entries = await self._store.get_session(session_key)
        return [{"role": e.role or "user", "content": e.content} for e in entries[-limit:]]

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Search all memories."""
        return await self._store.search(query=query, limit=limit)

    async def get_context_for_agent(self, max_chars: int = 4000) -> str:
        """
        Get memory context for injection into agent system prompt.

        Returns a formatted string with relevant memories.
        """
        parts = []

        # Long-term memories
        long_term = await self._store.get_by_type(MemoryType.LONG_TERM, limit=10)
        if long_term:
            parts.append("## Long-term Memory\n")
            for entry in long_term:
                parts.append(f"- {entry.content[:200]}")

        # Today's notes
        daily = await self._store.get_by_type(MemoryType.DAILY, limit=5)
        if daily:
            parts.append("\n## Today's Notes\n")
            for entry in daily:
                parts.append(f"- {entry.content[:200]}")

        context = "\n".join(parts)

        # Truncate if too long
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...(truncated)"

        return context

    async def get_compacted_history(
        self,
        session_key: str,
        recent_window: int = 10,
        char_budget: int = 8000,
        summary_chars: int = 150,
        llm_summarize: bool = False,
    ) -> list[dict[str, str]]:
        """Get session history with compaction.

        Keeps the last `recent_window` messages verbatim and collapses
        older messages into condensed one-liner extracts (Tier 1) or an
        LLM-generated summary (Tier 2, opt-in).

        Args:
            session_key: The session identifier.
            recent_window: Number of recent messages to keep verbatim.
            char_budget: Max total characters for the returned history.
            summary_chars: Max chars per older message extract (Tier 1).
            llm_summarize: Use LLM to summarize older messages (Tier 2).

        Returns:
            List of {"role": "...", "content": "..."} dicts.
        """
        entries = await self._store.get_session(session_key)
        if not entries:
            return []

        all_messages = [{"role": e.role or "user", "content": e.content} for e in entries]

        # Split into older and recent
        split_point = max(0, len(all_messages) - recent_window)
        older = all_messages[:split_point]
        recent = all_messages[split_point:]

        if not older:
            return self._enforce_budget(recent, char_budget)

        # Tier 2: Try LLM summary if enabled
        summary_block: str | None = None
        if llm_summarize:
            summary_block = await self._get_or_create_llm_summary(
                session_key, older, len(all_messages)
            )

        # Tier 1 fallback: one-liner extracts
        if summary_block is None:
            lines = []
            for msg in older:
                role = msg["role"].capitalize()
                text = msg["content"].replace("\n", " ").strip()
                if len(text) > summary_chars:
                    # Truncate at word boundary
                    truncated = text[:summary_chars].rsplit(" ", 1)[0]
                    text = truncated + "..."
                lines.append(f"{role}: {text}")
            summary_block = "\n".join(lines)

        compacted = [{"role": "user", "content": f"[Earlier conversation]\n{summary_block}"}]
        compacted.extend(recent)

        return self._enforce_budget(compacted, char_budget)

    @staticmethod
    def _enforce_budget(messages: list[dict[str, str]], char_budget: int) -> list[dict[str, str]]:
        """Drop oldest messages until total chars fit within budget.

        If a single message exceeds the budget, truncate it.
        """
        total = sum(len(m["content"]) for m in messages)
        if total <= char_budget:
            return messages

        # Drop from oldest until within budget
        result = list(messages)
        while len(result) > 1 and sum(len(m["content"]) for m in result) > char_budget:
            result.pop(0)

        # If single remaining message still exceeds budget, truncate it
        if result and len(result[0]["content"]) > char_budget:
            result[0] = {
                "role": result[0]["role"],
                "content": result[0]["content"][:char_budget],
            }

        return result

    async def _get_or_create_llm_summary(
        self,
        session_key: str,
        older_entries: list[dict[str, str]],
        current_total: int,
    ) -> str | None:
        """Get cached or create new LLM summary of older messages.

        Returns None on any error (caller falls back to Tier 1).
        """
        try:
            # Need sessions_path from the store (FileMemoryStore has it, Mem0 may not)
            if not hasattr(self._store, "sessions_path"):
                return None

            safe_key = session_key.replace(":", "_").replace("/", "_")
            cache_path: Path = self._store.sessions_path / f"{safe_key}_compaction.json"

            # Check cache
            if cache_path.exists():
                import json

                cache = json.loads(cache_path.read_text())
                if cache.get("watermark") == current_total:
                    return cache["summary"]

            # Build input for LLM (cap at 4000 chars)
            lines = []
            for msg in older_entries:
                lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
            input_text = "\n".join(lines)
            if len(input_text) > 4000:
                input_text = input_text[:4000]

            # Call Haiku via AsyncAnthropic
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Summarize the following conversation in 2-3 sentences. "
                            "Focus on key topics discussed, decisions made, and any "
                            "important context.\n\n"
                            f"{input_text}"
                        ),
                    }
                ],
            )
            summary = response.content[0].text

            # Write cache
            import json

            cache_path.write_text(
                json.dumps(
                    {
                        "watermark": current_total,
                        "summary": summary,
                        "older_count": len(older_entries),
                    },
                    indent=2,
                )
            )

            return summary

        except Exception:
            logger.debug("LLM summary failed, falling back to Tier 1", exc_info=True)
            return None

    async def clear_session(self, session_key: str) -> int:
        """Clear session history."""
        return await self._store.clear_session(session_key)


# Singleton
_manager: MemoryManager | None = None


def get_memory_manager(force_reload: bool = False) -> MemoryManager:
    """
    Get the global memory manager instance.

    Uses configuration from Settings to determine backend.

    Args:
        force_reload: Force recreation of the manager.

    Returns:
        MemoryManager instance
    """
    global _manager

    if _manager is None or force_reload:
        from pocketclaw.config import get_settings

        settings = get_settings()
        _manager = MemoryManager(
            backend=settings.memory_backend,
            use_inference=settings.memory_use_inference,
        )

    return _manager
