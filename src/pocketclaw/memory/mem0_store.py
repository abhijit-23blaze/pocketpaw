# Mem0-based memory store implementation.
# Created: 2026-02-04
# Provides semantic memory with LLM-powered fact extraction and search.
#
# Mem0 features:
# - Vector-based semantic search (Qdrant)
# - LLM-powered fact extraction and consolidation
# - Memory evolution (updates existing memories instead of duplicating)
# - Optional graph store for relationships (Neo4j)

import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from functools import partial

from pocketclaw.memory.protocol import MemoryStoreProtocol, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


class Mem0MemoryStore:
    """
    Mem0-based memory store implementing MemoryStoreProtocol.

    Uses Mem0 for semantic memory with:
    - Vector search for similarity-based retrieval
    - LLM-powered fact extraction (optional)
    - Memory consolidation and evolution

    Mapping to Mem0 concepts:
    - LONG_TERM memories -> user_id scoped (persistent facts)
    - DAILY memories -> user_id scoped with date metadata
    - SESSION memories -> run_id scoped (conversation history)
    """

    def __init__(
        self,
        user_id: str = "default",
        agent_id: str = "pocketpaw",
        data_path: Path | None = None,
        use_inference: bool = True,
    ):
        """
        Initialize Mem0 memory store.

        Args:
            user_id: Default user ID for memory scoping.
            agent_id: Agent ID for agent-specific memories.
            data_path: Path for Qdrant data storage.
            use_inference: If True, use LLM to extract facts from messages.
                          If False, store raw content.
        """
        self.user_id = user_id
        self.agent_id = agent_id
        self.use_inference = use_inference
        self._data_path = data_path or (Path.home() / ".pocketclaw" / "mem0_data")
        self._data_path.mkdir(parents=True, exist_ok=True)

        # Lazy initialization
        self._memory = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize Mem0 client."""
        if self._initialized:
            return

        try:
            from mem0 import Memory
            from mem0.configs.base import MemoryConfig

            # Configure Mem0 with local Qdrant storage
            config = MemoryConfig(
                vector_store={
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "pocketpaw_memory",
                        "path": str(self._data_path / "qdrant"),
                        "embedding_model_dims": 1536,
                    }
                },
                history_db_path=str(self._data_path / "history.db"),
                version="v1.1",
            )

            self._memory = Memory(config=config)
            self._initialized = True
            logger.info(f"Mem0 initialized with data at {self._data_path}")

        except ImportError:
            raise ImportError(
                "mem0ai package not installed. Install with: pip install mem0ai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            raise

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    # =========================================================================
    # MemoryStoreProtocol Implementation
    # =========================================================================

    async def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry using Mem0."""
        self._ensure_initialized()

        # Build metadata
        metadata = {
            "pocketpaw_type": entry.type.value,
            "tags": entry.tags,
            "created_at": entry.created_at.isoformat(),
            **entry.metadata,
        }

        # Determine scoping based on memory type
        if entry.type == MemoryType.SESSION:
            # Session memories use run_id for conversation isolation
            result = await self._run_sync(
                self._memory.add,
                entry.content,
                run_id=entry.session_key or "default_session",
                metadata=metadata,
                infer=False,  # Don't extract facts from conversation - store raw
            )
        elif entry.type == MemoryType.DAILY:
            # Daily notes scoped to user with date
            metadata["date"] = datetime.now().date().isoformat()
            result = await self._run_sync(
                self._memory.add,
                entry.content,
                user_id=self.user_id,
                metadata=metadata,
                infer=self.use_inference,
            )
        else:
            # Long-term memories - use full inference for fact extraction
            result = await self._run_sync(
                self._memory.add,
                entry.content,
                user_id=self.user_id,
                metadata=metadata,
                infer=self.use_inference,
            )

        # Extract memory ID from result
        if result and "results" in result and result["results"]:
            entry.id = result["results"][0].get("id", entry.id)

        logger.debug(f"Saved memory: {entry.id} ({entry.type.value})")
        return entry.id or ""

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID."""
        self._ensure_initialized()

        try:
            result = await self._run_sync(self._memory.get, entry_id)
            if result:
                return self._mem0_to_entry(result)
        except Exception as e:
            logger.warning(f"Failed to get memory {entry_id}: {e}")

        return None

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        self._ensure_initialized()

        try:
            await self._run_sync(self._memory.delete, entry_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete memory {entry_id}: {e}")
            return False

    async def search(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories using semantic search."""
        self._ensure_initialized()

        if not query:
            # Without a query, fall back to get_all with filters
            return await self._get_filtered(memory_type, tags, limit)

        # Build filters
        filters = {}
        if memory_type:
            filters["pocketpaw_type"] = memory_type.value

        try:
            result = await self._run_sync(
                self._memory.search,
                query,
                user_id=self.user_id,
                limit=limit,
                filters=filters if filters else None,
            )

            entries = []
            for item in result.get("results", []):
                entry = self._mem0_to_entry(item)
                # Filter by tags if specified
                if tags and not any(t in entry.tags for t in tags):
                    continue
                entries.append(entry)

            return entries[:limit]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _get_filtered(
        self,
        memory_type: MemoryType | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[MemoryEntry]:
        """Get memories with filters (no semantic search)."""
        filters = {}
        if memory_type:
            filters["pocketpaw_type"] = memory_type.value

        try:
            result = await self._run_sync(
                self._memory.get_all,
                user_id=self.user_id,
                limit=limit * 2,  # Get extra to filter
                filters=filters if filters else None,
            )

            entries = []
            for item in result.get("results", []):
                entry = self._mem0_to_entry(item)
                if tags and not any(t in entry.tags for t in tags):
                    continue
                entries.append(entry)
                if len(entries) >= limit:
                    break

            return entries

        except Exception as e:
            logger.error(f"Get filtered failed: {e}")
            return []

    async def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Get all memories of a specific type."""
        return await self._get_filtered(memory_type, None, limit)

    async def get_session(self, session_key: str) -> list[MemoryEntry]:
        """Get session history for a specific session."""
        self._ensure_initialized()

        try:
            result = await self._run_sync(
                self._memory.get_all,
                run_id=session_key,
                limit=1000,
            )

            entries = []
            for item in result.get("results", []):
                entry = self._mem0_to_entry(item)
                entry.session_key = session_key
                entries.append(entry)

            # Sort by creation time
            entries.sort(key=lambda e: e.created_at)
            return entries

        except Exception as e:
            logger.error(f"Get session failed: {e}")
            return []

    async def clear_session(self, session_key: str) -> int:
        """Clear session history."""
        self._ensure_initialized()

        try:
            # Get all session memories first to count
            result = await self._run_sync(
                self._memory.get_all,
                run_id=session_key,
                limit=1000,
            )
            count = len(result.get("results", []))

            # Delete all
            await self._run_sync(
                self._memory.delete_all,
                run_id=session_key,
            )

            return count

        except Exception as e:
            logger.error(f"Clear session failed: {e}")
            return 0

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _mem0_to_entry(self, mem0_item: dict) -> MemoryEntry:
        """Convert Mem0 memory item to MemoryEntry."""
        metadata = mem0_item.get("metadata", {})

        # Parse memory type
        type_str = metadata.get("pocketpaw_type", "long_term")
        try:
            mem_type = MemoryType(type_str)
        except ValueError:
            mem_type = MemoryType.LONG_TERM

        # Parse timestamps
        created_str = metadata.get("created_at")
        try:
            created_at = datetime.fromisoformat(created_str) if created_str else datetime.now()
        except (ValueError, TypeError):
            created_at = datetime.now()

        # Extract role for session memories
        role = metadata.get("role")

        return MemoryEntry(
            id=mem0_item.get("id", ""),
            type=mem_type,
            content=mem0_item.get("memory", ""),
            created_at=created_at,
            updated_at=datetime.now(),
            tags=metadata.get("tags", []),
            metadata={k: v for k, v in metadata.items()
                     if k not in ("pocketpaw_type", "tags", "created_at", "role")},
            role=role,
            session_key=metadata.get("session_key"),
        )

    # =========================================================================
    # Meta-Evolution Support (Future)
    # =========================================================================

    async def evolve_memories(self) -> dict[str, Any]:
        """
        Trigger memory evolution/consolidation.

        This is a hook for future meta-evolution capabilities:
        - Consolidate similar memories
        - Extract higher-level patterns
        - Prune outdated information

        Returns:
            Statistics about the evolution process.
        """
        # Mem0 handles some evolution automatically during add()
        # This method is a placeholder for more advanced meta-evolution
        logger.info("Memory evolution triggered (using Mem0's built-in consolidation)")
        return {"status": "ok", "message": "Using Mem0 built-in memory evolution"}

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about stored memories."""
        self._ensure_initialized()

        try:
            all_memories = await self._run_sync(
                self._memory.get_all,
                user_id=self.user_id,
                limit=10000,
            )

            results = all_memories.get("results", [])

            # Count by type
            type_counts = {}
            for item in results:
                mem_type = item.get("metadata", {}).get("pocketpaw_type", "unknown")
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

            return {
                "total_memories": len(results),
                "by_type": type_counts,
                "user_id": self.user_id,
                "data_path": str(self._data_path),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
