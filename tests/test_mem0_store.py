# Tests for Mem0 Memory Store Integration
# Created: 2026-02-04

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from pocketclaw.memory.protocol import MemoryEntry, MemoryType
from pocketclaw.memory.manager import create_memory_store, MemoryManager
from pocketclaw.memory.file_store import FileMemoryStore


class TestCreateMemoryStore:
    """Test the memory store factory function."""

    def test_create_file_store_default(self):
        """Default backend should be file store."""
        store = create_memory_store()
        assert isinstance(store, FileMemoryStore)

    def test_create_file_store_explicit(self):
        """Explicit file backend creates FileMemoryStore."""
        store = create_memory_store(backend="file")
        assert isinstance(store, FileMemoryStore)

    def test_create_unknown_backend_falls_back(self):
        """Unknown backend should fall back to file store."""
        store = create_memory_store(backend="unknown")
        assert isinstance(store, FileMemoryStore)


class TestMemoryManagerBackendSelection:
    """Test MemoryManager with different backends."""

    def test_manager_with_file_backend(self):
        """MemoryManager should use file backend by default."""
        manager = MemoryManager(backend="file")
        assert isinstance(manager._store, FileMemoryStore)

    def test_manager_with_custom_store(self):
        """MemoryManager should accept custom store."""
        mock_store = MagicMock()
        manager = MemoryManager(store=mock_store)
        assert manager._store is mock_store


# Only run Mem0-specific tests if mem0ai is installed
try:
    from mem0 import Memory
    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False


@pytest.mark.skipif(not HAS_MEM0, reason="mem0ai not installed")
class TestMem0MemoryStore:
    """Tests for Mem0MemoryStore (requires mem0ai package)."""

    @pytest.fixture
    def mock_mem0_memory(self):
        """Create a mock Mem0 Memory instance."""
        mock_instance = MagicMock()

        # Setup default return values
        mock_instance.add.return_value = {
            "results": [{"id": "test-id-123", "memory": "test content", "event": "ADD"}]
        }
        mock_instance.get.return_value = {
            "id": "test-id-123",
            "memory": "test content",
            "metadata": {"pocketpaw_type": "long_term", "tags": ["test"]}
        }
        mock_instance.search.return_value = {
            "results": [
                {
                    "id": "test-id-123",
                    "memory": "test content",
                    "metadata": {"pocketpaw_type": "long_term", "tags": ["test"]}
                }
            ]
        }
        mock_instance.get_all.return_value = {
            "results": [
                {
                    "id": "test-id-123",
                    "memory": "test content",
                    "metadata": {"pocketpaw_type": "long_term", "tags": ["test"]}
                }
            ]
        }
        mock_instance.delete.return_value = None
        mock_instance.delete_all.return_value = None

        return mock_instance

    @pytest.fixture
    def mem0_store(self, mock_mem0_memory, tmp_path):
        """Create a Mem0MemoryStore with mocked Memory."""
        from pocketclaw.memory.mem0_store import Mem0MemoryStore

        store = Mem0MemoryStore(
            user_id="test-user",
            data_path=tmp_path / "mem0_data",
            use_inference=False,
        )
        # Inject mock - bypass lazy initialization
        store._memory = mock_mem0_memory
        store._initialized = True
        return store

    @pytest.mark.asyncio
    async def test_save_long_term_memory(self, mem0_store, mock_mem0_memory):
        """Test saving a long-term memory."""
        entry = MemoryEntry(
            id="",
            type=MemoryType.LONG_TERM,
            content="User prefers dark mode",
            tags=["preferences"],
        )

        result_id = await mem0_store.save(entry)

        assert result_id == "test-id-123"
        mock_mem0_memory.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_session_memory(self, mem0_store, mock_mem0_memory):
        """Test saving a session memory."""
        entry = MemoryEntry(
            id="",
            type=MemoryType.SESSION,
            content="Hello, how are you?",
            role="user",
            session_key="test-session",
        )

        result_id = await mem0_store.save(entry)

        assert result_id == "test-id-123"
        # Session memories should use run_id
        call_kwargs = mock_mem0_memory.add.call_args[1]
        assert call_kwargs.get("run_id") == "test-session"
        assert call_kwargs.get("infer") is False  # Raw storage for sessions

    @pytest.mark.asyncio
    async def test_search_memories(self, mem0_store, mock_mem0_memory):
        """Test searching memories."""
        results = await mem0_store.search(query="dark mode", limit=5)

        assert len(results) == 1
        assert results[0].content == "test content"
        mock_mem0_memory.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_type(self, mem0_store, mock_mem0_memory):
        """Test getting memories by type."""
        results = await mem0_store.get_by_type(MemoryType.LONG_TERM)

        assert len(results) >= 0
        mock_mem0_memory.get_all.assert_called()

    @pytest.mark.asyncio
    async def test_delete_memory(self, mem0_store, mock_mem0_memory):
        """Test deleting a memory."""
        result = await mem0_store.delete("test-id-123")

        assert result is True
        mock_mem0_memory.delete.assert_called_once_with("test-id-123")

    @pytest.mark.asyncio
    async def test_clear_session(self, mem0_store, mock_mem0_memory):
        """Test clearing a session."""
        count = await mem0_store.clear_session("test-session")

        assert count == 1
        mock_mem0_memory.delete_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, mem0_store, mock_mem0_memory):
        """Test getting memory statistics."""
        stats = await mem0_store.get_memory_stats()

        assert "total_memories" in stats
        assert stats["user_id"] == "test-user"


class TestMemoryEntryConversion:
    """Test conversion between Mem0 format and MemoryEntry."""

    @pytest.mark.skipif(not HAS_MEM0, reason="mem0ai not installed")
    def test_mem0_to_entry_conversion(self, tmp_path):
        """Test converting Mem0 item to MemoryEntry."""
        from pocketclaw.memory.mem0_store import Mem0MemoryStore

        store = Mem0MemoryStore.__new__(Mem0MemoryStore)

        mem0_item = {
            "id": "test-id",
            "memory": "Test memory content",
            "metadata": {
                "pocketpaw_type": "long_term",
                "tags": ["test", "example"],
                "created_at": "2026-02-04T10:00:00",
                "custom_field": "custom_value",
            }
        }

        entry = store._mem0_to_entry(mem0_item)

        assert entry.id == "test-id"
        assert entry.content == "Test memory content"
        assert entry.type == MemoryType.LONG_TERM
        assert "test" in entry.tags
        assert "custom_field" in entry.metadata

    @pytest.mark.skipif(not HAS_MEM0, reason="mem0ai not installed")
    def test_mem0_to_entry_handles_missing_type(self, tmp_path):
        """Test conversion handles missing memory type gracefully."""
        from pocketclaw.memory.mem0_store import Mem0MemoryStore

        store = Mem0MemoryStore.__new__(Mem0MemoryStore)

        mem0_item = {
            "id": "test-id",
            "memory": "Test content",
            "metadata": {}
        }

        entry = store._mem0_to_entry(mem0_item)

        assert entry.type == MemoryType.LONG_TERM  # Default
