# Tests for Unified Agent Loop
# Created: 2026-02-02
# Updated: 2026-02-05 - Refactored to test router-based architecture

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pocketclaw.agents.loop import AgentLoop
from pocketclaw.bus import Channel, InboundMessage


@pytest.fixture
def mock_bus():
    bus = MagicMock()
    bus.consume_inbound = AsyncMock()
    bus.publish_outbound = AsyncMock()
    bus.publish_system = AsyncMock()
    return bus


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.add_to_session = AsyncMock()
    mem.get_session_history = AsyncMock(return_value=[])
    return mem


@pytest.fixture
def mock_router():
    """Mock AgentRouter that yields test responses."""
    router = MagicMock()

    async def mock_run(message):
        yield {"type": "message", "content": "Hello ", "metadata": {}}
        yield {"type": "message", "content": "world!", "metadata": {}}
        yield {
            "type": "tool_use",
            "content": "Using test_tool...",
            "metadata": {"name": "test_tool", "input": {}},
        }
        yield {
            "type": "tool_result",
            "content": "Tool completed",
            "metadata": {"name": "test_tool"},
        }
        yield {"type": "done", "content": ""}

    router.run = mock_run
    router.stop = AsyncMock()
    return router


@patch("pocketclaw.agents.loop.get_message_bus")
@patch("pocketclaw.agents.loop.get_memory_manager")
@patch("pocketclaw.agents.loop.AgentContextBuilder")
@patch("pocketclaw.agents.loop.AgentRouter")
@pytest.mark.asyncio
async def test_agent_loop_process_message(
    mock_router_cls,
    mock_builder_cls,
    mock_get_memory,
    mock_get_bus,
    mock_bus,
    mock_memory,
    mock_router,
):
    """Test that AgentLoop processes messages through the router."""
    # Setup mocks
    mock_get_bus.return_value = mock_bus
    mock_get_memory.return_value = mock_memory
    mock_router_cls.return_value = mock_router

    # Configure builder mock
    mock_builder_instance = mock_builder_cls.return_value
    mock_builder_instance.build_system_prompt = AsyncMock(return_value="System Prompt")

    # Mock settings
    with patch("pocketclaw.agents.loop.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_backend = "claude_agent_sdk"
        mock_settings.return_value = settings

        with patch("pocketclaw.agents.loop.Settings") as mock_settings_cls:
            mock_settings_cls.load.return_value = settings

            # Init loop
            loop = AgentLoop()

            # Create test message
            msg = InboundMessage(
                channel=Channel.CLI,
                sender_id="user1",
                chat_id="chat1",
                content="Hello",
            )

            # Test processing
            await loop._process_message(msg)

            # Verify memory was updated (user message saved)
            mock_memory.add_to_session.assert_called()

            # Verify outbound messages were published (streaming chunks)
            assert mock_bus.publish_outbound.call_count >= 2  # At least message chunks + stream_end

            # Verify system events were emitted (thinking, tool_start, tool_result)
            assert mock_bus.publish_system.call_count >= 1  # At least thinking event


@patch("pocketclaw.agents.loop.get_message_bus")
@patch("pocketclaw.agents.loop.get_memory_manager")
@patch("pocketclaw.agents.loop.AgentContextBuilder")
@pytest.mark.asyncio
async def test_agent_loop_reset_router(
    mock_builder_cls, mock_get_memory, mock_get_bus, mock_bus, mock_memory
):
    """Test that reset_router clears the router instance."""
    mock_get_bus.return_value = mock_bus
    mock_get_memory.return_value = mock_memory

    with patch("pocketclaw.agents.loop.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_backend = "claude_agent_sdk"
        mock_settings.return_value = settings

        loop = AgentLoop()

        # Initially no router
        assert loop._router is None

        # After reset, still None (lazy init)
        loop.reset_router()
        assert loop._router is None


@patch("pocketclaw.agents.loop.get_message_bus")
@patch("pocketclaw.agents.loop.get_memory_manager")
@patch("pocketclaw.agents.loop.AgentContextBuilder")
@patch("pocketclaw.agents.loop.AgentRouter")
@pytest.mark.asyncio
async def test_agent_loop_handles_error(
    mock_router_cls, mock_builder_cls, mock_get_memory, mock_get_bus, mock_bus, mock_memory
):
    """Test that AgentLoop handles errors gracefully."""
    mock_get_bus.return_value = mock_bus
    mock_get_memory.return_value = mock_memory

    # Router that raises an error
    error_router = MagicMock()

    async def mock_run_error(message):
        yield {"type": "error", "content": "Something went wrong", "metadata": {}}
        yield {"type": "done", "content": ""}

    error_router.run = mock_run_error
    mock_router_cls.return_value = error_router

    mock_builder_instance = mock_builder_cls.return_value
    mock_builder_instance.build_system_prompt = AsyncMock(return_value="System Prompt")

    with patch("pocketclaw.agents.loop.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_backend = "claude_agent_sdk"
        mock_settings.return_value = settings

        with patch("pocketclaw.agents.loop.Settings") as mock_settings_cls:
            mock_settings_cls.load.return_value = settings

            loop = AgentLoop()

            msg = InboundMessage(
                channel=Channel.CLI,
                sender_id="user1",
                chat_id="chat1",
                content="Hello",
            )

            # Should not raise
            await loop._process_message(msg)

            # Verify error was published via system event
            mock_bus.publish_system.assert_called()


@patch("pocketclaw.agents.loop.get_message_bus")
@patch("pocketclaw.agents.loop.get_memory_manager")
@patch("pocketclaw.agents.loop.AgentContextBuilder")
@patch("pocketclaw.agents.loop.AgentRouter")
@pytest.mark.asyncio
async def test_agent_loop_emits_tool_events(
    mock_router_cls,
    mock_builder_cls,
    mock_get_memory,
    mock_get_bus,
    mock_bus,
    mock_memory,
    mock_router,
):
    """Test that tool_use and tool_result events are emitted as SystemEvents."""
    mock_get_bus.return_value = mock_bus
    mock_get_memory.return_value = mock_memory
    mock_router_cls.return_value = mock_router

    mock_builder_instance = mock_builder_cls.return_value
    mock_builder_instance.build_system_prompt = AsyncMock(return_value="System Prompt")

    with patch("pocketclaw.agents.loop.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_backend = "claude_agent_sdk"
        mock_settings.return_value = settings

        with patch("pocketclaw.agents.loop.Settings") as mock_settings_cls:
            mock_settings_cls.load.return_value = settings

            loop = AgentLoop()

            msg = InboundMessage(
                channel=Channel.CLI,
                sender_id="user1",
                chat_id="chat1",
                content="Run a tool",
            )

            await loop._process_message(msg)

            # Check that system events include tool_start and tool_result
            system_calls = mock_bus.publish_system.call_args_list
            event_types = [call[0][0].event_type for call in system_calls]

            assert "thinking" in event_types
            assert "tool_start" in event_types
            assert "tool_result" in event_types
