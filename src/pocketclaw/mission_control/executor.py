"""Mission Control Task Executor.

Created: 2026-02-05
Updated: 2026-02-12 - Fixed execute_task_background self-defeating bug: removed
  stale _running_tasks check from execute_task, added guard + cleanup wrapper
  to execute_task_background to prevent zombie entries and 409 Conflicts.
  Previous: 2026-02-05 - Added task output persistence (auto-save deliverables on completion)

Enables execution of AI agents on tasks with real-time streaming via WebSocket.

Key features:
- Creates dedicated AgentRouter per task for isolation
- Uses agent's backend field (claude_agent_sdk, pocketpaw_native, open_interpreter)
- Streams execution to activity feed
- Updates task/agent status automatically
- Broadcasts events via MessageBus → WebSocket
- Auto-saves task output as deliverable document on completion

Security features:
- Max concurrent task limit (default: 5)
- UUID validation for task_id and agent_id
- Error message sanitization (no sensitive details exposed)
- Security audit logging

WebSocket Events:
- mc_task_started: Task execution begins
- mc_task_output: Agent produces output
- mc_task_completed: Execution ends (done/error)
- mc_activity_created: Activity logged
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any

# UUID validation pattern
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Security constants
MAX_CONCURRENT_TASKS = 5  # Prevent resource exhaustion
MAX_ERROR_MESSAGE_LENGTH = 200  # Truncate error messages

from pocketclaw.agents.router import AgentRouter
from pocketclaw.bus.events import SystemEvent
from pocketclaw.bus.queue import get_message_bus
from pocketclaw.config import Settings, get_settings
from pocketclaw.mission_control.manager import get_mission_control_manager
from pocketclaw.mission_control.models import (
    Activity,
    ActivityType,
    AgentStatus,
    TaskStatus,
    now_iso,
)

logger = logging.getLogger(__name__)


class MCTaskExecutor:
    """Executes Mission Control tasks with AI agents.

    Creates isolated agent instances per task and broadcasts execution
    events via the MessageBus for real-time WebSocket updates.

    Usage:
        executor = get_mc_task_executor()
        await executor.execute_task(task_id, agent_id)
    """

    def __init__(self):
        """Initialize the executor."""
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._agent_routers: dict[str, AgentRouter] = {}
        self._stop_flags: dict[str, bool] = {}

    async def execute_task(
        self,
        task_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """Execute a task with the specified agent.

        Creates a dedicated AgentRouter for the task, streams output
        via WebSocket, and updates task/agent status.

        Security:
        - Validates task_id and agent_id are valid UUIDs
        - Enforces max concurrent task limit
        - Sanitizes error messages before broadcast

        Args:
            task_id: ID of the task to execute
            agent_id: ID of the agent to run

        Returns:
            Dict with execution result:
            - status: "completed" | "error" | "stopped"
            - output: Full output from agent
            - error: Error message if failed
        """
        # Security: Validate input IDs are valid UUIDs
        if not self._is_valid_uuid(task_id):
            logger.warning(f"Security: Invalid task_id format: {task_id[:50]}")
            return {"status": "error", "error": "Invalid task ID format"}

        if not self._is_valid_uuid(agent_id):
            logger.warning(f"Security: Invalid agent_id format: {agent_id[:50]}")
            return {"status": "error", "error": "Invalid agent ID format"}

        # Security: Rate limit - check concurrent task count
        if len(self._running_tasks) >= MAX_CONCURRENT_TASKS:
            logger.warning(
                f"Security: Max concurrent tasks ({MAX_CONCURRENT_TASKS}) reached. "
                f"Rejecting task {task_id}"
            )
            return {
                "status": "error",
                "error": f"Maximum concurrent tasks ({MAX_CONCURRENT_TASKS}) reached. Please wait.",
            }

        manager = get_mission_control_manager()

        # Load task and agent
        task = await manager.get_task(task_id)
        if not task:
            return {"status": "error", "error": "Task not found"}

        agent = await manager.get_agent(agent_id)
        if not agent:
            return {"status": "error", "error": "Agent not found"}

        # Security: Log task execution start
        logger.info(
            f"Task execution starting: task={task_id}, agent={agent_id}, "
            f"agent_name={agent.name}, task_title={task.title}"
        )

        # Initialize stop flag
        self._stop_flags[task_id] = False

        # Build agent settings with the agent's backend.
        # bypass_permissions is ALWAYS True for task execution because
        # tasks run headlessly (no terminal for interactive prompts).
        # The PreToolUse hook still blocks dangerous commands.
        base_settings = get_settings()
        agent_settings = Settings(
            agent_backend=agent.backend,
            anthropic_api_key=base_settings.anthropic_api_key,
            anthropic_model=base_settings.anthropic_model,
            openai_api_key=base_settings.openai_api_key,
            openai_model=base_settings.openai_model,
            ollama_host=base_settings.ollama_host,
            ollama_model=base_settings.ollama_model,
            llm_provider=base_settings.llm_provider,
            bypass_permissions=True,
        )

        # Create dedicated router for this task
        router = AgentRouter(agent_settings)
        self._agent_routers[task_id] = router

        # Update task and agent status
        await manager.update_task_status(task_id, TaskStatus.IN_PROGRESS, agent_id)
        await manager.set_agent_status(agent_id, AgentStatus.ACTIVE, task_id)

        # Broadcast task started event
        await self._broadcast_event(
            "mc_task_started",
            {
                "task_id": task_id,
                "agent_id": agent_id,
                "agent_name": agent.name,
                "task_title": task.title,
                "timestamp": now_iso(),
            },
        )

        # Log activity
        activity = await self._log_activity(
            ActivityType.TASK_UPDATED,
            agent_id=agent_id,
            task_id=task_id,
            message=f"{agent.name} started working on '{task.title}'",
        )

        # Build the prompt for the agent
        prompt = self._build_task_prompt(task, agent)

        # Execute and collect output
        output_chunks: list[str] = []
        final_status = "completed"
        error_message = None

        try:
            async for chunk in router.run(prompt):
                # Check stop flag
                if self._stop_flags.get(task_id):
                    final_status = "stopped"
                    break

                chunk_type = chunk.get("type", "")
                content = chunk.get("content", "")

                if chunk_type == "message" and content:
                    output_chunks.append(content)
                    # Broadcast output chunk
                    await self._broadcast_event(
                        "mc_task_output",
                        {
                            "task_id": task_id,
                            "content": content,
                            "output_type": "message",
                            "timestamp": now_iso(),
                        },
                    )

                elif chunk_type == "tool_use":
                    tool_name = chunk.get("metadata", {}).get("name", "unknown")
                    await self._broadcast_event(
                        "mc_task_output",
                        {
                            "task_id": task_id,
                            "content": f"Using tool: {tool_name}",
                            "output_type": "tool_use",
                            "timestamp": now_iso(),
                        },
                    )

                elif chunk_type == "tool_result":
                    result = content[:200] if content else ""
                    await self._broadcast_event(
                        "mc_task_output",
                        {
                            "task_id": task_id,
                            "content": f"Tool result: {result}",
                            "output_type": "tool_result",
                            "timestamp": now_iso(),
                        },
                    )

                elif chunk_type == "error":
                    error_message = content
                    final_status = "error"
                    break

                elif chunk_type == "done":
                    break

        except Exception as e:
            logger.exception(f"Error executing task {task_id}")
            # Security: Sanitize error message - don't expose internal details
            error_message = self._sanitize_error(str(e))
            final_status = "error"

        finally:
            # Cleanup
            self._agent_routers.pop(task_id, None)
            self._running_tasks.pop(task_id, None)
            self._stop_flags.pop(task_id, None)

            # Update statuses
            new_task_status = TaskStatus.DONE if final_status == "completed" else TaskStatus.BLOCKED
            await manager.update_task_status(task_id, new_task_status, agent_id)
            await manager.set_agent_status(agent_id, AgentStatus.IDLE, None)

            # Broadcast completion
            await self._broadcast_event(
                "mc_task_completed",
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "status": final_status,
                    "error": error_message,
                    "timestamp": now_iso(),
                },
            )

            # Log completion activity
            if final_status == "completed":
                await self._log_activity(
                    ActivityType.TASK_COMPLETED,
                    agent_id=agent_id,
                    task_id=task_id,
                    message=f"{agent.name} completed '{task.title}'",
                )

                # Save task output as a deliverable document
                if output_chunks:
                    full_output = "".join(output_chunks)
                    await self._save_task_deliverable(
                        task_id=task_id,
                        agent_id=agent_id,
                        output=full_output,
                        task_title=task.title,
                    )

            elif final_status == "error":
                await self._log_activity(
                    ActivityType.TASK_UPDATED,
                    agent_id=agent_id,
                    task_id=task_id,
                    message=f"{agent.name} encountered an error on '{task.title}': {error_message}",
                )
            elif final_status == "stopped":
                await self._log_activity(
                    ActivityType.TASK_UPDATED,
                    agent_id=agent_id,
                    task_id=task_id,
                    message=f"Execution stopped for '{task.title}'",
                )

        full_output = "".join(output_chunks)
        return {
            "status": final_status,
            "output": full_output,
            "error": error_message,
        }

    async def execute_task_background(
        self,
        task_id: str,
        agent_id: str,
    ) -> None:
        """Start task execution in the background.

        Returns immediately. Task runs in a background asyncio task.
        Use stop_task() to cancel execution.

        Guards against double-dispatch: if task_id is already tracked in
        _running_tasks the call is silently skipped.  A cleanup wrapper
        ensures the tracking entry is removed even when execute_task
        returns early (e.g. validation failure), preventing zombie entries.

        Args:
            task_id: ID of the task to execute
            agent_id: ID of the agent to run
        """
        if task_id in self._running_tasks:
            logger.warning(f"Task {task_id} is already running, skipping duplicate dispatch")
            return

        async def _run_and_cleanup():
            try:
                return await self.execute_task(task_id, agent_id)
            finally:
                # Guarantee cleanup even if execute_task returns early
                self._running_tasks.pop(task_id, None)

        async_task = asyncio.create_task(_run_and_cleanup())
        self._running_tasks[task_id] = async_task

    async def stop_task(self, task_id: str) -> bool:
        """Stop a running task.

        Args:
            task_id: ID of the task to stop

        Returns:
            True if task was stopped, False if not running
        """
        if task_id not in self._running_tasks:
            return False

        # Set stop flag
        self._stop_flags[task_id] = True

        # Stop the agent router if exists
        router = self._agent_routers.get(task_id)
        if router:
            try:
                await router.stop()
            except Exception as e:
                logger.warning(f"Error stopping router for task {task_id}: {e}")

        # Cancel the asyncio task
        async_task = self._running_tasks.get(task_id)
        if async_task and not async_task.done():
            async_task.cancel()
            try:
                await async_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Stopped task execution: {task_id}")
        return True

    def is_task_running(self, task_id: str) -> bool:
        """Check if a task is currently running.

        Args:
            task_id: ID of the task to check

        Returns:
            True if task is running
        """
        return task_id in self._running_tasks

    def get_running_tasks(self) -> list[str]:
        """Get list of currently running task IDs.

        Returns:
            List of task IDs
        """
        return list(self._running_tasks.keys())

    def _is_valid_uuid(self, value: str) -> bool:
        """Validate that a string is a valid UUID.

        Security: Prevents injection via malformed IDs.

        Args:
            value: String to validate

        Returns:
            True if valid UUID format
        """
        if not value or not isinstance(value, str):
            return False
        return bool(UUID_PATTERN.match(value))

    def _sanitize_error(self, error: str) -> str:
        """Sanitize error message for safe broadcast.

        Security: Removes potentially sensitive information like:
        - File paths
        - API keys
        - Stack traces
        - Internal implementation details

        Args:
            error: Raw error message

        Returns:
            Sanitized error message
        """
        if not error:
            return "An error occurred"

        # Truncate to max length
        sanitized = error[:MAX_ERROR_MESSAGE_LENGTH]

        # Remove potential file paths
        sanitized = re.sub(r"/[^\s]+/[^\s]+", "[path]", sanitized)

        # Remove potential API keys or tokens
        sanitized = re.sub(
            r"(key|token|secret|password)[=:]\s*\S+",
            r"\1=[redacted]",
            sanitized,
            flags=re.IGNORECASE,
        )

        # If truncated, add indicator
        if len(error) > MAX_ERROR_MESSAGE_LENGTH:
            sanitized = sanitized.rstrip() + "..."

        return sanitized

    def _build_task_prompt(self, task, agent) -> str:
        """Build the prompt to send to the agent.

        Includes task context and agent instructions.
        """
        prompt_parts = [
            f"You are {agent.name}, a {agent.role}.",
        ]

        if agent.description:
            prompt_parts.append(f"Description: {agent.description}")

        if agent.specialties:
            prompt_parts.append(f"Specialties: {', '.join(agent.specialties)}")

        prompt_parts.extend(
            [
                "",
                "## Task",
                f"**Title:** {task.title}",
            ]
        )

        if task.description:
            prompt_parts.append(f"**Description:** {task.description}")

        prompt_parts.extend(
            [
                f"**Priority:** {task.priority.value}",
                "",
                "Please complete this task. Provide your work and findings.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _broadcast_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast an event via the MessageBus.

        Events are picked up by the WebSocket adapter and sent to clients.

        Args:
            event_type: Type of event (mc_task_started, mc_task_output, etc.)
            data: Event data
        """
        bus = get_message_bus()
        event = SystemEvent(
            event_type=event_type,
            data=data,
            timestamp=datetime.now(UTC),
        )
        await bus.publish_system(event)

    async def _log_activity(
        self,
        activity_type: ActivityType,
        agent_id: str | None = None,
        task_id: str | None = None,
        message: str = "",
    ) -> Activity:
        """Log an activity and broadcast it via WebSocket.

        Args:
            activity_type: Type of activity
            agent_id: Agent that triggered the activity
            task_id: Related task
            message: Human-readable description

        Returns:
            The created Activity
        """
        manager = get_mission_control_manager()

        activity = Activity(
            type=activity_type,
            agent_id=agent_id,
            task_id=task_id,
            message=message,
        )
        await manager._store.save_activity(activity)

        # Broadcast activity created event
        await self._broadcast_event(
            "mc_activity_created",
            {
                "activity": activity.to_dict(),
            },
        )

        return activity

    async def _save_task_deliverable(
        self,
        task_id: str,
        agent_id: str,
        output: str,
        task_title: str,
    ) -> None:
        """Save agent output as a deliverable document.

        Creates a Document of type DELIVERABLE linked to the task.
        This persists the agent's work for later review.

        Args:
            task_id: ID of the completed task
            agent_id: ID of the agent that completed the task
            output: Full text output from the agent
            task_title: Title of the task (for document title)
        """
        from pocketclaw.mission_control.models import Document, DocumentType

        if not output or not output.strip():
            return

        manager = get_mission_control_manager()

        # Create deliverable document
        document = Document(
            title=f"Deliverable: {task_title}",
            content=output,
            type=DocumentType.DELIVERABLE,
            author_id=agent_id,
            task_id=task_id,
            tags=["auto-generated", "task-output"],
        )

        await manager._store.save_document(document)

        logger.info(
            f"Saved task deliverable: doc_id={document.id}, task_id={task_id}, length={len(output)}"
        )

        # Log activity
        await self._log_activity(
            ActivityType.DOCUMENT_CREATED,
            agent_id=agent_id,
            task_id=task_id,
            message=f"Deliverable saved for '{task_title}'",
        )


# Singleton pattern
_executor_instance: MCTaskExecutor | None = None


def get_mc_task_executor() -> MCTaskExecutor:
    """Get or create the MC Task Executor singleton.

    Returns:
        The MCTaskExecutor instance
    """
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = MCTaskExecutor()
    return _executor_instance


def reset_mc_task_executor() -> None:
    """Reset the executor singleton (for testing)."""
    global _executor_instance
    _executor_instance = None
