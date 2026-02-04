"""
Claude Agent SDK wrapper for PocketPaw.

Uses the official Claude Agent SDK (pip install claude-agent-sdk) which provides:
- Built-in tools: Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
- Streaming responses
- PreToolUse hooks for security
- Permission management
- MCP server support for custom tools

Created: 2026-02-02
Changes:
  - 2026-02-02: Initial implementation with streaming support.
  - 2026-02-02: Added set_executor() for 2-layer architecture wiring.
  - 2026-02-02: Fixed streaming - properly handle all SDK message types.
  - 2026-02-02: REWRITE - Use official claude-agent-sdk properly with all features.
                Now uses real SDK imports (AssistantMessage, TextBlock, etc.)
"""

import logging
from pathlib import Path
from typing import AsyncIterator, Optional, Any

from pocketclaw.config import Settings
from pocketclaw.agents.protocol import AgentEvent, ExecutorProtocol

logger = logging.getLogger(__name__)

# Dangerous command patterns to block via PreToolUse hook
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf *",
    "sudo rm",
    "> /dev/",
    "format ",
    "mkfs",
    "chmod 777 /",
    ":(){ :|:& };:",  # Fork bomb
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "> /etc/passwd",
    "> /etc/shadow",
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
]

# PocketPaw system prompt for Claude Agent SDK
SYSTEM_PROMPT = """You are PocketPaw, a helpful AI assistant running locally on the user's computer.

You have powerful tools at your disposal:
- Bash: Run shell commands
- Read/Write/Edit: File operations
- Glob/Grep: Search files and content
- WebSearch: Search the web for information
- WebFetch: Fetch content from URLs

## Guidelines

1. **Be AGENTIC** - Don't just describe what to do, actually DO it using your tools.
2. **Query actual data** - When asked about calendar, emails, etc., use Bash with AppleScript/Python to QUERY the data.
3. **Be concise** - Give clear, helpful responses.
4. **Be safe** - Don't run destructive commands. Ask for confirmation if unsure.

## Examples of AGENTIC behavior

| User Says | You Should Do |
|-----------|---------------|
| "What's on my calendar today?" | Use Bash with AppleScript to query Calendar.app and return actual events |
| "Find files with TODO" | Use Grep to search and return matching files |
| "What's the weather?" | Use WebSearch to find current weather |
| "Download that PDF" | Use WebFetch or Bash with curl |

Always execute tasks and return results, don't just explain how to do them.
"""


class ClaudeAgentSDK:
    """Wraps Claude Agent SDK for autonomous task execution.

    This is the RECOMMENDED backend for PocketPaw - it provides:
    - All built-in tools (Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch)
    - Streaming responses for real-time feedback
    - PreToolUse hooks for security (block dangerous commands)
    - Permission management (can bypass for automation)

    Requires: pip install claude-agent-sdk
    """

    def __init__(self, settings: Settings, executor: Optional[ExecutorProtocol] = None):
        self.settings = settings
        self._executor = executor  # Optional - SDK has built-in execution
        self._stop_flag = False
        self._sdk_available = False
        self._cwd = Path.home()  # Default working directory

        # SDK imports (set during initialization)
        self._query = None
        self._ClaudeAgentOptions = None
        self._HookMatcher = None
        self._AssistantMessage = None
        self._UserMessage = None
        self._SystemMessage = None
        self._ResultMessage = None
        self._TextBlock = None
        self._ToolUseBlock = None
        self._ToolResultBlock = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Claude Agent SDK with all imports."""
        try:
            # Core SDK imports
            from claude_agent_sdk import (
                query,
                ClaudeAgentOptions,
                HookMatcher,
            )

            # Message type imports
            from claude_agent_sdk import (
                AssistantMessage,
                UserMessage,
                SystemMessage,
                ResultMessage,
            )

            # Content block imports
            from claude_agent_sdk import (
                TextBlock,
                ToolUseBlock,
                ToolResultBlock,
            )

            # Store references
            self._query = query
            self._ClaudeAgentOptions = ClaudeAgentOptions
            self._HookMatcher = HookMatcher
            self._AssistantMessage = AssistantMessage
            self._UserMessage = UserMessage
            self._SystemMessage = SystemMessage
            self._ResultMessage = ResultMessage
            self._TextBlock = TextBlock
            self._ToolUseBlock = ToolUseBlock
            self._ToolResultBlock = ToolResultBlock

            self._sdk_available = True
            logger.info("✓ Claude Agent SDK ready ─ cwd: %s", self._cwd)

        except ImportError as e:
            logger.warning("⚠️ Claude Agent SDK not installed ─ pip install claude-agent-sdk")
            logger.debug("Import error: %s", e)
            self._sdk_available = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Claude Agent SDK: {e}")
            self._sdk_available = False

    def set_executor(self, executor: ExecutorProtocol) -> None:
        """Inject an optional executor for custom tool execution.

        Note: Claude Agent SDK has built-in execution, so this is optional.
        Can be used for custom tools or fallback execution.
        """
        self._executor = executor
        logger.info("🔗 Optional executor connected to Claude Agent SDK")

    def set_working_directory(self, path: Path) -> None:
        """Set the working directory for file operations."""
        self._cwd = path
        logger.info(f"📂 Working directory set to: {path}")

    def _is_dangerous_command(self, command: str) -> Optional[str]:
        """Check if a command matches dangerous patterns.

        Args:
            command: Command string to check

        Returns:
            The matched pattern if dangerous, None otherwise
        """
        command_lower = command.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in command_lower:
                return pattern
        return None

    async def _block_dangerous_hook(
        self,
        input_data: dict,
        tool_use_id: str,
        context: dict
    ) -> dict:
        """PreToolUse hook to block dangerous commands.

        This hook is called before any Bash command is executed.
        Returns a deny decision for dangerous commands.

        Args:
            input_data: Contains tool_name and tool_input
            tool_use_id: Unique ID for this tool use
            context: Additional context from the SDK

        Returns:
            Empty dict to allow, or deny decision dict to block
        """
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Only check Bash commands
        if tool_name != "Bash":
            return {}

        command = str(tool_input.get("command", ""))

        matched = self._is_dangerous_command(command)
        if matched:
            logger.warning(f"🛑 BLOCKED dangerous command: {command[:100]}")
            logger.warning(f"   └─ Matched pattern: {matched}")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"PocketPaw security: '{matched}' pattern is blocked",
                }
            }

        logger.debug(f"✅ Allowed command: {command[:50]}...")
        return {}

    def _extract_text_from_message(self, message: Any) -> str:
        """Extract text content from an AssistantMessage.

        Args:
            message: AssistantMessage with content blocks

        Returns:
            Concatenated text from all TextBlocks
        """
        if not hasattr(message, 'content'):
            return ""

        content = message.content
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for block in content:
                # Check if it's a TextBlock
                if self._TextBlock and isinstance(block, self._TextBlock):
                    if hasattr(block, 'text') and block.text:
                        texts.append(block.text)
                # Fallback: check for text attribute
                elif hasattr(block, 'text') and isinstance(block.text, str):
                    texts.append(block.text)
            return "".join(texts)

        return ""

    def _extract_tool_info(self, message: Any) -> list[dict]:
        """Extract tool use information from an AssistantMessage.

        Args:
            message: AssistantMessage with content blocks

        Returns:
            List of tool use dicts with name and input
        """
        if not hasattr(message, 'content') or message.content is None:
            return []

        tools = []
        for block in message.content:
            if self._ToolUseBlock and isinstance(block, self._ToolUseBlock):
                tools.append({
                    "name": getattr(block, 'name', 'unknown'),
                    "input": getattr(block, 'input', {}),
                })
            elif hasattr(block, 'name') and hasattr(block, 'input'):
                # Fallback check
                tools.append({
                    "name": block.name,
                    "input": block.input,
                })
        return tools

    async def chat(self, message: str) -> AsyncIterator[AgentEvent]:
        """Process a message through Claude Agent SDK with streaming.

        Uses the SDK's built-in tools and streaming capabilities.

        Args:
            message: User message to process

        Yields:
            AgentEvent objects as the agent responds
        """
        if not self._sdk_available:
            yield AgentEvent(
                type="error",
                content="❌ Claude Agent SDK not available.\n\nInstall with: pip install claude-agent-sdk\n\nNote: Requires Claude Code CLI to be installed.",
            )
            return

        self._stop_flag = False

        try:
            # Build allowed tools list - all built-in tools
            allowed_tools = [
                # Shell & System
                "Bash",
                # File operations
                "Read", "Write", "Edit",
                # Search
                "Glob", "Grep",
                # Web (the killer features!)
                "WebSearch", "WebFetch",
            ]

            # Build hooks for security
            hooks = {
                "PreToolUse": [
                    self._HookMatcher(
                        matcher="Bash",  # Only hook Bash commands
                        hooks=[self._block_dangerous_hook],
                    )
                ]
            }

            # Build options
            options_kwargs = {
                "system_prompt": SYSTEM_PROMPT,
                "allowed_tools": allowed_tools,
                "hooks": hooks,
                "cwd": str(self._cwd),  # Working directory
            }

            # Permission mode based on settings
            if self.settings.bypass_permissions:
                options_kwargs["permission_mode"] = "bypassPermissions"
                logger.info("⚡ Permission bypass enabled")
            else:
                # Accept edits automatically but prompt for other things
                options_kwargs["permission_mode"] = "acceptEdits"

            # Create options
            options = self._ClaudeAgentOptions(**options_kwargs)

            logger.debug(f"🚀 Starting Claude Agent SDK query: {message[:100]}...")

            # Stream responses from the SDK
            async for event in self._query(prompt=message, options=options):
                if self._stop_flag:
                    logger.info("🛑 Stop flag set, breaking stream")
                    break

                # Handle different message types using isinstance checks

                # ========== SystemMessage - metadata, skip ==========
                if self._SystemMessage and isinstance(event, self._SystemMessage):
                    subtype = getattr(event, 'subtype', '')
                    logger.debug(f"SystemMessage: {subtype}")
                    continue

                # ========== UserMessage - echo, skip ==========
                if self._UserMessage and isinstance(event, self._UserMessage):
                    logger.debug("UserMessage (echo), skipping")
                    continue

                # ========== AssistantMessage - main content ==========
                if self._AssistantMessage and isinstance(event, self._AssistantMessage):
                    # Extract and yield text
                    text = self._extract_text_from_message(event)
                    if text:
                        yield AgentEvent(type="message", content=text)

                    # Emit tool_use events with proper metadata
                    tools = self._extract_tool_info(event)
                    for tool in tools:
                        logger.info(f"🔧 Tool: {tool['name']}")
                        yield AgentEvent(
                            type="tool_use",
                            content=f"Using {tool['name']}...",
                            metadata={"name": tool['name'], "input": tool['input']}
                        )
                    continue

                # ========== ResultMessage - final result ==========
                if self._ResultMessage and isinstance(event, self._ResultMessage):
                    is_error = getattr(event, 'is_error', False)
                    result = getattr(event, 'result', '')

                    if is_error:
                        logger.error(f"ResultMessage error: {result}")
                        yield AgentEvent(type="error", content=str(result))
                    else:
                        logger.debug(f"ResultMessage: {str(result)[:100]}...")
                        # Result is usually a summary, text was already streamed
                    continue

                # ========== Unknown event type - log it ==========
                event_class = event.__class__.__name__
                logger.debug(f"Unknown event type: {event_class}")

            yield AgentEvent(type="done", content="")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Claude Agent SDK error: {error_msg}")

            # Provide helpful error messages
            if "CLINotFoundError" in error_msg or "not found" in error_msg.lower():
                yield AgentEvent(
                    type="error",
                    content="❌ Claude Code CLI not found.\n\nInstall with: npm install -g @anthropic-ai/claude-code"
                )
            elif "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                yield AgentEvent(
                    type="error",
                    content="❌ Anthropic API key not configured.\n\nSet ANTHROPIC_API_KEY environment variable."
                )
            else:
                yield AgentEvent(type="error", content=f"❌ Agent error: {error_msg}")

    async def stop(self) -> None:
        """Stop the agent execution."""
        self._stop_flag = True
        logger.info("🛑 Claude Agent SDK stop requested")

    async def get_status(self) -> dict:
        """Get current agent status."""
        return {
            "backend": "claude_agent_sdk",
            "available": self._sdk_available,
            "running": not self._stop_flag,
            "cwd": str(self._cwd),
            "features": [
                "Bash", "Read", "Write", "Edit",
                "Glob", "Grep", "WebSearch", "WebFetch"
            ] if self._sdk_available else [],
        }


# Backwards-compatible wrapper for router
class ClaudeAgentSDKWrapper(ClaudeAgentSDK):
    """Wrapper to match existing agent interface expected by router.

    Provides the `run()` method that yields dicts instead of AgentEvents.
    """

    async def run(self, message: str) -> AsyncIterator[dict]:
        """Run the agent, yielding dict chunks for compatibility."""
        async for event in self.chat(message):
            yield {
                "type": event.type,
                "content": event.content,
                "metadata": getattr(event, 'metadata', None),
            }
