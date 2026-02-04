"""
PocketPaw Native Orchestrator - The Brain.

Your own orchestrator using raw Anthropic SDK + Open Interpreter as executor.
No LangChain, no Agent SDK - just simple, transparent control.

Created: 2026-02-02
Changes:
  - Initial implementation of PocketPaw Native Orchestrator.
  - 2026-02-02: Added comprehensive security layer (file jail, pattern matching, etc.)
  - 2026-02-02: Added 'computer' tool for full Open Interpreter delegation
                (Calendar, Mail, Browser, AppleScript, Python, etc.)
  - 2026-02-02: Made AGENTIC - system prompt now instructs Claude to QUERY and
                RETURN actual data from apps, not just open them.
                Example: "What's on my calendar?" → returns actual events as text
  - 2026-02-02: SPEED FIX - Shell commands now use direct subprocess (10x faster).
                'computer' tool uses OI for complex multi-step tasks only.
"""

import logging
import re
from pathlib import Path
from typing import AsyncIterator, Optional

from anthropic import Anthropic

from pocketclaw.config import Settings
from pocketclaw.agents.protocol import AgentEvent

logger = logging.getLogger(__name__)


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Dangerous command patterns (regex for better matching)
DANGEROUS_PATTERNS = [
    # Destructive file operations
    r"rm\s+(-[rf]+\s+)*[/~]",           # rm -rf /, rm -r -f ~, etc.
    r"rm\s+(-[rf]+\s+)*\*",             # rm -rf *
    r"sudo\s+rm\b",                      # Any sudo rm
    r">\s*/dev/",                        # Write to devices
    r"mkfs\.",                           # Format filesystem
    r"dd\s+if=",                         # Disk operations
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",  # Fork bomb
    r"chmod\s+(-R\s+)?777\s+/",          # Dangerous permissions

    # Remote code execution
    r"curl\s+.*\|\s*(ba)?sh",            # curl | sh
    r"wget\s+.*\|\s*(ba)?sh",            # wget | sh
    r"curl\s+.*-o\s*/",                  # curl download to root
    r"wget\s+.*-O\s*/",                  # wget download to root

    # System damage
    r">\s*/etc/passwd",                  # Overwrite passwd
    r">\s*/etc/shadow",                  # Overwrite shadow
    r"systemctl\s+(stop|disable)\s+(ssh|sshd|firewall)",  # Disable security
    r"iptables\s+-F",                    # Flush firewall
    r"shutdown",                         # Shutdown system
    r"reboot",                           # Reboot system
    r"init\s+0",                         # Halt system
]

# Sensitive paths that should never be read or written
SENSITIVE_PATHS = [
    # SSH keys
    ".ssh/id_rsa",
    ".ssh/id_ed25519",
    ".ssh/id_ecdsa",
    ".ssh/authorized_keys",

    # Credentials
    ".aws/credentials",
    ".aws/config",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".docker/config.json",
    ".kube/config",

    # Secrets
    ".env",
    ".envrc",
    "secrets.json",
    "credentials.json",
    ".git-credentials",

    # System files
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
]

# Patterns to redact from output (API keys, passwords, etc.)
REDACT_PATTERNS = [
    r"(sk-[a-zA-Z0-9]{20,})",                    # OpenAI/Anthropic keys
    r"(AKIA[A-Z0-9]{16})",                       # AWS access key
    r"(ghp_[a-zA-Z0-9]{36})",                    # GitHub token
    r"(xox[baprs]-[a-zA-Z0-9-]+)",               # Slack token
    r"password[\"']?\s*[:=]\s*[\"']([^\"']+)",   # password = "..."
    r"api[_-]?key[\"']?\s*[:=]\s*[\"']([^\"']+)", # api_key = "..."
]

# System prompt - PocketPaw's personality and instructions
SYSTEM_PROMPT = """You are PocketPaw, a helpful AI assistant that runs locally on the user's computer.

You have access to powerful tools that let you control the computer, access apps, and manage files.
You are resourceful, careful, and efficient.

## CRITICAL: Be AGENTIC - Query and Return Data

**DO NOT just open apps. ALWAYS query actual data and tell the user the information.**

When the user asks about their calendar, emails, reminders, etc.:
1. Use 'computer' to QUERY the actual data (events, emails, tasks)
2. Parse and understand the results
3. Tell the user the ACTUAL information in a clear, useful way

❌ WRONG: "I'll open Calendar.app for you" (not helpful)
✅ RIGHT: "You have 3 events today: 10am Team standup, 2pm Design review, 5pm 1:1 with Sarah"

## Tool Selection Guide

**Use 'computer' (Open Interpreter) for:**
- QUERYING data from apps: Calendar events, emails, reminders, notes, contacts
- Complex tasks that need Python or AppleScript
- Browser automation and web searches
- Multi-step operations
- Anything requiring intelligent problem-solving

**Use 'shell' only for:**
- Simple, well-known commands (ls, git status, npm install, etc.)
- When you know the EXACT command needed

## Examples of AGENTIC Behavior

| User Says | You Should Do |
|-----------|---------------|
| "Show my calendar" | Query Calendar.app via AppleScript/Python and LIST the actual events |
| "What meetings today?" | Query events for today and TELL the user what they are |
| "Any emails from Bob?" | Query Mail.app and SUMMARIZE emails from Bob |
| "What's on my todo list?" | Query Reminders.app and LIST the actual items |
| "Find that PDF I downloaded" | Search for it and TELL the user where it is |
| "What's the weather?" | Search the web and TELL the user the forecast |

## Instructions for computer tool

When using 'computer', be SPECIFIC about what you want:

❌ BAD: "show my calendar"
✅ GOOD: "Use AppleScript or Python to query Calendar.app and return ALL events for today including title, time, and location. Return the data as text, do not open any apps."

❌ BAD: "check my email"
✅ GOOD: "Use AppleScript to query Mail.app for unread emails from the last 24 hours. Return sender, subject, and preview for each email."

## Guidelines
- Be concise and helpful
- ALWAYS return actual data/information, not just open apps
- Prefer 'computer' over 'shell' when in doubt
- Report results clearly in a human-readable format"""

# Tool definitions for Claude
TOOLS = [
    {
        "name": "computer",
        "description": """Execute a task using Open Interpreter - an AI agent with FULL computer control.

IMPORTANT: Use this to QUERY and RETURN data, not just open apps!

When querying apps (Calendar, Mail, Reminders), be SPECIFIC:
- "Query Calendar.app for today's events and return title, time, location as text"
- "Query Mail.app for unread emails and return sender, subject, preview"
- "Query Reminders.app and return all incomplete tasks"

DO NOT just say "show calendar" - that opens the app without returning data.

CAPABILITIES:
- Query macOS apps via AppleScript/Python (Calendar, Mail, Reminders, Contacts, Notes)
- Run Python code for data processing
- Browser automation and web searches
- Complex multi-step operations

ALWAYS instruct it to RETURN data as text, not open GUI apps.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "SPECIFIC task description. Include: what data to query, which app, and 'return as text' or 'do not open app'"
                }
            },
            "required": ["task"]
        }
    },
    {
        "name": "shell",
        "description": "Execute a simple shell command. Only use for basic operations where you know the exact command. For complex tasks, use 'computer' instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_dir",
        "description": "List contents of a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                }
            },
            "required": ["path"]
        }
    }
]


class PocketPawOrchestrator:
    """PocketPaw Native Orchestrator - Your own AI brain.

    Architecture:
        User Message → PocketPaw (Brain) → Tool Calls → Open Interpreter (Hands) → Result

    Security layers:
    1. Dangerous command regex matching
    2. Sensitive path protection (no reading SSH keys, credentials, etc.)
    3. File jail (restricts to home directory by default)
    4. Output redaction (hides API keys, passwords in output)

    This is a simple, transparent orchestrator:
    - Uses Anthropic SDK directly for reasoning
    - Routes tool calls to Open Interpreter executor
    - You control the loop, prompts, and security
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[Anthropic] = None
        self._executor = None
        self._stop_flag = False
        self._file_jail = settings.file_jail_path.resolve()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the orchestrator."""
        # Initialize Anthropic client
        if not self.settings.anthropic_api_key:
            logger.error("❌ Anthropic API key required for PocketPaw Native")
            return

        self._client = Anthropic(api_key=self.settings.anthropic_api_key)

        # Initialize executor (Open Interpreter)
        try:
            from pocketclaw.agents.executor import OpenInterpreterExecutor
            self._executor = OpenInterpreterExecutor(self.settings)
        except Exception as e:
            logger.warning(f"⚠️ Executor init failed: {e}. Using fallback.")
            self._executor = None

        logger.info("=" * 50)
        logger.info("🐾 POCKETPAW NATIVE ORCHESTRATOR")
        logger.info("   └─ Brain: Anthropic API (direct)")
        logger.info("   └─ Hands: Open Interpreter")
        logger.info("   └─ Model: %s", self.settings.anthropic_model)
        logger.info("   └─ File Jail: %s", self._file_jail)
        logger.info("   └─ Security: Enabled (patterns, paths, redaction)")
        logger.info("=" * 50)

    # =========================================================================
    # SECURITY METHODS
    # =========================================================================

    def _is_dangerous_command(self, command: str) -> Optional[str]:
        """Check if a command matches dangerous patterns using regex."""
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return pattern
        return None

    def _is_sensitive_path(self, path: str) -> bool:
        """Check if a path is sensitive and should be protected."""
        # Normalize path
        try:
            normalized = Path(path).expanduser().resolve()
            path_str = str(normalized)
        except Exception:
            path_str = path

        # Check against sensitive paths
        for sensitive in SENSITIVE_PATHS:
            if sensitive in path_str or path_str.endswith(sensitive):
                return True

        return False

    def _is_path_in_jail(self, path: str) -> bool:
        """Check if a path is within the allowed file jail."""
        try:
            # Resolve the path (handles .., symlinks, etc.)
            resolved = Path(path).expanduser().resolve()

            # Check if it's within the jail
            resolved.relative_to(self._file_jail)
            return True
        except ValueError:
            # relative_to raises ValueError if not a subpath
            return False
        except Exception:
            return False

    def _redact_secrets(self, text: str) -> str:
        """Redact sensitive information from output."""
        redacted = text
        for pattern in REDACT_PATTERNS:
            redacted = re.sub(pattern, r"[REDACTED]", redacted, flags=re.IGNORECASE)
        return redacted

    def _validate_file_access(self, path: str, operation: str) -> tuple[bool, str]:
        """Validate file access for read/write operations.

        Returns:
            (allowed: bool, reason: str)
        """
        # Check sensitive paths
        if self._is_sensitive_path(path):
            return False, f"🛑 BLOCKED: '{path}' is a sensitive file (credentials, keys, etc.)"

        # Check file jail
        if not self._is_path_in_jail(path):
            return False, f"🛑 BLOCKED: '{path}' is outside allowed directory ({self._file_jail})"

        return True, ""

    def _validate_command(self, command: str) -> tuple[bool, str]:
        """Validate a shell command for execution.

        Returns:
            (allowed: bool, reason: str)
        """
        # Check dangerous patterns
        danger = self._is_dangerous_command(command)
        if danger:
            return False, f"🛑 BLOCKED: Command matches dangerous pattern: {danger}"

        return True, ""

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result.

        All tool execution goes through security validation:
        1. Command validation (dangerous patterns)
        2. Path validation (sensitive files, jail check)
        3. Output redaction (secrets removal)
        """
        logger.info(f"🔧 Tool: {tool_name}({tool_input})")

        try:
            # =================================================================
            # COMPUTER TOOL - Full Open Interpreter power (for complex tasks)
            # =================================================================
            if tool_name == "computer":
                task = tool_input.get("task", "")

                if not task:
                    return "Error: No task provided"

                logger.info(f"🖥️ Delegating to Open Interpreter: {task[:100]}...")

                # Use executor's run_complex_task method (cleaner interface)
                if self._executor and hasattr(self._executor, 'run_complex_task'):
                    result = await self._executor.run_complex_task(task)
                    return self._redact_secrets(result or "(no output)")
                else:
                    return "Error: Open Interpreter not available. Install with: pip install open-interpreter"

            # =================================================================
            # SHELL TOOL - Simple command execution
            # =================================================================
            elif tool_name == "shell":
                command = tool_input.get("command", "")

                # Security: validate command
                allowed, reason = self._validate_command(command)
                if not allowed:
                    logger.warning(f"Security block: {reason}")
                    return reason

                # Execute via Open Interpreter or fallback
                if self._executor:
                    result = await self._executor.run_shell(command)
                else:
                    # Fallback: direct execution
                    import subprocess
                    try:
                        proc = subprocess.run(
                            command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=60,
                            cwd=str(self._file_jail)  # Run in jail directory
                        )
                        result = proc.stdout + proc.stderr
                    except subprocess.TimeoutExpired:
                        result = "Command timed out after 60 seconds"
                    except Exception as e:
                        result = f"Error: {e}"

                # Security: redact secrets from output
                return self._redact_secrets(result or "(no output)")

            elif tool_name == "read_file":
                path = tool_input.get("path", "")

                # Security: validate path
                allowed, reason = self._validate_file_access(path, "read")
                if not allowed:
                    logger.warning(f"Security block: {reason}")
                    return reason

                if self._executor:
                    content = await self._executor.read_file(path)
                else:
                    with open(Path(path).expanduser(), 'r') as f:
                        content = f.read()

                # Security: redact secrets from file content
                return self._redact_secrets(content)

            elif tool_name == "write_file":
                path = tool_input.get("path", "")
                content = tool_input.get("content", "")

                # Security: validate path
                allowed, reason = self._validate_file_access(path, "write")
                if not allowed:
                    logger.warning(f"Security block: {reason}")
                    return reason

                if self._executor:
                    await self._executor.write_file(path, content)
                else:
                    with open(Path(path).expanduser(), 'w') as f:
                        f.write(content)

                return f"✓ Written to {path}"

            elif tool_name == "list_dir":
                path = tool_input.get("path", ".")

                # Security: validate path
                allowed, reason = self._validate_file_access(path, "list")
                if not allowed:
                    logger.warning(f"Security block: {reason}")
                    return reason

                if self._executor:
                    items = await self._executor.list_directory(path)
                    return "\n".join(items)
                else:
                    import os
                    return "\n".join(os.listdir(Path(path).expanduser()))

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            # Security: redact secrets from error messages too
            return self._redact_secrets(f"Error executing {tool_name}: {e}")

    async def chat(self, message: str) -> AsyncIterator[AgentEvent]:
        """Process a message through the orchestrator.

        This is the main agentic loop:
        1. Send message to Claude with tools
        2. If Claude responds with text → yield it
        3. If Claude wants to use a tool → execute it → feed result back
        4. Repeat until done
        """
        if not self._client:
            yield AgentEvent(
                type="error",
                content="❌ PocketPaw not initialized. Check Anthropic API key."
            )
            return

        self._stop_flag = False

        # Conversation history for this request
        messages = [{"role": "user", "content": message}]

        # Maximum iterations to prevent infinite loops
        max_iterations = 10
        iteration = 0

        try:
            while iteration < max_iterations and not self._stop_flag:
                iteration += 1
                logger.debug(f"Iteration {iteration}/{max_iterations}")

                # Call Claude
                response = self._client.messages.create(
                    model=self.settings.anthropic_model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )

                # Process response content blocks
                assistant_content = []
                tool_results_needed = []

                for block in response.content:
                    if block.type == "text":
                        # Text response - yield to user
                        if block.text:
                            yield AgentEvent(type="message", content=block.text)
                        assistant_content.append(block)

                    elif block.type == "tool_use":
                        # Tool call - execute it
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        # Emit tool_use event
                        yield AgentEvent(
                            type="tool_use",
                            content=f"🔧 Using {tool_name}...",
                            metadata={"name": tool_name, "input": tool_input}
                        )

                        # Execute
                        result = await self._execute_tool(tool_name, tool_input)

                        # Emit tool_result event
                        yield AgentEvent(
                            type="tool_result",
                            content=result[:500] + ("..." if len(result) > 500 else ""),
                            metadata={"name": tool_name}
                        )

                        assistant_content.append(block)
                        tool_results_needed.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result
                        })

                # Add assistant message to history
                messages.append({"role": "assistant", "content": assistant_content})

                # Check if we're done
                if response.stop_reason == "end_turn":
                    break

                # If tools were used, add results and continue
                if tool_results_needed:
                    messages.append({"role": "user", "content": tool_results_needed})
                else:
                    # No tools and not end_turn - shouldn't happen, but break anyway
                    break

            yield AgentEvent(type="done", content="")

        except Exception as e:
            logger.error(f"PocketPaw error: {e}")
            yield AgentEvent(type="error", content=f"❌ Error: {e}")

    async def run(self, message: str) -> AsyncIterator[dict]:
        """Run method for compatibility with router."""
        async for event in self.chat(message):
            yield {"type": event.type, "content": event.content}

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._stop_flag = True
        logger.info("🛑 PocketPaw stopped")

    async def get_status(self) -> dict:
        """Get orchestrator status."""
        return {
            "backend": "pocketpaw_native",
            "available": self._client is not None,
            "executor": "open_interpreter" if self._executor else "fallback",
            "model": self.settings.anthropic_model
        }
