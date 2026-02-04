"""Open Interpreter agent wrapper.

Changes:
  2026-02-04 - Filter out verbose console output, only show messages and final results
  2026-02-02 - Added executor layer logging for architecture visibility.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from pocketclaw.config import Settings

logger = logging.getLogger(__name__)


class OpenInterpreterAgent:
    """Wraps Open Interpreter for autonomous task execution.

    In the Agent SDK architecture, this serves as the EXECUTOR layer:
    - Executes code and system commands
    - Handles file operations
    - Provides sandboxed execution environment
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._interpreter = None
        self._stop_flag = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Open Interpreter instance."""
        try:
            from interpreter import interpreter

            # Configure interpreter
            interpreter.auto_run = True  # Don't ask for confirmation
            interpreter.loop = True  # Allow multi-step execution

            # Set LLM based on settings
            provider = self.settings.llm_provider

            # Explicit provider selection
            if provider == "anthropic" and self.settings.anthropic_api_key:
                interpreter.llm.model = self.settings.anthropic_model
                interpreter.llm.api_key = self.settings.anthropic_api_key
                logger.info(f"🤖 Using Anthropic: {self.settings.anthropic_model}")
            elif provider == "openai" and self.settings.openai_api_key:
                interpreter.llm.model = self.settings.openai_model
                interpreter.llm.api_key = self.settings.openai_api_key
                logger.info(f"🤖 Using OpenAI: {self.settings.openai_model}")
            elif provider == "ollama":
                interpreter.llm.model = f"ollama/{self.settings.ollama_model}"
                interpreter.llm.api_base = self.settings.ollama_host
                logger.info(f"🤖 Using Ollama: {self.settings.ollama_model}")
            # Auto mode: prioritize cloud APIs, fallback to Ollama
            elif provider == "auto":
                if self.settings.anthropic_api_key:
                    interpreter.llm.model = self.settings.anthropic_model
                    interpreter.llm.api_key = self.settings.anthropic_api_key
                    logger.info(f"🤖 Auto-selected Anthropic: {self.settings.anthropic_model}")
                elif self.settings.openai_api_key:
                    interpreter.llm.model = self.settings.openai_model
                    interpreter.llm.api_key = self.settings.openai_api_key
                    logger.info(f"🤖 Auto-selected OpenAI: {self.settings.openai_model}")
                else:
                    interpreter.llm.model = f"ollama/{self.settings.ollama_model}"
                    interpreter.llm.api_base = self.settings.ollama_host
                    logger.info(f"🤖 Auto-selected Ollama: {self.settings.ollama_model}")

            # Safety settings
            interpreter.safe_mode = "ask"  # Will still ask before dangerous ops

            self._interpreter = interpreter
            logger.info("=" * 50)
            logger.info("🔧 EXECUTOR: Open Interpreter initialized")
            logger.info("   └─ Role: Code execution, file ops, system commands")
            logger.info("=" * 50)

        except ImportError:
            logger.error("❌ Open Interpreter not installed. Run: pip install open-interpreter")
            self._interpreter = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Open Interpreter: {e}")
            self._interpreter = None

    async def run(self, message: str, system_message: str | None = None) -> AsyncIterator[dict]:
        """Run a message through Open Interpreter with real-time streaming."""
        if not self._interpreter:
            yield {"type": "message", "content": "❌ Open Interpreter not available."}
            return

        self._stop_flag = False

        # Apply system message if provided
        if system_message:
            # We prepend to keep OI's functional instructions
            # interpreter usually has its own long system_message
            self._interpreter.system_message = (
                f"{system_message}\n\n{self._interpreter.system_message}"
            )

        # Use a queue to stream chunks from the sync thread to the async generator
        chunk_queue: asyncio.Queue = asyncio.Queue()

        def run_sync():
            """Run interpreter in a thread, push chunks to queue.

            Open Interpreter chunk types:
            - role: "assistant", type: "message" -> Text to show user
            - role: "assistant", type: "code" -> Code being written
            - role: "computer", type: "console", start: true -> Execution starting
            - role: "computer", type: "console", format: "output" -> Final output
            - role: "computer", type: "console", end: true -> Execution done
            """
            current_message = []
            current_language = None
            shown_running = False

            try:
                for chunk in self._interpreter.chat(message, stream=True):
                    if self._stop_flag:
                        break

                    if isinstance(chunk, dict):
                        chunk_role = chunk.get("role", "")
                        chunk_type = chunk.get("type", "")
                        content = chunk.get("content", "")
                        chunk_format = chunk.get("format", "")
                        is_start = chunk.get("start", False)
                        is_end = chunk.get("end", False)

                        # Handle computer/console chunks - show progress indicators
                        if chunk_role == "computer":
                            if chunk_type == "console":
                                if is_start and current_language and not shown_running:
                                    # Show "Running X..." indicator
                                    lang_display = current_language.title()
                                    asyncio.run_coroutine_threadsafe(
                                        chunk_queue.put(
                                            {
                                                "type": "message",
                                                "content": f"\n⚡ *Running {lang_display}...*\n",
                                            }
                                        ),
                                        loop,
                                    )
                                    shown_running = True
                                elif is_end:
                                    # Reset for next code block
                                    shown_running = False
                                # Skip verbose active_line, intermediate output
                            continue

                        # Handle assistant chunks
                        if chunk_role == "assistant":
                            if chunk_type == "code":
                                # Capture language for progress indicator
                                current_language = chunk_format or "code"
                                # Flush any pending message
                                if current_message:
                                    asyncio.run_coroutine_threadsafe(
                                        chunk_queue.put(
                                            {"type": "message", "content": "".join(current_message)}
                                        ),
                                        loop,
                                    )
                                    current_message = []
                                # Don't show raw code fragments
                            elif chunk_type == "message" and content:
                                # Stream message chunks
                                asyncio.run_coroutine_threadsafe(
                                    chunk_queue.put({"type": "message", "content": content}), loop
                                )
                    elif isinstance(chunk, str) and chunk:
                        current_message.append(chunk)

                # Flush remaining message
                if current_message:
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put({"type": "message", "content": "".join(current_message)}),
                        loop,
                    )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put({"type": "error", "content": f"Agent error: {str(e)}"}), loop
                )
            finally:
                # Signal completion
                asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop)

        try:
            loop = asyncio.get_event_loop()

            # Start the sync function in a thread
            executor_future = loop.run_in_executor(None, run_sync)

            # Yield chunks as they arrive
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=60.0)
                    if chunk is None:  # End signal
                        break
                    yield chunk
                except TimeoutError:
                    yield {"type": "message", "content": "⏳ Still processing..."}

            # Wait for executor to finish
            await executor_future

        except Exception as e:
            logger.error(f"Open Interpreter error: {e}")
            yield {"type": "error", "content": f"❌ Agent error: {str(e)}"}

    async def stop(self) -> None:
        """Stop the agent execution."""
        self._stop_flag = True
        if self._interpreter:
            try:
                self._interpreter.reset()
            except Exception:
                pass
