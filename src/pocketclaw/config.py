"""Configuration management for PocketPaw.

Changes:
  - 2026-02-02: Added claude_agent_sdk to agent_backend options.
  - 2026-02-02: Simplified backends - removed 2-layer mode.
  - 2026-02-02: claude_agent_sdk is now RECOMMENDED (uses official SDK).
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_config_dir() -> Path:
    """Get the config directory, creating if needed."""
    config_dir = Path.home() / ".pocketclaw"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the config file path."""
    return get_config_dir() / "config.json"


def get_token_path() -> Path:
    """Get the access token file path."""
    return get_config_dir() / "access_token"


class Settings(BaseSettings):
    """PocketPaw settings with env and file support."""

    model_config = SettingsConfigDict(env_prefix="POCKETCLAW_", env_file=".env", extra="ignore")

    # Telegram
    telegram_bot_token: str | None = Field(
        default=None, description="Telegram Bot Token from @BotFather"
    )
    allowed_user_id: int | None = Field(
        default=None, description="Telegram User ID allowed to control the bot"
    )

    # Agent Backend
    agent_backend: str = Field(
        default="claude_agent_sdk",
        description="Agent backend: 'claude_agent_sdk' (recommended), 'pocketpaw_native', or 'open_interpreter'",
    )

    # LLM Configuration
    llm_provider: str = Field(
        default="auto", description="LLM provider: 'auto', 'ollama', 'openai', 'anthropic'"
    )
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama API host")
    ollama_model: str = Field(default="llama3.2", description="Ollama model to use")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-4-5-20250929", description="Anthropic model to use"
    )

    # Memory Backend
    memory_backend: str = Field(
        default="file",
        description="Memory backend: 'file' (simple markdown), 'mem0' (semantic with LLM)",
    )
    memory_use_inference: bool = Field(
        default=True, description="Use LLM to extract facts from memories (only for mem0 backend)"
    )

    # Session History Compaction
    compaction_recent_window: int = Field(
        default=10, description="Number of recent messages to keep verbatim"
    )
    compaction_char_budget: int = Field(
        default=8000, description="Max total chars for compacted history"
    )
    compaction_summary_chars: int = Field(
        default=150, description="Max chars per older message one-liner extract"
    )
    compaction_llm_summarize: bool = Field(
        default=False, description="Use Haiku to summarize older messages (opt-in)"
    )

    # Tool Policy
    tool_profile: str = Field(
        default="full", description="Tool profile: 'minimal', 'coding', or 'full'"
    )
    tools_allow: list[str] = Field(
        default_factory=list, description="Explicit tool allow list (merged with profile)"
    )
    tools_deny: list[str] = Field(
        default_factory=list, description="Explicit tool deny list (highest priority)"
    )

    # Discord
    discord_bot_token: str | None = Field(default=None, description="Discord bot token")
    discord_allowed_guild_ids: list[int] = Field(
        default_factory=list, description="Discord guild IDs allowed to use the bot"
    )
    discord_allowed_user_ids: list[int] = Field(
        default_factory=list, description="Discord user IDs allowed to use the bot"
    )

    # Slack
    slack_bot_token: str | None = Field(
        default=None, description="Slack Bot OAuth token (xoxb-...)"
    )
    slack_app_token: str | None = Field(
        default=None, description="Slack App-Level token for Socket Mode (xapp-...)"
    )
    slack_allowed_channel_ids: list[str] = Field(
        default_factory=list, description="Slack channel IDs allowed to use the bot"
    )

    # WhatsApp
    whatsapp_mode: str = Field(
        default="personal",
        description="WhatsApp mode: 'personal' (QR scan via neonize) or 'business' (Cloud API)",
    )
    whatsapp_neonize_db: str = Field(
        default="",
        description="Path to neonize SQLite credential store",
    )
    whatsapp_access_token: str | None = Field(
        default=None, description="WhatsApp Business Cloud API access token"
    )
    whatsapp_phone_number_id: str | None = Field(
        default=None, description="WhatsApp Business phone number ID"
    )
    whatsapp_verify_token: str | None = Field(
        default=None, description="WhatsApp webhook verification token"
    )
    whatsapp_allowed_phone_numbers: list[str] = Field(
        default_factory=list, description="WhatsApp phone numbers allowed to use the bot"
    )

    # Web Search
    web_search_provider: str = Field(
        default="tavily", description="Web search provider: 'tavily' or 'brave'"
    )
    tavily_api_key: str | None = Field(default=None, description="Tavily search API key")
    brave_search_api_key: str | None = Field(default=None, description="Brave Search API key")
    parallel_api_key: str | None = Field(default=None, description="Parallel AI API key")
    url_extract_provider: str = Field(
        default="auto", description="URL extract provider: 'auto', 'parallel', or 'local'"
    )

    # Image Generation
    google_api_key: str | None = Field(default=None, description="Google API key (for Gemini)")
    image_model: str = Field(
        default="gemini-2.0-flash-exp", description="Google image generation model"
    )

    # Security
    bypass_permissions: bool = Field(
        default=False, description="Skip permission prompts for agent actions (use with caution)"
    )
    file_jail_path: Path = Field(
        default_factory=Path.home, description="Root path for file operations"
    )
    injection_scan_enabled: bool = Field(
        default=True, description="Enable prompt injection scanning on inbound messages"
    )
    injection_scan_llm: bool = Field(
        default=False, description="Use LLM deep scan for suspicious content (requires API key)"
    )
    injection_scan_llm_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model for LLM-based injection deep scan",
    )

    # Smart Model Routing
    smart_routing_enabled: bool = Field(
        default=False, description="Enable automatic model selection based on task complexity"
    )
    model_tier_simple: str = Field(
        default="claude-haiku-4-5-20251001", description="Model for simple tasks (greetings, facts)"
    )
    model_tier_moderate: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Model for moderate tasks (coding, analysis)",
    )
    model_tier_complex: str = Field(
        default="claude-opus-4-6", description="Model for complex tasks (planning, debugging)"
    )

    # Plan Mode
    plan_mode: bool = Field(default=False, description="Require approval before executing tools")
    plan_mode_tools: list[str] = Field(
        default_factory=lambda: ["shell", "write_file", "edit_file"],
        description="Tools that require approval in plan mode",
    )

    # Self-Audit Daemon
    self_audit_enabled: bool = Field(default=True, description="Enable daily self-audit daemon")
    self_audit_schedule: str = Field(
        default="0 3 * * *", description="Cron schedule for self-audit (default: 3 AM daily)"
    )

    # OAuth
    google_oauth_client_id: str | None = Field(
        default=None, description="Google OAuth 2.0 client ID"
    )
    google_oauth_client_secret: str | None = Field(
        default=None, description="Google OAuth 2.0 client secret"
    )

    # Voice/TTS
    tts_provider: str = Field(
        default="openai", description="TTS provider: 'openai' or 'elevenlabs'"
    )
    elevenlabs_api_key: str | None = Field(default=None, description="ElevenLabs API key for TTS")
    tts_voice: str = Field(
        default="alloy", description="TTS voice name (OpenAI: alloy/echo/fable/onyx/nova/shimmer)"
    )

    # Web Server
    web_host: str = Field(default="127.0.0.1", description="Web server host")
    web_port: int = Field(default=8888, description="Web server port")

    def save(self) -> None:
        """Save settings to config file.

        Merges with existing config to preserve API keys if not set in current instance.
        """
        config_path = get_config_path()

        # Load existing config to preserve API keys if not set
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except (json.JSONDecodeError, Exception):
                pass

        data = {
            "telegram_bot_token": self.telegram_bot_token or existing.get("telegram_bot_token"),
            "allowed_user_id": self.allowed_user_id or existing.get("allowed_user_id"),
            "agent_backend": self.agent_backend,
            "memory_backend": self.memory_backend,
            "memory_use_inference": self.memory_use_inference,
            "compaction_recent_window": self.compaction_recent_window,
            "compaction_char_budget": self.compaction_char_budget,
            "compaction_summary_chars": self.compaction_summary_chars,
            "compaction_llm_summarize": self.compaction_llm_summarize,
            "llm_provider": self.llm_provider,
            "ollama_host": self.ollama_host,
            "ollama_model": self.ollama_model,
            "openai_api_key": self.openai_api_key or existing.get("openai_api_key"),
            "openai_model": self.openai_model,
            "anthropic_api_key": self.anthropic_api_key or existing.get("anthropic_api_key"),
            "anthropic_model": self.anthropic_model,
            # Discord
            "discord_bot_token": (self.discord_bot_token or existing.get("discord_bot_token")),
            "discord_allowed_guild_ids": self.discord_allowed_guild_ids,
            "discord_allowed_user_ids": self.discord_allowed_user_ids,
            # Slack
            "slack_bot_token": self.slack_bot_token or existing.get("slack_bot_token"),
            "slack_app_token": self.slack_app_token or existing.get("slack_app_token"),
            "slack_allowed_channel_ids": self.slack_allowed_channel_ids,
            # Web Search
            "web_search_provider": self.web_search_provider,
            "tavily_api_key": self.tavily_api_key or existing.get("tavily_api_key"),
            "brave_search_api_key": (
                self.brave_search_api_key or existing.get("brave_search_api_key")
            ),
            "parallel_api_key": self.parallel_api_key or existing.get("parallel_api_key"),
            "url_extract_provider": self.url_extract_provider,
            # Image Generation
            "google_api_key": self.google_api_key or existing.get("google_api_key"),
            "image_model": self.image_model,
            # WhatsApp
            "whatsapp_mode": self.whatsapp_mode,
            "whatsapp_neonize_db": self.whatsapp_neonize_db,
            "whatsapp_access_token": (
                self.whatsapp_access_token or existing.get("whatsapp_access_token")
            ),
            "whatsapp_phone_number_id": (
                self.whatsapp_phone_number_id or existing.get("whatsapp_phone_number_id")
            ),
            "whatsapp_verify_token": (
                self.whatsapp_verify_token or existing.get("whatsapp_verify_token")
            ),
            "whatsapp_allowed_phone_numbers": self.whatsapp_allowed_phone_numbers,
            # Tool policy
            "tool_profile": self.tool_profile,
            "tools_allow": self.tools_allow,
            "tools_deny": self.tools_deny,
            # Security
            "injection_scan_enabled": self.injection_scan_enabled,
            "injection_scan_llm": self.injection_scan_llm,
            "injection_scan_llm_model": self.injection_scan_llm_model,
            # Smart routing
            "smart_routing_enabled": self.smart_routing_enabled,
            "model_tier_simple": self.model_tier_simple,
            "model_tier_moderate": self.model_tier_moderate,
            "model_tier_complex": self.model_tier_complex,
            # Plan mode
            "plan_mode": self.plan_mode,
            "plan_mode_tools": self.plan_mode_tools,
            # Self-audit
            "self_audit_enabled": self.self_audit_enabled,
            "self_audit_schedule": self.self_audit_schedule,
            # OAuth
            "google_oauth_client_id": (
                self.google_oauth_client_id or existing.get("google_oauth_client_id")
            ),
            "google_oauth_client_secret": (
                self.google_oauth_client_secret or existing.get("google_oauth_client_secret")
            ),
            # Voice/TTS
            "tts_provider": self.tts_provider,
            "elevenlabs_api_key": (self.elevenlabs_api_key or existing.get("elevenlabs_api_key")),
            "tts_voice": self.tts_voice,
        }
        config_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from config file, falling back to env/defaults."""
        config_path = get_config_path()
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                return cls(**data)
            except (json.JSONDecodeError, Exception):
                pass
        return cls()


@lru_cache
def get_settings(force_reload: bool = False) -> Settings:
    """Get cached settings instance."""
    if force_reload:
        get_settings.cache_clear()
    return Settings.load()


def get_access_token() -> str:
    """
    Get the current access token.
    If it doesn't exist, generate a new one.
    """
    token_path = get_token_path()
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    return regenerate_token()


def regenerate_token() -> str:
    """
    Generate a new secure access token and save it.
    Invalidates previous tokens.
    """
    import uuid

    token = str(uuid.uuid4())
    token_path = get_token_path()
    token_path.write_text(token)
    return token
