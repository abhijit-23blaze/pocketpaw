"""Configuration management for PocketPaw.

Changes:
  - 2026-02-02: Added claude_agent_sdk to agent_backend options.
  - 2026-02-02: Simplified backends - removed 2-layer mode.
  - 2026-02-02: claude_agent_sdk is now RECOMMENDED (uses official SDK).
"""

import json
from pathlib import Path
from typing import Optional
from functools import lru_cache

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
    
    model_config = SettingsConfigDict(
        env_prefix="POCKETCLAW_",
        env_file=".env",
        extra="ignore"
    )
    
    # Telegram
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram Bot Token from @BotFather")
    allowed_user_id: Optional[int] = Field(default=None, description="Telegram User ID allowed to control the bot")
    
    # Agent Backend
    agent_backend: str = Field(
        default="claude_agent_sdk",
        description="Agent backend: 'claude_agent_sdk' (recommended), 'pocketpaw_native', or 'open_interpreter'"
    )
    
    # LLM Configuration
    llm_provider: str = Field(default="auto", description="LLM provider: 'auto', 'ollama', 'openai', 'anthropic'")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama API host")
    ollama_model: str = Field(default="llama3.2", description="Ollama model to use")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-sonnet-4-5-20250929", description="Anthropic model to use")
    
    # Memory Backend
    memory_backend: str = Field(
        default="file",
        description="Memory backend: 'file' (simple markdown), 'mem0' (semantic with LLM)"
    )
    memory_use_inference: bool = Field(
        default=True,
        description="Use LLM to extract facts from memories (only for mem0 backend)"
    )

    # Security
    bypass_permissions: bool = Field(default=False, description="Skip permission prompts for agent actions (use with caution)")
    file_jail_path: Path = Field(default_factory=Path.home, description="Root path for file operations")
    
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
            "llm_provider": self.llm_provider,
            "ollama_host": self.ollama_host,
            "ollama_model": self.ollama_model,
            "openai_api_key": self.openai_api_key or existing.get("openai_api_key"),
            "openai_model": self.openai_model,
            "anthropic_api_key": self.anthropic_api_key or existing.get("anthropic_api_key"),
            "anthropic_model": self.anthropic_model,
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
