"""PocketPaw entry point.

Changes:
  - 2026-02-06: Web dashboard is now the default mode (no flags needed).
  - 2026-02-06: Added --telegram flag for legacy Telegram-only mode.
  - 2026-02-06: Added --discord, --slack, --whatsapp CLI modes.
  - 2026-02-02: Added Rich logging for beautiful console output.
  - 2026-02-03: Handle port-in-use gracefully with automatic port finding.
"""

import argparse
import asyncio
import logging
import webbrowser

from pocketclaw.config import Settings, get_settings
from pocketclaw.logging_setup import setup_logging

# Setup beautiful logging with Rich
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


async def run_telegram_mode(settings: Settings) -> None:
    """Run in Telegram bot mode."""
    from pocketclaw.bot_gateway import run_bot
    from pocketclaw.web_server import find_available_port, run_pairing_server

    # Check if we need to run pairing flow
    if not settings.telegram_bot_token or not settings.allowed_user_id:
        logger.info("🔧 First-time setup: Starting pairing server...")

        # Find available port before showing instructions
        try:
            port = find_available_port(settings.web_port)
        except OSError:
            logger.error(
                "❌ Could not find an available port. Please close other applications and try again."
            )
            return

        print("\n" + "=" * 50)
        print("🐾 POCKETPAW SETUP")
        print("=" * 50)
        print("\n1. Create a Telegram bot via @BotFather")
        print("2. Copy the bot token")
        print(f"3. Open http://localhost:{port} in your browser")
        print("4. Paste the token and scan the QR code\n")

        # Open browser automatically with correct port
        webbrowser.open(f"http://localhost:{port}")

        # Run pairing server (blocks until pairing complete)
        await run_pairing_server(settings)

        # Reload settings after pairing
        settings = get_settings(force_reload=True)

    # Start the bot
    logger.info("🚀 Starting PocketPaw bot...")
    await run_bot(settings)


async def run_multi_channel_mode(settings: Settings, args: argparse.Namespace) -> None:
    """Run one or more channel adapters sharing a single bus and AgentLoop."""
    from pocketclaw.agents.loop import AgentLoop
    from pocketclaw.bus import get_message_bus

    bus = get_message_bus()
    adapters = []

    if args.discord:
        if not settings.discord_bot_token:
            logger.error("Discord bot token not configured. Set POCKETCLAW_DISCORD_BOT_TOKEN.")
        else:
            from pocketclaw.bus.adapters.discord_adapter import DiscordAdapter

            adapters.append(
                DiscordAdapter(
                    token=settings.discord_bot_token,
                    allowed_guild_ids=settings.discord_allowed_guild_ids,
                    allowed_user_ids=settings.discord_allowed_user_ids,
                )
            )

    if args.slack:
        if not settings.slack_bot_token or not settings.slack_app_token:
            logger.error(
                "Slack tokens not configured. Set POCKETCLAW_SLACK_BOT_TOKEN "
                "and POCKETCLAW_SLACK_APP_TOKEN."
            )
        else:
            from pocketclaw.bus.adapters.slack_adapter import SlackAdapter

            adapters.append(
                SlackAdapter(
                    bot_token=settings.slack_bot_token,
                    app_token=settings.slack_app_token,
                    allowed_channel_ids=settings.slack_allowed_channel_ids,
                )
            )

    if args.whatsapp:
        if not settings.whatsapp_access_token or not settings.whatsapp_phone_number_id:
            logger.error(
                "WhatsApp not configured. Set POCKETCLAW_WHATSAPP_ACCESS_TOKEN "
                "and POCKETCLAW_WHATSAPP_PHONE_NUMBER_ID."
            )
        else:
            from pocketclaw.bus.adapters.whatsapp_adapter import WhatsAppAdapter

            adapters.append(
                WhatsAppAdapter(
                    access_token=settings.whatsapp_access_token,
                    phone_number_id=settings.whatsapp_phone_number_id,
                    verify_token=settings.whatsapp_verify_token or "",
                    allowed_phone_numbers=settings.whatsapp_allowed_phone_numbers,
                )
            )

    if not adapters:
        logger.error("No channel adapters could be started. Check your configuration.")
        return

    agent_loop = AgentLoop()

    for adapter in adapters:
        await adapter.start(bus)
        logger.info(f"Started {adapter.channel.value} adapter")

    loop_task = asyncio.create_task(agent_loop.start())

    # If WhatsApp is one of the adapters, start a minimal webhook server
    whatsapp_server = None
    if args.whatsapp:
        import uvicorn

        import pocketclaw.whatsapp_gateway as wa_gw
        from pocketclaw.whatsapp_gateway import create_whatsapp_app

        # Point the gateway module at our adapter
        for a in adapters:
            if a.channel.value == "whatsapp":
                wa_gw._whatsapp_adapter = a
                break

        wa_app = create_whatsapp_app(settings)
        config = uvicorn.Config(
            wa_app, host=settings.web_host, port=settings.web_port, log_level="info"
        )
        whatsapp_server = uvicorn.Server(config)
        asyncio.create_task(whatsapp_server.serve())

    try:
        await loop_task
    except asyncio.CancelledError:
        logger.info("Stopping channels...")
    finally:
        await agent_loop.stop()
        for adapter in adapters:
            await adapter.stop()


def run_dashboard_mode(settings: Settings, port: int) -> None:
    """Run in web dashboard mode."""
    from pocketclaw.dashboard import run_dashboard

    print("\n" + "=" * 50)
    print("🦀 POCKETCLAW WEB DASHBOARD")
    print("=" * 50)
    print(f"\n🌐 Open http://localhost:{port} in your browser\n")

    webbrowser.open(f"http://localhost:{port}")
    run_dashboard(host="127.0.0.1", port=port)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🐾 PocketPaw - The AI agent that runs on your laptop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pocketpaw                          Start web dashboard (default)
  pocketpaw --telegram               Start in Telegram-only mode
  pocketpaw --discord                Start headless Discord bot
  pocketpaw --slack                  Start headless Slack bot (Socket Mode)
  pocketpaw --whatsapp               Start headless WhatsApp webhook server
  pocketpaw --discord --slack        Run Discord + Slack simultaneously
""",
    )

    parser.add_argument(
        "--web",
        "-w",
        action="store_true",
        help="Run web dashboard (same as default, kept for compatibility)",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Run Telegram-only mode (legacy pairing flow)",
    )
    parser.add_argument("--discord", action="store_true", help="Run headless Discord bot")
    parser.add_argument("--slack", action="store_true", help="Run headless Slack bot (Socket Mode)")
    parser.add_argument(
        "--whatsapp", action="store_true", help="Run headless WhatsApp webhook server"
    )
    parser.add_argument(
        "--security-audit",
        action="store_true",
        help="Run security audit and print report",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix fixable issues found by --security-audit",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8888, help="Port for web server (default: 8888)"
    )
    parser.add_argument("--version", "-v", action="version", version="%(prog)s 0.2.0")

    args = parser.parse_args()
    settings = get_settings()

    has_channel_flag = args.discord or args.slack or args.whatsapp

    try:
        if args.security_audit:
            from pocketclaw.security.audit_cli import run_security_audit

            exit_code = asyncio.run(run_security_audit(fix=args.fix))
            raise SystemExit(exit_code)
        elif args.telegram:
            asyncio.run(run_telegram_mode(settings))
        elif has_channel_flag:
            asyncio.run(run_multi_channel_mode(settings, args))
        else:
            # Default: web dashboard (also handles --web flag)
            run_dashboard_mode(settings, args.port)
    except KeyboardInterrupt:
        logger.info("👋 PocketPaw stopped.")


if __name__ == "__main__":
    main()
