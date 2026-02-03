"""PocketPaw entry point.

Changes:
  - 2026-02-02: Added Rich logging for beautiful console output.
  - 2026-02-03: Handle port-in-use gracefully with automatic port finding.
"""

import argparse
import asyncio
import logging
import webbrowser
from pathlib import Path

from pocketclaw.config import get_settings, Settings
from pocketclaw.logging_setup import setup_logging

# Setup beautiful logging with Rich
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


async def run_telegram_mode(settings: Settings) -> None:
    """Run in Telegram bot mode."""
    from pocketclaw.web_server import run_pairing_server, find_available_port
    from pocketclaw.bot_gateway import run_bot

    # Check if we need to run pairing flow
    if not settings.telegram_bot_token or not settings.allowed_user_id:
        logger.info("🔧 First-time setup: Starting pairing server...")

        # Find available port before showing instructions
        try:
            port = find_available_port(settings.web_port)
        except OSError:
            logger.error("❌ Could not find an available port. Please close other applications and try again.")
            return

        print("\n" + "="*50)
        print("🐾 POCKETPAW SETUP")
        print("="*50)
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


def run_dashboard_mode(settings: Settings, port: int) -> None:
    """Run in web dashboard mode."""
    from pocketclaw.dashboard import run_dashboard
    
    print("\n" + "="*50)
    print("🦀 POCKETCLAW WEB DASHBOARD")
    print("="*50)
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
  pocketclaw              Start in Telegram mode (default)
  pocketclaw --web        Start web dashboard for testing
  pocketclaw --web --port 9000   Web dashboard on custom port
"""
    )
    
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Run web dashboard instead of Telegram bot"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8888,
        help="Port for web server (default: 8888)"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.2.0"
    )

    args = parser.parse_args()
    settings = get_settings()

    try:
        if args.web:
            run_dashboard_mode(settings, args.port)
        else:
            asyncio.run(run_telegram_mode(settings))
    except KeyboardInterrupt:
        logger.info("👋 PocketPaw stopped.")


if __name__ == "__main__":
    main()
