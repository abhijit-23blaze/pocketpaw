#!/bin/sh
# PocketPaw Installer Bootstrap
# Usage: curl -fsSL https://pocketpaw.com/install.sh | sh
# POSIX sh — no bashisms

set -e

# ── Banner ──────────────────────────────────────────────────────────────
printf '\n'
printf '  \033[1;35m┌─────────────────────────────────────────┐\033[0m\n'
printf '  \033[1;35m│\033[0m  \033[1m🐾  PocketPaw Installer\033[0m                  \033[1;35m│\033[0m\n'
printf '  \033[1;35m│\033[0m  The AI agent that runs on your laptop   \033[1;35m│\033[0m\n'
printf '  \033[1;35m└─────────────────────────────────────────┘\033[0m\n'
printf '\n'

# ── OS Detection ────────────────────────────────────────────────────────
OS="$(uname -s 2>/dev/null || echo Unknown)"
case "$OS" in
    CYGWIN*|MINGW*|MSYS*|Windows_NT)
        printf '\033[31mError:\033[0m Native Windows is not supported.\n'
        printf '       Please use WSL (Windows Subsystem for Linux):\n'
        printf '       https://learn.microsoft.com/windows/wsl/install\n'
        exit 1
        ;;
esac

# ── Find Python 3.11+ ──────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python3.13 python3.12 python3.11 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    printf '\033[31mError:\033[0m Python 3.11+ is required but not found.\n'
    printf '       Install from https://www.python.org/downloads/\n'
    case "$OS" in
        Darwin) printf '       Or: brew install python@3.12\n' ;;
        Linux)  printf '       Or: sudo apt install python3.12 (Debian/Ubuntu)\n'
                printf '           sudo dnf install python3.12 (Fedora)\n' ;;
    esac
    exit 1
fi

PYTHON_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
printf '  Python:  %s (%s)\n' "$PYTHON_VER" "$(command -v "$PYTHON")"

# ── Find package installer (prefer uv > pip3 > pip) ────────────────────
PIP_CMD=""
if command -v uv >/dev/null 2>&1; then
    PIP_CMD="uv pip"
    printf '  Installer: uv pip\n'
elif "$PYTHON" -m pip --version >/dev/null 2>&1; then
    PIP_CMD="$PYTHON -m pip"
    printf '  Installer: pip (%s)\n' "$("$PYTHON" -m pip --version 2>/dev/null | cut -d' ' -f2)"
elif command -v pip3 >/dev/null 2>&1; then
    PIP_CMD="pip3"
    printf '  Installer: pip3\n'
elif command -v pip >/dev/null 2>&1; then
    PIP_CMD="pip"
    printf '  Installer: pip\n'
else
    printf '\033[31mError:\033[0m No pip or uv found. Install pip:\n'
    printf '       %s -m ensurepip --upgrade\n' "$PYTHON"
    exit 1
fi

printf '\n'

# ── Download installer.py ──────────────────────────────────────────────
TMPDIR="${TMPDIR:-/tmp}"
INSTALLER="$TMPDIR/pocketpaw_installer.py"

# Clean up on exit
cleanup() { rm -f "$INSTALLER"; }
trap cleanup EXIT INT TERM

INSTALLER_URL="https://raw.githubusercontent.com/pocketpaw/pocketpaw/dev/installer/installer.py"

if command -v curl >/dev/null 2>&1; then
    DOWNLOAD="curl -fsSL"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD="wget -qO-"
else
    printf '\033[31mError:\033[0m Neither curl nor wget found.\n'
    exit 1
fi

printf '  Downloading installer...\n'
if ! $DOWNLOAD "$INSTALLER_URL" > "$INSTALLER" 2>/dev/null; then
    printf '\033[33mWarn:\033[0m Primary download failed, trying fallback...\n'
    FALLBACK_URL="https://pocketpaw.com/installer.py"
    if ! $DOWNLOAD "$FALLBACK_URL" > "$INSTALLER" 2>/dev/null; then
        printf '\033[31mError:\033[0m Could not download installer.\n'
        printf '       Try manually: %s\n' "$INSTALLER_URL"
        exit 1
    fi
fi

# Verify it looks like Python
if ! head -1 "$INSTALLER" | grep -q "^#\|^\"\"\"\|^import\|^from\|^def\|^class"; then
    printf '\033[31mError:\033[0m Downloaded file does not look like a Python script.\n'
    printf '       Check your network connection and try again.\n'
    exit 1
fi

printf '  Launching interactive installer...\n\n'

# ── Run installer ──────────────────────────────────────────────────────
"$PYTHON" "$INSTALLER" --pip-cmd "$PIP_CMD" "$@"
