#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_PATH="${VENV_PATH:-.venv}"

echo "[+] Checking venv at ${VENV_PATH}"
if [ ! -d "${VENV_PATH}" ]; then
  echo "[+] Creating venv..."
  "${PYTHON}" -m venv "${VENV_PATH}"
fi

echo "[+] Activating venv"
source "${VENV_PATH}/bin/activate"

echo "[+] Upgrading pip and installing requirements"
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

if [ ! -f .env ]; then
  echo "[i] Create a .env with TELEGRAM_BOT_TOKEN and OPENAI_API_KEY"
fi

echo "[+] Starting bot"
python main.py

