Param(
  [string]$Python = "python",
  [string]$VenvPath = ".venv"
)

Write-Host "[+] Checking venv at $VenvPath"
if (-not (Test-Path $VenvPath)) {
  Write-Host "[+] Creating venv..."
  & $Python -m venv $VenvPath
}

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
  Write-Error "Activation script not found at $activate"
  exit 1
}

Write-Host "[+] Activating venv"
. $activate

Write-Host "[+] Upgrading pip and installing requirements"
python -m pip install --upgrade pip
if (Test-Path "requirements.txt") {
  pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
  Write-Host "[i] Create a .env with TELEGRAM_BOT_TOKEN and OPENAI_API_KEY"
}

Write-Host "[+] Starting bot"
python main.py

