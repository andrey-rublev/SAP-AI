@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    >&2 echo The project Python environment is missing.
    >&2 echo Run: python -m venv .venv
    >&2 echo Then: .\.venv\Scripts\python.exe -m pip install -e ".[live]"
    >&2 echo Tesseract OCR must also be installed; see docs\desktop.md.
    exit /b 1
)
".venv\Scripts\python.exe" desktop.py play %*
exit /b %errorlevel%
