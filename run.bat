@echo off
rem Start live Armenian-to-English transcription. Options pass through, e.g.
rem   run.bat --list-devices
rem   run.bat --mic 1 --output 5
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment is missing. Set it up once with:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

rem UTF-8 console code page, so Armenian text shows correctly.
chcp 65001 >nul
".venv\Scripts\python.exe" -m live_hy_en %*
