@echo off
REM Start Majestic LLM. Double-click this file, or run it from a terminal.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py %*
) else (
    python run.py %*
)

REM Keep the window open if the server exits with an error, so the message
REM is readable instead of vanishing with the console.
if errorlevel 1 pause
