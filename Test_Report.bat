@echo off
REM Double-click after you have run and stopped the app.
REM Writes TEST_REPORT.md next to this file - that is the only thing to send back.
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" session_report.py
) else (
    python session_report.py
)
echo.
echo ============================================================
echo   Done. TEST_REPORT.md is in this folder.
echo   Add anything you noticed under "What I saw" at the bottom.
echo ============================================================
pause
