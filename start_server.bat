@echo off
echo ========================================
echo Text to Audio Converter Server
echo ========================================
echo.
echo Starting Flask server...
echo.
echo Once the server starts, open index.html in your browser
echo or navigate to http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Kill any existing processes on port 5000
echo Cleaning up any existing servers...
for /f "tokens=5" %%A in ('netstat -ano 2^>nul ^| findstr ":5000 "') do (
    taskkill /PID %%A /T /F 2>nul
)
timeout /t 1 /nobreak >nul 2>&1
echo.

py -3.11 server.py
pause
