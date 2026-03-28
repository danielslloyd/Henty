@echo off
REM Henty Launcher Script for Windows
REM Checks dependencies (Python, ffmpeg), then starts the server and opens the app

echo.
echo ================================================
echo   Henty Audiobook Creation Suite
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if server.py exists
if not exist "server.py" (
    echo [ERROR] server.py not found
    echo Please run this script from the Henty directory
    pause
    exit /b 1
)

REM Check if ffmpeg is available, install if not
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] ffmpeg not found - required for audio transcription scoring
    echo [INFO] Attempting to install via winget...
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo [WARNING] winget install failed. Please install ffmpeg manually:
        echo   1. Download from https://ffmpeg.org/download.html
        echo   2. Extract and add the bin\ folder to your system PATH
        echo   Transcription scoring will not work until ffmpeg is installed.
        echo.
    ) else (
        echo [SUCCESS] ffmpeg installed. You may need to restart the launcher
        echo           once for PATH changes to take effect.
        echo.
    )
) else (
    echo [OK] ffmpeg found
)

REM Detect local network IP for mobile access
set LOCAL_IP=
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /C:"IPv4 Address" ^| findstr /V "127.0.0.1"') do (
    if not defined LOCAL_IP (
        set RAW_IP=%%A
        setlocal enabledelayedexpansion
        set LOCAL_IP=!RAW_IP: =!
        endlocal & set LOCAL_IP=%LOCAL_IP%
    )
)

REM Detect Tailscale IP
set TAILSCALE_IP=
tailscale ip -4 >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%A in ('tailscale ip -4 2^>nul') do (
        if not defined TAILSCALE_IP set TAILSCALE_IP=%%A
    )
)

echo [INFO] Starting server...
echo.

REM Start the server in a new window
start "Henty Server" python server.py

REM Wait for server to start
echo [INFO] Waiting for server to initialize...
timeout /t 3 /nobreak >nul

REM Open browser
echo [INFO] Opening browser...
start http://localhost:5000/app.html

echo.
echo ================================================
echo   Henty is ready!
echo ================================================
echo.
echo   Desktop:  http://localhost:5000/app.html
echo.
echo   Mobile listener (/listen page):
if defined LOCAL_IP (
    echo   Local WiFi:  http://%LOCAL_IP%:5000/listen
) else (
    echo   Local WiFi:  [could not detect local IP]
)
if defined TAILSCALE_IP (
    echo   Tailscale:   http://%TAILSCALE_IP%:5000/listen
    echo   [Tailscale active - works from any network]
) else (
    echo   Tailscale:   [not detected]
    echo   Install Tailscale for remote access: https://tailscale.com/download
)
echo.
echo ================================================
echo.
echo The server is running in a separate window.
echo Close that window to stop the server.
echo.
echo Press any key to close this launcher window...
pause >nul
