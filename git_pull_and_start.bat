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

echo [INFO] Starting server on http://localhost:5000...
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
echo App URL:      http://localhost:5000/app.html
echo Server URL:   http://localhost:5000
echo.
echo The server is running in a separate window.
echo Close that window to stop the server.
echo.
echo Press any key to close this launcher window...
pause >nul
