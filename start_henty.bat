@echo off
REM Henty Launcher Script for Windows
REM Starts the server and opens the landing page in your default browser

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

echo [INFO] Starting server on http://localhost:5000...
echo.

REM Start the server in a new window
start "Henty Server" python server.py

REM Wait for server to start
echo [INFO] Waiting for server to initialize...
timeout /t 3 /nobreak >nul

REM Open browser
echo [INFO] Opening browser...
start http://localhost:5000/index.html

echo.
echo ================================================
echo   Henty is ready!
echo ================================================
echo.
echo Landing Page: http://localhost:5000/index.html
echo Server URL:   http://localhost:5000
echo.
echo The server is running in a separate window.
echo Close that window to stop the server.
echo.
echo Press any key to close this launcher window...
pause >nul
