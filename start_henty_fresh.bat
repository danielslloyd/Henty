@echo off
REM Henty Launcher Script for Windows (Fresh Start)
REM Kills any running instances, checks dependencies, then starts the server and opens the app

echo.
echo ================================================
echo   Henty Audiobook Creation Suite
echo ================================================
echo.

REM Kill any previously running Henty server instances
echo [INFO] Stopping any previously running Henty server...
taskkill /FI "WINDOWTITLE eq Henty Server*" /T /F 2>nul
for /f "tokens=5" %%A in ('netstat -ano ^| findstr ":5000 " 2^>nul') do (
    taskkill /PID %%A /T /F 2>nul
)
timeout /t 1 /nobreak >nul 2>&1

REM --- Find the Python that has Flask installed ---
REM Check candidates in order: common install paths, then fall back to PATH
set HENTY_PYTHON=

REM Try Python locations most likely to have the packages
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%PROGRAMFILES%\Python311\python.exe"
    "%PROGRAMFILES%\Python312\python.exe"
    "python"
) do (
    if not defined HENTY_PYTHON (
        %%P -c "import flask" >nul 2>&1
        if not errorlevel 1 (
            set HENTY_PYTHON=%%P
        )
    )
)

if not defined HENTY_PYTHON (
    echo [ERROR] Could not find a Python installation with Flask.
    echo         Please install Flask: pip install flask
    pause
    exit /b 1
)

echo [INFO] Using Python: %HENTY_PYTHON%

REM Check if server.py exists
if not exist "server.py" (
    echo [ERROR] server.py not found. Run this script from the Henty directory.
    pause
    exit /b 1
)

REM --- Run pre-flight dependency check (auto-repairs broken packages) ---
echo.
echo [INFO] Running pre-flight dependency check...
%HENTY_PYTHON% precheck.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pre-flight check failed. See messages above.
    pause
    exit /b 1
)

REM Check if ffmpeg is available, install if not
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] ffmpeg not found - attempting to install via winget...
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo [WARNING] winget install failed. Install ffmpeg manually for transcription scoring.
        echo.
    ) else (
        echo [OK] ffmpeg installed.
        echo.
    )
) else (
    echo [OK] ffmpeg found
)

REM Detect local network IP
set LOCAL_IP=
for /f "delims=" %%A in ('powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } | Sort-Object InterfaceIndex | Select-Object -First 1).IPAddress" 2^>nul') do set LOCAL_IP=%%A

REM Detect public IP
set PUBLIC_IP=
for /f "delims=" %%A in ('curl -s --max-time 4 https://ifconfig.me 2^>nul') do set PUBLIC_IP=%%A

echo.
echo [INFO] Starting server...
echo.

REM Start the server using the detected Python, logging output to server_log.txt
REM Window stays open after crash so you can see the error
if exist server_log.txt del server_log.txt
start "Henty Server" cmd /k "%HENTY_PYTHON% server.py > server_log.txt 2>&1 || (echo. & echo ====== SERVER CRASHED - check server_log.txt ====== & echo.)"

REM Wait for server to start
echo [INFO] Waiting for server to initialize...
timeout /t 4 /nobreak >nul

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
if defined PUBLIC_IP (
    echo   External:    http://%PUBLIC_IP%:5000/listen
    echo   [Requires port 5000 forwarded on your router]
) else (
    echo   External:    [could not detect public IP]
)
echo.
echo ================================================
echo.
echo The server is running in a separate window.
echo Close that window to stop the server.
echo.
echo Press any key to close this launcher window...
pause >nul
