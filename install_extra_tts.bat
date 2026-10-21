@echo off
REM ============================================================
REM  Install optional TTS engines for Henty: Kokoro + Orpheus
REM  Installs into the SAME Python that runs the Henty server
REM  (the one that already has Flask). Both engines are optional;
REM  the server starts without them and disables whichever is absent.
REM ============================================================

echo.
echo ================================================
echo   Henty - Optional TTS Engine Installer
echo ================================================
echo.

REM --- Find the Python that has Flask installed (matches the launcher) ---
set HENTY_PYTHON=
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
        if not errorlevel 1 set HENTY_PYTHON=%%P
    )
)

if not defined HENTY_PYTHON (
    echo [ERROR] Could not find the Henty Python (the one with Flask installed^).
    echo         Run this from the Henty directory after the server has worked once.
    pause
    exit /b 1
)

echo [INFO] Using Python: %HENTY_PYTHON%
echo.

REM --- Kokoro-82M: lightweight engine with named voices ---
echo [INFO] Installing Kokoro (kokoro + misaki[en])...
%HENTY_PYTHON% -m pip install "kokoro" "misaki[en]"
if errorlevel 1 (echo [WARNING] Kokoro pip install failed - see above.) else (echo [OK] Kokoro packages installed.)
echo.

echo [INFO] Checking for espeak-ng (required by Kokoro)...
where espeak-ng >nul 2>&1
if errorlevel 1 (
    echo [INFO] espeak-ng not found - attempting install via winget...
    winget install --id eSpeak-NG.eSpeak-NG -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [WARNING] Could not auto-install espeak-ng.
        echo           Get it from https://github.com/espeak-ng/espeak-ng/releases and add it to PATH.
    ) else (
        echo [OK] espeak-ng installed - reopen your terminal so PATH updates.
    )
) else (
    echo [OK] espeak-ng found.
)
echo.

REM --- Orpheus-3b: heavy (~6GB) engine with named speakers, SNAC-decoded ---
echo [INFO] Installing Orpheus dependencies (snac + transformers)...
%HENTY_PYTHON% -m pip install "snac" "transformers"
if errorlevel 1 (echo [WARNING] Orpheus dependency install failed - see above.) else (echo [OK] Orpheus dependencies installed. The 3B model downloads on first use.)
echo.

echo ================================================
echo   Verifying engine availability
echo ================================================
%HENTY_PYTHON% -c "import importlib.util as u; print('  kokoro      :', 'OK' if u.find_spec('kokoro') else 'MISSING'); print('  snac        :', 'OK' if u.find_spec('snac') else 'MISSING'); print('  transformers:', 'OK' if u.find_spec('transformers') else 'MISSING')"
echo.
echo Done. Restart the server (start_henty_fresh.bat) to pick up the new engines.
echo.
pause
