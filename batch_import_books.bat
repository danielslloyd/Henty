@echo off
REM Batch import all books from BOOKS_DIR (creates projects/chunks without UI)

echo.
echo ================================================
echo   Batch Import Books
echo ================================================
echo.

setlocal enabledelayedexpansion

REM --- Find Python with Flask installed ---
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
echo.

REM Check if script exists
if not exist "batch_import_books.py" (
    echo [ERROR] batch_import_books.py not found.
    pause
    exit /b 1
)

REM Silence urllib3 warning
set PYTHONWARNINGS=ignore:urllib3

REM Run the import
%HENTY_PYTHON% batch_import_books.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Import failed
    pause
    exit /b 1
)

echo.
echo Press any key to close this window...
pause >nul
