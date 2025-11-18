@echo off
REM Quick start script for Gutenberg Text Annotator UI (Windows)

echo ==========================================
echo Gutenberg Text Annotator UI
echo ==========================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Ollama found

    REM Check if Ollama is running
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% == 0 (
        echo ✓ Ollama is running
        echo.
        echo Available Ollama models:
        ollama list
    ) else (
        echo ⚠ Ollama is not running
        echo.
        echo To start Ollama, run in another terminal:
        echo   ollama serve
        echo.
        echo Then pull a model:
        echo   ollama pull llama3.2       # Fast (3B)
        echo   ollama pull llama3.1:8b    # Better (8B)
        echo   ollama pull qwen2.5:14b    # Best (14B)
        echo.
    )
) else (
    echo ⚠ Ollama not found
    echo.
    echo To use local models (free), install Ollama:
    echo   https://ollama.ai/download
    echo.
    echo Or use Anthropic (cloud, paid):
    echo   set ANTHROPIC_API_KEY=your-key-here
    echo.
)

REM Check for Anthropic API key
if defined ANTHROPIC_API_KEY (
    echo ✓ Anthropic API key found
) else (
    echo ℹ No Anthropic API key (optional)
)

echo.
echo ==========================================
echo Starting UI...
echo ==========================================
echo.
echo The interface will open at:
echo   http://localhost:7860
echo.
echo Press Ctrl+C to stop
echo.

REM Launch the UI
python annotator_ui.py

pause
