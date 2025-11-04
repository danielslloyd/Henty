@echo off
echo ========================================
echo Voice Sample Fix - Cleanup and Restart
echo ========================================
echo.

echo Step 1: Deleting old voice samples...
if exist "voice_samples\Dan_Test1.wav" (
    del "voice_samples\Dan_Test1.wav"
    echo   - Deleted Dan_Test1.wav
) else (
    echo   - Dan_Test1.wav not found (already deleted?)
)
echo.

echo Step 2: Starting server...
echo.
echo IMPORTANT: After the server starts:
echo   1. Open index.html in your browser
echo   2. Re-upload or re-record your voice sample
echo   3. Try generating audio with voice cloning
echo.
echo Press Ctrl+C to stop the server when done.
echo.

py -3.11 server.py
pause
