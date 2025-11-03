# Troubleshooting Guide

## How to Debug Issues

### Step 1: Check the Browser Console

1. Open your browser (Chrome, Edge, or Firefox)
2. Press **F12** to open Developer Tools
3. Click on the **Console** tab
4. Look for error messages (they'll be red)

The console will show:
- Which files are being processed
- Server response status
- Detailed error messages
- Network requests

### Step 2: Check the Server Terminal

Look at the terminal/command prompt where you ran `start_server.bat`:
- You'll see detailed logs for every request
- File processing steps
- Error traces with full details
- Model loading messages

### Common Issues and Solutions

## Issue: "Generated audio for 0 file(s). Failed: 1"

**Causes:**
1. Server not running or crashed
2. Model failed to load
3. Text file encoding issues
4. Missing dependencies

**Debug Steps:**
1. Check the server terminal for error messages
2. Press F12 in browser and check Console tab
3. Look for specific error like:
   - `ModuleNotFoundError` - Missing Python package
   - `CUDA error` - GPU issue (will fall back to CPU)
   - `UnicodeDecodeError` - File encoding problem
   - `Connection refused` - Server not running

**Solutions:**
- **Server crashed:** Restart `start_server.bat`
- **Module not found:** Run `py -3.11 -m pip install <missing-module>`
- **CUDA error:** This is OK, it will use CPU (slower but works)
- **Encoding error:** Make sure your .txt files are UTF-8 encoded

## Issue: "Unable to connect to server"

**Cause:** Flask server isn't running

**Solution:**
1. Open `start_server.bat`
2. Wait for message: `Running on http://0.0.0.0:5000`
3. Then open `index.html` in your browser

## Issue: Audio generation is VERY slow or hangs

**Causes:**
1. First time - downloading TTS model (~1-2GB)
2. CPU processing (no GPU)
3. Model loading takes time

**What's Normal:**
- **First run:** 5-10 minutes to download model
- **Model loading:** 30-60 seconds on first request
- **Per file:** 10-30 seconds on CPU, 2-5 seconds on GPU

**Solutions:**
- Be patient on first run (one-time download)
- Check server terminal - it shows progress
- CPU is slower but works fine

## Issue: "ModuleNotFoundError: No module named 'chatterbox'"

**Cause:** Chatterbox TTS not installed properly

**Solution:**
```bash
py -3.11 -m pip install chatterbox-tts --no-deps
py -3.11 -m pip install librosa==0.11.0 transformers==4.46.3 diffusers==0.29.0
py -3.11 -m pip install resemble-perth==1.0.1 conformer==0.3.2 safetensors==0.5.3
py -3.11 -m pip install pykakasi s3tokenizer
```

## Issue: "Cannot open include file: 'longintrepr.h'"

**Cause:** pkuseg (Chinese support) can't compile on Windows

**Impact:** Only affects Chinese language - English works fine!

**Solution:** This is expected and doesn't affect English TTS

## Issue: Browser shows "Select a file to generate and play audio"

**Causes:**
1. Folder has no .txt files
2. Files don't have .txt extension
3. Browser folder selection didn't work

**Solutions:**
- Make sure files end with `.txt`
- Try a different folder
- Check browser console (F12) for errors

## Issue: Port 5000 already in use

**Error:** `OSError: [Errno 48] Address already in use`

**Solutions:**
1. Close other programs using port 5000
2. Kill existing Python processes
3. Restart your computer
4. Edit `server.py` and change `port=5000` to `port=5001`

## How to Get Detailed Logs

### Browser Logs:
1. Press **F12**
2. Go to **Console** tab
3. Click settings (gear icon)
4. Enable "Preserve log"
5. Try your action again
6. Copy all red error messages

### Server Logs:
1. Look at the terminal running `start_server.bat`
2. Scroll up to see all messages
3. Copy the entire error section (starts with "=== ERROR ===")

## Still Having Issues?

### Create a test file:

1. Create a file called `test.txt` with this content:
```
Hello, this is a test of the text to speech system.
```

2. Try loading just this one file
3. Check both browser console (F12) and server terminal
4. Report the specific error messages

### Check Python Version:
```bash
py -3.11 --version
```
Should show: `Python 3.11.0` or similar

### Check if packages are installed:
```bash
py -3.11 -m pip list | findstr chatterbox
py -3.11 -m pip list | findstr torch
py -3.11 -m pip list | findstr flask
```

All should show version numbers

## Performance Tips

- **CPU is slow but works:** 10-30 seconds per file is normal
- **GPU is faster:** If you have NVIDIA GPU with CUDA, it's 5x faster
- **Cache is your friend:** Second time generating same file is instant
- **Shorter text is faster:** Files under 500 chars process quicker

## Error Messages Explained

| Error | Meaning | Fix |
|-------|---------|-----|
| `No file provided` | Upload failed | Check browser console, retry |
| `Only .txt files are supported` | Wrong file type | Use .txt files only |
| `ModuleNotFoundError` | Missing Python package | Install with pip |
| `Connection refused` | Server not running | Start server first |
| `JSON parse error` | Server crashed | Check server terminal |
| `500 Internal Server Error` | Server-side error | Check server logs |

## Getting Help

When asking for help, provide:
1. Browser console logs (F12 → Console)
2. Server terminal output
3. Python version (`py -3.11 --version`)
4. Steps to reproduce the issue
5. Sample text file content (if relevant)
