# Quick Start Guide

## Installation

1. **Install Python** (if not already installed)
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Install dependencies**
   - Open Command Prompt or PowerShell in this folder
   - Run: `pip install -r requirements.txt`
   - This will install all required packages (may take a few minutes)

## Running the App

### Easy Way (Windows)
1. Double-click `start_server.bat`
2. Wait for the server to start
3. Open `index.html` in your browser (Chrome, Edge, or Firefox)

### Manual Way
1. Open Command Prompt in this folder
2. Run: `python server.py`
3. Open `index.html` in your browser

## Using the App

1. **Click "Browse Folder"** - Select any folder on your computer
2. **All .txt files appear automatically** - No need to click anything else
3. **First file loads automatically** - Text appears on the left, audio generates on the right
4. **Select other files** - Use the dropdown to switch between files
5. **Generate all audio** - Click the button to process all files at once

## Troubleshooting

### "Python not found"
- Make sure Python is installed and added to PATH
- Restart your computer after installing Python

### "Unable to connect to server"
- Make sure `server.py` is running
- Check that the server shows "Running on http://0.0.0.0:5000"

### "No .txt files found"
- Make sure your folder actually contains .txt files
- Check that files have the .txt extension

### Audio generation is slow
- First time will be slower as it downloads the TTS model
- CPU processing is slower than GPU
- Consider using a CUDA-capable GPU for faster generation

## Features

- Automatic .txt file detection
- Side-by-side text and audio display
- Audio caching (generated once, reused forever)
- Batch processing
- Modern, beautiful UI
- Works completely offline after initial model download

## Next Steps

- Try different text files
- Generate audio for an entire folder
- Share the generated audio files (saved in `generated_audio` folder)

Enjoy converting your text to speech!
