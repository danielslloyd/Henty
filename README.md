# Text to Audio Converter - Chatterbox TTS

A Python application that converts text files to audio using Chatterbox TTS from Hugging Face. Features both a Gradio interface and a standalone HTML web interface with side-by-side text and audio display.

## Features

- Browse and select directories containing .txt files
- Automatically find all text files in a directory
- Display text content alongside generated audio
- Generate audio for individual files or all files at once
- Uses state-of-the-art Chatterbox TTS model from Resemble AI
- Caches generated audio files to avoid regeneration

## Requirements

- Python 3.11 or higher (recommended)
- CUDA-capable GPU (optional, but recommended for faster processing)
- See [requirements.txt](requirements.txt) for full dependencies

## Installation

1. Clone or download this repository

2. Make sure you have Python 3.11+ installed. Check with:
```bash
python --version
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Note for Windows users:** If `python` command is not found, you may need to install Python from [python.org](https://www.python.org/downloads/) and make sure to check "Add Python to PATH" during installation.

## Usage

### Option 1: HTML Interface (Recommended)

1. Start the Flask server:
```bash
python server.py
```
   Or on Windows, simply double-click `start_server.bat`

2. Open [index.html](index.html) in your web browser, or navigate to `http://localhost:5000`

3. In the web interface:
   - Click "Browse Folder" to select a folder containing .txt files
   - All .txt files will automatically appear in the dropdown
   - The first file will be automatically loaded and displayed
   - Select any file from the dropdown to view its content and generate/play audio
   - Click "Generate Audio for All Files" to batch process all text files at once

### Option 2: Gradio Interface

1. Start the Gradio application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

3. Use the same interface controls as described above

## How It Works

- The app scans the specified directory for all .txt files
- When you select a file, it displays the text content on the left side
- Audio is automatically generated using Chatterbox TTS and displayed on the right side
- Generated audio files are saved in the `generated_audio` folder
- Previously generated audio is reused to save time

## Notes

- The first run will download the Chatterbox TTS model (this may take a few minutes)
- Very long text files are truncated to 1000 characters for demo purposes (you can adjust this in the code)
- Audio files are saved as WAV format
- GPU acceleration is automatically used if available

## Technologies Used

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) - State-of-the-art open-source TTS from Resemble AI
- [Flask](https://flask.palletsprojects.com/) - Backend API server
- [Gradio](https://gradio.app/) - Alternative web UI framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Torchaudio](https://pytorch.org/audio/) - Audio processing
- HTML/CSS/JavaScript - Standalone web interface

## License

This project uses the Chatterbox TTS model, which is provided by Resemble AI. Please refer to their repository for license information.

## Troubleshooting

- If you get CUDA errors, the app will automatically fall back to CPU processing
- Make sure you have sufficient disk space for the model and generated audio files
- If the model fails to load, try reinstalling the dependencies
