# Henty Dependencies — Locked Versions

**Last Updated:** 2026-05-28  
**Environment:** Windows 11 + RTX 5070 Ti (CUDA 13.0)

## Python Runtime
- **Python:** 3.11.0

## GPU/ML Stack (CUDA 13.0 — Essential)
- **PyTorch:** 2.12.0+cu130
- **torchaudio:** 2.11.0+cu130
- **torchvision:** 0.27.0+cu130
- **CUDA Version:** 13.0

## TTS Engine
- **chatterbox-tts:** 0.1.7

## Scientific Computing
- **numpy:** 2.2.6 (must be <2.3 for numba/chatterbox compatibility)
- **scipy:** 1.16.3

## Web Framework
- **Flask:** 3.0.0
- **Flask-CORS:** 6.0.1
- **Flask-SocketIO:** 5.5.1
- **python-socketio:** 5.15.0

## UI/API
- **gradio:** 6.8.0

## Audio Processing
- **pydub:** 0.25.1
- **librosa:** 0.11.0
- **soundfile:** 0.13.1

## Web/Data Utilities
- **requests:** 2.31.0
- **beautifulsoup4:** 4.14.2
- **python-dotenv:** 1.0.1

## Optional/LLM
- **anthropic:** (not currently installed; add if needed)
- **ollama:** (not currently installed; add if needed)

## Critical Notes

1. **numpy constraint:** Must be <2.3 for numba (Chatterbox dependency). Current 2.2.6 is correct.
2. **PyTorch GPU:** Install with `--index-url https://download.pytorch.org/whl/cu130` for CUDA 13.0 support.
3. **Chatterbox version lock:** 0.1.7 works with torch 2.12.0 despite its declared requirement of torch==2.6.0 (tested and verified).
4. **Full dependency tree:** See `pip freeze` output in version control for complete transitive dependencies.
