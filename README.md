# Henty Project - Enhanced Project Gutenberg Reader

A comprehensive suite of tools for downloading, annotating, and experiencing Project Gutenberg texts with AI-powered context and audio narration.

## 🌟 Features

### 📚 AI-Powered Text Annotation
- **Automatic entity detection**: Places, people, historical topics, and archaic terms
- **Rich annotations**: Coordinates, biographies, definitions, and sources
- **Local or Cloud LLMs**: Use free Ollama models or Anthropic Claude
- **Beautiful HTML reader**: Responsive design with interactive annotations

### 🔊 Text-to-Speech
- **High-quality TTS**: Chatterbox TTS from Resemble AI
- **Batch processing**: Generate audio for entire books
- **Web interface**: Side-by-side text and audio display

### 📖 Complete Reading Experience
- **Download from Gutenberg**: Automatic processing of books
- **Interactive reader**: Hover annotations, adjustable fonts
- **Mobile-friendly**: Responsive design for all devices
- **Offline capable**: Works locally without internet

## 🚀 Quick Start

### One-Line Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the annotator UI (recommended)
python annotator_ui.py
```

Then open http://localhost:7860 in your browser!

### Or Use Convenience Scripts

**Linux/Mac:**
```bash
./start_annotator.sh
```

**Windows:**
```batch
start_annotator.bat
```

## 📦 What's Included

### 1. Text Annotator UI (`annotator_ui.py`)
**The main interface** - Gradio UI for annotating texts

**Features:**
- Choose between Ollama (local/free) and Anthropic (cloud/paid) models
- Download directly from Project Gutenberg URLs
- Upload your own text files
- Process full books or test with sample paragraphs
- Automatic entity detection and enrichment
- Progress tracking and error handling

**See:** [`ANNOTATOR_UI_GUIDE.md`](ANNOTATOR_UI_GUIDE.md) for full guide

### 2. HTML Reader (`reader.html`)
Beautiful, responsive reader for annotated texts

**Features:**
- Main reading column + annotation sidebar
- Adjustable font size (12-32px)
- Color-coded annotations (place, person, topic, term)
- Links to Wikipedia, Britannica, and other sources
- Embedded images from Wikimedia Commons
- Keyboard shortcuts (Cmd/Ctrl +/- for font size)

**See:** [`READER_DOCUMENTATION.md`](READER_DOCUMENTATION.md) for details

### 3. Gutenberg Processor (`scripts/gutenberg_processor.py`)
Download and process Project Gutenberg texts

**Features:**
- Download from Gutenberg URLs
- Automatic title extraction
- Remove metadata headers/footers
- Split into chapters
- Batch processing

### 4. Text-to-Speech (`app.py`, `server.py`)
Convert text to high-quality audio narration

**Features:**
- Chatterbox TTS model
- Web interface with audio player
- Batch audio generation
- Audio caching

## 🎯 Complete Workflow

### From Gutenberg Book to Interactive Reader (5 minutes)

```bash
# 1. Start the annotator UI
python annotator_ui.py

# 2. In the UI:
#    - Choose your model (Ollama or Anthropic)
#    - Paste a Gutenberg URL
#    - Click "Download & Process"
#    - Select a chapter
#    - Click "Annotate"
#
# 3. Click "Open in Reader" and load the JSON file
#
# Done! Hover over highlighted text to see annotations
```

### Example URLs to Try

- **"With Clive in India" by G.A. Henty:**
  `https://www.gutenberg.org/cache/epub/4932/pg4932.txt`

- **"The Adventures of Sherlock Holmes":**
  `https://www.gutenberg.org/cache/epub/1661/pg1661.txt`

- **"Pride and Prejudice":**
  `https://www.gutenberg.org/cache/epub/1342/pg1342.txt`

## 🤖 Model Options

### Local Models (Ollama) - FREE

**Install Ollama:**
```bash
# Visit https://ollama.ai/download
# Or on Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull models
ollama pull llama3.2        # 3B - Fast
ollama pull llama3.1:8b     # 8B - Better quality
ollama pull qwen2.5:14b     # 14B - Best local quality
```

**Pros:**
- ✅ Completely free
- ✅ Private (runs locally)
- ✅ No API limits
- ✅ Works offline

**Cons:**
- ⚠️ Requires disk space (2-14GB per model)
- ⚠️ Slower on CPU
- ⚠️ Lower quality than Claude

### Cloud Models (Anthropic) - PAID

**Setup:**
```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-key-here"
```

**Pros:**
- ✅ Highest quality annotations
- ✅ Very fast
- ✅ No local resources needed
- ✅ Best for historical accuracy

**Cons:**
- ⚠️ Costs ~$1-2 per chapter
- ⚠️ Requires internet
- ⚠️ API rate limits

## 📋 Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB+ recommended for local models)
- **Disk**: 10GB+ for Ollama models
- **GPU**: Optional but recommended for TTS and large Ollama models

See [`requirements.txt`](requirements.txt) for Python dependencies.

## 📚 Documentation

- **[ANNOTATOR_UI_GUIDE.md](ANNOTATOR_UI_GUIDE.md)** - Complete UI guide with tips & troubleshooting
- **[READER_DOCUMENTATION.md](READER_DOCUMENTATION.md)** - Technical details for the HTML reader
- **[READER_QUICKSTART.md](READER_QUICKSTART.md)** - Quick examples and workflows

## 🎨 What Gets Annotated

The AI automatically identifies and annotates:

### 📍 Places (Green)
- Geographic locations with coordinates
- Modern names if changed
- Historical context

**Example:** "Madras" → Shows coordinates (13.08°N, 80.27°E), notes modern name "Chennai"

### 👤 People (Red)
- Historical figures
- Birth/death years
- Key accomplishments
- Portrait images

**Example:** "Robert Clive" → Biography, dates (1725-1774), portrait from Wikipedia

### 📖 Topics (Yellow)
- Historical events
- Time periods
- Cultural concepts

**Example:** "British East India Company" → Founding date, purpose, historical significance

### 🔤 Terms (Purple)
- Archaic words
- Etymology
- Modern equivalents
- Definitions

**Example:** "fortnight" → Etymology ("fourteen nights"), modern equivalent ("two weeks")

## 💡 Use Cases

- **Education**: Enhance historical texts for students
- **Research**: Quick context for historical documents
- **Reading Groups**: Shared annotations for book clubs
- **Personal Enrichment**: Deeper understanding of classic literature
- **Accessibility**: Audio narration for all texts
- **Digital Humanities**: Structured annotations for analysis

## 🛠️ Advanced Usage

### Command Line

```bash
# Annotate with Ollama (default)
python scripts/text_annotator.py input.txt output.json

# Use specific model
python scripts/text_annotator.py input.txt output.json --model llama3.1:8b

# Use Anthropic
python scripts/text_annotator.py input.txt output.json --backend anthropic

# Download from Gutenberg
python scripts/gutenberg_processor.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt
```

### Customize Annotations

Edit `scripts/text_annotator.py` to modify:
- Entity types to detect
- Annotation detail level
- Source preferences
- Output format

### Web Server Integration

The reader can be served via the included Flask server:

```bash
python server.py
# Visit http://localhost:5000
```

## 📊 Performance

### Annotation Speed

| Model | Speed (per paragraph) | Quality | RAM |
|-------|----------------------|---------|-----|
| Llama 3.2 (3B) | 5-10 sec | Good | 8GB |
| Llama 3.1 (8B) | 15-30 sec | Great | 16GB |
| Qwen 2.5 (14B) | 30-60 sec | Excellent | 32GB |
| Claude Sonnet 4.5 | 3-5 sec | Best | N/A |

*On CPU. GPU is 3-10x faster.*

### Cost Estimates (Anthropic)

| Content | Typical Cost |
|---------|--------------|
| Sample (3 para) | $0.10-0.20 |
| Full chapter (~50 para) | $1-2 |
| Full book (20 chapters) | $20-40 |

**Ollama is completely free!**

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional annotation types (quotes, references, etc.)
- Map integration for place annotations
- Timeline view for historical events
- Multi-language support
- OCR for scanned texts
- Export to EPUB/PDF

## 📄 License

This project uses several open-source components:

- **Chatterbox TTS**: See [Resemble AI](https://github.com/resemble-ai/chatterbox) for license
- **Project Gutenberg texts**: Public domain
- **Ollama**: MIT License
- **Other dependencies**: See respective packages

The code in this repository is provided as-is for educational purposes.

## 🙏 Acknowledgments

- **Project Gutenberg**: For preserving and providing free access to classic literature
- **Anthropic**: For Claude AI models
- **Ollama**: For making local LLMs accessible
- **Resemble AI**: For Chatterbox TTS
- **G.A. Henty**: Inspiration for the project name

## 📧 Support

Having issues? Check:

1. **[ANNOTATOR_UI_GUIDE.md](ANNOTATOR_UI_GUIDE.md)** - Troubleshooting section
2. **Console output** - Look for error messages
3. **Model status** - Ensure Ollama is running or API key is set
4. **Dependencies** - Try reinstalling: `pip install -r requirements.txt --upgrade`

## 🚀 Roadmap

- [ ] Map view with all annotated places
- [ ] Character relationship graphs
- [ ] Reading statistics and progress tracking
- [ ] Social features (share annotations)
- [ ] Mobile app
- [ ] PDF/EPUB export
- [ ] OCR for scanned books
- [ ] Multi-language support
- [ ] Voice selection for TTS
- [ ] Collaborative annotation

---

**Happy Reading!** 📖✨

Transform your reading experience with AI-powered context and understanding.
