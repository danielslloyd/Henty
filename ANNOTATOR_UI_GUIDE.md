# Gutenberg Text Annotator UI - Quick Start Guide

## 🚀 One-Step Setup & Launch

The Annotator UI provides a simple, all-in-one interface for annotating Project Gutenberg texts with AI. Choose between **local models (Ollama)** or **cloud models (Anthropic)**.

## Quick Start (2 minutes)

### Option 1: Local Models (Ollama) - FREE

```bash
# 1. Install Ollama (if not already installed)
# Visit: https://ollama.ai/download
# Or on Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama
ollama serve

# 3. Pull a model (in another terminal)
ollama pull llama3.2        # 3B - Fast
# OR
ollama pull llama3.1:8b     # 8B - Better quality
# OR
ollama pull qwen2.5:14b     # 14B - Best local quality

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Launch the UI
python annotator_ui.py
```

The UI will open in your browser at `http://localhost:7860`

### Option 2: Cloud Models (Anthropic) - PAID

```bash
# 1. Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-key-here"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the UI
python annotator_ui.py
```

## Using the UI

### 1. Choose Your Model

The UI automatically detects:
- **Ollama models** (local, free, private)
- **Anthropic models** (cloud, paid, highest quality)

Select your preferred model from the dropdown.

### 2. Two Ways to Annotate

#### **Option A: From Gutenberg URL** (Recommended for books)

1. Go to [Project Gutenberg](https://www.gutenberg.org/)
2. Find a book and copy its `.txt` URL
   - Example: `https://www.gutenberg.org/cache/epub/4932/pg4932.txt`
3. Paste URL in the UI
4. Click "Download & Process"
5. Select a chapter
6. Click "Annotate Selected Chapter"

#### **Option B: Upload Text File** (For custom text)

1. Click "Upload Text File" tab
2. Upload any `.txt` file
3. Click "Annotate File"

### 3. Adjust Settings (Optional)

- **Max Paragraphs**: Limit processing for testing
  - Set to `3-5` for quick tests
  - Set to `0` to process entire file
- **API Key**: Only needed if using Anthropic

### 4. View Results

Once annotation completes:
1. Click "Open in Reader"
2. In the reader, click "Choose File"
3. Select the generated JSON file
4. Hover over highlighted text to see annotations!

## Model Comparison

| Model | Type | Speed | Quality | Cost | Privacy |
|-------|------|-------|---------|------|---------|
| Llama 3.2 (3B) | Ollama | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | Free | 🔒 Local |
| Llama 3.1 (8B) | Ollama | ⚡⚡ Medium | ⭐⭐⭐⭐ Great | Free | 🔒 Local |
| Qwen 2.5 (14B) | Ollama | ⚡ Slow | ⭐⭐⭐⭐⭐ Excellent | Free | 🔒 Local |
| Claude Sonnet 4.5 | Anthropic | ⚡⚡⚡ Fast | ⭐⭐⭐⭐⭐ Best | ~$1-2/chapter | ☁️ Cloud |

## Features

### What Gets Annotated

The AI automatically identifies and annotates:

- **📍 Places** (Green)
  - Geographic locations
  - Coordinates (lat/lon)
  - Modern names if changed
  - Historical context

- **👤 People** (Red)
  - Historical figures
  - Birth/death years
  - Key accomplishments
  - Portraits (when available)

- **📖 Topics** (Yellow)
  - Historical events
  - Time periods
  - Cultural concepts
  - Background context

- **🔤 Terms** (Purple)
  - Archaic words
  - Etymology
  - Modern equivalents
  - Definitions

### All Annotations Include

- Short summary
- Detailed explanation
- Authoritative sources (Wikipedia, Britannica, etc.)
- Links for further reading
- Images (when available)

## Tips for Best Results

### 1. Model Selection

**For Testing** (fastest):
- Use Llama 3.2 (3B)
- Set max paragraphs to 3-5

**For Production** (balanced):
- Use Llama 3.1 (8B) or Qwen 2.5 (14B)
- Process full chapters

**For Best Quality**:
- Use Claude Sonnet 4.5
- Note: Costs ~$1-2 per chapter

### 2. Processing Strategy

**Testing a Book**:
```
1. Download book from Gutenberg
2. Select first chapter
3. Set max paragraphs to 3
4. Run with fast model (Llama 3.2)
5. Review results
```

**Processing Full Book**:
```
1. Test with first chapter (see above)
2. Once satisfied, process each chapter:
   - Set max paragraphs to 0 (all)
   - Use quality model (Llama 3.1 8B or better)
   - Process overnight for large books
```

### 3. Ollama Performance

**GPU**:
- Recommended: 8GB+ VRAM for 8B models
- 16GB+ VRAM for 14B+ models
- Much faster inference

**CPU** (no GPU):
- Works fine, just slower
- 3B models: ~5-10 sec/paragraph
- 8B models: ~15-30 sec/paragraph
- 14B+ models: ~30-60 sec/paragraph

**RAM**:
- 3B models: 8GB RAM minimum
- 8B models: 16GB RAM recommended
- 14B+ models: 32GB RAM recommended

### 4. Cost Estimates (Anthropic)

Using Claude Sonnet 4.5:

| Paragraphs | Typical Cost |
|------------|--------------|
| 3 (testing) | $0.10-0.20 |
| Full chapter (~50 para) | $1-2 |
| Full book (20 chapters) | $20-40 |

**Note**: Ollama models are completely free!

## Troubleshooting

### "No Ollama models found"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve

# Pull a model:
ollama pull llama3.2
```

### "Cannot connect to Ollama"

1. Make sure Ollama is running: `ollama serve`
2. Check firewall settings
3. Verify port 11434 is not blocked

### "Anthropic API key required"

Either:
- Enter key in the UI (under Settings)
- Or set environment variable:
  ```bash
  export ANTHROPIC_API_KEY="your-key-here"
  ```

### Slow annotation speed

Local models (Ollama):
- Use smaller model (3B instead of 8B)
- Reduce max paragraphs
- Use GPU if available
- Close other applications

Cloud models (Anthropic):
- Should be fast (<5 sec/paragraph)
- Check internet connection
- Verify API rate limits

### JSON parsing errors

This can happen with smaller or less capable models:
- Try a larger model (8B instead of 3B)
- Or use Anthropic for more reliable JSON
- Check console for detailed error

## Advanced: Command Line

You can also use the annotator from command line:

```bash
# With Ollama (default)
python scripts/text_annotator.py input.txt output.json

# With specific model
python scripts/text_annotator.py input.txt output.json --model llama3.1:8b

# With Anthropic
python scripts/text_annotator.py input.txt output.json --backend anthropic
```

## File Locations

After annotation, files are saved to:

```
annotated_output/
├── book_name/                    # From Gutenberg downloads
│   ├── chapter_01.txt
│   ├── chapter_02.txt
│   └── ...
├── chapter_01_annotated.json     # Annotated files
├── chapter_02_annotated.json
└── ...
```

## Next Steps

1. **Annotate a book**: Try the full workflow with a Gutenberg URL
2. **Customize**: Edit prompts in `scripts/text_annotator.py`
3. **Integrate**: Use annotations in your own projects
4. **Contribute**: Share improvements or request features

## Ollama Model Recommendations

### For Historical Texts (like Henty books)

**Best overall**: `qwen2.5:14b`
- Excellent at historical context
- Good with archaic terms
- Accurate coordinates

**Fast & good**: `llama3.1:8b`
- Good balance of speed and quality
- Decent historical knowledge
- Works well on most systems

**Testing**: `llama3.2` (3B)
- Very fast
- Good for quick tests
- Lower quality but acceptable

### Other Good Options

- `mistral:7b` - Fast, general purpose
- `neural-chat:7b` - Good for explanations
- `openchat:7b` - Conversational style

### Pull Models

```bash
# Recommended for this project
ollama pull qwen2.5:14b         # Best quality (14GB download)
ollama pull llama3.1:8b         # Good balance (4.7GB download)
ollama pull llama3.2            # Fast testing (2GB download)

# List installed models
ollama list

# Remove a model
ollama rm model-name
```

## FAQ

**Q: Do I need a GPU?**
A: No, but it helps. Ollama works on CPU, just slower.

**Q: Can I use both Ollama and Anthropic?**
A: Yes! Switch between them in the UI dropdown.

**Q: How much does Anthropic cost?**
A: ~$1-2 per chapter. Ollama is free.

**Q: Can I run this on a server?**
A: Yes, the UI supports remote access. Use `--server-name 0.0.0.0`

**Q: How do I improve annotation quality?**
A: Use a larger model (14B+ or Claude) and provide good source texts.

**Q: Can I annotate non-English texts?**
A: Yes, but quality varies by model. Claude is best for multiple languages.

**Q: Where are annotations stored?**
A: In JSON files in `annotated_output/` directory.

**Q: Can I edit annotations?**
A: Yes, just edit the JSON file in any text editor.

## Support

For issues or questions:
1. Check this guide
2. Review `READER_DOCUMENTATION.md`
3. Check console output for errors
4. Verify Ollama/API key setup

## Enjoy!

You now have a complete system for creating richly annotated, interactive readers for historical texts. Have fun exploring!
