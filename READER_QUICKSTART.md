# HTML Reader Quick Start Guide

## 🚀 Quick Start (5 minutes)

### Option 1: Try with Sample Text

```bash
# 1. Set your API key (get one from https://console.anthropic.com/)
export ANTHROPIC_API_KEY="your-key-here"

# 2. Annotate the sample text
python scripts/text_annotator.py test_sample.txt sample_annotated.json

# 3. Open reader.html in your browser
open reader.html  # Mac
# or
start reader.html  # Windows
# or just double-click reader.html

# 4. Click "Choose File" and select sample_annotated.json
```

### Option 2: Full Gutenberg Workflow

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY="your-key-here"

# 3. Run demo workflow
python scripts/demo_workflow.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt

# 4. Open reader.html and load the generated JSON files
```

## 📖 What You Get

### **A) Rich Data Structure**
- JSON format with inline annotations
- Supports footnotes, images, hyperlinks
- Structured metadata and indexing
- See: `schemas/annotated_text_schema.json`

### **B) AI-Powered Annotations**
The system automatically identifies and annotates:

- **📍 Places**: With coordinates and descriptions
  - Example: "Madras (now Chennai), British colonial settlement"
  - Includes: lat/lon, modern name, historical context

- **👤 People**: Historical figures with bios
  - Example: "Robert Clive (1725-1774), British officer..."
  - Includes: birth/death years, accomplishments, portrait

- **📚 Topics**: Historical events and concepts
  - Example: "The British East India Company..."
  - Includes: period, significance, related events

- **📖 Archaic Terms**: Old words with definitions
  - Example: "fortnight - a period of two weeks"
  - Includes: etymology, modern equivalent, usage

All with **authoritative sources** (Wikipedia, Britannica, etc.)

### **C) Beautiful Web Reader**

**Features**:
- ✅ Responsive layout (works on phone, tablet, desktop)
- ✅ Adjustable font size (12px - 32px)
- ✅ Main reading column + annotation sidebar
- ✅ Hover over highlighted text to see details
- ✅ Color-coded annotations by type
- ✅ Links to sources and references
- ✅ Smooth animations and transitions
- ✅ Keyboard shortcuts (Cmd/Ctrl +/- for font size)

## 🎯 Key Features

### Typography & Readability
- Optimized serif font for reading
- Responsive column width (narrows with larger text)
- Perfect line height and spacing
- High contrast for accessibility

### Interactive Annotations
```
Text: "In 1745, Robert Clive sailed to Madras..."
      ↓
Hover over "Madras" → Sidebar shows:
📍 PLACE
Madras (now Chennai), major British colonial settlement on
India's eastern coast
Coordinates: 13.0827, 80.2707
Modern name: Chennai, India
📚 Wikipedia: Chennai
```

### Responsive Design
- **Desktop**: Main column + sidebar (side-by-side)
- **Tablet**: Narrower layout, collapsible sidebar
- **Mobile**: Full-width text, annotations below

## 📂 File Structure

```
Henty/
├── reader.html                    # Main HTML reader
├── schemas/
│   ├── annotated_text_schema.json # JSON schema
│   └── example_annotated_document.json
├── scripts/
│   ├── gutenberg_processor.py    # Download & process
│   ├── text_annotator.py         # AI annotation
│   └── demo_workflow.py          # Complete demo
├── test_sample.txt               # Sample text for testing
├── READER_DOCUMENTATION.md       # Full documentation
└── READER_QUICKSTART.md         # This file
```

## 🔧 Requirements

- Python 3.8+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Anthropic API key (for annotation generation)

```bash
pip install anthropic requests
```

## 💡 Usage Examples

### Annotate a Single Chapter
```bash
python scripts/text_annotator.py books/MyBook/chapter_01.txt output.json
```

### Annotate Multiple Chapters
```bash
for file in books/MyBook/*.txt; do
    output="${file%.txt}_annotated.json"
    python scripts/text_annotator.py "$file" "$output"
done
```

### Process Gutenberg URL
```bash
# Download and split into chapters
python scripts/gutenberg_processor.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt

# Annotate first chapter
python scripts/text_annotator.py books/With_Clive_in_India/chapter_01.txt annotated.json
```

## ⚙️ Customization

### Change Reader Colors
Edit CSS variables in `reader.html`:
```css
:root {
    --color-bg: #fafaf8;           /* Background */
    --color-text: #2c2c2c;         /* Text */
    --color-annotation: #0066cc;   /* Annotation links */
}
```

### Adjust Annotation Detail
Edit prompts in `text_annotator.py`:
- Change `max_tokens` for longer/shorter annotations
- Modify prompts for different types of information
- Adjust `max_paragraphs` for partial processing

### Font Preferences
```css
body {
    font-family: 'Georgia', 'Times New Roman', serif; /* Change to your preference */
}
```

## 🎨 Color Coding

Annotations are color-coded by type:
- 🟢 **Green**: Places (geography, locations)
- 🔴 **Red**: People (historical figures)
- 🟡 **Yellow**: Topics (events, concepts)
- 🟣 **Purple**: Terms (archaic words)

## ⌨️ Keyboard Shortcuts

- `Cmd/Ctrl + Plus`: Increase font size
- `Cmd/Ctrl + Minus`: Decrease font size
- `Cmd/Ctrl + 0`: Reset font size

## 🚨 Troubleshooting

### "No annotations appearing"
- Check that JSON file has `annotations_index` populated
- Verify annotation IDs match between segments and index
- Check browser console for errors

### "Error loading document"
- Ensure JSON file follows the schema
- Validate JSON syntax (use jsonlint.com)
- Check file encoding is UTF-8

### "Annotation taking too long"
- Reduce `max_paragraphs` in annotation script
- Check API rate limits
- Process chapters in smaller batches

### "API key error"
```bash
# Make sure API key is set
echo $ANTHROPIC_API_KEY

# If empty, set it:
export ANTHROPIC_API_KEY="your-key-here"
```

## 💰 Cost Estimate

Using Claude Sonnet 4.5:
- ~2-3 API calls per paragraph
- ~1000-2000 tokens per paragraph
- Typical chapter (50 paragraphs): $0.50-$2.00
- Full book (20 chapters): $10-$40

**Optimization Tips**:
- Process overnight for large books
- Use `max_paragraphs` to test first
- Cache common entities (future enhancement)

## 🌟 Next Steps

1. **Try the demo**: Run with sample text
2. **Process a book**: Use full Gutenberg workflow
3. **Customize styling**: Edit colors, fonts, layout
4. **Extend features**: Add maps, timelines, etc.
5. **Integrate audio**: Combine with existing TTS system

## 📚 Full Documentation

See `READER_DOCUMENTATION.md` for:
- Complete technical details
- Integration with existing system
- Advanced customization
- Future enhancements
- API reference

## 🎓 Example Output

Input text:
```
"In 1745, Robert Clive sailed to Madras aboard the Winchester..."
```

Output annotations:
- **1745**: Historical period context
- **Robert Clive**: Biography with portrait
- **Madras**: Location with coordinates (13.08, 80.27)
- **Winchester**: Ship details and class

Each with sources, images, and detailed context!

---

**Ready to enhance your reading experience?**

Start with `python scripts/demo_workflow.py test_sample.txt` and open `reader.html`!
