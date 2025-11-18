# Project Gutenberg HTML Reader - Complete Guide

## Overview

This system provides a complete workflow for creating richly annotated, interactive HTML readers for Project Gutenberg texts. It uses AI to automatically identify and annotate places, people, historical topics, and archaic terminology with contextual information, sources, and links.

## System Components

### 1. Data Structure (`schemas/`)

**`annotated_text_schema.json`**
- JSON schema defining the structure for annotated documents
- Supports inline annotations, footnotes, images, and hyperlinks
- Includes metadata, content blocks, and annotation indices

**`example_annotated_document.json`**
- Example document showing the annotation format
- Demonstrates places, people, and topics with coordinates, sources, and details

### 2. Annotation Generator (`scripts/text_annotator.py`)

**Purpose**: Uses Claude AI to parse plain text and generate rich annotations

**Features**:
- Identifies places, people, historical topics, and archaic terms
- Generates latitude/longitude coordinates for locations
- Creates biographical summaries for historical figures
- Provides definitions for archaic terminology
- Fetches authoritative sources (Wikipedia, Britannica, etc.)
- Finds relevant images from Wikimedia Commons

**Usage**:
```bash
# Install required packages
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Process a text file
python scripts/text_annotator.py input.txt output.json
```

**Workflow**:
1. Splits text into paragraphs
2. For each paragraph, identifies entities using Claude
3. Enriches each entity with detailed information and sources
4. Creates annotated segments with inline references
5. Builds complete JSON document following the schema

### 3. HTML Reader (`reader.html`)

**Purpose**: Beautiful, responsive web interface for reading annotated texts

**Features**:
- **Responsive Layout**:
  - Main reading column (center)
  - Annotation sidebar (right)
  - Adapts to screen size (mobile, tablet, desktop)

- **Adjustable Typography**:
  - Font size slider (12px - 32px)
  - Keyboard shortcuts (Cmd/Ctrl +/- to resize)
  - Responsive column width (narrows with larger text)
  - Optimized line height and spacing

- **Interactive Annotations**:
  - Hover over highlighted text to view annotations
  - Color-coded by type (place, person, topic, term)
  - Sidebar displays detailed information
  - Links to authoritative sources
  - Embedded images where available

- **Visual Design**:
  - Serif font optimized for reading
  - Clean, minimal interface
  - Smooth animations and transitions
  - High contrast for accessibility

**Controls**:
- Font size slider and +/- buttons
- Show/hide annotations sidebar
- File upload for loading JSON documents

## Complete Workflow

### Step 1: Download Gutenberg Text

Use the existing `gutenberg_processor.py`:

```bash
python scripts/gutenberg_processor.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt
```

This will:
- Download the text file
- Extract the title
- Remove Gutenberg metadata
- Split into chapters
- Save to `books/[title]/` directory

### Step 2: Annotate the Text

Process one or more chapter files with the annotator:

```bash
# Single file
python scripts/text_annotator.py books/With_Clive_in_India/chapter_01.txt annotated_chapter_01.json

# Or process all chapters in a loop
for file in books/With_Clive_in_India/*.txt; do
    output=$(echo "$file" | sed 's/.txt/_annotated.json/')
    python scripts/text_annotator.py "$file" "$output"
done
```

**Note**: The annotator uses Claude AI, which:
- Requires an ANTHROPIC_API_KEY environment variable
- Makes multiple API calls per paragraph (can be rate-limited)
- Has a cost per token (see Anthropic pricing)
- Processes about 1-2 paragraphs per minute

**Configuration Options**:
- `max_paragraphs`: Limit processing for testing (default: None)
- `batch_size`: Group paragraphs to reduce API calls (default: 5)

### Step 3: View in the Reader

1. Open `reader.html` in a web browser
2. Click "Choose File" in the header
3. Select an annotated JSON file
4. Read and interact with annotations

## Annotation Types

The system identifies four types of annotations:

### 1. **Places** (Green)
- Geographic locations (cities, regions, countries)
- Includes:
  - Latitude/longitude coordinates
  - Modern name if changed
  - Brief description
  - Sources and links

**Example**:
```json
{
  "id": "place_madras",
  "type": "place",
  "summary": "Madras (now Chennai), major British colonial settlement",
  "details": {
    "coordinates": {"lat": 13.0827, "lon": 80.2707},
    "modern_equivalent": "Chennai, India"
  },
  "sources": [...]
}
```

### 2. **People** (Red)
- Historical figures and notable persons
- Includes:
  - Birth/death years
  - Key accomplishments
  - Brief biography
  - Portrait images when available
  - Sources and links

**Example**:
```json
{
  "id": "person_clive",
  "type": "person",
  "summary": "Robert Clive (1725-1774), British officer...",
  "details": {
    "born": "1725",
    "died": "1774"
  },
  "image_url": "https://...",
  "sources": [...]
}
```

### 3. **Topics** (Yellow)
- Historical events, periods, concepts
- Includes:
  - Contextual explanation
  - Time period
  - Related topics
  - Sources and links

### 4. **Terms** (Purple)
- Archaic words and phrases
- Includes:
  - Definition
  - Etymology
  - Modern equivalent
  - Usage examples
  - Sources and links

**Example**:
```json
{
  "id": "term_fortnight",
  "type": "term",
  "summary": "A period of two weeks",
  "details": {
    "etymology": "Old English 'feowertyne niht' (fourteen nights)",
    "modern_equivalent": "two weeks"
  }
}
```

## Customization

### Styling

Edit the CSS variables in `reader.html`:

```css
:root {
    --font-size-base: 18px;
    --main-column-width: 65ch;
    --sidebar-width: 350px;
    --color-bg: #fafaf8;
    --color-text: #2c2c2c;
    --color-annotation: #0066cc;
    /* ... */
}
```

### Annotation Prompts

Modify the prompts in `text_annotator.py` to change:
- Types of entities identified
- Level of detail in annotations
- Source requirements
- Summary length

### Reader Features

The JavaScript in `reader.html` can be extended to add:
- Bookmarking and progress saving
- Search functionality
- Night mode / dark theme
- Print-optimized view
- Export to PDF
- Annotation filtering

## Integration with Existing System

This reader system integrates with your existing Henty project:

1. **Gutenberg Processor** (`scripts/gutenberg_processor.py`)
   - Already downloads and processes texts
   - Outputs chapter files ready for annotation

2. **Audio Generation** (`app.py`)
   - Could be extended to generate audio for annotated texts
   - Could speak annotation details on hover

3. **Server** (`server.py`)
   - Could serve the reader as a web endpoint
   - Could provide API for on-demand annotation

## Example Integration Workflow

```python
# In server.py or a new endpoint

@app.route('/reader/<book_id>/<chapter_id>')
def serve_reader(book_id, chapter_id):
    """Serve the HTML reader for a specific chapter"""
    # Load annotated JSON
    json_path = f"books/{book_id}/{chapter_id}_annotated.json"

    # Return reader HTML with embedded data
    return render_template('reader.html', document=json_data)
```

## Performance Considerations

### Annotation Generation
- **Rate Limits**: Claude API has rate limits (contact Anthropic for details)
- **Cost**: Each paragraph requires 2-3 API calls (identification + enrichment per entity)
- **Time**: ~30-60 seconds per paragraph depending on entity count
- **Optimization**: Batch process chapters overnight for large books

### Reader Performance
- **File Size**: Annotated documents can be 5-10x larger than plain text
- **Load Time**: Large documents (100+ paragraphs) load instantly
- **Memory**: Keeps full document in memory (not an issue for typical chapters)
- **Mobile**: Fully responsive, works well on phones and tablets

## Future Enhancements

### Potential Features
1. **Map Integration**: Display places on an interactive map
2. **Timeline View**: Show chronological events from the text
3. **Related Readings**: Suggest related Gutenberg texts
4. **Reading Statistics**: Track reading time, progress
5. **Social Features**: Share annotations, discuss with others
6. **Offline Mode**: Service worker for offline reading
7. **Multi-language**: Support for non-English texts
8. **Audio Sync**: Highlight text as audio plays

### Advanced Annotations
1. **Character Tracking**: Build character relationship graphs
2. **Sentiment Analysis**: Track emotional arcs
3. **Reading Level**: Adjust complexity for different audiences
4. **Cultural Context**: Explain cultural references and norms
5. **Historical Accuracy**: Cross-reference with historical records

## Troubleshooting

### "Error loading document"
- Ensure JSON file follows the schema
- Check for valid JSON syntax
- Verify all required fields are present

### Annotations Not Appearing
- Check browser console for JavaScript errors
- Verify `annotations_index` is populated
- Ensure annotation IDs match between segments and index

### Slow Annotation Generation
- Reduce `max_paragraphs` for testing
- Check API key and rate limits
- Simplify prompts to reduce response time
- Consider caching common entities

### Font Size Not Changing
- Check browser compatibility (modern browsers required)
- Verify CSS variables are supported
- Try hard refresh (Cmd+Shift+R / Ctrl+Shift+R)

## Browser Support

**Fully Supported**:
- Chrome/Edge 88+
- Firefox 78+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

**Required Features**:
- CSS Custom Properties (variables)
- ES6 JavaScript
- Flexbox layout
- File API for document loading

## License and Credits

This reader system is part of the Henty project for enhancing Project Gutenberg texts with modern educational tools.

**Technologies Used**:
- Claude AI (Anthropic) for annotation generation
- Vanilla JavaScript (no dependencies)
- Modern CSS (CSS Grid, Flexbox, Custom Properties)
- Project Gutenberg public domain texts

**Data Sources**:
- Wikipedia (encyclopedic content)
- Wikimedia Commons (images)
- Britannica (biographical data)
- Various historical and linguistic databases

## Getting Started

**Quick Start** (5 minutes):

1. Download a Gutenberg text:
   ```bash
   python scripts/gutenberg_processor.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt
   ```

2. Annotate a chapter (requires ANTHROPIC_API_KEY):
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   python scripts/text_annotator.py books/With_Clive_in_India/chapter_01.txt test.json
   ```

3. Open `reader.html` in browser and load `test.json`

**Full Workflow** (30+ minutes):

1. Process entire book with `gutenberg_processor.py`
2. Annotate all chapters with `text_annotator.py` (run overnight for large books)
3. Set up web server to serve reader and documents
4. Integrate with audio generation for multimedia experience

## Support

For questions, issues, or feature requests:
1. Check this documentation
2. Review example files in `schemas/`
3. Inspect browser console for errors
4. Check API key and rate limits for annotation issues

Enjoy reading with enhanced context and understanding!
