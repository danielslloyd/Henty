# Henty - Audiobook Creator

## Project Overview
Web-based audiobook creation tool. Loads Project Gutenberg texts, splits them into chapters/chunks, generates TTS audio, and provides a reader interface with playback.

## Tech Stack
- **Backend**: Python/Flask (`server.py`) with `TextToAudioConverter` class
- **Frontend**: Single HTML file (`app.html`) with vanilla JS tab modules (`gutenberg_tab.js`, `reader_tab.js`, `tts_tab.js`)
- **Text Processing**: `scripts/gutenberg_processor.py` — downloads and parses Gutenberg plain text + EPUB files

## Architecture

### Data Flow
1. User provides a Gutenberg URL or book ID
2. Server downloads plain text + EPUB (cached as `source.epub` in project dir)
3. Text is stripped of Gutenberg metadata, carriage returns processed, saved as `raw_text.txt`
4. Chapter parsing runs (see below), producing chapter structures with chunks
5. Chapters stored in `project.json` under `metadata.chapters`
6. Markdown editor allows manual editing of chapter/chunk structure
7. TTS tab generates audio per chunk; Reader tab plays back with highlighting

### Chapter Parsing System (5 Methods)
Located in `scripts/gutenberg_processor.py` on the `GutenbergProcessor` class.
Selectable via the UI dropdown in the Markdown pane, or via `POST /api/project/reparse-chapters`.

| Method | Key | Description |
|--------|-----|-------------|
| EPUB TOC | `epub_toc` | Uses NCX/nav chapter titles from the EPUB to locate chapter boundaries in the plain text |
| EPUB Spine + HTML | `epub_spine_html` | Walks the EPUB spine order reading h1-h3 headings and body text from each content file |
| Regex Headings | `regex_headings` | Scans plain text for "Chapter N", Roman numerals, "Part N", ALL-CAPS headings, etc. |
| Blank-Line Sections | `blank_line_sections` | Splits on 3+ consecutive blank lines (common Gutenberg section separator) |
| Hybrid EPUB + Regex | `hybrid_epub_regex` | Tries EPUB TOC first; fills gaps and validates with regex heading detection |

All methods return `[(title, body_text), ...]` and accept a `log` list for verbose output.
The verbose log is displayed in a collapsible panel in the UI (color-coded by status).

### Key API Endpoints
- `GET /api/project/parsing-methods` — lists available methods with descriptions
- `POST /api/project/reparse-chapters` — re-parses with `{ "method": "<key>" }`, returns chapters + log
- `POST /api/project/add-gutenberg-url` — initial load from Gutenberg URL (caches EPUB)
- `POST /api/project/save-markdown` — saves edited markdown back to chapter structure
- `GET /api/project/get-text-files` — returns current chapters

### Key Files
- `app.html` — main UI (single page, all tabs)
- `gutenberg_tab.js` — Gutenberg text loading, markdown editing, parsing method selector
- `reader_tab.js` — book reader with audio playback
- `tts_tab.js` — TTS generation controls
- `server.py` — Flask backend, all API endpoints, `TextToAudioConverter` class
- `scripts/gutenberg_processor.py` — `GutenbergProcessor` with EPUB/text parsing

## Development Notes
- No build system; edit files directly
- The EPUB is cached as `source.epub` in the project directory for re-parsing
- Raw text is cached as `raw_text.txt` in the project directory
- The `detect_chapters()` method on `TextToAudioConverter` is the legacy fallback parser
- Parsing method verbose logs print to both server console and the UI log panel
