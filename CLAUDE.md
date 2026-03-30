# Henty - Audiobook Creator

## Project Overview
Web-based audiobook creation tool. Loads Project Gutenberg texts, splits them into chapters/chunks, generates TTS audio, and provides a reader interface with playback.

## Tech Stack
- **Backend**: Python/Flask (`server.py`) with `TextToAudioConverter` class
- **Frontend**: Single HTML file (`app.html`) with vanilla JS tab modules (`gutenberg_tab.js`, `reader_tab.js`, `tts_tab.js`)
- **Text Processing**: `scripts/gutenberg_processor.py` — downloads and parses Gutenberg plain text + EPUB files
- **TTS Engine**: Chatterbox TTS (standard) and Chatterbox Turbo (with paralinguistic emotion tags)

## Architecture

### Data Flow
1. User provides a Gutenberg URL or book ID
2. Server downloads plain text + EPUB (cached as `source.epub` in project dir)
3. Text is stripped of Gutenberg metadata, carriage returns processed, saved as `raw_text.txt`
4. Chapter parsing runs (see below), producing chapter structures with chunks
5. Chapters stored in `project.json` under `metadata.chapters`
6. User locks chapters (saves `chapters_original.txt`), then moves to text editing
7. Markdown editor allows manual editing of chunk text and pronunciation markup (auto-saves with undo)
8. TTS tab generates audio per chunk; Reader tab plays back with highlighting

### Chapter Locking Workflow
After chapter divisions look correct, the user clicks "Lock Chapters" to:
1. Save the original chapter-divided text to `chapters_original.txt`
2. Set `chapters_locked: true` in project metadata
3. Hide the parsing method selector (prevents accidental re-parse)
4. Focus the UI on text editing and TTS generation

The user can "Unlock" chapters at any time to re-parse with a different method.

### Chapter Parsing System (5 Methods)
Located in `scripts/gutenberg_processor.py` on the `GutenbergProcessor` class.
Selectable via the UI dropdown in the Markdown pane, or via `POST /api/project/reparse-chapters`.

| Method | Key | Description |
|--------|-----|-------------|
| EPUB TOC | `epub_toc` | Uses NCX/nav chapter titles from the EPUB to locate chapter boundaries in the plain text |
| EPUB Spine + HTML | `epub_spine_html` | Walks the EPUB spine order reading h1-h3 headings from each content file, then splits plain text at matching positions |
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
- `POST /api/project/lock-chapters` — locks chapter divisions, saves original text
- `POST /api/project/unlock-chapters` — unlocks to allow re-parsing

### Key Files
- `app.html` — main UI (single page, all tabs)
- `gutenberg_tab.js` — Gutenberg text loading, markdown editing, parsing method selector, auto-save, undo
- `reader_tab.js` — book reader with audio playback
- `tts_tab.js` — TTS generation controls, model selection (standard/turbo)
- `server.py` — Flask backend, all API endpoints, `TextToAudioConverter` class
- `scripts/gutenberg_processor.py` — `GutenbergProcessor` with EPUB/text parsing
- `config.py` — Server configuration (reads from `.env`)

### Pronunciation & Emotion Markup — `{display|spoken}`
Inline annotations for TTS pronunciation overrides and emotion tags. The display text is shown to the reader; the spoken text is sent to the TTS engine.

**Pronunciation Syntax:** `{display_text|spoken_text}`

**Examples:**
```
{Beauchamp|BEE-chum}
{St.|Saint}
{Leicestershire|Lester-sher}
```

**Emotion Tag Syntax (Chatterbox Turbo):** `{display_text|[tag]}` or `{|[tag]}`

When an emotion tag is present, the system automatically uses Chatterbox Turbo for that chunk.

**Supported Tags:** `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[clear throat]`, `[shush]`

**Examples:**
```
{ha ha|[laugh]} That's funny!
He cleared his throat {|[clear throat]} and began speaking.
```

**Pipeline:**
- **Editor** (`gutenberg_tab.js`): In markup view, `{display|spoken}` highlighted live with purple background as user types. Clean View toggle strips annotations. Auto-saves with debounce (1.5s), undo stack (5 levels).
- **Server** (`server.py`): `process_pronunciation_markup()` replaces `{display|spoken}` → spoken text before TTS. `text_has_emotion_tags()` detects `[tag]` patterns and forces Turbo model.
- **Reader** (`reader_tab.js`): `processChunkText()` strips `{display|spoken}` → display text for reader view.
- **TTS tab** (`tts_tab.js`): Chunk previews strip markup. Model selector (Standard/Turbo) at project and take level. Auto-detects emotion tags → forces Turbo.
- **Storage**: Full markup stored in `project.json` chunk text. Clean view is purely client-side rendering.
- **Important**: Pronunciation edits must NOT span across `</chunk>` boundaries.

### TTS Models
- **Chatterbox** (Standard): 0.5B params, `exaggeration` + `cfg_weight` controls. Best for narration.
- **Chatterbox Turbo**: 350M params, faster, supports paralinguistic emotion tags (`[laugh]`, etc.). Set via project defaults, per-chunk override, or auto-detected from emotion tags.
- Default model configurable in `.env` via `DEFAULT_TTS_MODEL=chatterbox` or `chatterbox_turbo`.

## Development Notes
- No build system; edit files directly
- The EPUB is cached as `source.epub` in the project directory for re-parsing
- Raw text is cached as `raw_text.txt` in the project directory
- Locked chapter text is cached as `chapters_original.txt`
- The `detect_chapters()` method on `TextToAudioConverter` is the legacy fallback parser
- Parsing method verbose logs print to both server console and the UI log panel
- Editor auto-saves after 1.5s of inactivity; undo reverts last 5 saves
- Dirty indicators in TTS tab update instantly when editor auto-saves (no page reload)
