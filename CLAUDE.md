# Henty - Audiobook Creator

## Project Overview
Web-based audiobook creation tool. Henty imports a **pre-structured book** (a `book.json`
produced upstream by Claude Cowork / the Rewriter pipeline), lays its chapters and chunks
out in an aligned grid, and generates TTS audio takes per chunk using Chatterbox.

Henty does **not** acquire or structure text. There is no Gutenberg downloading, no
EPUB/regex chapter parsing, and no Markdown/stage editor — that pipeline lives in Cowork.
Do not reintroduce it.

## Tech Stack
- **Backend**: Python/Flask (`server.py`) with the `TextToAudioConverter` class.
- **Frontend**: Single page (`app.html`) + two vanilla-JS modules: `book_tab.js` (the grid)
  and `reader_tab.js` (the reader pane).
- **TTS Engine**: Chatterbox (standard) and Chatterbox Turbo (paralinguistic emotion tags).
- **GPU-only**: CUDA is required; the server raises at startup if CUDA is unavailable.
- Exact pinned versions: see `README.md` → "Locked Environment".

## Architecture

### Data flow
1. A book folder under `BOOKS_DIR` contains `book.json` (a list of `blocks`: `para`,
   `verse`, `heading`).
2. `POST /api/project/import-book` reads the blocks and builds chapters/chunks:
   `heading` → new chapter (title = first chunk); `para`/`verse` → text chunks (oversized
   blocks split via `smart_chunk_text`). The variant (`original` | `rewrite`) selects which
   text field is spoken.
3. The book's own folder becomes the **project directory** — `project.json` and `audio/`
   are written alongside `book.json`. Chapters are imported locked.
4. The grid renders chapters/chunks; takes are generated and stored on each chunk under
   `generated_audios` (each take has `audio_file`, `audio_url`, `is_best_take`, settings).

### Data model (`project.json`)
```
metadata.chapters[] = { id, title, name, order, non_voiced, chunks[], timeline? }
chunk = { id, type:'text'|'pause'|'common_file', text, enriched_text?, notes?, nickname, dirty, generated_audios[] }
take  = { audio_file, audio_url, is_best_take, voice_sample, exaggeration, cfg_weight, audio_duration_seconds, ... }
timeline = { published_file, total_ms, generated_at, chunks:[{ chunk_id, type, start_ms, end_ms, duration_ms }] }
```
Chunk `id` is unique within its chapter. The grid references chunks by `chapter_id` + `chunk_id`.

### Chapter timeline (passage ↔ audio mapping)
`chapter.timeline` is baked by `publish_chapter()` when a chapter is stitched: it walks the
chunks in published order, using each best take's `audio_duration_seconds` (and pause
durations), so `start_ms`/`end_ms` line up exactly with the published WAV. The reader uses it
for click-to-seek and scroll-sync. The shape is forward-compatible — a later forced-alignment
pass can add an optional `words:[{w,start_ms,end_ms}]` per chunk entry without breaking readers.

### Notes & enrichment (authored in Cowork, rendered by Henty)
Enrichment is **authored upstream** and baked into `book.json` blocks; Henty only renders it
(it does not generate notes). `build_chapters_from_rewriter_blocks()` carries two optional
block fields onto chunks, the same way `image`/`caption` are preserved:
- `block.notes` → `chunk.notes`: `[{ id, marker, type:'footnote'|'endnote'|'sidenote'|'gloss',
  term?, body }]`. `marker` is the inline token in the chunk text (e.g. `[^1]`) where the
  superscript anchor is placed; `body` is markdown. A note is only rendered by the chunk whose
  text contains its `marker` (so a paragraph split across chunks renders each note once).
- `block.enriched` / `block.enriched_text` → `chunk.enriched_text`: an alternate markdown
  rendering shown when the reader's **Enriched** toggle is on (falls back to `text`).
Both are absent in older books and render exactly as before.

### Pronunciation & Emotion Markup — `{display|spoken}` (core TTS feature, do not remove)
Inline annotations in chunk text. Display text is shown to the reader; spoken text is sent to TTS.
- **Pronunciation:** `{display|spoken}` (e.g. `{Beauchamp|BEE-chum}`).
- **Emotion tags (Turbo):** `{display|[tag]}` / `{|[tag]}` — `[laugh] [chuckle] [cough] [sigh]
  [gasp] [groan] [sniff] [clear throat] [shush]`. A chunk with an emotion tag forces Turbo.
- Server: `process_pronunciation_markup()` resolves markup before TTS;
  `text_has_paralinguistic_tags()` detects tags and forces the Turbo model.
- Reader: `chunkSource()` strips `{display|spoken}` → display text before markdown rendering.

### Reader (markdown + typography)
`reader_tab.js` renders each chunk as markdown via the vendored `marked` (`vendor/marked.min.js`,
no CDN) and sanitises the output through a tag/attribute whitelist (`ReaderTab.sanitize`). The
reading surface in `app.html` (`.reader-doc`) follows Butterick's *Practical Typography*: a
constrained measure (~33rem), serif body, ~1.5 leading, curly punctuation, and paragraph rhythm
by first-line indent. Sidenotes float into a right-hand gutter when the pane is wide; otherwise
notes render inline as footnote blocks. An **Original ⇄ Enriched** toggle (persisted in
`localStorage`) switches `chunk.text` vs `chunk.enriched_text`.

### Voice safety (hard rule)
The Chatterbox built-in default voice must **never** be used. All generation endpoints resolve
the voice via `resolve_voice_sample_path()` (project voice → `DEFAULT_VOICE` fallback) and
return a 400 error if none resolves. Voice names are extension-agnostic.

### Filenames
Audio filenames are sanitized with `_safe_filename_part()` — strips `:` and other characters
that on Windows would create an NTFS alternate data stream (which silently produces 0-byte files).

### Key files
- `server.py` — Flask backend, all `/api/*` endpoints, `TextToAudioConverter`.
- `app.html` — book picker landing → aligned grid + collapsible reader pane.
- `book_tab.js` — grid render, per-chunk/queue generation, take management, merge/split, inline edit.
- `reader_tab.js` — reader pane (best-take playback + highlighting).
- `config.py` — config from `.env` (incl. `BOOKS_DIR`, `DEFAULT_VOICE`).
- `app.py` — standalone Gradio preview app (independent of the server).

### Key API endpoints
- `GET /api/books` — list book folders under `BOOKS_DIR`.
- `GET /api/queue`, `PUT /api/queue` — read/write the priority queue (ordered list of book
  folder names) persisted to `<BOOKS_DIR>/.henty_priority_queue.json`. The UI floats queued
  books to the top of the picker; the **Run Queue** button processes them top-to-bottom
  (import/chunk if needed → `generate-entire-book`), sequentially since generation is GPU-bound.
- `POST /api/project/import-book` — `{folder, variant}` build/open project in the book folder.
- `POST /api/project/load`, `GET /api/project/info` — load / read current project.
- `POST /api/project/generate-chunk-audio` — one take (per-chunk).
- `POST /api/project/chapter/generate-all` — all chunks in a chapter.
- `POST /api/project/chapter/publish` — `{chapter_id}` stitch the chapter's best takes (and
  pauses) into one WAV under `<project>/published/`, served at `/api/published/<file>`.
- `POST /api/project/generate-entire-book` — every chunk, skips existing takes.
- `POST /api/project/split-chunk`, `merge-chunk` — restructure chunks (operate on
  `project.json` directly, by `chapter_id` + `chunk_id`).
- `POST /api/project/update-chunk-text` — edit chunk text.
- `POST /api/project/set-chunk-best-take`, `delete-audio` — manage takes.

## Development Notes
- No build system; edit files directly.
- GPU is mandatory. To run, use `start_henty_fresh.bat` (logs stream to the window and
  `server_log.txt`). `SKIP_PRECHECK=1` in `.env` skips the dependency check.
- `smart_chunk_text` is the only remaining text-splitting helper (used by import for
  oversized blocks). Chaptering/parsing helpers were removed.
