# Henty — Audiobook Creator

Henty turns a pre-structured book into a narrated audiobook. Books are produced
upstream (by Claude Cowork / the Rewriter pipeline) as a `book.json` file; Henty
imports that, lays the text out in an aligned grid, and generates TTS takes per chunk
with [Chatterbox](https://github.com/resemble-ai/chatterbox).

Henty no longer does any text acquisition or chaptering — no Gutenberg downloading,
no EPUB/regex chapter parsing, no Markdown editor. Those steps live in Cowork now.

---

## What it does

1. **Open a book** — pick a folder under `BOOKS_DIR` (each contains a `book.json`).
   Importing builds a locked chapter/chunk project saved **in that folder**
   (`project.json` + `audio/` live alongside `book.json`).
2. **Review in the grid** — one continuous grid: chapter titles as header rows, then
   one row per chunk with the **text on the left and its takes on the right**, always
   aligned. Hover the left gutter to **merge** (↑/↓) or **split** (✂) a chunk.
3. **Generate audio** — per chunk, per chapter, or **Generate Entire Book** (skips
   chunks that already have a take). The first take becomes the best take; you can
   change the best take or delete takes.
4. **Reader pane** — a right-side reading view (hidden by default; toggle "Reader").
   Plays best takes in order with text highlighting.

### Pronunciation & emotion markup (core TTS feature)
Chunk text may contain inline annotations; both are sent through the TTS pipeline:

- **Pronunciation:** `{display|spoken}` → the reader sees *display*, TTS speaks *spoken*.
  e.g. `{Beauchamp|BEE-chum}`, `{St.|Saint}`.
- **Paralinguistic tags (Chatterbox Turbo):** `{display|[tag]}` or `{|[tag]}` — supported
  tags: `[laugh] [chuckle] [cough] [sigh] [gasp] [groan] [sniff] [clear throat] [shush]`.
  A chunk containing an emotion tag auto-selects the Turbo model.

---

## Running

GPU is **required** — the server refuses to start without CUDA.

```
start_henty_fresh.bat
```

The launcher kills any old instance, runs `precheck.py`, starts the server (logs stream
live to the window and to `server_log.txt`), and opens `http://localhost:5000/app.html`.

Set `SKIP_PRECHECK=1` in `.env` to skip the dependency check after the first good run.

### `.env` settings
```
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
REQUIRE_AUTH=False
API_KEY=...
BOOKS_DIR=C:\Users\danie\OneDrive\Documents\Claude\Projects\Rewriter\books
DEFAULT_VOICE=Haggard          # must match a file in voice_samples/ (extension-agnostic)
DEFAULT_TTS_MODEL=chatterbox   # chatterbox | chatterbox_turbo
MAX_PARALLEL_GENERATIONS=3
MAX_CHUNK_SIZE=500
SKIP_PRECHECK=0
```

> **Voice safety:** Henty never uses the Chatterbox built-in default voice. If a voice
> can't be resolved (project voice → `DEFAULT_VOICE` fallback), generation errors out.

There is also a small standalone **Gradio app** (`app.py`) for quick one-off audio
previewing, independent of the main server.

---

## Locked Environment (verified working — reproduce exactly)

This stack is fiddly to assemble; pin these versions.

**Platform:** Windows 11 + NVIDIA RTX 5070 Ti, CUDA 13.0

| Component | Version |
|-----------|---------|
| Python | **3.11.0** |
| PyTorch | **2.12.0+cu130** |
| torchaudio | **2.11.0+cu130** |
| torchvision | **0.27.0+cu130** |
| CUDA | **13.0** |
| chatterbox-tts | **0.1.7** |
| numpy | **2.2.6** (must be `<2.3` for numba/Chatterbox) |
| scipy | 1.16.3 |
| Flask | 3.0.0 |
| Flask-CORS | 6.0.1 |
| Flask-SocketIO | 5.5.1 |
| python-socketio | 5.15.0 |
| gradio | 6.8.0 (for `app.py` preview) |
| pydub | 0.25.1 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| requests | 2.31.0 |
| beautifulsoup4 | 4.14.2 |
| python-dotenv | 1.0.1 |

**Install the CUDA build of PyTorch explicitly:**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**Critical notes**
1. **numpy must be `<2.3`** — numba (a Chatterbox dependency) breaks otherwise. 2.2.6 is correct.
2. **PyTorch must be the cu130 wheel** — the default PyPI wheel is CPU-only and Henty will refuse to start.
3. **chatterbox-tts 0.1.7 works with torch 2.12.0** despite declaring `torch==2.6.0` (tested/verified).
4. For the complete transitive dependency tree, see `pip freeze` output committed to version control.

---

## Project layout

| Path | Role |
|------|------|
| `server.py` | Flask backend, `TextToAudioConverter`, all `/api/*` endpoints |
| `app.html` | Single-page UI: book picker → aligned grid + reader pane |
| `book_tab.js` | The grid: chunk rows, takes, generation queue, merge/split, inline edit |
| `reader_tab.js` | Reader pane: best-take playback with highlighting |
| `config.py` | Configuration (reads `.env`) |
| `auth.py` | API-key auth |
| `precheck.py` | Pre-flight dependency/GPU check |
| `app.py` | Standalone Gradio preview app |
| `voice_samples/` | Reference voices (the spoken voice prompt) |
| `BOOKS_DIR/<book>/` | A project: `book.json`, `project.json`, `audio/` |

### Key API endpoints
- `GET /api/books` — list importable book folders under `BOOKS_DIR`
- `POST /api/project/import-book` — `{folder, variant}` → build & open a project
- `POST /api/project/load` — load an existing project by path
- `GET /api/project/info` — current project metadata (chapters/chunks/takes)
- `POST /api/project/generate-chunk-audio` — generate one take
- `POST /api/project/chapter/generate-all` — generate a whole chapter
- `POST /api/project/generate-entire-book` — generate every chunk (skips existing)
- `POST /api/project/split-chunk` / `merge-chunk` — restructure chunks
- `POST /api/project/update-chunk-text` — edit chunk text
- `POST /api/project/set-chunk-best-take` / `delete-audio` — manage takes
