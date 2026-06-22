/**
 * Reader Tab Logic
 *
 * Renders the book as a calm, Claude-preview-style reading surface (markdown via the
 * vendored `marked`, Butterick-flavoured typography from app.html), parses & links
 * footnotes / sidenotes / endnotes baked into the book by Cowork, lets the reader toggle
 * between the Original and Enriched editions, and keeps the text in sync with the
 * published audiobook using the per-chapter `timeline` baked into project.json.
 */

class ReaderTab {
    constructor() {
        this.chapters = [];
        this.audio = null;          // single shared <audio> element
        this.isPlaying = false;
        this.mode = localStorage.getItem('henty.readerMode') || 'original';
        // Queue playback ("Play All Best Takes") — used when chapters aren't published.
        this.audioQueue = [];
        this.currentChunkIndex = 0;
        this.pauseTimer = null;
        // Timeline playback (published chapter WAV) — drives click-to-seek + scroll sync.
        this.syncChapterIndex = null;
        this.syncTimeline = null;   // array of {chunk_id, start_ms, end_ms}
    }

    // ---- markdown + sanitising ------------------------------------------------------

    /** Render markdown to a sanitised HTML string. */
    renderMarkdown(md, inline = false) {
        if (!md) return '';
        const raw = (typeof marked !== 'undefined')
            ? (inline ? marked.parseInline(md) : marked.parse(md))
            : md;
        return this.sanitize(raw);
    }

    /** Whitelist-based sanitiser: keeps the small set of tags marked emits, drops the rest. */
    sanitize(html) {
        const ALLOWED = new Set(['P','BR','H1','H2','H3','H4','EM','STRONG','I','B',
            'BLOCKQUOTE','UL','OL','LI','A','CODE','PRE','HR','SUP','SUB','SPAN','DEL']);
        const doc = new DOMParser().parseFromString(`<body>${html}</body>`, 'text/html');
        const walk = (node) => {
            [...node.childNodes].forEach(child => {
                if (child.nodeType === 1) {
                    if (!ALLOWED.has(child.tagName)) {
                        // Unwrap disallowed element: keep its (sanitised) children.
                        walk(child);
                        while (child.firstChild) node.insertBefore(child.firstChild, child);
                        node.removeChild(child);
                        return;
                    }
                    // Strip every attribute except a safe href/class/id allowlist.
                    [...child.attributes].forEach(attr => {
                        const name = attr.name.toLowerCase();
                        const ok = (name === 'class' || name === 'id')
                            || (child.tagName === 'A' && (name === 'href' || name === 'title'));
                        if (!ok) { child.removeAttribute(attr.name); return; }
                        if (name === 'href' && /^\s*javascript:/i.test(attr.value)) {
                            child.removeAttribute(attr.name);
                        }
                    });
                    walk(child);
                }
            });
        };
        walk(doc.body);
        return doc.body.innerHTML;
    }

    // ---- text + note processing -----------------------------------------------------

    /** Resolve the text for a chunk in the current mode and strip pronunciation markup. */
    chunkSource(chunk) {
        let text = (this.mode === 'enriched' && chunk.enriched_text)
            ? chunk.enriched_text : (chunk.text || '');
        // Strip pronunciation / emotion markup: {display|spoken} -> display text only.
        text = text.replace(/\{([^|}]+)\|[^}]*\}/g, '$1');
        return text;
    }

    /**
     * Replace a chunk's inline note markers with superscript anchors and return the
     * rendered note HTML to attach. Notes whose marker isn't present in this chunk's
     * text are ignored (a paragraph split across chunks duplicates `notes`).
     * Returns { text, inlineNotes (html), endNotes:[{html}] }.
     */
    applyNotes(text, chunk, chapterIndex) {
        const notes = chunk.notes || [];
        if (!notes.length) return { text, inlineNotes: '', endNotes: [] };

        let inline = '';
        const endNotes = [];
        notes.forEach((note, i) => {
            const marker = note.marker;
            if (!marker || !text.includes(marker)) return;
            const cid = chapterIndex;
            const nid = (note.id != null ? note.id : i);
            const num = (this._noteCounter = (this._noteCounter || 0) + 1);
            const refId = `ref-${cid}-${nid}`;
            const noteId = `note-${cid}-${nid}`;
            const anchor = `<sup class="noteref" id="${refId}"><a href="#${noteId}">${num}</a></sup>`;
            text = text.replace(marker, anchor);

            const type = note.type || 'footnote';
            const term = note.term ? `<span class="note-term">${this.escape(note.term)}: </span>` : '';
            const back = `<a href="#${refId}" class="noteref">↩</a>`;
            const body = this.renderMarkdown(note.body || '', true);
            const html = `<div class="note ${this.escape(type)}" id="${noteId}">`
                + `<sup class="noteref">${num}</sup> ${term}${body} ${back}</div>`;
            if (type === 'endnote') endNotes.push(html);
            else inline += html;
        });
        return { text, inlineNotes: inline, endNotes };
    }

    escape(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // ---- loading + rendering --------------------------------------------------------

    async init() { await this.loadChapters(); this.renderContent(); }

    async refresh() {
        console.log('[READER] Refreshing content...');
        await this.loadChapters();
        this.renderContent();
    }

    async loadChapters() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/info`, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (response.ok) {
                const projectInfo = await response.json();
                this.chapters = projectInfo.metadata?.chapters || projectInfo.chapters || [];
                console.log('[READER] Loaded', this.chapters.length, 'chapters');
            }
        } catch (error) {
            console.error('Error loading chapters:', error);
        }
    }

    renderContent() {
        const container = document.getElementById('readerContent');
        if (!container) return;

        if (this.chapters.length === 0) {
            container.innerHTML = '<div class="reader-doc"><p>No content available. '
                + 'Import a book and generate audio first.</p></div>';
            return;
        }

        this._noteCounter = 0;
        const docs = this.chapters.map((chapter, chapterIndex) => {
            const chunks = chapter.chunks || [];
            const endNotes = [];

            const body = chunks.map((chunk, chunkIndex) => {
                if ((chunk.type || 'text') !== 'text') return ''; // pauses aren't displayed
                const hasBestTake = chunk.generated_audios?.some(a => a.is_best_take);

                let text = this.chunkSource(chunk);
                const applied = this.applyNotes(text, chunk, chapterIndex);
                text = applied.text;
                applied.endNotes.forEach(n => endNotes.push(n));

                const isVerse = chunk.type === 'verse' || /\n/.test(text.trim());
                const inner = this.renderMarkdown(text);
                const classes = ['chunk'];
                if (hasBestTake) classes.push('has-audio');
                if (isVerse) classes.push('verse');

                return `<div class="${classes.join(' ')}"
                              id="readerChunk_${chapterIndex}_${chunkIndex}"
                              data-chapter-index="${chapterIndex}"
                              data-chunk-index="${chunkIndex}"
                              data-chapter-id="${this.escape(chapter.id)}"
                              data-chunk-id="${this.escape(chunk.id)}"
                              onclick="readerTab.onChunkClick(${chapterIndex}, ${chunkIndex})"
                       >${inner}${applied.inlineNotes}</div>`;
            }).join('');

            const title = chapter.title || chapter.name || `Chapter ${chapterIndex + 1}`;
            const endNotesHtml = endNotes.length
                ? `<div class="reader-endnotes"><h3>Notes</h3>${endNotes.join('')}</div>` : '';

            return `<section class="reader-doc">
                        <h2 class="chapter-title">${this.escape(title)}</h2>
                        ${body}
                        ${endNotesHtml}
                    </section>`;
        }).join('');

        container.innerHTML = docs;
        this.applyMode();
        this.applyFontSize();
        this.updateGutter();
    }

    updateGutter() {
        // Open the sidenote gutter only when the pane is wide enough to host it.
        const wide = (document.getElementById('readerPane')?.clientWidth || 0) >= 600;
        document.querySelectorAll('#readerContent .reader-doc').forEach(d =>
            d.classList.toggle('has-gutter', wide));
    }

    // ---- mode + font ----------------------------------------------------------------

    setMode(mode) {
        if (mode !== 'original' && mode !== 'enriched') return;
        this.mode = mode;
        localStorage.setItem('henty.readerMode', mode);
        this.renderContent();
    }

    applyMode() {
        document.querySelectorAll('#readerModeToggle button').forEach(b =>
            b.classList.toggle('active', b.dataset.mode === this.mode));
    }

    adjustFontSize(delta) {
        const span = document.getElementById('readerFontSize');
        let size = parseInt(span.textContent) || 19;
        size = Math.max(14, Math.min(30, size + delta));
        span.textContent = size;
        localStorage.setItem('henty.readerFontSize', size);
        this.applyFontSize();
    }

    applyFontSize() {
        const span = document.getElementById('readerFontSize');
        const stored = localStorage.getItem('henty.readerFontSize');
        if (stored && span) span.textContent = stored;
        const size = (parseInt(span?.textContent) || 19) + 'px';
        document.querySelectorAll('#readerContent .reader-doc').forEach(d =>
            d.style.setProperty('--reader-fs', size));
    }

    // ---- audio: timeline sync + click-to-seek ---------------------------------------

    /** A chunk in a published chapter plays that chapter's WAV and seeks to its offset. */
    onChunkClick(chapterIndex, chunkIndex) {
        const chapter = this.chapters[chapterIndex];
        const chunk = chapter?.chunks?.[chunkIndex];
        if (!chapter || !chunk) return;

        const tl = chapter.timeline;
        const entry = tl?.chunks?.find(c => String(c.chunk_id) === String(chunk.id));
        if (tl?.published_file && entry) {
            this.playChapterFrom(chapterIndex, entry.start_ms / 1000);
            return;
        }
        // No timeline yet: play just this chunk's best take, if any.
        const best = chunk.generated_audios?.find(a => a.is_best_take);
        if (best) {
            this.stopPlayback();
            this.highlightChunk(chapterIndex, chunkIndex);
            this.audio = new Audio(SERVER_URL + best.audio_url);
            this.isPlaying = true;
            this.audio.play().catch(e => console.error('Playback error:', e));
            this.audio.onended = () => this.stopPlayback();
        }
    }

    playChapterFrom(chapterIndex, seconds) {
        const chapter = this.chapters[chapterIndex];
        const tl = chapter?.timeline;
        if (!tl?.published_file) return;
        this.stopPlayback();

        this.syncChapterIndex = chapterIndex;
        this.syncTimeline = tl.chunks || [];
        this.audio = new Audio(`${SERVER_URL}/api/published/${encodeURIComponent(tl.published_file)}`);
        this.audio.currentTime = Math.max(0, seconds || 0);
        this.isPlaying = true;
        this.setPlayButton(true);

        this.audio.ontimeupdate = () => this.syncToTime(this.audio.currentTime * 1000);
        this.audio.onended = () => this.stopPlayback();
        this.audio.onerror = () => this.stopPlayback();
        this.audio.play().catch(e => { console.error('Playback error:', e); this.stopPlayback(); });
    }

    /** Highlight the chunk whose [start,end) contains the given ms within the synced chapter. */
    syncToTime(ms) {
        if (this.syncChapterIndex == null || !this.syncTimeline) return;
        const entry = this.syncTimeline.find(c => ms >= c.start_ms && ms < c.end_ms);
        if (!entry) return;
        const idx = this.chapters[this.syncChapterIndex].chunks
            .findIndex(c => String(c.id) === String(entry.chunk_id));
        if (idx >= 0) this.highlightChunk(this.syncChapterIndex, idx);
    }

    highlightChunk(chapterIndex, chunkIndex) {
        document.querySelectorAll('#readerContent .chunk.playing')
            .forEach(el => el.classList.remove('playing'));
        const el = document.querySelector(
            `#readerContent .chunk[data-chapter-index="${chapterIndex}"][data-chunk-index="${chunkIndex}"]`);
        if (el && !el.classList.contains('playing')) {
            el.classList.add('playing');
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    // ---- audio: "Play All Best Takes" (queue, cross-chapter fallback) ----------------

    async playAllBestTakes() {
        if (this.isPlaying) { this.stopPlayback(); return; }

        this.audioQueue = [];
        this.chapters.forEach((chapter, chapterIndex) => {
            (chapter.chunks || []).forEach((chunk, chunkIndex) => {
                if ((chunk.type || 'text') === 'pause') {
                    const ms = chunk.duration_ms ?? (chunk.duration ? Math.round(chunk.duration * 1000) : 500);
                    this.audioQueue.push({ chapterIndex, chunkIndex, pause: true, durationMs: ms });
                    return;
                }
                const best = chunk.generated_audios?.find(a => a.is_best_take);
                if (best) this.audioQueue.push({ chapterIndex, chunkIndex, audioUrl: best.audio_url });
            });
        });

        if (this.audioQueue.length === 0) {
            alert('No best takes available. Generate and set best takes first.');
            return;
        }

        this.isPlaying = true;
        this.currentChunkIndex = 0;
        this.setPlayButton(true);
        this.playNext();
    }

    playNext() {
        if (!this.isPlaying || this.currentChunkIndex >= this.audioQueue.length) {
            this.stopPlayback();
            return;
        }
        const item = this.audioQueue[this.currentChunkIndex];
        if (item.pause) {
            this.pauseTimer = setTimeout(() => { this.currentChunkIndex++; this.playNext(); }, item.durationMs || 500);
            return;
        }
        this.highlightChunk(item.chapterIndex, item.chunkIndex);
        this.audio = new Audio(SERVER_URL + item.audioUrl);
        const advance = () => { this.currentChunkIndex++; this.playNext(); };
        this.audio.onended = advance;
        this.audio.onerror = (e) => { console.error('Error playing audio:', e); advance(); };
        this.audio.play().catch(err => { console.error('Playback error:', err); advance(); });
    }

    stopPlayback() {
        this.isPlaying = false;
        this.syncChapterIndex = null;
        this.syncTimeline = null;
        if (this.pauseTimer) { clearTimeout(this.pauseTimer); this.pauseTimer = null; }
        if (this.audio) { this.audio.pause(); this.audio = null; }
        document.querySelectorAll('#readerContent .chunk.playing')
            .forEach(el => el.classList.remove('playing'));
        this.setPlayButton(false);
    }

    setPlayButton(playing) {
        const btn = document.querySelector('.play-btn');
        if (btn) btn.innerHTML = `<span class="material-symbols-outlined">${playing ? 'stop' : 'play_arrow'}</span>`;
    }
}

// Global instance
const readerTab = new ReaderTab();

// Keep the sidenote gutter in step with pane resizes.
window.addEventListener('resize', () => readerTab.updateGutter());

// Exports for HTML onclick handlers
window.playAllBestTakes = function () { readerTab.playAllBestTakes(); };
window.adjustReaderFontSize = function (delta) { readerTab.adjustFontSize(delta); };
window.setReaderMode = function (mode) { readerTab.setMode(mode); };
