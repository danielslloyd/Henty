/**
 * Book Tab — aligned grid with the classic take layout.
 *
 * One grid: [gutter | chunk text | takes]. Chapter titles are full-width header rows;
 * each chunk is one grid row, so the text on the left always lines up with its takes on
 * the right. The takes column reuses the original .chunk-take-row look (best-take ring,
 * settings, play, delete). The text column is editable and has a toolbar to insert
 * pronunciation `{display|spoken}` and paralinguistic `[tag]` markup.
 *
 * GPU-only server-side; the voice sample is always resolved (the server refuses the
 * Chatterbox default voice).
 */
const EMOTION_TAGS = ['laugh', 'chuckle', 'cough', 'sigh', 'gasp', 'groan', 'sniff', 'clear throat', 'shush'];
const PARA_TAG_RE = /\{[^}]*\|\s*\[(?:laugh|chuckle|cough|sigh|gasp|groan|sniff|clear throat|shush)\]\s*\}/;

class BookTab {
    constructor() {
        this.chapters = [];
        this.voiceSamples = [];
        this.projectDefaults = { exaggeration: 0.6, cfg_weight: 0.4, voice_sample: '', temperature: 0.8, tts_model: 'chatterbox' };
        this.activeGenerations = {};
        this.generationQueue = [];
        this.maxParallelGenerations = 3;
        this.currentAudio = null;
        this.currentPlayBtn = null;
    }

    async init() {
        await this.loadVoiceSamples();
        await this.loadDefaultVoice();
        await this.loadProjectDefaults();
        await this.loadDeviceStatus();
        await this.refresh();
    }

    // ---- Loading -----------------------------------------------------------
    async loadVoiceSamples() {
        try { const r = await fetch(`${SERVER_URL}/api/voice-samples`, { headers: { 'X-API-Key': API_KEY } });
            if (r.ok) this.voiceSamples = (await r.json()).samples || []; }
        catch (e) { console.error('voice samples:', e); this.voiceSamples = []; }
    }
    async loadDeviceStatus() {
        try { const r = await fetch(`${SERVER_URL}/api/device-status`, { headers: { 'X-API-Key': API_KEY } });
            if (r.ok) this.maxParallelGenerations = (await r.json()).max_parallel_generations || 3; } catch (e) {}
    }
    async loadDefaultVoice() {
        try {
            const r = await fetch(`${SERVER_URL}/api/config`); if (!r.ok) return;
            const cfg = await r.json();
            if (cfg.default_tts_model) this.projectDefaults.tts_model = cfg.default_tts_model;
            const dv = cfg.default_voice || '';
            const m = this.voiceSamples.find(v => v.name.replace(/\.[^/.]+$/, '') === dv || v.name === dv);
            if (m) this.projectDefaults.voice_sample = m.name;
            else if (dv) showToast(`Configured voice "${dv}" not found in voice_samples`, 'error');
        } catch (e) { console.error('default voice:', e); }
    }
    async loadProjectDefaults() {
        try {
            const r = await fetch(`${SERVER_URL}/api/project/info`, { headers: { 'X-API-Key': API_KEY } });
            if (!r.ok) return;
            const saved = (await r.json()).metadata?.default_audio_settings;
            if (saved) {
                const voice = saved.voice_sample && saved.voice_sample !== 'none' ? saved.voice_sample : this.projectDefaults.voice_sample;
                this.projectDefaults = { ...this.projectDefaults, ...saved, voice_sample: voice };
            }
        } catch (e) { console.error('project defaults:', e); }
    }
    async refresh() {
        const scroller = document.getElementById('gridScroll');
        const top = scroller ? scroller.scrollTop : 0;
        try {
            const r = await fetch(`${SERVER_URL}/api/project/info`, { headers: { 'X-API-Key': API_KEY } });
            if (r.ok) this.chapters = (await r.json()).metadata?.chapters || [];
        } catch (e) { console.error('refresh:', e); }
        this.render();
        if (scroller && top) scroller.scrollTop = top;
        if (typeof readerTab !== 'undefined' && document.body.classList.contains('reader-open')) readerTab.refresh();
    }

    // ---- Rendering ---------------------------------------------------------
    render() {
        const grid = document.getElementById('bookGrid');
        if (!grid) return;
        if (!this.chapters.length) { grid.innerHTML = '<div style="color:#94a3b8;text-align:center;padding:60px;">No chapters.</div>'; return; }
        this._pronOptions = this._buildPronOptions();
        const rows = [];
        this.chapters.forEach(ch => {
            const chId = this._esc(ch.id);
            rows.push(`<div class="chapter-row">
                <span class="chapter-title">${this._h(ch.title || ch.name || 'Untitled')}</span>
                <span class="chapter-actions">
                    <button class="chapter-gen-btn" id="play-${chId}" onclick="bookTab.playChapter('${chId}')" title="Play all best takes in this chapter">▶ Play chapter</button>
                    <button class="chapter-gen-btn" onclick="bookTab.generateChapter('${chId}')">Generate chapter</button>
                    <button class="chapter-gen-btn" id="publish-${chId}" onclick="bookTab.publishChapter('${chId}')" title="Compile all best takes into one audio file and publish">Publish chapter</button>
                </span>
            </div>`);
            (ch.chunks || []).forEach(c => rows.push(this._row(ch, c)));
        });
        grid.innerHTML = rows.join('');
    }

    _row(chapter, chunk) {
        const chId = this._esc(chapter.id);
        const cid = chunk.id;
        const type = chunk.type || 'text';
        const key = `${chId}:${cid}`;
        const adder = `<button class="pause-adder" title="Insert a pause after this chunk" onclick="bookTab.insertPause('${chId}',${cid})"><span class="material-symbols-outlined">more_horiz</span> pause</button>`;

        if (type === 'pause') {
            const ms = chunk.duration_ms ?? (chunk.duration ? Math.round(chunk.duration * 1000) : 500);
            return `<div class="chunk-row pause-chunk" data-key="${key}">
                <div class="row-gutter"></div>
                <div class="pause-cell">
                    <span class="pause-pill">
                        <span class="material-symbols-outlined" style="font-size:16px;">pause</span>
                        <input type="number" class="pause-ms" min="0" step="100" value="${ms}"
                               onchange="bookTab.setPauseDuration('${chId}',${cid},this.value)"> ms pause
                        <button class="pause-del" title="Delete pause" onclick="bookTab.deletePause('${chId}',${cid})"><span class="material-symbols-outlined" style="font-size:16px;">close</span></button>
                    </span>
                </div>
                ${adder}
            </div>`;
        }

        const isText = type === 'text';
        const generating = !!this.activeGenerations[key];

        const gutter = `<div class="row-gutter">
            <button class="gutter-btn" title="Merge with previous" onmousedown="event.preventDefault()" onclick="bookTab.mergeChunk('${chId}',${cid},'prev')">⤒</button>
            <button class="gutter-btn" title="Split at cursor" onmousedown="event.preventDefault()" onclick="bookTab.splitChunkAtCursor('${chId}',${cid})">✂</button>
            <button class="gutter-btn" title="Merge with next" onmousedown="event.preventDefault()" onclick="bookTab.mergeChunk('${chId}',${cid},'next')">⤓</button>
            <button class="gutter-btn gutter-del" title="Delete this chunk" onmousedown="event.preventDefault()" onclick="bookTab.deleteChunk('${chId}',${cid})">🗑</button>
        </div>`;

        const textCell = isText ? `<div class="text-cell">
            <div class="chunk-edit${chunk.dirty ? ' dirty' : ''}" contenteditable="true" spellcheck="false"
                 data-chapter="${chId}" data-chunk="${cid}"
                 onkeyup="bookTab.updateTagButton(this)" onclick="bookTab.updateTagButton(this)" onfocus="bookTab.updateTagButton(this)"
                 onblur="bookTab.saveChunkText('${chId}',${cid},this)">${this._h(chunk.text || '')}</div>
            <div class="insert-toolbar">
                <button class="insert-btn" onmousedown="event.preventDefault()" onclick="bookTab.insertPron('${chId}',${cid})" title="Wrap selection as {display|spoken}">{ ᵃ | ᵇ } Pronounce</button>
                <button class="insert-btn" onmousedown="event.preventDefault()" onclick="bookTab.properCaseSelection('${chId}',${cid})" title="Keep the selection as shown but speak it in proper case (fixes ALL-CAPS TTS)">Aa Proper case</button>
                <select onmousedown="event.stopPropagation()" onchange="if(this.value){bookTab.insertTag('${chId}',${cid},this.value);this.value='';}" title="Insert paralinguistic tag">
                    <option value="">+ tag…</option>
                    ${EMOTION_TAGS.map(t => `<option value="${t}">[${t}]</option>`).join('')}
                </select>
                <select class="pron-select" onmousedown="event.stopPropagation()" onchange="if(this.value){bookTab.applyPron('${chId}',${cid},this.value);this.value='';}" title="Apply an existing pronunciation">
                    <option value="">📖 pronunciations…</option>
                    ${this._pronOptions || ''}
                </select>
                <button class="insert-btn replace-pron" style="display:none" onmousedown="event.preventDefault()" onclick="bookTab.applyPronToAll('${chId}',${cid})" title="Find &amp; replace every other plain occurrence of this word across the book">⮂ replace all</button>
                <button class="insert-btn remove-tag" style="display:none" onmousedown="event.preventDefault()" onclick="bookTab.removeTagAtCursor('${chId}',${cid})" title="Remove the tag the cursor is in">✕ tag</button>
                <button class="insert-btn" onmousedown="event.preventDefault()" onclick="bookTab.showOriginal('${chId}',${cid})" title="Show the original un-edited text">Original</button>
            </div></div>`
            : `<div class="text-cell non-text"><div class="chunk-edit">[${this._h(chunk.type)}]</div></div>`;

        const takesCell = isText
            ? `<div class="takes-cell" id="takes-${chId}-${cid}">${this._takes(chapter, chunk, generating)}</div>`
            : `<div class="takes-cell"></div>`;

        return `<div class="chunk-row${generating ? ' generating' : ''}" data-key="${key}">${gutter}${textCell}${takesCell}${adder}</div>`;
    }

    _takes(chapter, chunk, generating) {
        const chId = this._esc(chapter.id);
        const cid = chunk.id;
        const takes = [...(chunk.generated_audios || [])].sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
        const settingsId = `cfg-${chId}-${cid}`;

        const actions = `<div class="takes-actions">
            <button class="chunk-icon" title="Tune settings" onclick="bookTab.toggle('${settingsId}')"><span class="material-symbols-outlined">tune</span></button>
            <button class="chunk-icon" title="Generate take" onclick="bookTab.generateChunk('${chId}',${cid})"><span class="material-symbols-outlined">text_to_speech</span></button>
        </div>`;
        const placeholder = generating ? this._generatingPlaceholder() : '';
        const emptyLabel = (takes.length || generating) ? '' :
            '<div class="takes-empty-label" style="padding:6px 8px;">No takes yet</div>';

        const settingsPanel = `<div class="take-settings" id="${settingsId}">
            <div class="setting-row"><span class="setting-label">Voice</span><div class="setting-control">
                <select id="voice-${chId}-${cid}">${this.voiceSamples.map(s => `<option value="${s.name}" ${this.projectDefaults.voice_sample === s.name ? 'selected' : ''}>${s.name}</option>`).join('')}</select></div></div>
            <div class="setting-row"><span class="setting-label">Exaggeration</span><div class="setting-control">
                <input type="range" id="exag-${chId}-${cid}" min="0" max="1" step="0.1" value="${this.projectDefaults.exaggeration}"
                       oninput="document.getElementById('exagv-${chId}-${cid}').textContent=this.value">
                <span class="setting-value" id="exagv-${chId}-${cid}">${this.projectDefaults.exaggeration}</span></div></div>
            <div class="setting-row"><span class="setting-label">CFG Weight</span><div class="setting-control">
                <input type="range" id="cfg-w-${chId}-${cid}" min="0" max="1" step="0.1" value="${this.projectDefaults.cfg_weight}"
                       oninput="document.getElementById('cfgv-${chId}-${cid}').textContent=this.value">
                <span class="setting-value" id="cfgv-${chId}-${cid}">${this.projectDefaults.cfg_weight}</span></div></div>
            <div class="setting-row"><span class="setting-label">TTS Model</span><div class="setting-control">
                <select id="model-${chId}-${cid}">
                    <option value="chatterbox" ${this.projectDefaults.tts_model === 'chatterbox' ? 'selected' : ''}>Chatterbox</option>
                    <option value="chatterbox_turbo" ${this.projectDefaults.tts_model === 'chatterbox_turbo' ? 'selected' : ''}>Turbo</option>
                </select></div></div>
        </div>`;

        const takeRows = takes.map((t, i) => this._takeRow(chapter, chunk, t, i)).join('');
        return `<div class="takes-inner">${actions}<div class="takes-list">${settingsPanel}${placeholder}${emptyLabel}${takeRows}</div></div>`;
    }

    // Full-size placeholder card with an indeterminate progress bar, shown while a take generates.
    _generatingPlaceholder() {
        return `<div class="chunk-take-row generating take-placeholder">
            <div class="chunk-take-header">
                <span class="take-label generating-text">Generating take…</span>
            </div>
            <div class="take-progress"><div class="take-progress-bar"></div></div>
        </div>`;
    }

    _takeRow(chapter, chunk, take, i) {
        const chId = this._esc(chapter.id);
        const cid = chunk.id;
        const file = this._esc(take.audio_file || '');
        const url = SERVER_URL + (take.audio_url || `/api/audio/${take.audio_file}`);
        const isBest = !!take.is_best_take;
        const outdated = take.input_text && take.input_text !== chunk.text;
        const sid = `tk-${chId}-${cid}-${i}`;
        const model = take.tts_model === 'chatterbox_turbo' ? 'Turbo' : 'Standard';
        return `<div class="chunk-take-row ${isBest ? 'best-take' : 'non-best-take'}${outdated ? ' outdated-take' : ''}">
            <div class="chunk-take-header">
                <span class="take-label">Take ${i + 1}</span>
                <div class="take-icons">
                    ${outdated ? '<span class="material-symbols-outlined outdated-icon" title="Text changed since this take">edit_off</span>' : ''}
                    <button class="take-icon check-circle ${isBest ? 'best' : ''}" title="${isBest ? 'Best take' : 'Set as best take'}"
                            onclick="bookTab.setBestTake('${chId}',${cid},'${file}')"><span class="material-symbols-outlined">${isBest ? 'check_circle' : 'radio_button_unchecked'}</span></button>
                    <button class="take-icon settings" title="Take settings" onclick="bookTab.toggle('${sid}')"><span class="material-symbols-outlined">settings</span></button>
                    <button class="take-icon transcript" title="Show exact text sent to the model" onclick="bookTab.showTakeInput('${chId}',${cid},'${file}')"><span class="material-symbols-outlined">speaker_notes</span></button>
                    <button class="take-icon play" title="Play" onclick="bookTab.playTake('${url}', this)"><span class="material-symbols-outlined">play_arrow</span></button>
                    <button class="take-icon delete" title="Delete" onclick="bookTab.askDeleteTake('${chId}',${cid},'${file}',this)"><span class="material-symbols-outlined">delete</span></button>
                </div>
            </div>
            <div class="take-settings" id="${sid}">
                <div class="setting-row"><span class="setting-label">Voice</span><span class="setting-value">${this._h(take.voice_sample || '—')}</span></div>
                <div class="setting-row"><span class="setting-label">Model</span><span class="setting-value">${model}${take.tts_model_emotion_forced ? ' <span class="model-forced-badge">(auto)</span>' : ''}</span></div>
                <div class="setting-row"><span class="setting-label">Exaggeration</span><span class="setting-value">${take.exaggeration ?? '—'}</span></div>
                <div class="setting-row"><span class="setting-label">CFG Weight</span><span class="setting-value">${take.cfg_weight ?? '—'}</span></div>
                ${take.audio_duration_seconds ? `<div class="setting-row"><span class="setting-label">Duration</span><span class="setting-value">${take.audio_duration_seconds}s</span></div>` : ''}
                <div class="setting-row"><span class="setting-label">File</span><span class="setting-value" style="word-break:break-all;">${this._h(take.audio_file || '—')}</span></div>
                ${take.possibly_truncated ? '<div class="setting-row warning-row"><span class="warning-text">May be truncated (40s TTS limit)</span></div>' : ''}
            </div>
        </div>`;
    }

    toggle(id) { const el = document.getElementById(id); if (el) el.classList.toggle('expanded'); }

    // ---- Generation --------------------------------------------------------
    _genSettings(chId, cid) {
        const val = (pfx) => document.getElementById(`${pfx}-${chId}-${cid}`)?.value;
        const chunk = this.chapters.find(c => String(c.id) === String(chId))?.chunks.find(c => String(c.id) === String(cid));
        let model = val('model') || this.projectDefaults.tts_model || 'chatterbox';
        if (chunk && PARA_TAG_RE.test(chunk.text)) model = 'chatterbox_turbo';   // emotion tags force Turbo
        return {
            voice: val('voice') || this.projectDefaults.voice_sample,
            exaggeration: parseFloat(val('exag') ?? this.projectDefaults.exaggeration),
            cfg_weight: parseFloat(val('cfg-w') ?? this.projectDefaults.cfg_weight),
            tts_model: model
        };
    }

    async generateChunk(chapterId, chunkId) {
        const s = this._genSettings(chapterId, chunkId);
        if (!s.voice) { showToast('No voice set. Choose a voice before generating.', 'error'); return; }
        const key = `${chapterId}:${chunkId}`;
        if (this.activeGenerations[key]) return;
        if (Object.keys(this.activeGenerations).length >= this.maxParallelGenerations) { this.generationQueue.push({ chapterId, chunkId }); return; }

        const chunk = this.chapters.find(c => String(c.id) === String(chapterId))?.chunks.find(c => String(c.id) === String(chunkId));
        if (!chunk) return;
        this.activeGenerations[key] = true; this._markGenerating(key, true);
        let result = null;
        try {
            const r = await fetch(`${SERVER_URL}/api/project/generate-chunk-audio`, {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ text_file_id: chapterId, chunk_id: chunkId, chunk_text: chunk.text,
                    voice_sample: s.voice, exaggeration: s.exaggeration, cfg_weight: s.cfg_weight,
                    temperature: this.projectDefaults.temperature, tts_model: s.tts_model })
            });
            const data = await r.json();
            if (!r.ok) throw new Error(data.error || `Server error ${r.status}`);
            result = data;
        } catch (e) { console.error('generate:', e); showToast('Generation failed: ' + e.message, 'error'); }
        finally {
            delete this.activeGenerations[key];
            // Patch local state from the server's returned chunk and re-render only this
            // cell — full refresh per chunk is far too heavy during a whole-book run.
            if (result && result.chunk) {
                chunk.generated_audios = result.chunk.generated_audios || [];
                chunk.dirty = result.chunk.dirty;
            }
            this._renderTakesCell(chapterId, chunkId);
            if (this._bookGen) this._tickBookProgress();
            else if (typeof readerTab !== 'undefined' && document.body.classList.contains('reader-open')) readerTab.refresh();
            this._drain();
        }
    }

    _drain() {
        while (this.generationQueue.length && Object.keys(this.activeGenerations).length < this.maxParallelGenerations) {
            const { chapterId, chunkId } = this.generationQueue.shift();
            this.generateChunk(chapterId, chunkId);
        }
    }

    generateChapter(chapterId) {
        const ch = this.chapters.find(c => String(c.id) === String(chapterId));
        if (ch) (ch.chunks || []).filter(c => (c.type || 'text') === 'text').forEach(c => this.generateChunk(chapterId, c.id));
    }

    // Drive whole-book generation from the client, one chunk at a time through the same
    // per-chunk generator the buttons use. Each take appears live as it finishes and
    // in-progress chunks show the placeholder bar; the parallel queue throttles to the
    // GPU's capacity. (The headless Run Queue still uses the server-side endpoint.)
    async generateEntireBook() {
        if (!this.projectDefaults.voice_sample) { showToast('No project voice is set.', 'error'); return; }
        if (this._bookGen) { showToast('Already generating the book.', 'info'); return; }
        const work = [];
        for (const ch of this.chapters) {
            if (ch.non_voiced) continue;
            for (const c of (ch.chunks || [])) {
                if ((c.type || 'text') !== 'text') continue;
                if (!(c.generated_audios || []).length || c.dirty) work.push({ chId: ch.id, cid: c.id });
            }
        }
        if (!work.length) { showToast('Every chunk already has a take. Nothing to generate.', 'info'); return; }
        if (!confirm(`Generate audio for ${work.length} chunk${work.length > 1 ? 's' : ''}?\n\nChunks that already have a take are skipped. Takes appear live as they finish; this can take a long time.`)) return;

        this._bookGen = { total: work.length, done: 0 };
        this._setBookGenUI();
        for (const w of work) this.generateChunk(w.chId, w.cid);
    }

    _tickBookProgress() {
        const g = this._bookGen;
        if (!g) return;
        g.done++;
        this._setBookGenUI();
        if (g.done >= g.total && !Object.keys(this.activeGenerations).length && !this.generationQueue.length) {
            this._bookGen = null;
            this._setBookGenUI();
            showToast('Entire book generated.', 'success');
            if (typeof readerTab !== 'undefined' && document.body.classList.contains('reader-open')) readerTab.refresh();
        }
    }

    _setBookGenUI() {
        const btn = document.getElementById('genBookBtn');
        if (!btn) return;
        const icon = '<span class="material-symbols-outlined">graphic_eq</span>';
        if (this._bookGen) { btn.disabled = true; btn.innerHTML = `${icon} Generating… ${this._bookGen.done}/${this._bookGen.total}`; }
        else { btn.disabled = false; btn.innerHTML = `${icon} Generate Entire Book`; }
    }

    // Compile every best take in a chapter into one audio file and publish it.
    async publishChapter(chapterId) {
        const btn = document.getElementById(`publish-${this._esc(chapterId)}`) || document.getElementById(`publish-${chapterId}`);
        if (btn) { btn.disabled = true; btn.textContent = 'Publishing…'; }
        try {
            const d = await this._post('/api/project/chapter/publish', { chapter_id: chapterId }, 'Publish chapter');
            if (d) {
                const url = SERVER_URL + d.url;
                const kb = Math.round((d.bytes || 0) / 1024);
                const warn = d.missing_chunk_ids && d.missing_chunk_ids.length
                    ? ` (${d.missing_chunk_ids.length} chunk${d.missing_chunk_ids.length > 1 ? 's' : ''} had no take and were skipped)` : '';
                showToast(`Published ${d.file} — ${kb} KB${warn}`, warn ? 'warning' : 'success');
                window.open(url, '_blank');
            }
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = 'Publish chapter'; }
        }
    }

    // Delete every take except each chunk's best take, across the whole book.
    async deleteNonBestTakes() {
        let extra = 0;
        for (const ch of this.chapters)
            for (const c of (ch.chunks || []))
                extra += Math.max(0, (c.generated_audios || []).length - 1);
        if (!extra) { showToast('No non-best takes to delete.', 'info'); return; }
        if (!confirm(`Delete ${extra} non-best take${extra > 1 ? 's' : ''} across the book?\n\nThe best take of each chunk is kept. The audio files are permanently removed (undoable in the session).`)) return;
        const d = await this._post('/api/project/delete-non-best-takes', {}, 'Delete non-best takes');
        if (d) {
            showToast(`Removed ${d.removed_takes} take${d.removed_takes === 1 ? '' : 's'} (${d.removed_files} files deleted)`, 'success');
            await this.refresh();
        }
    }

    // ---- Takes -------------------------------------------------------------
    async setBestTake(chapterId, chunkId, audioFile) {
        await this._post('/api/project/set-chunk-best-take', { text_file_id: chapterId, chunk_id: chunkId, audio_filename: audioFile }, 'Set best take');
        await this.refresh();
    }
    // First click swaps the take's buttons for an inline confirm; second click deletes.
    askDeleteTake(chapterId, chunkId, audioFile, btn) {
        const icons = btn.closest('.take-icons');
        if (!icons) return;
        const original = icons.innerHTML;
        icons.innerHTML = `<button class="take-icon delete" style="width:auto;gap:4px;padding:5px 10px;" title="Click again to delete"
            onclick="bookTab.deleteTake('${chapterId}',${chunkId},'${audioFile}')"><span class="material-symbols-outlined">delete</span>Confirm delete</button>`;
        const cancel = (e) => {
            if (!icons.contains(e.target)) { icons.innerHTML = original; document.removeEventListener('mousedown', cancel, true); }
        };
        setTimeout(() => document.addEventListener('mousedown', cancel, true), 0);
    }

    async deleteTake(chapterId, chunkId, audioFile) {
        this._busyRow(chapterId, chunkId, true);
        await this._post('/api/project/delete-audio', { text_file_id: chapterId, chunk_id: chunkId, audio_file: audioFile, audio_filename: audioFile }, 'Delete take');
        await this.refresh();
    }
    playTake(url, btn) {
        const icon = btn.querySelector('.material-symbols-outlined');
        if (this.currentAudio && this.currentPlayBtn === btn && !this.currentAudio.paused) {
            this.currentAudio.pause(); icon.textContent = 'play_arrow'; return;
        }
        if (this.currentAudio) { this.currentAudio.pause(); if (this.currentPlayBtn) this.currentPlayBtn.querySelector('.material-symbols-outlined').textContent = 'play_arrow'; }
        const audio = new Audio(url); this.currentAudio = audio; this.currentPlayBtn = btn; icon.textContent = 'pause';
        audio.play().catch(err => { console.error('playback:', err); showToast('Playback failed', 'error'); });
        const reset = () => { icon.textContent = 'play_arrow'; if (this.currentAudio === audio) this.currentAudio = null; };
        audio.addEventListener('ended', reset); audio.addEventListener('error', reset);
    }

    // ---- Chapter preview playback -----------------------------------------
    playChapter(chapterId) {
        if (this._chapterPlay && this._chapterPlay.chapterId === chapterId) { this.stopChapterPlay(); return; }
        this.stopChapterPlay();
        const ch = this.chapters.find(c => String(c.id) === String(chapterId));
        if (!ch) return;
        const queue = [];
        (ch.chunks || []).forEach(c => {
            if ((c.type || 'text') === 'pause') {
                queue.push({ pause: true, ms: c.duration_ms ?? (c.duration ? Math.round(c.duration * 1000) : 500) });
            } else {
                const best = (c.generated_audios || []).find(a => a.is_best_take)
                    || (c.generated_audios || []).slice(-1)[0];
                if (best) queue.push({ url: SERVER_URL + (best.audio_url || `/api/audio/${best.audio_file}`) });
            }
        });
        if (!queue.length) { showToast('No takes to play in this chapter', 'warning'); return; }
        this._chapterPlay = { chapterId, queue, idx: 0, audio: null, timer: null };
        this._setPlayBtn(chapterId, true);
        this._chapterNext();
    }

    _chapterNext() {
        const p = this._chapterPlay; if (!p) return;
        if (p.idx >= p.queue.length) { this.stopChapterPlay(); return; }
        const item = p.queue[p.idx++];
        if (item.pause) { p.timer = setTimeout(() => this._chapterNext(), item.ms || 500); return; }
        const audio = new Audio(item.url); p.audio = audio;
        const next = () => this._chapterNext();
        audio.addEventListener('ended', next);
        audio.addEventListener('error', next);
        audio.play().catch(err => { console.error('chapter playback:', err); this._chapterNext(); });
    }

    stopChapterPlay() {
        const p = this._chapterPlay; if (!p) return;
        if (p.timer) clearTimeout(p.timer);
        if (p.audio) { p.audio.pause(); p.audio = null; }
        this._setPlayBtn(p.chapterId, false);
        this._chapterPlay = null;
    }

    _setPlayBtn(chapterId, playing) {
        const btn = document.getElementById(`play-${this._esc(chapterId)}`) || document.getElementById(`play-${chapterId}`);
        if (btn) btn.textContent = playing ? '◼ Stop' : '▶ Play chapter';
    }

    // ---- Editing / structure ----------------------------------------------
    async saveChunkText(chapterId, chunkId, el) {
        const newText = el.innerText.replace(/ /g, ' ').replace(/\s+$/,'');
        const chunk = this.chapters.find(c => String(c.id) === String(chapterId))?.chunks.find(c => String(c.id) === String(chunkId));
        if (!chunk || newText === (chunk.text || '')) return;
        const d = await this._post('/api/project/update-chunk-text', {
            text_file_id: chapterId, chunk_id: chunkId, new_text: newText,
            new_nickname: newText.slice(0, 50) + (newText.length > 50 ? '...' : '')
        }, 'Save text');
        // On rejection (e.g. too long) keep the user's text in the box so they can fix it.
        if (d) await this.refresh();
    }

    _caretOffset(el) {
        const sel = window.getSelection();
        if (!sel || !sel.rangeCount || !el.contains(sel.anchorNode)) return null;
        const r = sel.getRangeAt(0).cloneRange();
        r.selectNodeContents(el); r.setEnd(sel.anchorNode, sel.anchorOffset);
        return r.toString().length;
    }

    // Returns the {…} markup span enclosing `offset`, or null.
    _tagAt(text, offset) {
        if (offset == null) return null;
        for (const m of text.matchAll(/\{[^{}]*\}/g)) {
            const start = m.index, end = start + m[0].length;
            if (offset >= start && offset <= end) return { start, end, body: m[0] };
        }
        return null;
    }

    updateTagButton(el) {
        const cell = el.closest('.text-cell');
        if (!cell) return;
        const tag = this._tagAt(el.firstChild ? el.firstChild.textContent : '', this._caretOffset(el));
        const rm = cell.querySelector('.remove-tag');
        if (rm) rm.style.display = tag ? 'inline-flex' : 'none';
        // The "replace all" button only applies to pronunciation pairs (not emotion tags).
        const repl = cell.querySelector('.replace-pron');
        if (repl) repl.style.display = (tag && this._parsePron(tag.body)) ? 'inline-flex' : 'none';
    }

    // Parse "{display|spoken}" → {display, spoken}. Returns null for emotion tags / non-pairs.
    _parsePron(body) {
        const inner = body.slice(1, -1);
        const bar = inner.indexOf('|');
        if (bar < 0) return null;
        const display = inner.slice(0, bar), spoken = inner.slice(bar + 1);
        if (!display || !spoken || /^\s*\[.*\]\s*$/.test(spoken)) return null;  // skip emotion tags
        return { display, spoken };
    }

    // Collect unique {display|spoken} pronunciation pairs across the whole book.
    _collectPronunciations() {
        const map = new Map();   // key "display spoken" → {display, spoken, count}
        for (const ch of this.chapters) for (const c of (ch.chunks || [])) {
            for (const m of (c.text || '').matchAll(/\{[^{}]*\}/g)) {
                const p = this._parsePron(m[0]);
                if (!p) continue;
                const k = p.display + ' ' + p.spoken;
                const e = map.get(k) || { ...p, count: 0 };
                e.count++; map.set(k, e);
            }
        }
        return [...map.values()].sort((a, b) => a.display.toLowerCase().localeCompare(b.display.toLowerCase()));
    }

    _buildPronOptions() {
        return this._collectPronunciations()
            .map(p => `<option value="${this._h(p.display + ' ' + p.spoken)}">${this._h(p.display)} → ${this._h(p.spoken)} (${p.count})</option>`)
            .join('');
    }

    // Apply a chosen pronunciation pair to the current selection / cursor.
    applyPron(chapterId, chunkId, value) {
        const i = value.indexOf(' ');
        if (i < 0) return;
        const display = value.slice(0, i), spoken = value.slice(i + 1);
        this._insert(chapterId, chunkId, sel => `{${sel || display}|${spoken}}`);
    }

    _escapeRegExp(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

    // Wrap every plain (un-bracketed) occurrence of `display` in `text` as {display|spoken}.
    _wrapOccurrences(text, display, spoken) {
        const re = new RegExp(`(?<![\\w'’])${this._escapeRegExp(display)}(?![\\w'’])`, 'g');
        let count = 0;
        const out = text.split(/(\{[^{}]*\})/).map(part => {
            if (part.startsWith('{') && part.endsWith('}')) return part;   // leave existing tags alone
            return part.replace(re, m => { count++; return `{${m}|${spoken}}`; });
        }).join('');
        return { out, count };
    }

    // Find & replace: wrap every other plain occurrence of the word the cursor is in.
    async applyPronToAll(chapterId, chunkId) {
        const el = document.querySelector(`.chunk-edit[data-chapter="${chapterId}"][data-chunk="${chunkId}"]`);
        if (!el || !el.firstChild) return;
        const tag = this._tagAt(el.firstChild.textContent || '', this._caretOffset(el));
        const p = tag && this._parsePron(tag.body);
        if (!p) { showToast('Put the cursor inside a {display|spoken} pronunciation first.', 'warning'); return; }

        const edits = [];
        for (const ch of this.chapters) for (const c of (ch.chunks || [])) {
            if ((c.type || 'text') !== 'text') continue;
            const { out, count } = this._wrapOccurrences(c.text || '', p.display, p.spoken);
            if (count > 0 && out !== c.text) edits.push({ chapterId: ch.id, chunkId: c.id, text: out, count });
        }
        const total = edits.reduce((n, e) => n + e.count, 0);
        if (!total) { showToast(`No other plain occurrences of "${p.display}" found.`, 'info'); return; }
        if (!confirm(`Replace ${total} occurrence${total > 1 ? 's' : ''} of "${p.display}" across ${edits.length} chunk${edits.length > 1 ? 's' : ''} with {${p.display}|${p.spoken}}?`)) return;

        for (const e of edits) {
            await this._post('/api/project/update-chunk-text', {
                text_file_id: e.chapterId, chunk_id: e.chunkId, new_text: e.text,
                new_nickname: e.text.slice(0, 50) + (e.text.length > 50 ? '...' : '')
            }, 'Replace');
        }
        showToast(`Applied pronunciation to ${total} occurrence${total > 1 ? 's' : ''}.`, 'success');
        await this.refresh();
    }

    removeTagAtCursor(chapterId, chunkId) {
        const el = document.querySelector(`.chunk-edit[data-chapter="${chapterId}"][data-chunk="${chunkId}"]`);
        if (!el || !el.firstChild) return;
        const text = el.firstChild.textContent || '';
        const tag = this._tagAt(text, this._caretOffset(el));
        if (!tag) return;
        // Unwrap to the display text: {display|spoken} → display, {|[tag]} → ''.
        const inner = tag.body.slice(1, -1);
        const display = inner.includes('|') ? inner.slice(0, inner.indexOf('|')) : inner;
        const newText = text.slice(0, tag.start) + display + text.slice(tag.end);
        el.textContent = newText;
        this.updateTagButton(el);
        this.saveChunkText(chapterId, chunkId, el);
    }

    insertPron(chapterId, chunkId) { this._insert(chapterId, chunkId, sel => sel ? `{${sel}|${sel}}` : '{|}', true); }

    // "HENTY PUBLISHING" → "Henty Publishing". Lower-cases then capitalises each word.
    _properCase(s) { return s.toLowerCase().replace(/\b[a-z]/g, c => c.toUpperCase()); }

    // Show the selection as written but speak it in proper case — fixes ALL-CAPS TTS.
    properCaseSelection(chapterId, chunkId) {
        const el = document.querySelector(`.chunk-edit[data-chapter="${chapterId}"][data-chunk="${chunkId}"]`);
        const sel = window.getSelection();
        const selText = el && sel && sel.rangeCount && el.contains(sel.anchorNode) ? sel.toString() : '';
        if (!selText.trim()) { showToast('Select the text to convert to proper case first.', 'warning'); return; }
        this._insert(chapterId, chunkId, s => `{${s}|${this._properCase(s)}}`, true);
    }
    insertTag(chapterId, chunkId, tag) { this._insert(chapterId, chunkId, sel => sel ? `{${sel}|[${tag}]}` : `{|[${tag}]}`); }

    _insert(chapterId, chunkId, build, selectSpoken = false) {
        const el = document.querySelector(`.chunk-edit[data-chapter="${chapterId}"][data-chunk="${chunkId}"]`);
        if (!el) return;
        el.focus();
        const sel = window.getSelection();
        const selText = sel && sel.rangeCount ? sel.toString() : '';
        // Where the snippet will start: the offset of the selection's start within el
        // (independent of selection direction, unlike the anchor-based caret offset).
        const selStart = this._rangeStartOffset(el);
        const snippet = build(selText);
        document.execCommand('insertText', false, snippet);   // replaces selection / inserts at cursor
        // Select the spoken part (right of the last pipe) so the user can type to replace it.
        // In this case skip the immediate save+refresh (which would rebuild the DOM and drop
        // the selection) — the chunk's onblur handler persists the edit instead.
        if (selectSpoken && selStart != null) {
            const bar = snippet.lastIndexOf('|'), brace = snippet.indexOf('}', bar);
            if (bar >= 0 && brace > bar) {
                el.classList.add('dirty');
                this._selectRange(el, selStart + bar + 1, selStart + brace);
                return;
            }
        }
        this.saveChunkText(chapterId, chunkId, el);
    }

    // Character offset of the selection's START within el (0 if none / collapsed at start).
    _rangeStartOffset(el) {
        const sel = window.getSelection();
        if (!sel || !sel.rangeCount) return null;
        const range = sel.getRangeAt(0);
        if (!el.contains(range.startContainer)) return null;
        const r = range.cloneRange();
        r.selectNodeContents(el);
        r.setEnd(range.startContainer, range.startOffset);
        return r.toString().length;
    }

    // Select characters [start, end) of a single-text-node contenteditable.
    _selectRange(el, start, end) {
        el.normalize();
        const node = el.firstChild;
        if (!node || node.nodeType !== Node.TEXT_NODE) return;
        const len = node.textContent.length;
        const r = document.createRange();
        r.setStart(node, Math.min(start, len));
        r.setEnd(node, Math.min(end, len));
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
    }

    async mergeChunk(chapterId, chunkId, direction) {
        const ci = this.chapters.findIndex(c => String(c.id) === String(chapterId));
        const ch = this.chapters[ci];
        if (!ch) return;
        const idx = (ch.chunks || []).findIndex(c => String(c.id) === String(chunkId));
        const atStart = idx === 0, atEnd = idx === ch.chunks.length - 1;
        let cross = false;
        if ((direction === 'prev' && atStart) || (direction === 'next' && atEnd)) {
            const adj = direction === 'prev' ? ci - 1 : ci + 1;
            if (adj < 0 || adj >= this.chapters.length) { showToast('No adjacent chapter to merge into', 'warning'); return; }
            const where = direction === 'prev' ? 'previous' : 'next';
            const msg = ch.chunks.length === 1
                ? `Merge this chunk into the ${where} chapter and remove the chapter "${ch.title || ch.name}"?`
                : `This chunk is at the chapter boundary. Merge it into the ${where} chapter?`;
            if (!confirm(msg)) return;
            cross = true;
        }
        this._busyRow(chapterId, chunkId, true);
        const d = await this._post('/api/project/merge-chunk',
            { chapter_id: chapterId, chunk_id: chunkId, direction, cross_chapter: cross }, 'Merge');
        if (d) await this.refresh(); else this._busyRow(chapterId, chunkId, false);
    }
    // Delete a whole chunk (text or other), after a browser confirmation. Undoable (Ctrl+Z).
    async deleteChunk(chapterId, chunkId) {
        const chunk = this.chapters.find(c => String(c.id) === String(chapterId))?.chunks.find(c => String(c.id) === String(chunkId));
        if (!chunk) return;
        const takes = (chunk.generated_audios || []).length;
        const preview = (chunk.text || '').trim().slice(0, 60);
        const msg = `Delete this chunk?` +
            (preview ? `\n\n"${preview}${(chunk.text || '').trim().length > 60 ? '…' : ''}"` : '') +
            (takes ? `\n\nIts ${takes} take${takes > 1 ? 's' : ''} will be removed from the chunk.` : '') +
            `\n\nThis can be undone with Ctrl+Z.`;
        if (!confirm(msg)) return;
        this._busyRow(chapterId, chunkId, true);
        const d = await this._post('/api/project/delete-chunk', { chapter_id: chapterId, chunk_id: chunkId }, 'Delete chunk');
        if (d) await this.refresh(); else this._busyRow(chapterId, chunkId, false);
    }

    // ---- Pauses ------------------------------------------------------------
    async insertPause(chapterId, afterChunkId) {
        const ch = this.chapters.find(c => String(c.id) === String(chapterId));
        const idx = ch ? ch.chunks.findIndex(c => String(c.id) === String(afterChunkId)) : -1;
        if (idx < 0) return;
        const d = await this._post('/api/project/insert-chunk',
            { text_file_id: chapterId, chapter_id: chapterId, insert_after_id: idx, type: 'pause', duration_ms: 500 }, 'Insert pause');
        if (d) await this.refresh();
    }
    async setPauseDuration(chapterId, chunkId, ms) {
        const d = await this._post('/api/project/set-pause-duration',
            { chapter_id: chapterId, chunk_id: chunkId, duration_ms: parseInt(ms, 10) || 0 }, 'Set pause');
        if (d) { const c = this.chapters.find(c => String(c.id) === String(chapterId))?.chunks.find(c => String(c.id) === String(chunkId)); if (c) c.duration_ms = d.duration_ms; }
    }
    async deletePause(chapterId, chunkId) {
        const d = await this._post('/api/project/delete-chunk', { chapter_id: chapterId, chunk_id: chunkId }, 'Delete pause');
        if (d) await this.refresh();
    }

    // ---- Split with a movable marker --------------------------------------
    splitChunkAtCursor(chapterId, chunkId) {
        const el = document.querySelector(`.chunk-edit[data-chapter="${chapterId}"][data-chunk="${chunkId}"]`);
        if (!el) return;
        el.normalize();                       // merge any fragmented text nodes from editing
        const text = el.textContent || '';
        if (!text) return;
        // Initial offset: the live caret if it's inside this chunk, else the midpoint.
        let offset = this._caretOffset(el);
        if (offset == null) offset = Math.floor(text.length / 2);
        this._enterSplitMode(el, chapterId, chunkId, this._snapWord(text, offset));
    }

    // Snap an offset to the nearest word boundary. Any whitespace — including newlines —
    // counts, so the start of a line is a valid (and common) landing spot.
    _snapWord(text, offset) {
        offset = Math.max(0, Math.min(offset, text.length));
        const ws = (ch) => ch === undefined || /\s/.test(ch);
        if (offset === 0 || offset === text.length) return offset;
        if (ws(text[offset]) || ws(text[offset - 1])) return offset;   // already at a gap / line start
        let l = offset; while (l > 0 && !/\s/.test(text[l - 1])) l--;   // start of current word
        let r = offset; while (r < text.length && !/\s/.test(text[r])) r++; // end of current word
        return (offset - l) <= (r - offset) ? l : r;
    }

    _enterSplitMode(el, chapterId, chunkId, offset) {
        this._cancelSplit();
        const cell = el.closest('.text-cell');
        const text = el.firstChild.textContent || '';
        el.setAttribute('contenteditable', 'false');
        el.classList.add('splitting');

        const marker = document.createElement('div');
        marker.className = 'split-marker';
        marker.innerHTML = '<span class="material-symbols-outlined">content_cut</span>';
        cell.appendChild(marker);

        const bar = document.createElement('div');
        bar.className = 'split-confirm';
        bar.innerHTML = '<button class="ok">Split here</button><button class="cancel">Cancel</button>';
        cell.appendChild(bar);

        const state = { el, cell, marker, bar, text, offset, chapterId, chunkId };
        this._split = state;
        this._positionMarker();

        state.onMove = (e) => {
            const pos = this._offsetFromPoint(el, e.clientX, e.clientY);
            if (pos != null) { state.offset = this._snapWord(text, pos); this._positionMarker(); }
        };
        state.onKey = (e) => { if (e.key === 'Escape') this._cancelSplit(); };
        el.addEventListener('mousemove', state.onMove);
        document.addEventListener('keydown', state.onKey);
        bar.querySelector('.ok').onclick = () => this._confirmSplit();
        bar.querySelector('.cancel').onclick = () => this._cancelSplit();
        el.addEventListener('click', () => this._confirmSplit(), { once: true });
    }

    _offsetFromPoint(el, x, y) {
        let range = null;
        if (document.caretRangeFromPoint) range = document.caretRangeFromPoint(x, y);
        else if (document.caretPositionFromPoint) { const p = document.caretPositionFromPoint(x, y); if (p) return p.offset; }
        if (range && range.startContainer === el.firstChild) return range.startOffset;
        return null;
    }

    _positionMarker() {
        const s = this._split; if (!s) return;
        const node = s.el.firstChild;
        const range = document.createRange();
        range.setStart(node, Math.min(s.offset, node.textContent.length));
        range.collapse(true);
        const rect = range.getBoundingClientRect();
        const cellRect = s.cell.getBoundingClientRect();
        const left = (rect.left || cellRect.left) - cellRect.left;
        const top = (rect.top || cellRect.top) - cellRect.top;
        s.marker.style.left = left + 'px';
        s.marker.style.top = top + 'px';
        s.marker.style.height = (rect.height || 20) + 'px';
        s.bar.style.left = Math.max(0, left - 30) + 'px';
        s.bar.style.top = (top + (rect.height || 20) + 4) + 'px';
    }

    async _confirmSplit() {
        const s = this._split; if (!s) return;
        const { chapterId, chunkId, offset } = s;
        this._cancelSplit();
        this._busyRow(chapterId, chunkId, true);
        const d = await this._post('/api/project/split-chunk', { chapter_id: chapterId, chunk_id: chunkId, char_offset: offset }, 'Split');
        if (d) await this.refresh(); else this._busyRow(chapterId, chunkId, false);
    }

    _cancelSplit() {
        const s = this._split; if (!s) return;
        s.el.removeEventListener('mousemove', s.onMove);
        document.removeEventListener('keydown', s.onKey);
        s.el.setAttribute('contenteditable', 'true');
        s.el.classList.remove('splitting');
        s.marker.remove(); s.bar.remove();
        this._split = null;
    }

    // ---- Undo + original text ---------------------------------------------
    async undo() {
        const d = await this._post('/api/project/undo', {}, 'Undo');
        if (!d) return;
        if (d.empty) { showToast('Nothing to undo', 'info'); return; }
        await this.refresh();
        showToast('Undone', 'info');
    }

    showOriginal(chapterId, chunkId) {
        const chunk = this.chapters.find(c => String(c.id) === String(chapterId))?.chunks.find(c => String(c.id) === String(chunkId));
        if (!chunk) return;
        const original = chunk.original_text;
        const text = (original != null && original !== '') ? original : (chunk.text || '');
        const note = (original == null || original === '') ? 'No original recorded for this chunk (showing current text). Re-import to capture originals.' : '';
        this._showTextPopup('Original text', text, note);
    }

    showTakeInput(chapterId, chunkId, audioFile) {
        const chunk = this.chapters.find(c => String(c.id) === String(chapterId))?.chunks.find(c => String(c.id) === String(chunkId));
        const take = chunk?.generated_audios?.find(a => a.audio_file === audioFile);
        if (!take) return;
        const text = take.tts_input_text || take.input_text || '';
        const note = take.tts_input_text ? '' : 'Exact model input not recorded for this take (showing stored input text).';
        this._showTextPopup('Exact text sent to the model', text || '(no input text recorded)', note);
    }

    _showTextPopup(title, text, note) {
        const overlay = document.createElement('div');
        overlay.className = 'orig-overlay';
        overlay.innerHTML = `<div class="orig-box">
            <div class="orig-head"><span></span><button class="orig-close" title="Close">✕</button></div>
            ${note ? '<div class="orig-note"></div>' : ''}
            <div class="orig-text"></div>
            <div class="orig-actions">
                <button class="orig-copy">Copy</button>
                <button class="orig-dismiss">Close</button>
            </div>
        </div>`;
        overlay.querySelector('.orig-head span').textContent = title;
        if (note) overlay.querySelector('.orig-note').textContent = note;
        overlay.querySelector('.orig-text').textContent = text;
        const close = () => overlay.remove();
        overlay.querySelector('.orig-close').onclick = close;
        overlay.querySelector('.orig-dismiss').onclick = close;
        overlay.onclick = (e) => { if (e.target === overlay) close(); };
        overlay.querySelector('.orig-copy').onclick = async (e) => {
            try { await navigator.clipboard.writeText(text); e.target.textContent = 'Copied!'; setTimeout(() => e.target.textContent = 'Copy', 1200); }
            catch (err) { showToast('Copy failed', 'error'); }
        };
        document.body.appendChild(overlay);
    }

    // ---- Helpers -----------------------------------------------------------
    async _post(path, body, label) {
        try {
            const r = await fetch(`${SERVER_URL}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY }, body: JSON.stringify(body) });
            const data = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(data.error || `Server error ${r.status}`);
            return data;
        } catch (e) { console.error(`${label}:`, e); showToast(`${label} failed: ${e.message}`, 'error'); return null; }
    }
    _markGenerating(key, on) {
        const i = key.indexOf(':');
        this._renderTakesCell(key.slice(0, i), key.slice(i + 1));
    }

    // Re-render just one chunk's takes cell (and its row's generating tint) from local
    // state, so completions and the in-progress placeholder appear live without rebuilding
    // the whole grid — essential during whole-book generation (thousands of chunks).
    _renderTakesCell(chId, cid) {
        const chapter = this.chapters.find(c => String(c.id) === String(chId));
        const chunk = chapter && (chapter.chunks || []).find(c => String(c.id) === String(cid));
        if (!chunk) return;
        const key = `${chId}:${cid}`;
        const generating = !!this.activeGenerations[key];
        const cell = document.getElementById(`takes-${chId}-${cid}`);
        if (cell) cell.innerHTML = this._takes(chapter, chunk, generating);
        const row = document.querySelector(`.chunk-row[data-key="${key}"]`);
        if (row) row.classList.toggle('generating', generating);
    }
    // Instant visual feedback while the server processes a structural/delete change.
    _busyRow(chapterId, chunkId, on) {
        const row = document.querySelector(`.chunk-row[data-key="${chapterId}:${chunkId}"]`);
        if (row) row.classList.toggle('busy', on);
    }
    _esc(s) { return String(s).replace(/'/g, "\\'"); }
    _h(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
}

const bookTab = new BookTab();
