/**
 * Gutenberg Processing Tab Logic
 * Handles raw text loading, Markdown editing, chapter processing, and validation
 *
 * Markdown Format:
 *   ## Chapter Title     <- Chapter header (also becomes first chunk)
 *   </chunk>             <- Chunk boundary marker
 *   Text content here    <- Chunk text (no escaping needed)
 *   </chunk>
 *   [pause:1.5]          <- Pause marker
 *   </chunk>
 *   More text...
 *
 * Pronunciation/Emotion Markup:
 *   {display|spoken}     <- Pronunciation override (purple highlight)
 *   {display|[laugh]}    <- Emotion tag (auto-uses Turbo model)
 *   {|[cough]}           <- Standalone emotion (no display text)
 */

/**
 * StructureEditor — stage-1 interactive editor.
 *
 * Parses a stage-1 annotated book.txt (with <delete>…</delete> blocks and
 * self-closing <chapter title="…"/> markers) and renders it as an interactive
 * DOM view where:
 *   • <delete> blocks are shown in red and can be toggled (keep / discard)
 *   • <chapter/> markers are draggable horizontal bars
 *   • Between every paragraph an "Add chapter break here" affordance appears
 *     on hover (or when a drag is in progress)
 */
class StructureEditor {
    constructor(containerEl) {
        this.container = containerEl;
        this.blocks = [];      // [{type, ...}]
        this._dragIdx = null;  // index of block being dragged
        this._dropIdx = null;  // target insertion index
        this._dragGhost = null;
        this._boundMouseMove = this._onMouseMove.bind(this);
        this._boundMouseUp   = this._onMouseUp.bind(this);
    }

    // ------------------------------------------------------------------ //
    //  Parse stage-1 content into a blocks array                         //
    // ------------------------------------------------------------------ //
    parse(content) {
        this.blocks = [];
        // Match <delete>...</delete> and <chapter title="..."/>
        const re = /(<delete>[\s\S]*?<\/delete>|<chapter\s+title="([^"]*)"[^/]*\/>)/g;
        let lastIndex = 0;
        let m;

        while ((m = re.exec(content)) !== null) {
            if (m.index > lastIndex) {
                const raw = content.slice(lastIndex, m.index);
                this._pushTextBlock(raw);
            }
            if (m[0].startsWith('<delete>')) {
                const inner = m[0].replace(/^<delete>/, '').replace(/<\/delete>$/, '');
                this.blocks.push({ type: 'delete', content: inner, active: true });
            } else {
                const title = (m[2] || '').replace(/&quot;/g, '"');
                this.blocks.push({ type: 'chapter', title });
            }
            lastIndex = re.lastIndex;
        }
        if (lastIndex < content.length) {
            this._pushTextBlock(content.slice(lastIndex));
        }
    }

    _pushTextBlock(raw) {
        // Split raw text on blank lines → individual paragraph strings
        const paras = raw.split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
        if (paras.length) {
            this.blocks.push({ type: 'text', paragraphs: paras });
        }
    }

    // ------------------------------------------------------------------ //
    //  Serialise back to stage-1 format                                  //
    // ------------------------------------------------------------------ //
    serialise() {
        return this.blocks.map(b => {
            if (b.type === 'delete') {
                return b.active ? `<delete>${b.content}</delete>` : b.content;
            }
            if (b.type === 'chapter') {
                const t = b.title.replace(/"/g, '&quot;');
                return `<chapter title="${t}"/>`;
            }
            // text
            return b.paragraphs.join('\n\n');
        }).join('\n\n');
    }

    // ------------------------------------------------------------------ //
    //  Render                                                             //
    // ------------------------------------------------------------------ //
    render() {
        this.container.innerHTML = '';
        this.blocks.forEach((block, idx) => {
            this.container.appendChild(this._makeEl(block, idx));
        });
        // Trailing drop zone
        this.container.appendChild(this._makeDropZone(this.blocks.length));
    }

    _makeEl(block, idx) {
        if (block.type === 'delete')  return this._makeDeleteEl(block, idx);
        if (block.type === 'chapter') return this._makeChapterEl(block, idx);
        return this._makeTextEl(block, idx);
    }

    _makeDeleteEl(block, idx) {
        const wrap = document.createElement('div');
        wrap.className = 'se-delete-block' + (block.active ? '' : ' se-delete-kept');
        wrap.dataset.idx = idx;

        const header = document.createElement('div');
        header.className = 'se-delete-header';
        const badge = document.createElement('span');
        badge.className = 'se-delete-badge';
        badge.textContent = block.active ? 'WILL DELETE' : 'KEPT';
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'se-delete-toggle';
        toggleBtn.textContent = block.active ? 'Keep instead' : 'Mark for deletion';
        toggleBtn.onclick = () => {
            block.active = !block.active;
            wrap.classList.toggle('se-delete-kept', !block.active);
            badge.textContent    = block.active ? 'WILL DELETE' : 'KEPT';
            toggleBtn.textContent = block.active ? 'Keep instead' : 'Mark for deletion';
        };
        header.appendChild(badge);
        header.appendChild(toggleBtn);

        const pre = document.createElement('pre');
        pre.className = 'se-delete-content';
        const preview = block.content.trim();
        pre.textContent = preview.length > 500 ? preview.slice(0, 500) + '\n…' : preview;

        wrap.appendChild(header);
        wrap.appendChild(pre);
        return wrap;
    }

    _makeChapterEl(block, idx) {
        const wrap = document.createElement('div');
        wrap.className = 'se-chapter-marker';
        wrap.dataset.idx = idx;

        const handle = document.createElement('span');
        handle.className = 'se-drag-handle';
        handle.title = 'Drag to move chapter break';
        handle.textContent = '⠿';
        handle.addEventListener('mousedown', (e) => this._startDrag(e, idx));

        const input = document.createElement('input');
        input.type  = 'text';
        input.className = 'se-chapter-title-input';
        input.value = block.title;
        input.addEventListener('change', () => { block.title = input.value; });

        const removeBtn = document.createElement('button');
        removeBtn.className = 'se-chapter-remove';
        removeBtn.textContent = '×';
        removeBtn.title = 'Remove chapter break';
        removeBtn.onclick = () => {
            this.blocks.splice(idx, 1);
            this._mergeAdjacentText();
            this.render();
        };

        wrap.appendChild(handle);
        wrap.appendChild(input);
        wrap.appendChild(removeBtn);

        // Drop zone below this chapter marker
        wrap.appendChild(this._makeDropZone(idx + 1));
        return wrap;
    }

    _makeTextEl(block, idx) {
        const wrap = document.createElement('div');
        const totalChars = block.paragraphs.join(' ').length;
        const isShort = totalChars <= 120;
        wrap.className = 'se-text-block' + (isShort ? ' se-short-block' : '');
        wrap.dataset.idx = idx;

        if (isShort) {
            const badge = document.createElement('span');
            badge.className = 'se-short-badge';
            badge.title = 'Short block — likely a section heading. Delete the break above to merge it into the previous chapter.';
            badge.textContent = 'heading?';
            wrap.appendChild(badge);
        }

        block.paragraphs.forEach((para, pIdx) => {
            // Drop zone before each paragraph
            const dz = this._makeDropZone(idx, pIdx);
            wrap.appendChild(dz);

            const p = document.createElement('p');
            p.className = 'se-paragraph';
            p.textContent = para;
            wrap.appendChild(p);
        });
        // Drop zone after last paragraph
        wrap.appendChild(this._makeDropZone(idx, block.paragraphs.length));
        return wrap;
    }

    _makeDropZone(blockIdx, paraIdx) {
        const dz = document.createElement('div');
        dz.className = 'se-drop-zone';
        dz.dataset.blockIdx = blockIdx;
        if (paraIdx !== undefined) dz.dataset.paraIdx = paraIdx;

        dz.addEventListener('mouseenter', () => {
            if (this._dragIdx !== null) dz.classList.add('se-drop-zone-active');
        });
        dz.addEventListener('mouseleave', () => {
            dz.classList.remove('se-drop-zone-active');
        });
        // Click to add a new chapter break
        dz.addEventListener('click', () => {
            if (this._dragIdx !== null) return; // ignore during drag
            const title = prompt('Chapter title:', 'New Chapter');
            if (title === null) return; // cancelled
            const pIdx = paraIdx !== undefined ? paraIdx : 0;
            this.addChapterMarker(title, blockIdx, pIdx);
        });
        return dz;
    }

    // ------------------------------------------------------------------ //
    //  Drag logic                                                         //
    // ------------------------------------------------------------------ //
    _startDrag(e, idx) {
        e.preventDefault();
        this._dragIdx = idx;

        // Ghost label
        this._dragGhost = document.createElement('div');
        this._dragGhost.className = 'se-drag-ghost';
        this._dragGhost.textContent = '↕ ' + (this.blocks[idx].title || 'Chapter');
        document.body.appendChild(this._dragGhost);
        this._moveDragGhost(e);

        document.addEventListener('mousemove', this._boundMouseMove);
        document.addEventListener('mouseup',   this._boundMouseUp);

        // Show all drop zones
        this.container.classList.add('se-dragging');
    }

    _onMouseMove(e) {
        if (this._dragIdx === null) return;
        this._moveDragGhost(e);

        // Highlight nearest drop zone
        document.querySelectorAll('.se-drop-zone-active').forEach(el =>
            el.classList.remove('se-drop-zone-active'));

        const best = this._nearestDropZone(e.clientX, e.clientY);
        if (best) {
            best.classList.add('se-drop-zone-active');
            this._dropIdx = { blockIdx: parseInt(best.dataset.blockIdx),
                              paraIdx:  best.dataset.paraIdx !== undefined
                                        ? parseInt(best.dataset.paraIdx) : null };
        }
    }

    _onMouseUp(e) {
        document.removeEventListener('mousemove', this._boundMouseMove);
        document.removeEventListener('mouseup',   this._boundMouseUp);
        this.container.classList.remove('se-dragging');
        document.querySelectorAll('.se-drop-zone-active').forEach(el =>
            el.classList.remove('se-drop-zone-active'));
        if (this._dragGhost) { this._dragGhost.remove(); this._dragGhost = null; }

        if (this._dragIdx !== null && this._dropIdx !== null) {
            this._dropChapterMarker(this._dragIdx, this._dropIdx);
        }
        this._dragIdx = null;
        this._dropIdx = null;
    }

    _moveDragGhost(e) {
        if (!this._dragGhost) return;
        this._dragGhost.style.left = (e.clientX + 12) + 'px';
        this._dragGhost.style.top  = (e.clientY + 4)  + 'px';
    }

    _nearestDropZone(x, y) {
        let best = null, bestDist = Infinity;
        this.container.querySelectorAll('.se-drop-zone').forEach(el => {
            const r = el.getBoundingClientRect();
            const cy = (r.top + r.bottom) / 2;
            const dist = Math.abs(y - cy);
            if (dist < bestDist) { bestDist = dist; best = el; }
        });
        return best;
    }

    /**
     * Move a chapter marker (at blocks[fromIdx]) to the position described by dropTarget.
     * dropTarget = {blockIdx, paraIdx}
     *   blockIdx — the text block to drop into (or boundary between blocks)
     *   paraIdx  — if not null, split the text block at this paragraph index
     */
    _dropChapterMarker(fromIdx, dropTarget) {
        const movedBlock = this.blocks.splice(fromIdx, 1)[0];

        // After removal, adjust indices
        let { blockIdx, paraIdx } = dropTarget;
        if (fromIdx < blockIdx) blockIdx--;

        const target = this.blocks[blockIdx];

        if (target && target.type === 'text' && paraIdx !== null && paraIdx > 0
                && paraIdx < target.paragraphs.length) {
            // Split the text block at paraIdx
            const before = { type: 'text', paragraphs: target.paragraphs.slice(0, paraIdx) };
            const after  = { type: 'text', paragraphs: target.paragraphs.slice(paraIdx) };
            this.blocks.splice(blockIdx, 1, before, movedBlock, after);
        } else {
            // Insert before the target block (or at end)
            const insertAt = Math.min(Math.max(blockIdx, 0), this.blocks.length);
            this.blocks.splice(insertAt, 0, movedBlock);
        }

        this._mergeAdjacentText();
        this.render();
    }

    /** Merge consecutive text blocks that ended up adjacent */
    _mergeAdjacentText() {
        const merged = [];
        for (const b of this.blocks) {
            const prev = merged[merged.length - 1];
            if (b.type === 'text' && prev && prev.type === 'text') {
                prev.paragraphs = prev.paragraphs.concat(b.paragraphs);
            } else {
                merged.push(b);
            }
        }
        this.blocks = merged;
    }

    /** Add a new chapter marker before a given text-block paragraph */
    addChapterMarker(title, blockIdx, paraIdx) {
        const target = this.blocks[blockIdx];
        const newMarker = { type: 'chapter', title: title || 'New Chapter' };
        if (target && target.type === 'text' && paraIdx > 0
                && paraIdx < target.paragraphs.length) {
            const before = { type: 'text', paragraphs: target.paragraphs.slice(0, paraIdx) };
            const after  = { type: 'text', paragraphs: target.paragraphs.slice(paraIdx) };
            this.blocks.splice(blockIdx, 1, before, newMarker, after);
        } else {
            this.blocks.splice(blockIdx, 0, newMarker);
        }
        this._mergeAdjacentText();
        this.render();
    }

    /**
     * Auto-collapse headings: remove chapter breaks that separate short text
     * blocks from their adjacent content.  A "short block" is any text block
     * with ≤ 120 chars total.
     *
     * Pattern removed:
     *   [chapter marker]  ← delete this marker when the text block that follows
     *   [short text block] ← this (heading) gets merged into previous block
     *
     * Merged text blocks are joined with a blank line so the heading is still
     * visible in the body text.  The chapter marker BEFORE the long block that
     * comes after the heading is renamed to include the heading text.
     *
     * Returns the number of markers removed.
     */
    autoCollapseHeadings() {
        const SHORT = 120;

        const totalChars = (block) =>
            block.type === 'text' ? block.paragraphs.join(' ').length : Infinity;

        let removed = 0;
        let changed = true;

        // Keep iterating until nothing more to collapse (handles chained headings)
        while (changed) {
            changed = false;
            for (let i = 0; i < this.blocks.length - 2; i++) {
                const marker = this.blocks[i];
                const textBlock = this.blocks[i + 1];

                // Pattern: [chapter] [short-text] [chapter] [any-text]
                // → remove the first [chapter], absorb short-text into prev block
                if (marker.type === 'chapter'
                        && textBlock.type === 'text'
                        && totalChars(textBlock) <= SHORT) {

                    // Remove the chapter marker and absorb short text into
                    // the previous text block (or just remove both if at start)
                    const prevBlock = i > 0 ? this.blocks[i - 1] : null;
                    if (prevBlock && prevBlock.type === 'text') {
                        prevBlock.paragraphs.push(...textBlock.paragraphs);
                        this.blocks.splice(i, 2);  // remove marker + short block
                    } else if (i === 0) {
                        // Very first block is a short heading — remove its marker,
                        // keep the text so it's visible in the editor
                        this.blocks.splice(i, 1);  // remove only the marker
                    } else {
                        // Previous block is a chapter marker — just remove this marker
                        // and let the short text sit under the previous chapter
                        this.blocks.splice(i, 1);
                    }
                    removed++;
                    changed = true;
                    break;  // restart scan after splice
                }
            }
        }

        if (removed > 0) {
            this._mergeAdjacentText();
            this.render();
        }
        return removed;
    }

    _esc(s) {
        return String(s || '')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
}


class GutenbergTab {
    constructor() {
        this.markdownContent = '';
        this.chapters = [];
        this.defaultGutenbergUrl = '';
        this.cleanViewActive = false;
        this.chaptersLocked = false;

        // Undo stack (max 5 levels)
        this.undoStack = [];
        this.maxUndoLevels = 5;

        // Auto-save debounce
        this._autoSaveTimer = null;
        this._autoSaveDelay = 1500; // ms after last keystroke
        this._lastSavedContent = '';

        // Structure editor (stage 1)
        this.structureEditor = null;

        // Stage 3 text lock
        this._textUnlocked = false;
    }

    async init() {
        await this.loadDefaultUrl();
        await this.loadMarkdown();
        this._setupEditorListeners();
    }

    /**
     * Set up input listeners for live highlighting and auto-save.
     * Also tracks the last selection inside the editor so pronunciation/tag
     * buttons can restore it after stealing focus.
     * At stage 3 with chapters locked, prevents edits to plain text while allowing tag manipulation.
     */
    _setupEditorListeners() {
        const editor = document.getElementById('pseudoXmlEditor');
        if (!editor) return;

        // Store original content at stage 3 to detect text-only changes
        let lastKnownGoodContent = editor.textContent;

        // Live re-highlight on input (preserves cursor position)
        editor.addEventListener('input', () => {
            if (this.cleanViewActive) return;

            // At stage 3 with chapters locked, detect if plain text (not tags) was changed
            if (this._isStage3() && this.chaptersLocked && !this._textUnlocked) {
                const currentRaw = editor.textContent || editor.innerText;
                // Extract plain text (remove all markup)
                const lastGoodPlain = this._extractPlainText(lastKnownGoodContent);
                const currentPlain = this._extractPlainText(currentRaw);

                // If plain text changed (beyond tags being modified), revert
                if (lastGoodPlain !== currentPlain) {
                    showToast('Text is locked. Only tags and chunks can be modified.', 'warning');
                    editor.textContent = lastKnownGoodContent;
                    this._liveHighlight(editor);
                    return;
                }
            }

            lastKnownGoodContent = editor.textContent || editor.innerText;
            this._liveHighlight(editor);
            this._scheduleAutoSave(editor);
        });

        // Save the last selection range inside this editor so insertion
        // buttons can restore it after the button click steals focus.
        document.addEventListener('selectionchange', () => {
            const sel = window.getSelection();
            if (!sel || !sel.rangeCount) return;
            const r = sel.getRangeAt(0);
            if (editor.contains(r.commonAncestorContainer)) {
                this._savedRange = r.cloneRange();
            }
        });
    }

    /**
     * Extract plain text by removing all markup (chunks, pronunciation, etc.)
     */
    _extractPlainText(text) {
        let plain = text || '';
        // Remove chunk tags
        plain = plain.replace(/<chunk[^>]*>/g, '');
        plain = plain.replace(/<\/chunk>/g, '');
        // Remove pronunciation markup
        plain = plain.replace(/\{[^|}]*\|[^}]*\}/g, (match) => {
            // Extract just the display text from {display|spoken}
            const parts = match.slice(1, -1).split('|');
            return parts[0] || '';
        });
        return plain;
    }

    /**
     * Restore the last saved editor selection and focus the editor.
     * Call this before any programmatic text insertion.
     */
    _restoreEditorSelection() {
        const editor = document.getElementById('pseudoXmlEditor');
        if (!editor) return;
        editor.focus();
        if (this._savedRange) {
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(this._savedRange);
        }
    }

    /**
     * Re-apply purple/yellow highlighting while preserving caret position.
     * Uses a fast innerHTML swap with cursor save/restore.
     */
    _liveHighlight(editor) {
        // Get plain text and cursor offset
        const sel = window.getSelection();
        if (!sel.rangeCount) return;

        const range = sel.getRangeAt(0);
        // Calculate text offset from start of editor
        const preRange = document.createRange();
        preRange.selectNodeContents(editor);
        preRange.setEnd(range.startContainer, range.startOffset);
        const cursorOffset = preRange.toString().length;

        // Get raw text content
        const raw = editor.textContent || editor.innerText;

        // Build highlighted HTML
        let escaped = raw
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Highlight <delete>...</delete> blocks (red background)
        escaped = escaped.replace(
            /(&lt;delete&gt;)([\s\S]*?)(&lt;\/delete&gt;)/g,
            '<span style="background:#fee2e2;color:#991b1b;border-radius:3px;padding:0 2px">$1$2$3</span>'
        );

        // Highlight <chapter title="..."> and </chapter> tags (blue)
        escaped = escaped.replace(
            /(&lt;chapter[^&]*&gt;)/g,
            '<span style="background:#dbeafe;color:#1e40af;border-radius:3px;padding:0 3px;font-weight:600">$1</span>'
        );
        escaped = escaped.replace(
            /(&lt;\/chapter&gt;)/g,
            '<span style="background:#dbeafe;color:#1e40af;border-radius:3px;padding:0 3px;font-weight:600">$1</span>'
        );

        // Highlight legacy markdown ## chapter headers
        escaped = escaped.replace(
            /(^#{2,3}\s+.+$)/gm,
            '<span style="background:#dbeafe;color:#1e40af;font-weight:600">$1</span>'
        );

        // Highlight pronunciation/paralinguistic markup: {display|spoken} or {|[tag]}
        escaped = escaped.replace(
            /\{([^|}]*)\|([^}]*)\}/g,
            (match, display, spoken) => {
                const origDisplay = display.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"');
                const markupId = Math.random().toString(16).substr(2, 8);
                return `<span class="pp-markup" data-markup-id="${markupId}" data-original-display="${origDisplay.replace(/"/g, '&quot;')}" style="background:#ddd6fe;border-radius:3px;padding:0 2px;display:inline-block;cursor:pointer">{${display}|<span style="color:#7c3aed;font-weight:600">${spoken}</span>}</span>`
            }
        );

        // Highlight <chunk id="HEX"> open tags — carry hex ID as data attribute for popup menu
        escaped = escaped.replace(
            /&lt;chunk\s+id="([a-f0-9]{8})"([^&]*)&gt;/g,
            (_, hexId, rest) =>
                `<span class="chunk-open-tag" data-hex-id="${hexId}" style="background:#fde68a;color:#92400e;border-radius:3px;padding:0 3px;font-size:0.85em;cursor:pointer;">&lt;chunk id="${hexId}"${rest}&gt;</span>`
        );
        // Highlight </chunk> close tags — annotate with preceding open-tag hex ID in sequence
        {
            const chunkIdSeq = [];
            (raw.match(/<chunk\s+id="([a-f0-9]{8})"/g) || []).forEach(m => {
                const hm = m.match(/id="([a-f0-9]{8})"/);
                if (hm) chunkIdSeq.push(hm[1]);
            });
            let closeIdx = 0;
            escaped = escaped.replace(
                /&lt;\/chunk&gt;/g,
                () => {
                    const hexId = chunkIdSeq[closeIdx++] || '';
                    return `<span class="chunk-close-tag" data-hex-id="${hexId}" style="background:#fde68a;color:#92400e;border-radius:3px;padding:0 3px;font-size:0.85em;cursor:pointer;">&lt;/chunk&gt;</span>`;
                }
            );
        }

        editor.innerHTML = escaped;

        // Attach popup menu handlers to chunk and p/p markup
        this._attachChunkPopupMenus(editor);
        this._attachMarkupPopupMenus(editor);

        // Restore cursor position
        this._restoreCursor(editor, cursorOffset);
    }

    /**
     * Restore cursor to a text offset within a contenteditable element
     */
    _restoreCursor(editor, targetOffset) {
        const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null, false);
        let offset = 0;
        let node;
        while ((node = walker.nextNode())) {
            const len = node.textContent.length;
            if (offset + len >= targetOffset) {
                const sel = window.getSelection();
                const range = document.createRange();
                range.setStart(node, targetOffset - offset);
                range.collapse(true);
                sel.removeAllRanges();
                sel.addRange(range);
                return;
            }
            offset += len;
        }
    }

    /**
     * Attach click handlers to chunk tags that show a popup menu for merging.
     */
    _attachChunkPopupMenus(editor) {
        // Create (or reuse) a singleton popup element
        let popup = document.getElementById('chunkMergePopup');
        if (!popup) {
            popup = document.createElement('div');
            popup.id = 'chunkMergePopup';
            popup.style.cssText = [
                'position:fixed',
                'background:#1e293b',
                'color:#f1f5f9',
                'padding:8px 0',
                'border-radius:4px',
                'font-size:12px',
                'font-family:sans-serif',
                'cursor:pointer',
                'z-index:9999',
                'display:none',
                'pointer-events:all',
                'min-width:140px',
                'box-shadow:0 2px 8px rgba(0,0,0,0.3)'
            ].join(';');
            document.body.appendChild(popup);
        }

        let hideTimer = null;
        const showPopup = (hexId, rect) => {
            clearTimeout(hideTimer);
            popup.innerHTML = `
                <div style="padding:6px 16px;cursor:pointer;transition:background 0.15s" data-action="merge-prev" data-hex-id="${hexId}">⬆ Merge ↑</div>
                <div style="padding:6px 16px;cursor:pointer;transition:background 0.15s" data-action="merge-next" data-hex-id="${hexId}">⬇ Merge ↓</div>
            `;
            popup.querySelectorAll('div').forEach(item => {
                item.addEventListener('mouseenter', () => item.style.background = '#334155');
                item.addEventListener('mouseleave', () => item.style.background = '');
                item.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const action = item.dataset.action;
                    const id = item.dataset.hexId;
                    popup.style.display = 'none';
                    if (action === 'merge-prev') await this._mergeChunk(id, 'prev');
                    if (action === 'merge-next') await this._mergeChunk(id, 'next');
                });
            });
            popup.style.display = 'block';
            popup.style.left = rect.left + 'px';
            popup.style.top = (rect.bottom + 4) + 'px';
        };
        const hidePopup = () => {
            hideTimer = setTimeout(() => { popup.style.display = 'none'; }, 100);
        };

        editor.querySelectorAll('.chunk-open-tag, .chunk-close-tag').forEach(span => {
            span.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const hexId = span.dataset.hexId;
                if (hexId) showPopup(hexId, span.getBoundingClientRect());
            });
        });

        popup.addEventListener('mouseenter', () => clearTimeout(hideTimer));
        popup.addEventListener('mouseleave', hidePopup);
        document.addEventListener('click', hidePopup);
    }

    /**
     * Attach click handlers to p/p markup that show a popup menu for removing.
     */
    _attachMarkupPopupMenus(editor) {
        // Create (or reuse) a singleton popup element
        let popup = document.getElementById('markupRemovePopup');
        if (!popup) {
            popup = document.createElement('div');
            popup.id = 'markupRemovePopup';
            popup.style.cssText = [
                'position:fixed',
                'background:#1e293b',
                'color:#f1f5f9',
                'padding:8px 16px',
                'border-radius:4px',
                'font-size:12px',
                'font-family:sans-serif',
                'cursor:pointer',
                'z-index:9999',
                'display:none',
                'pointer-events:all',
                'box-shadow:0 2px 8px rgba(0,0,0,0.3)'
            ].join(';');
            popup.addEventListener('mouseenter', (e) => clearTimeout(popup._hideTimer));
            popup.addEventListener('mouseleave', (e) => {
                popup._hideTimer = setTimeout(() => { popup.style.display = 'none'; }, 100);
            });
            document.body.appendChild(popup);
        }

        editor.querySelectorAll('.pp-markup').forEach(span => {
            span.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const markupId = span.dataset.markupId;
                if (!markupId) return;

                popup.textContent = '✕ Remove markup';
                popup.style.display = 'block';
                popup.style.left = (e.clientX) + 'px';
                popup.style.top = (e.clientY + 4) + 'px';
                popup.onclick = async (evt) => {
                    evt.stopPropagation();
                    popup.style.display = 'none';

                    const originalDisplay = span.dataset.originalDisplay || '';
                    const textNode = document.createTextNode(originalDisplay);
                    span.replaceWith(textNode);
                    this._scheduleAutoSave(editor);
                };
            });
        });

        document.addEventListener('click', () => {
            clearTimeout(popup._hideTimer);
            popup.style.display = 'none';
        });
    }

    async _mergeChunk(hexId, direction) {
        try {
            const resp = await fetch(`${SERVER_URL}/api/project/merge-chunk`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ hex_id: hexId, direction })
            });
            const result = await resp.json();
            if (!resp.ok || result.error) {
                showToast('Merge failed: ' + (result.error || 'Unknown error'), 'error');
                return;
            }
            await this.loadMarkdown();
            showToast('Chunks merged', 'success');
        } catch (e) {
            showToast('Error merging chunks: ' + e.message, 'error');
        }
    }

    /**
     * Schedule auto-save after debounce period
     */
    _scheduleAutoSave(editor) {
        if (this._autoSaveTimer) clearTimeout(this._autoSaveTimer);
        this._autoSaveTimer = setTimeout(async () => {
            // Don't auto-save at stage 3 when text is locked
            if (this._isStage3() && !this._textUnlocked) return;

            const currentText = editor.textContent || editor.innerText;
            if (currentText === this._lastSavedContent) return;
            if (currentText.length === 0) return;

            // Push to undo stack before saving
            if (this._lastSavedContent && this._lastSavedContent !== currentText) {
                this.undoStack.push(this._lastSavedContent);
                if (this.undoStack.length > this.maxUndoLevels) {
                    this.undoStack.shift();
                }
                this._updateUndoButton();
            }

            console.log('[EDITOR] Auto-saving...');
            await this._performSave(currentText);
        }, this._autoSaveDelay);
    }

    /**
     * Undo last edit (restore from undo stack)
     */
    async undo() {
        if (this.undoStack.length === 0) {
            showToast('Nothing to undo', 'error');
            return;
        }
        const previous = this.undoStack.pop();
        this._updateUndoButton();

        const editor = document.getElementById('pseudoXmlEditor');
        this.markdownContent = previous;
        this._lastSavedContent = previous;
        this.displayMarkdown();

        // Save the reverted content
        await this._performSave(previous);
        showToast('Undone', 'success');
    }

    _updateUndoButton() {
        const btn = document.getElementById('undoBtn');
        if (btn) {
            btn.disabled = this.undoStack.length === 0;
            btn.title = this.undoStack.length > 0
                ? `Undo (${this.undoStack.length} available)`
                : 'Nothing to undo';
        }
    }

    /**
     * Internal save — sends markdown to server, refreshes TTS dirty indicators
     */
    async _performSave(markdownContent) {
        try {
            // Use book-file endpoint for new format, fall back to save-markdown for legacy
            const endpoint = this._useBookFile
                ? `${SERVER_URL}/api/project/save-book-file`
                : `${SERVER_URL}/api/project/save-markdown`;
            const body = this._useBookFile
                ? JSON.stringify({ content: markdownContent })
                : JSON.stringify({ markdown_content: markdownContent });

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body
            });

            if (response.ok) {
                this.markdownContent = markdownContent;
                this._lastSavedContent = markdownContent;

                // Refresh chapters data silently (no re-render of editor)
                const chapResp = await fetch(`${SERVER_URL}/api/project/get-text-files`, {
                    headers: { 'X-API-Key': API_KEY }
                });
                if (chapResp.ok) {
                    const data = await chapResp.json();
                    this.chapters = data.chapters || [];
                }

                // Notify TTS tab to update dirty indicators AND reload current chapter
                if (typeof ttsTab !== 'undefined') {
                    if (ttsTab.updateDirtyIndicators) {
                        ttsTab.updateDirtyIndicators(this.chapters);
                    }
                    // Reload the currently displayed chapter so emotion tags are detected
                    if (ttsTab.currentChapterIndex !== null) {
                        await ttsTab.loadChapter(ttsTab.currentChapterIndex);
                    }
                }

                // Refresh Reader tab if available
                if (typeof readerTab !== 'undefined' && readerTab.refresh) {
                    readerTab.refresh();
                }

                console.log('[EDITOR] Auto-saved successfully');
            } else {
                const error = await response.json();
                console.error('[EDITOR] Auto-save failed:', error.error);
            }
        } catch (error) {
            console.error('[EDITOR] Auto-save error:', error);
        }
    }

    async loadDefaultUrl() {
        try {
            const response = await fetch(`${SERVER_URL}/api/config`);
            if (response.ok) {
                const config = await response.json();
                this.defaultGutenbergUrl = config.default_gutenberg_url || 'https://www.gutenberg.org/cache/epub/4932/pg4932.txt';
                console.log('[GUTENBERG] Default URL loaded:', this.defaultGutenbergUrl);
            }
        } catch (error) {
            console.error('Error loading default URL:', error);
            this.defaultGutenbergUrl = 'https://www.gutenberg.org/cache/epub/4932/pg4932.txt';
        }
    }

    async loadMarkdown() {
        try {
            // Try book-file (new format) first
            const bookResp = await fetch(`${SERVER_URL}/api/project/book-file`, {
                headers: { 'X-API-Key': API_KEY }
            });

            if (bookResp.ok) {
                const data = await bookResp.json();
                this._useBookFile = true;
                this.markdownContent = data.content || '';
                this._lastSavedContent = this.markdownContent;
                this._bookFileStage = data.stage || 0;
                console.log('[LOAD] Loaded book.txt (stage', this._bookFileStage, ')');

                // Also fetch chapters for TTS/reader tab awareness
                const chapResp = await fetch(`${SERVER_URL}/api/project/get-text-files`, {
                    headers: { 'X-API-Key': API_KEY }
                });
                if (chapResp.ok) {
                    const chapData = await chapResp.json();
                    this.chapters = chapData.chapters || [];
                }
            } else {
                // Fall back to legacy markdown from chapters
                this._useBookFile = false;
                const response = await fetch(`${SERVER_URL}/api/project/get-text-files`, {
                    headers: { 'X-API-Key': API_KEY }
                });

                if (response.ok) {
                    const data = await response.json();
                    this.chapters = data.chapters || [];
                    console.log('[LOAD MARKDOWN] Loaded chapters (legacy):', this.chapters.length);

                    if (this.chapters.length > 0) {
                        this.markdownContent = this.chaptersToMarkdown(this.chapters);
                        this._lastSavedContent = this.markdownContent;
                    } else {
                        this.markdownContent = '';
                        this._lastSavedContent = '';
                    }
                }
            }

            this.displayMarkdown();
        } catch (error) {
            console.error('Error loading markdown:', error);
        }
    }

    _isStage1() { return this._bookFileStage === 1; }
    _isStage2() { return this._bookFileStage === 2; }
    _isStage3() { return this._bookFileStage >= 3; }

    /**
     * Display markdown in the editor (or structure editor at stage 1)
     */
    displayMarkdown() {
        const structureContainer = document.getElementById('structureEditorContainer');
        const editorContainer    = document.getElementById('markdownEditorContainer');
        const editor             = document.getElementById('pseudoXmlEditor');

        this._updateStageToolbar();

        if (this._isStage1()) {
            // Show structure editor, hide markdown editor
            if (structureContainer) structureContainer.style.display = '';
            if (editorContainer)    editorContainer.style.display    = 'none';
            this._renderStructureEditor();
            return;
        }

        // Normal markdown editor (stages 0, 2, 3)
        if (structureContainer) structureContainer.style.display = 'none';
        if (editorContainer)    editorContainer.style.display    = '';

        if (!this.markdownContent || this._bookFileStage === 0) {
            editor.innerHTML = '<div class="xml-empty-state">No content. Click Load Book to begin.</div>';
            editor.contentEditable = 'false';
            return;
        }

        // Stage 3: always editable (for tag/chunk manipulation), but plain text is protected by event handlers
        const editable = true; // Always true at stage 3 to allow tag manipulation

        if (this.cleanViewActive) {
            let cleanText = this.markdownContent;
            cleanText = cleanText.replace(/<delete>[\s\S]*?<\/delete>/g, '');
            cleanText = cleanText.replace(/<chapter[^>]*>\n?/g, '');
            cleanText = cleanText.replace(/<\/chapter>/g, '');
            cleanText = cleanText.replace(/\{([^|}]*)\|[^}]*\}/g, '$1');
            // Strip both old </chunk> and new <chunk id="...">...</chunk> formats
            cleanText = cleanText.replace(/<chunk[^>]*>/g, '');
            cleanText = cleanText.replace(/<\/chunk>/g, '');
            cleanText = cleanText.replace(/^#{2,3}\s+.+$/gm, '');
            const escaped = cleanText
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            editor.innerHTML = escaped;
            editor.contentEditable = 'false';
        } else {
            let escaped = this.markdownContent
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Highlight <delete>...</delete> blocks (red)
            escaped = escaped.replace(
                /(&lt;delete&gt;)([\s\S]*?)(&lt;\/delete&gt;)/g,
                '<span style="background:#fee2e2;color:#991b1b;border-radius:3px;padding:0 2px">$1$2$3</span>'
            );

            // Highlight <chapter ...> and </chapter> tags (blue)
            escaped = escaped.replace(
                /(&lt;chapter[^&]*&gt;)/g,
                '<span style="background:#dbeafe;color:#1e40af;border-radius:3px;padding:0 3px;font-weight:600">$1</span>'
            );
            escaped = escaped.replace(
                /(&lt;\/chapter&gt;)/g,
                '<span style="background:#dbeafe;color:#1e40af;border-radius:3px;padding:0 3px;font-weight:600">$1</span>'
            );

            // Highlight legacy markdown ## chapter headers
            escaped = escaped.replace(
                /(^#{2,3}\s+.+$)/gm,
                '<span style="background:#dbeafe;color:#1e40af;font-weight:600">$1</span>'
            );

            // Highlight pronunciation/paralinguistic markup: {display|spoken}
            escaped = escaped.replace(
                /\{([^|}]*)\|([^}]*)\}/g,
                (match, display, spoken) => {
                    const origDisplay = display.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"');
                    const markupId = Math.random().toString(16).substr(2, 8);
                    return `<span class="pp-markup" data-markup-id="${markupId}" data-original-display="${origDisplay.replace(/"/g, '&quot;')}" style="background:#ddd6fe;border-radius:3px;padding:0 2px;display:inline-block;cursor:pointer">{${display}|<span style="color:#7c3aed;font-weight:600">${spoken}</span>}</span>`;
                }
            );

            // Highlight <chunk id="..."> open tags — carry hex ID as data attribute for popup
            escaped = escaped.replace(
                /&lt;chunk\s+id="([a-f0-9]{8})"([^&]*)&gt;/g,
                (_, hexId, rest) =>
                    `<span class="chunk-open-tag" data-hex-id="${hexId}" style="background:#fde68a;color:#92400e;border-radius:3px;padding:0 3px;font-size:0.85em;cursor:pointer;">&lt;chunk id="${hexId}"${rest}&gt;</span>`
            );

            // Highlight </chunk> tags
            {
                const chunkIdSeq = [];
                (this.markdownContent.match(/<chunk\s+id="([a-f0-9]{8})"/g) || []).forEach(m => {
                    const hm = m.match(/id="([a-f0-9]{8})"/);
                    if (hm) chunkIdSeq.push(hm[1]);
                });
                let closeIdx = 0;
                escaped = escaped.replace(
                    /&lt;\/chunk&gt;/g,
                    () => {
                        const hexId = chunkIdSeq[closeIdx++] || '';
                        return `<span class="chunk-close-tag" data-hex-id="${hexId}" style="background:#fde68a;color:#92400e;border-radius:3px;padding:0 3px;font-size:0.85em;cursor:pointer;">&lt;/chunk&gt;</span>`;
                    }
                );
            }

            editor.innerHTML = escaped;
            editor.contentEditable = editable ? 'true' : 'false';

            // Attach popup menu handlers after setting innerHTML
            this._attachChunkPopupMenus(editor);
            this._attachMarkupPopupMenus(editor);
        }
    }

    /**
     * Convert chapters array to markdown format
     */
    chaptersToMarkdown(chapters) {
        let markdown = '';

        for (let i = 0; i < chapters.length; i++) {
            const chapter = chapters[i];
            const title = chapter.title || chapter.name || 'Untitled';
            const isNonVoiced = chapter.non_voiced || false;

            if (i > 0) {
                markdown += '\n\n';
            }

            if (isNonVoiced) {
                markdown += `### ${title} [non-voiced]\n`;
            } else {
                markdown += `## ${title}\n`;
            }

            if (chapter.chunks && chapter.chunks.length > 0) {
                let startIdx = 0;
                if (chapter.chunks[0]?.type === 'text' && chapter.chunks[0]?.text === title) {
                    startIdx = 1;
                }

                for (let j = startIdx; j < chapter.chunks.length; j++) {
                    const chunk = chapter.chunks[j];

                    if (chunk.type === 'pause') {
                        markdown += `[pause:${chunk.duration || 1.0}]</chunk>`;
                    } else if (chunk.type === 'common_file') {
                        markdown += `[file:${chunk.path || ''}]</chunk>`;
                    } else {
                        const text = chunk.text || '';
                        markdown += text;
                        if (j < chapter.chunks.length - 1) {
                            markdown += '</chunk>';
                        }
                    }
                }
            }
        }

        return markdown.trim();
    }

    /**
     * Manual save (kept for backward compat but auto-save handles most cases)
     */
    async saveCode() {
        const editor = document.getElementById('pseudoXmlEditor');
        const markdownContent = editor.textContent || editor.innerText;

        // Push undo
        if (this._lastSavedContent && this._lastSavedContent !== markdownContent) {
            this.undoStack.push(this._lastSavedContent);
            if (this.undoStack.length > this.maxUndoLevels) {
                this.undoStack.shift();
            }
            this._updateUndoButton();
        }

        await this._performSave(markdownContent);

        // Full refresh of TTS and Reader tabs
        if (typeof ttsTab !== 'undefined' && ttsTab.refreshChapters) {
            await ttsTab.refreshChapters();
            if (ttsTab.currentChapterIndex !== null) {
                await ttsTab.loadChapter(ttsTab.currentChapterIndex);
            }
        }
        if (typeof readerTab !== 'undefined' && readerTab.refresh) {
            await readerTab.refresh();
        }

        showToast('Saved successfully!', 'success');
    }

    // Insert a new chapter break at cursor
    insertChapterBreak() {
        const editor = document.getElementById('pseudoXmlEditor');
        const insertText = '\n\n## New Chapter\n\nChapter text goes here.\n';
        document.execCommand('insertText', false, insertText);
        editor.focus();
    }

    // Insert a chunk boundary at cursor
    insertChunkBreak() {
        const editor = document.getElementById('pseudoXmlEditor');
        const insertText = '\n\n</chunk>\n\n';
        document.execCommand('insertText', false, insertText);
        editor.focus();
    }

    // Insert a pause marker at cursor
    insertPause() {
        const editor = document.getElementById('pseudoXmlEditor');
        const insertText = '\n\n[pause:1.0]\n\n</chunk>\n\n';
        document.execCommand('insertText', false, insertText);
        editor.focus();
    }

    // Insert a pronunciation override: requires highlighted text.
    // Creates {display|spoken} where spoken defaults to the same as display.
    // Automatically selects the spoken part for immediate editing.
    insertPronunciation() {
        if (this.cleanViewActive) {
            showToast('Switch to Markup View to insert pronunciation markers', 'error');
            return;
        }
        if (this._isStage1()) {
            showToast('Confirm the chapter structure first before adding pronunciation', 'error');
            return;
        }
        // Capture selected text BEFORE restoring selection (in case saved range has it)
        const selectedText = this._savedRange ? this._savedRange.toString() : '';
        if (!selectedText) {
            showToast('Select the word or phrase to annotate first, then click Pronunciation', 'error');
            return;
        }
        this._restoreEditorSelection();
        // Insert the markup
        document.execCommand('insertText', false, `{${selectedText}|${selectedText}}`);

        // After insertion, trigger highlight to render the markup, then select the spoken text
        const editor = document.getElementById('pseudoXmlEditor');
        setTimeout(() => {
            this._liveHighlight(editor);
            // Find and select the replacement text (the part after |)
            const raw = editor.textContent || editor.innerText;
            const lastMarkup = `{${selectedText}|${selectedText}}`;
            const idx = raw.lastIndexOf(lastMarkup);
            if (idx !== -1) {
                const startPos = idx + selectedText.length + 2; // {display|
                const endPos = startPos + selectedText.length;
                this._selectRange(editor, startPos, endPos);
            }
        }, 10);
    }

    /**
     * Select a range of text by character offset in editor.
     */
    _selectRange(editor, startOffset, endOffset) {
        const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null, false);
        let offset = 0;
        let startNode = null, startNodeOffset = null;
        let endNode = null, endNodeOffset = null;
        let node;

        while ((node = walker.nextNode())) {
            const len = node.textContent.length;
            if (!startNode && offset + len >= startOffset) {
                startNode = node;
                startNodeOffset = startOffset - offset;
            }
            if (!endNode && offset + len >= endOffset) {
                endNode = node;
                endNodeOffset = endOffset - offset;
                break;
            }
            offset += len;
        }

        if (startNode && endNode) {
            const sel = window.getSelection();
            const range = document.createRange();
            range.setStart(startNode, startNodeOffset);
            range.setEnd(endNode, endNodeOffset);
            sel.removeAllRanges();
            sel.addRange(range);
        }
    }

    // Insert a paralinguistic tag at cursor: {|[tag]}
    insertParalinguisticTag(tag) {
        if (this.cleanViewActive) {
            showToast('Switch to Markup View to insert paralinguistic tags', 'error');
            return;
        }
        this._restoreEditorSelection();
        document.execCommand('insertText', false, `{|[${tag}]}`);
    }

    // Toggle between clean view and markup view
    toggleCleanView() {
        const editor = document.getElementById('pseudoXmlEditor');
        const btn = document.getElementById('cleanViewToggle');

        if (!this.cleanViewActive) {
            // Switching TO clean view: save current editor content first
            this.markdownContent = editor.textContent || editor.innerText;
            this.cleanViewActive = true;
            if (btn) {
                btn.textContent = 'Markup View';
                btn.style.background = '#7c3aed';
            }
        } else {
            // Switching back TO markup view
            this.cleanViewActive = false;
            if (btn) {
                btn.textContent = 'Clean View';
                btn.style.background = '';
            }
        }
        this.displayMarkdown();
    }

    // ----------------------------------------------------------------
    // Chapter Locking
    // ----------------------------------------------------------------

    async checkLockState() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/info`, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (response.ok) {
                const info = await response.json();
                this.chaptersLocked = info.metadata?.chapters_locked || false;
                this._updateLockUI();
            }
        } catch (e) {
            console.error('Error checking lock state:', e);
        }
    }

    async lockChapters() {
        if (this.chapters.length === 0) {
            showToast('No chapters to lock', 'error');
            return;
        }
        if (!confirm('Lock chapter divisions? This saves the original text with chapter boundaries. You can unlock later to re-parse.')) return;

        try {
            const response = await fetch(`${SERVER_URL}/api/project/lock-chapters`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY }
            });
            if (response.ok) {
                this.chaptersLocked = true;
                this._updateLockUI();
                showToast('Chapters locked. Original text saved.', 'success');
            } else {
                const err = await response.json();
                showToast('Lock failed: ' + (err.error || 'Unknown error'), 'error');
            }
        } catch (e) {
            showToast('Lock error: ' + e.message, 'error');
        }
    }

    async unlockChapters() {
        if (!confirm('Unlock chapters? This allows re-parsing but will not restore the original text.')) return;

        try {
            const response = await fetch(`${SERVER_URL}/api/project/unlock-chapters`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY }
            });
            if (response.ok) {
                this.chaptersLocked = false;
                this._updateLockUI();
                showToast('Chapters unlocked. You can now re-parse.', 'success');
            } else {
                const err = await response.json();
                showToast('Unlock failed: ' + (err.error || 'Unknown error'), 'error');
            }
        } catch (e) {
            showToast('Unlock error: ' + e.message, 'error');
        }
    }

    _updateLockUI() {
        const lockBtn = document.getElementById('lockChaptersBtn');
        const unlockBtn = document.getElementById('unlockChaptersBtn');
        const lockStatus = document.getElementById('chapterLockStatus');

        if (this.chaptersLocked) {
            if (lockBtn) lockBtn.style.display = 'none';
            if (unlockBtn) unlockBtn.style.display = '';
            if (lockStatus) {
                lockStatus.style.display = '';
                lockStatus.textContent = '🔒 Text locked — chunks and pronunciation tags are still editable';
            }
        } else {
            if (lockBtn) lockBtn.style.display = '';
            if (unlockBtn) unlockBtn.style.display = 'none';
            if (lockStatus) lockStatus.style.display = 'none';
        }
    }

    async validateChunks() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/validate-chunks`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({})
            });

            if (response.ok) {
                const result = await response.json();
                if (result.oversized_chunks && result.oversized_chunks.length > 0) {
                    const msg = `Found ${result.oversized_chunks.length} oversized chunks:\n\n` +
                        result.oversized_chunks.map(c =>
                            `Chapter: ${c.chapter_title}, Chunk ${c.chunk_id}: ${c.size} chars`
                        ).join('\n');
                    alert(msg);
                } else {
                    alert('All chunks are valid!');
                }
            } else {
                throw new Error('Failed to validate chunks');
            }
        } catch (error) {
            console.error('Error validating chunks:', error);
            alert('Error validating chunks: ' + error.message);
        }
    }

    async autoRechunk() {
        try {
            const confirmation = confirm('This will automatically split oversized chunks. Continue?');
            if (!confirmation) return;

            const response = await fetch(`${SERVER_URL}/api/project/auto-rechunk`, {
                method: 'POST',
                headers: { 'X-API-Key': API_KEY }
            });

            if (response.ok) {
                const result = await response.json();
                alert(`Rechunking complete! Fixed ${result.chunks_fixed || 0} chunks.`);
                await this.loadMarkdown();
            } else {
                throw new Error('Failed to auto-rechunk');
            }
        } catch (error) {
            console.error('Error auto-rechunking:', error);
            alert('Error auto-rechunking: ' + error.message);
        }
    }

    // ------------------------------------------------------------------ //
    //  Stage-1 structure editor helpers                                   //
    // ------------------------------------------------------------------ //

    _renderStructureEditor() {
        const container = document.getElementById('structureEditorContainer');
        if (!container) return;

        if (!this.structureEditor) {
            this.structureEditor = new StructureEditor(container);
        } else {
            // Re-use existing instance but point at possibly-new container
            this.structureEditor.container = container;
        }

        this.structureEditor.parse(this.markdownContent || '');
        this.structureEditor.render();
    }

    /** Show the correct toolbar and header buttons for the current stage. */
    _updateStageToolbar() {
        const stage = this._bookFileStage || 0;

        // Toolbar divs
        const toolbars = {
            stage0Toolbar: stage === 0,
            stage1Toolbar: stage === 1,
            stage2Toolbar: stage === 2,
            stage3Toolbar: stage >= 3,
        };
        for (const [id, visible] of Object.entries(toolbars)) {
            const el = document.getElementById(id);
            if (el) el.style.display = visible ? (el.id === 'stage3Toolbar' ? 'flex' : '') : 'none';
        }

        // Header buttons (loadBook / confirmStructure / runChunking only;
        // editTextBtn / saveBtn / undoBtn now live inside stage3Toolbar)
        const btns = {
            loadBookBtn:         stage === 0,
            confirmStructureBtn: stage === 1,
            runChunkingBtn:      stage === 2,
        };
        for (const [id, visible] of Object.entries(btns)) {
            const el = document.getElementById(id);
            if (el) el.style.display = visible ? '' : 'none';
        }

        // Lock toggle button (in stage3Toolbar)
        const editBtn = document.getElementById('editTextBtn');
        if (editBtn) {
            if (this._textUnlocked) {
                editBtn.textContent = '🔓 Unlocked';
                editBtn.style.background = '#fee2e2';
                editBtn.style.color = '#991b1b';
            } else {
                editBtn.textContent = '🔒 Locked';
                editBtn.style.background = '#e2e8f0';
                editBtn.style.color = '#475569';
            }
        }

        // Editing buttons: only visible when text is unlocked
        const editOnlyBtns = ['insertPronunciationBtn', 'paralinguisticSelect', 'splitChunkBtn'];
        editOnlyBtns.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = this._textUnlocked ? '' : 'none';
        });

        // Pane subtitle
        const subtitle = document.getElementById('markdownPaneSubtitle');
        if (subtitle) {
            const labels = { 0: '', 1: 'Step 1 — Review structure', 2: 'Step 2 — Chunking…', 3: 'Step 3 — Pronunciation & TTS' };
            subtitle.textContent = labels[Math.min(stage, 3)] || '';
        }

        // Gate TTS pane on chunking completion (stage ≥ 3)
        const ttsToggle = document.getElementById('toggle-tts');
        const ttsToggleItem = ttsToggle?.closest('.toggle-item');
        if (ttsToggle && ttsToggleItem) {
            if (stage < 3) {
                ttsToggle.checked = false;
                ttsToggle.disabled = true;
                ttsToggleItem.style.opacity = '0.4';
                ttsToggleItem.style.pointerEvents = 'none';
                ttsToggleItem.title = 'Available after chunking';
            } else {
                ttsToggle.disabled = false;
                ttsToggleItem.style.opacity = '';
                ttsToggleItem.style.pointerEvents = '';
                ttsToggleItem.title = '';
            }
        }
        if (typeof updatePaneVisibility === 'function') updatePaneVisibility();
    }

    /** Auto-collapse heading blocks in the structure editor */
    autoCollapseHeadings() {
        if (!this.structureEditor) return;
        const removed = this.structureEditor.autoCollapseHeadings();
        if (removed > 0) {
            showToast(`Collapsed ${removed} heading block${removed === 1 ? '' : 's'}`, 'success');
            this._saveStructureEditorContent();
        } else {
            showToast('No short heading blocks found to collapse', 'info');
        }
    }

    /** Save current structure-editor state back to book.txt */
    async _saveStructureEditorContent() {
        if (!this.structureEditor) return;
        const content = this.structureEditor.serialise();
        this.markdownContent = content;
        this._lastSavedContent = content;
        await this._performSave(content);
    }

    /**
     * Confirm the stage-1 structure: convert to stage-2 chapter-wrapped format.
     * Called when user clicks "OK / Confirm Structure".
     */
    async confirmStructure() {
        if (!this.structureEditor) {
            showToast('No structure to confirm', 'error');
            return;
        }

        // Save any unsaved drag changes first
        await this._saveStructureEditorContent();

        // Retrieve current book.txt content (already saved above)
        const content = this.markdownContent;

        // Count chapter markers before confirming
        const markerCount = (content.match(/<chapter\s+title="/g) || []).length;
        if (markerCount === 0) {
            showToast('No chapter breaks found. Add at least one chapter break before confirming.', 'error');
            return;
        }

        if (!confirm(
            `Confirm structure?\n\n` +
            `• ${markerCount} chapter(s) will be created\n` +
            `• Sections marked for deletion will be removed\n\n` +
            `You can still edit text and pronunciation after this step.`
        )) return;

        try {
            const resp = await fetch(`${SERVER_URL}/api/project/confirm-structure`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ content })
            });
            const result = await resp.json();
            if (!resp.ok || result.error) {
                showToast('Confirm failed: ' + (result.error || 'Unknown error'), 'error');
                return;
            }

            // Stage 2 is a transient internal state — immediately run chunking
            this._bookFileStage = 2;
            this._updateStageToolbar();
            showToast(`Structure confirmed — running chunking…`, 'info');

            const chunkResp = await fetch(`${SERVER_URL}/api/project/tag-chunks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY }
            });
            const chunkResult = await chunkResp.json();
            if (!chunkResp.ok || chunkResult.error) {
                showToast('Chunking failed: ' + (chunkResult.error || 'Unknown error'), 'error');
                return;
            }

            this._bookFileStage = 3;
            this._textUnlocked = false;
            await this.loadMarkdown();
            if (typeof ttsTab !== 'undefined' && ttsTab.refreshChapters) {
                await ttsTab.refreshChapters();
            }
            showToast(`Done — ${chunkResult.chapter_count} chapters chunked and ready for TTS.`, 'success');
        } catch (e) {
            showToast('Error confirming structure: ' + e.message, 'error');
        }
    }

    /** Toggle text lock at stage 3 (now just disables the protection, keeping tags editable) */
    toggleTextLock() {
        this._textUnlocked = !this._textUnlocked;
        this._updateStageToolbar();
        if (this._textUnlocked) {
            showToast('Text protection disabled — you can now edit all text', 'warning');
        } else {
            showToast('Text protection enabled — only tags and chunks are editable', 'success');
        }
    }

    /** Run chunking on stage-2 book.txt, advancing to stage 3 */
    async runChunking() {
        if (!confirm('Run automatic chunking? This will split each chapter into TTS-sized segments.')) return;

        try {
            const resp = await fetch(`${SERVER_URL}/api/project/tag-chunks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY }
            });
            const result = await resp.json();
            if (!resp.ok || result.error) {
                showToast('Chunking failed: ' + (result.error || 'Unknown error'), 'error');
                return;
            }
            this._bookFileStage = 3;
            this._textUnlocked = false;
            await this.loadMarkdown();
            if (typeof ttsTab !== 'undefined' && ttsTab.refreshChapters) {
                await ttsTab.refreshChapters();
            }
            showToast(`Chunking complete — ${result.chapter_count} chapters ready for TTS.`, 'success');
        } catch (e) {
            showToast('Error running chunking: ' + e.message, 'error');
        }
    }

    /**
     * Split the chunk at cursor position (stage 3).
     * Reads the cursor's character offset in the editor, maps it to a chunk ID
     * and offset within that chunk, then calls the server split-chunk endpoint.
     */
    async splitChunkAtCursor() {
        if (!this._isStage3()) {
            showToast('Chunking must be complete before splitting', 'error');
            return;
        }

        const editor = document.getElementById('pseudoXmlEditor');
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0) {
            showToast('Position cursor inside a chunk first', 'error');
            return;
        }

        // Get cursor offset in the plain text
        const range = sel.getRangeAt(0);
        const preRange = document.createRange();
        preRange.selectNodeContents(editor);
        preRange.setEnd(range.startContainer, range.startOffset);
        const globalOffset = preRange.toString().length;

        // Parse the raw text to find which chunk we're in and where
        const raw = editor.textContent || editor.innerText;
        // Match new-format chunks: <chunk id="XXXXXXXX">text</chunk>
        const chunkRe = /<chunk\s+id="([a-f0-9]{8})"[^>]*>([\s\S]*?)<\/chunk>/g;
        let m;
        let found = null;
        let pos = 0;

        while ((m = chunkRe.exec(raw)) !== null) {
            const tagStart = m.index;
            const contentStart = tagStart + m[0].indexOf('>') + 1;
            const contentEnd = contentStart + m[2].length;

            // Approximate: map global offset to within this chunk
            // (editor shows escaped/highlighted content, so use raw text positions)
            if (globalOffset >= contentStart && globalOffset <= contentEnd) {
                found = {
                    hex_id: m[1],
                    char_offset: globalOffset - contentStart
                };
                break;
            }
        }

        if (!found) {
            showToast('Cursor must be inside a chunk (between <chunk> tags)', 'error');
            return;
        }

        try {
            const resp = await fetch(`${SERVER_URL}/api/project/split-chunk`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ hex_id: found.hex_id, char_offset: found.char_offset })
            });
            const result = await resp.json();
            if (!resp.ok || result.error) {
                showToast('Split failed: ' + (result.error || 'Unknown error'), 'error');
                return;
            }
            await this.loadMarkdown();
            showToast('Chunk split successfully', 'success');
        } catch (e) {
            showToast('Error splitting chunk: ' + e.message, 'error');
        }
    }

    async loadGutenbergUrl(prefilledUrl) {
        const url = prefilledUrl !== undefined
            ? prefilledUrl
            : prompt('Enter Project Gutenberg URL:', this.defaultGutenbergUrl);
        if (!url) return;

        console.log('[GUTENBERG] Loading URL:', url);

        // ── Step 1: scan newline distribution ──────────────────────────────
        let scanResult;
        try {
            const scanResp = await fetch(`${SERVER_URL}/api/project/add-gutenberg-url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ url, scan_only: true })
            });
            if (!scanResp.ok) {
                const err = await scanResp.json();
                throw new Error(err.error || 'Scan failed');
            }
            scanResult = await scanResp.json();
        } catch (error) {
            console.error('Error scanning Gutenberg text:', error);
            alert('Error scanning Gutenberg text: ' + error.message);
            return;
        }

        // ── Step 2: show newline-rules dialog ──────────────────────────────
        const rules = await this._showNewlineRulesDialog(
            scanResult.newline_distribution,
            scanResult.default_rules
        );
        if (rules === null) return;  // user cancelled

        // ── Step 3: process with chosen rules ─────────────────────────────
        try {
            const response = await fetch(`${SERVER_URL}/api/project/add-gutenberg-url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({ url, newline_rules: rules })
            });

            console.log('[GUTENBERG] Response status:', response.status);

            if (response.ok) {
                const result = await response.json();
                const markerCount = result.chapter_marker_count || 0;
                const shortCount  = result.short_block_count || 0;
                console.log('[GUTENBERG] Stage-1 book loaded, blocks:', markerCount, 'short:', shortCount);

                if (result.debug) {
                    const d = result.debug;
                    console.log('[LOAD] Book ID:', d.book_id, '| Markers:', markerCount,
                        '| EPUB titles:', d.epub_titles?.length || 0,
                        '| Source:', d.chapter_source);
                }

                await this.loadMarkdown();

                showToast(
                    `Loaded! ${markerCount} blocks detected ` +
                    `(${shortCount} short headings highlighted in yellow). ` +
                    `Click ⚡ Collapse Headings, then review and lock.`,
                    'success'
                );
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to load Gutenberg text');
            }
        } catch (error) {
            console.error('Error loading Gutenberg text:', error);
            alert('Error loading Gutenberg text: ' + error.message);
        }
    }

    /**
     * Show a modal asking the user how to treat each newline count ≥2.
     * Returns a rules object { "2": action, "3": action, ... } or null if cancelled.
     */
    _showNewlineRulesDialog(distribution, defaultRules) {
        return new Promise(resolve => {
            // Build sorted list of distinct counts present in the text
            const allKeys = Object.keys(distribution).sort((a, b) => {
                const na = a.endsWith('+') ? 999 : parseInt(a);
                const nb = b.endsWith('+') ? 999 : parseInt(b);
                return na - nb;
            });

            const ACTIONS = [
                { value: 'chapter',   label: 'Chapter break' },
                { value: 'paragraph', label: 'Paragraph separator' },
                { value: 'trailing',  label: 'Trailing space after chapter' },
                { value: 'ignore',    label: 'Ignore (remove)' },
            ];

            const getDefault = key => {
                if (key in defaultRules) return defaultRules[key];
                const n = parseInt(key);
                if (!isNaN(n)) {
                    for (let k = n; k >= 2; k--) {
                        if (String(k) in defaultRules) return defaultRules[String(k)];
                    }
                }
                return defaultRules['4+'] || 'chapter';
            };

            // Build rows
            const rows = allKeys.map(key => {
                const count = distribution[key];
                const def = getDefault(key);
                const radioName = `nl_${key}`;
                const radios = ACTIONS.map(a =>
                    `<label style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;cursor:pointer;">
                        <input type="radio" name="${radioName}" value="${a.value}"${a.value === def ? ' checked' : ''}>
                        ${a.label}
                    </label>`
                ).join('');
                const countStr = count.toLocaleString();
                return `<tr style="border-bottom:1px solid #e2e8f0;">
                    <td style="padding:8px 12px;white-space:nowrap;font-family:monospace;font-weight:600;color:#92400e;">
                        ${key} newline${key === '1' ? '' : 's'}
                        <span style="font-weight:normal;color:#64748b;font-family:sans-serif;font-size:11px;"> (${countStr}×)</span>
                    </td>
                    <td style="padding:8px 12px;">${radios}</td>
                </tr>`;
            }).join('');

            const modal = document.createElement('div');
            modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);z-index:10000;display:flex;align-items:center;justify-content:center;';
            modal.innerHTML = `
                <div style="background:white;border-radius:12px;padding:28px;max-width:680px;width:95%;max-height:90vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,0.4);">
                    <h3 style="margin:0 0 6px;color:#1e293b;font-size:17px;">Newline handling</h3>
                    <p style="margin:0 0 16px;color:#64748b;font-size:13px;">
                        Choose what to do with each run of consecutive blank lines found in this text.
                        Single \\n (hard line-wrap) is always converted to a space.
                    </p>
                    <table style="width:100%;border-collapse:collapse;font-size:13px;">
                        <thead>
                            <tr style="background:#f8fafc;">
                                <th style="padding:6px 12px;text-align:left;color:#475569;font-size:12px;">Blank lines</th>
                                <th style="padding:6px 12px;text-align:left;color:#475569;font-size:12px;">Treatment</th>
                            </tr>
                        </thead>
                        <tbody>${rows || '<tr><td colspan="2" style="padding:12px;color:#94a3b8;">No multi-line runs found.</td></tr>'}</tbody>
                    </table>
                    <div style="margin-top:20px;display:flex;justify-content:flex-end;gap:10px;">
                        <button id="nlCancel" style="padding:8px 18px;border:1px solid #cbd5e1;border-radius:6px;background:white;cursor:pointer;font-size:14px;">Cancel</button>
                        <button id="nlProcess" style="padding:8px 18px;border:none;border-radius:6px;background:#1e293b;color:white;cursor:pointer;font-size:14px;font-weight:600;">Process book</button>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);

            modal.querySelector('#nlCancel').onclick = () => {
                modal.remove();
                resolve(null);
            };
            modal.querySelector('#nlProcess').onclick = () => {
                const result = {};
                allKeys.forEach(key => {
                    const checked = modal.querySelector(`input[name="nl_${key}"]:checked`);
                    if (checked) result[key] = checked.value;
                });
                modal.remove();
                resolve(result);
            };
            // Click outside = cancel
            modal.onclick = e => { if (e.target === modal) { modal.remove(); resolve(null); } };
        });
    }

}

// Create global instance
const gutenbergTab = new GutenbergTab();

// Global helper functions for onclick handlers
function loadGutenbergText() {
    gutenbergTab.loadGutenbergUrl();
}

function validateChunks() {
    gutenbergTab.validateChunks();
}

function showAnnotator() {
    alert('Annotator feature coming soon!');
}
