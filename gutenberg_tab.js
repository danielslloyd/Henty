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
 */

class GutenbergTab {
    constructor() {
        this.rawText = '';
        this.markdownContent = '';
        this.chapters = [];
        this.defaultGutenbergUrl = '';
        this.parsingMethods = {};
        this.cleanViewActive = false;
    }

    async init() {
        // Load default Gutenberg URL from config
        await this.loadDefaultUrl();
        // Load raw text from project if available
        await this.loadRawText();
        // Load markdown content from project
        await this.loadMarkdown();
        // Initialize parsing method selector
        await this.initParsingMethods();
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

    async loadRawText() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/raw-text`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.rawText = data.raw_text || '';
                this.displayRawText();
            }
        } catch (error) {
            console.error('Error loading raw text:', error);
        }
    }

    displayRawText() {
        const display = document.getElementById('rawTextDisplay');
        if (this.rawText) {
            display.textContent = this.rawText;
        } else {
            display.innerHTML = '<div style="color: #999; padding: 20px;">No text loaded. Load a Project Gutenberg text or upload a file to begin.</div>';
        }
    }

    async loadMarkdown() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/get-text-files`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.chapters = data.chapters || [];

                console.log('[LOAD MARKDOWN] Loaded chapters:', this.chapters.length);

                // Always generate markdown from chapters (chapters are the source of truth)
                if (this.chapters.length > 0) {
                    console.log('[LOAD MARKDOWN] Generating markdown from chapters...');
                    this.markdownContent = this.chaptersToMarkdown(this.chapters);
                    console.log('[LOAD MARKDOWN] Generated markdown first 500 chars:', this.markdownContent.substring(0, 500));
                } else {
                    this.markdownContent = '';
                }

                this.displayMarkdown();
            }
        } catch (error) {
            console.error('Error loading markdown:', error);
        }
    }

    /**
     * Display markdown in the editor
     * No escaping needed - text displays as-is
     */
    displayMarkdown() {
        const editor = document.getElementById('pseudoXmlEditor');

        if (!this.markdownContent) {
            editor.innerHTML = '<div class="xml-empty-state">No content. Load a Project Gutenberg text or upload a file to begin.</div>';
            return;
        }

        if (this.cleanViewActive) {
            // Clean view: strip {display|spoken} → display text only, hide </chunk> tags
            let cleanText = this.markdownContent;
            cleanText = cleanText.replace(/\{([^|}]+)\|[^}]*\}/g, '$1');
            cleanText = cleanText.replace(/<\/chunk>/g, '');
            const escaped = cleanText
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            editor.innerHTML = escaped;
            editor.contentEditable = 'false';
        } else {
            // Markup view: highlight {display|spoken} and </chunk> tags with color
            let escaped = this.markdownContent
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Highlight pronunciation markup: {display|spoken}
            escaped = escaped.replace(
                /\{([^|}]+)\|([^}]*)\}/g,
                '<span style="background:#ddd6fe;border-radius:3px;padding:0 2px">{$1|<span style="color:#7c3aed;font-weight:600">$2</span>}</span>'
            );

            // Highlight </chunk> tags
            escaped = escaped.replace(
                /&lt;\/chunk&gt;/g,
                '<span style="background:#fde68a;color:#92400e;border-radius:3px;padding:0 3px;font-size:0.85em;">&lt;/chunk&gt;</span>'
            );

            editor.innerHTML = escaped;
            editor.contentEditable = 'true';
        }
    }

    /**
     * Convert chapters array to markdown format
     * Format:
     *   ## Chapter Title      <- This becomes chunk 0 when parsed
     *   Chunk text here.</chunk>Next chunk text.</chunk>More text...
     *   [pause:1.5]</chunk>
     *   Text after pause...
     *
     * Notes:
     * - </chunk> markers are inline, not on separate lines
     * - Newlines in text represent real paragraph breaks (from <p> tags)
     * - The ## header line is automatically added as chunk 0 by the parser,
     *   so we skip outputting chunk 0 if its text matches the title.
     */
    chaptersToMarkdown(chapters) {
        let markdown = '';

        for (let i = 0; i < chapters.length; i++) {
            const chapter = chapters[i];
            const title = chapter.title || chapter.name || 'Untitled';
            const isNonVoiced = chapter.non_voiced || false;

            // Add blank line between chapters (except first)
            if (i > 0) {
                markdown += '\n\n';
            }

            // Chapter header (non-voiced uses ###)
            // This header becomes chunk 0 when parsed
            if (isNonVoiced) {
                markdown += `### ${title} [non-voiced]\n`;
            } else {
                markdown += `## ${title}\n`;
            }

            if (chapter.chunks && chapter.chunks.length > 0) {
                // Determine starting index - skip chunk 0 if it's the title
                let startIdx = 0;
                if (chapter.chunks[0]?.type === 'text' && chapter.chunks[0]?.text === title) {
                    startIdx = 1;  // Skip the title chunk since ## header represents it
                }

                for (let j = startIdx; j < chapter.chunks.length; j++) {
                    const chunk = chapter.chunks[j];

                    if (chunk.type === 'pause') {
                        markdown += `[pause:${chunk.duration || 1.0}]</chunk>`;
                    } else if (chunk.type === 'common_file') {
                        markdown += `[file:${chunk.path || ''}]</chunk>`;
                    } else {
                        // Text chunk - no escaping needed!
                        const text = chunk.text || '';
                        markdown += text;
                        // Add </chunk> after text (except for last chunk in chapter)
                        if (j < chapter.chunks.length - 1) {
                            markdown += '</chunk>';
                        }
                    }
                }
            }
        }

        return markdown.trim();
    }

    async saveCode() {
        try {
            const editor = document.getElementById('pseudoXmlEditor');
            const markdownContent = editor.textContent || editor.innerText;

            // Save the markdown content to the server
            const response = await fetch(`${SERVER_URL}/api/project/save-markdown`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    markdown_content: markdownContent
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.markdownContent = markdownContent;
                await this.loadMarkdown();  // Reload to sync chapters

                // Refresh TTS tab to pick up chunk text changes
                if (typeof ttsTab !== 'undefined' && ttsTab.refreshChapters) {
                    console.log('[MARKDOWN EDITOR] Refreshing TTS tab after save...');
                    await ttsTab.refreshChapters();
                    // Reload current chapter if one is selected
                    if (ttsTab.currentChapterIndex !== null) {
                        await ttsTab.loadChapter(ttsTab.currentChapterIndex);
                    }
                }

                // Refresh Reader tab if available
                if (typeof readerTab !== 'undefined' && readerTab.refresh) {
                    console.log('[MARKDOWN EDITOR] Refreshing Reader tab after save...');
                    await readerTab.refresh();
                }

                showToast('Saved successfully!', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to save');
            }
        } catch (error) {
            console.error('Error saving markdown:', error);
            showToast('Error saving: ' + error.message, 'error');
        }
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

    // Insert a pronunciation override at cursor: {display|spoken}
    insertPronunciation() {
        if (this.cleanViewActive) {
            showToast('Switch to Markup View to insert pronunciation markers', 'error');
            return;
        }
        const editor = document.getElementById('pseudoXmlEditor');
        // Get selected text to pre-fill the display portion
        const selection = window.getSelection();
        const selectedText = selection.toString();
        if (selectedText) {
            // Replace selection with markup template
            document.execCommand('insertText', false, `{${selectedText}|}`);
        } else {
            document.execCommand('insertText', false, '{|}');
        }
        editor.focus();
    }

    // Toggle between clean view (annotations hidden) and markup view
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
            // Switching back TO markup view: restore full markup
            this.cleanViewActive = false;
            if (btn) {
                btn.textContent = 'Clean View';
                btn.style.background = '';
            }
        }
        this.displayMarkdown();
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
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const result = await response.json();
                alert(`Rechunking complete! Fixed ${result.chunks_fixed || 0} chunks.`);
                await this.loadMarkdown();  // Reload markdown
            } else {
                throw new Error('Failed to auto-rechunk');
            }
        } catch (error) {
            console.error('Error auto-rechunking:', error);
            alert('Error auto-rechunking: ' + error.message);
        }
    }

    async loadGutenbergUrl() {
        const url = prompt('Enter Project Gutenberg URL:', this.defaultGutenbergUrl);
        if (!url) return;

        console.log('[GUTENBERG] Loading URL:', url);

        try {
            const response = await fetch(`${SERVER_URL}/api/project/add-gutenberg-url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({ url })
            });

            console.log('[GUTENBERG] Response status:', response.status);

            if (response.ok) {
                const result = await response.json();
                console.log('[GUTENBERG] Success! Result:', result);
                console.log('[GUTENBERG] Chapters received:', result.chapters?.length || 0);

                // Show debug info in parse log
                if (result.debug) {
                    const d = result.debug;
                    const logLines = [
                        `[LOAD] Book ID: ${d.book_id}`,
                        `[LOAD] Plain text URL: ${d.txt_url}`,
                        `[LOAD] Plain text: ${d.txt_chars?.toLocaleString()} chars`,
                        `[LOAD] EPUB downloaded: ${d.epub_downloaded} (${d.epub_bytes?.toLocaleString()} bytes)`,
                        `[LOAD] EPUB titles found: ${d.epub_titles?.length || 0}`,
                    ];
                    if (d.epub_titles?.length) {
                        d.epub_titles.forEach((t, i) => logLines.push(`[LOAD]   title[${i}]: "${t}"`));
                    }
                    logLines.push(`[LOAD] Titles matched in text: ${d.epub_titles_matched || 0}`);
                    logLines.push(`[LOAD] Chapter source: ${d.chapter_source || 'unknown'}`);
                    logLines.push(`[LOAD] Final chapters: ${d.final_chapter_count || result.chapters?.length || 0}`);
                    if (d.epub_error) {
                        logLines.push(`[LOAD] EPUB error: ${d.epub_error}`);
                    }
                    logLines.push(`[LOAD] Use "Re-Parse Chapters" with different methods to try alternative parsing.`);
                    this.appendToLog(logLines);
                }

                // Reload both raw text and markdown
                await this.loadRawText();
                await this.loadMarkdown();
                this.displayRawText();
                this.displayMarkdown();

                showToast(`Loaded! ${result.chapters?.length || 0} chapters created. Try different parsing methods if needed.`, 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to load Gutenberg text');
            }
        } catch (error) {
            console.error('Error loading Gutenberg text:', error);
            alert('Error loading Gutenberg text: ' + error.message);
        }
    }

    async uploadTextFile() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.txt';

        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            try {
                const text = await file.text();
                this.rawText = text;
                this.displayRawText();

                // Save to project
                await this.saveRawText();
                alert('Text file loaded successfully!');
            } catch (error) {
                console.error('Error loading file:', error);
                alert('Error loading file: ' + error.message);
            }
        };

        input.click();
    }

    async saveRawText() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/raw-text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    raw_text: this.rawText
                })
            });

            if (!response.ok) {
                throw new Error('Failed to save raw text');
            }
        } catch (error) {
            console.error('Error saving raw text:', error);
            alert('Failed to save raw text: ' + error.message);
        }
    }

    async processText() {
        if (!this.rawText) {
            alert('Please load text first');
            return;
        }

        // Check if chapters already exist
        if (this.chapters.length > 0) {
            const overwrite = confirm(
                `This project already has ${this.chapters.length} chapter(s).\n\n` +
                'Processing the text will REPLACE all existing chapters with a new single chapter.\n\n' +
                'Are you sure you want to continue?'
            );
            if (!overwrite) return;
        } else {
            const confirmation = confirm('Process text into chapters and chunks? This will create a new chapter structure.');
            if (!confirmation) return;
        }

        try {
            // Save the raw text first
            await this.saveRawText();

            // Use a simple approach: create a text blob and upload it
            const blob = new Blob([this.rawText], { type: 'text/plain' });
            const formData = new FormData();
            formData.append('file', blob, 'manual_text.txt');

            const response = await fetch(`${SERVER_URL}/api/project/add-text-file`, {
                method: 'POST',
                headers: {
                    'X-API-Key': API_KEY
                },
                body: formData
            });

            if (response.ok) {
                await this.loadMarkdown();

                // Refresh TTS tab chapters dropdown after processing
                if (typeof ttsTab !== 'undefined' && ttsTab.refreshChapters) {
                    console.log('[GUTENBERG] Refreshing TTS dropdown after chapter processing...');
                    await ttsTab.refreshChapters();
                }

                alert('Text processed successfully!');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to process text');
            }
        } catch (error) {
            console.error('Error processing text:', error);
            alert('Error processing text: ' + error.message);
        }
    }

    // ----------------------------------------------------------------
    // Chapter Parsing Method Selector & Verbose Log
    // ----------------------------------------------------------------

    async initParsingMethods() {
        // Load method descriptions from server
        try {
            const response = await fetch(`${SERVER_URL}/api/project/parsing-methods`, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (response.ok) {
                const data = await response.json();
                this.parsingMethods = data.methods || {};
                console.log('[GUTENBERG] Parsing methods loaded:', Object.keys(this.parsingMethods));
                this.updateMethodDescription();
            }
        } catch (error) {
            console.error('[GUTENBERG] Error loading parsing methods:', error);
        }

        // Listen for dropdown changes
        const select = document.getElementById('parsingMethodSelect');
        if (select) {
            select.addEventListener('change', () => this.updateMethodDescription());
        }
    }

    updateMethodDescription() {
        const select = document.getElementById('parsingMethodSelect');
        const descDiv = document.getElementById('parsingMethodDesc');
        if (!select || !descDiv) return;

        const method = select.value;
        const desc = this.parsingMethods?.[method] || '';
        descDiv.textContent = desc;
    }

    toggleParseLog() {
        const panel = document.getElementById('parseLogPanel');
        const btn = document.getElementById('toggleLogBtn');
        if (!panel) return;

        if (panel.style.display === 'none') {
            panel.style.display = 'block';
            if (btn) btn.textContent = 'Hide Log';
        } else {
            panel.style.display = 'none';
            if (btn) btn.textContent = 'Show Log';
        }
    }

    appendToLog(lines) {
        const logContent = document.getElementById('parseLogContent');
        if (!logContent) return;

        // Color-code log lines
        const colored = lines.map(line => {
            if (line.includes('ERROR') || line.includes('EXCEPTION')) {
                return `<span style="color: #f38ba8;">${this.escapeHtml(line)}</span>`;
            } else if (line.includes('✓') || line.includes('OK')) {
                return `<span style="color: #a6e3a1;">${this.escapeHtml(line)}</span>`;
            } else if (line.includes('✗') || line.includes('NOT FOUND')) {
                return `<span style="color: #fab387;">${this.escapeHtml(line)}</span>`;
            } else if (line.includes('[PARSE]')) {
                return `<span style="color: #89b4fa;">${this.escapeHtml(line)}</span>`;
            } else {
                return this.escapeHtml(line);
            }
        }).join('\n');

        logContent.innerHTML = colored;

        // Auto-show log panel
        const panel = document.getElementById('parseLogPanel');
        const btn = document.getElementById('toggleLogBtn');
        if (panel) panel.style.display = 'block';
        if (btn) btn.textContent = 'Hide Log';

        // Scroll to bottom
        if (panel) panel.scrollTop = panel.scrollHeight;
    }

    escapeHtml(text) {
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    async reparseChapters() {
        const select = document.getElementById('parsingMethodSelect');
        if (!select) return;

        const method = select.value;
        console.log(`[GUTENBERG] Re-parsing chapters with method: ${method}`);

        // Show loading state
        const logContent = document.getElementById('parseLogContent');
        if (logContent) logContent.innerHTML = `<span style="color: #cba6f7;">Re-parsing with method "${method}"...</span>`;
        const panel = document.getElementById('parseLogPanel');
        const btn = document.getElementById('toggleLogBtn');
        if (panel) panel.style.display = 'block';
        if (btn) btn.textContent = 'Hide Log';

        try {
            const response = await fetch(`${SERVER_URL}/api/project/reparse-chapters`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({ method })
            });

            const result = await response.json();

            // Display log lines
            if (result.log && result.log.length > 0) {
                this.appendToLog(result.log);
            }

            if (result.success) {
                console.log(`[GUTENBERG] Re-parsed: ${result.chapter_count} chapters with method ${method}`);

                // Reload markdown view
                await this.loadMarkdown();

                // Refresh TTS tab
                if (typeof ttsTab !== 'undefined' && ttsTab.refreshChapters) {
                    await ttsTab.refreshChapters();
                }

                // Refresh Reader tab
                if (typeof readerTab !== 'undefined' && readerTab.refresh) {
                    await readerTab.refresh();
                }

                showToast(`Re-parsed: ${result.chapter_count} chapters (${method})`, 'success');
            } else {
                const errMsg = result.error || 'Unknown error';
                console.error('[GUTENBERG] Re-parse failed:', errMsg);
                showToast('Re-parse failed: ' + errMsg, 'error');
            }
        } catch (error) {
            console.error('[GUTENBERG] Re-parse error:', error);
            this.appendToLog([`[REPARSE] Network/JS error: ${error.message}`]);
            showToast('Error re-parsing: ' + error.message, 'error');
        }
    }

    generateId() {
        return 'id_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

// Create global instance
const gutenbergTab = new GutenbergTab();

// Global helper functions for onclick handlers
function loadGutenbergText() {
    gutenbergTab.loadGutenbergUrl();
}

function uploadTextFile() {
    gutenbergTab.uploadTextFile();
}

function processText() {
    gutenbergTab.processText();
}

function validateChunks() {
    gutenbergTab.validateChunks();
}

function showAnnotator() {
    alert('Annotator feature coming soon!');
}
