/**
 * Gutenberg Processing Tab Logic
 * Handles raw text loading, chapter processing, editing, and validation
 */

class GutenbergTab {
    constructor() {
        this.rawText = '';
        this.chapters = [];
        this.selectedChapterId = null;
        this.pseudoXmlMode = false;
    }

    async init() {
        // Load raw text from project if available
        await this.loadRawText();
        // Load chapters from project
        await this.loadChapters();
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

    async loadChapters() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/text-files`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.chapters = data.chapters || [];
                console.log('[LOAD CHAPTERS] Loaded chapters:', this.chapters.length);
                this.displayChapters();
            }
        } catch (error) {
            console.error('Error loading chapters:', error);
        }
    }

    displayChapters() {
        const container = document.getElementById('chaptersContainer');
        const placeholder = document.getElementById('chapterPlaceholder');
        const chapterList = document.getElementById('chapterList');

        console.log('[DISPLAY CHAPTERS] Called');
        console.log('[DISPLAY CHAPTERS] this.chapters:', this.chapters);
        console.log('[DISPLAY CHAPTERS] chapters length:', this.chapters.length);

        if (this.chapters.length === 0) {
            console.log('[DISPLAY CHAPTERS] No chapters, showing placeholder');
            chapterList.style.display = 'none';
            placeholder.style.display = 'block';
            return;
        }

        console.log('[DISPLAY CHAPTERS] Displaying', this.chapters.length, 'chapters');
        chapterList.style.display = 'block';
        placeholder.style.display = 'none';

        container.innerHTML = this.chapters.map((chapter, index) => {
            const chunkCount = chapter.chunks ? chapter.chunks.length : 0;
            const totalChars = chapter.chunks ?
                chapter.chunks.reduce((sum, chunk) => sum + (chunk.text?.length || 0), 0) : 0;

            return `
                <div class="chapter-item ${this.selectedChapterId === chapter.id ? 'active' : ''}"
                     onclick="gutenbergTab.selectChapter('${chapter.id}')">
                    <div>
                        <strong>${chapter.name || chapter.title || `Chapter ${index + 1}`}</strong>
                        <div style="font-size: 12px; color: #888; margin-top: 4px;">
                            ${chunkCount} chunks, ${totalChars} characters
                        </div>
                    </div>
                    <div class="chapter-actions">
                        <button class="icon-btn" onclick="event.stopPropagation(); gutenbergTab.editChapterName('${chapter.id}')" title="Rename">
                            ✏️
                        </button>
                        <button class="icon-btn" onclick="event.stopPropagation(); gutenbergTab.splitChapter('${chapter.id}')" title="Split">
                            ✂️
                        </button>
                        ${index < this.chapters.length - 1 ? `
                        <button class="icon-btn" onclick="event.stopPropagation(); gutenbergTab.mergeWithNext('${chapter.id}')" title="Merge with next">
                            🔗
                        </button>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }

    selectChapter(chapterId) {
        this.selectedChapterId = chapterId;
        const chapter = this.chapters.find(ch => ch.id === chapterId);

        if (chapter) {
            this.displayChapterContent(chapter);
        }

        this.displayChapters(); // Refresh to show active state
    }

    displayChapterContent(chapter) {
        const editor = document.getElementById('pseudoXmlEditor');
        editor.style.display = 'block';

        // Convert chapter to pseudo-XML format
        const xml = this.chapterToPseudoXml(chapter);
        editor.value = xml;
    }

    chapterToPseudoXml(chapter) {
        let xml = `<chapter id="${chapter.id}" name="${this.escapeXml(chapter.name || 'Untitled')}">\n`;

        if (chapter.chunks) {
            chapter.chunks.forEach(chunk => {
                xml += `  <chunk id="${chunk.id}" start="${chunk.start_pos || 0}" end="${chunk.end_pos || 0}">\n`;
                xml += `    ${this.escapeXml(chunk.text || '')}\n`;
                xml += `  </chunk>\n`;
            });
        }

        xml += `</chapter>`;
        return xml;
    }

    escapeXml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&apos;');
    }

    async loadGutenbergUrl() {
        const url = prompt('Enter Project Gutenberg URL (must be .txt file):');
        if (!url) return;

        if (!url.includes('gutenberg.org') || !url.endsWith('.txt')) {
            alert('Please provide a valid Project Gutenberg .txt URL');
            return;
        }

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

                alert('Gutenberg text loaded successfully!');

                console.log('[GUTENBERG] Reloading raw text...');
                await this.loadRawText();
                console.log('[GUTENBERG] Raw text loaded. Length:', this.rawText.length);

                console.log('[GUTENBERG] Reloading chapters...');
                await this.loadChapters();
                console.log('[GUTENBERG] Chapters loaded. Count:', this.chapters.length);

                console.log('[GUTENBERG] Displaying content...');
                this.displayRawText();
                this.displayChapters();
                console.log('[GUTENBERG] Complete!');
            } else {
                const error = await response.json();
                console.error('[GUTENBERG] Error:', error);
                alert('Error: ' + (error.error || 'Failed to load text'));
            }
        } catch (error) {
            console.error('[GUTENBERG] Exception:', error);
            alert('Network error: ' + error.message);
        }
    }

    async uploadTextFile() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.txt';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (file) {
                this.rawText = await file.text();
                await this.saveRawText();
                this.displayRawText();
            }
        };
        input.click();
    }

    async saveRawText() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/save-raw-text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({ raw_text: this.rawText })
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
                await this.loadChapters();
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

    async addChapterToProject(chapter) {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/add-text-file`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    file_path: chapter.path || 'generated',
                    file_name: chapter.name,
                    chunks: chapter.chunks
                })
            });

            if (!response.ok) {
                throw new Error('Failed to add chapter to project');
            }
        } catch (error) {
            console.error('Error adding chapter:', error);
            throw error;
        }
    }

    async validateChunks() {
        const maxSize = prompt('Maximum chunk size (characters):', '500');
        if (!maxSize) return;

        try {
            const response = await fetch(`${SERVER_URL}/api/project/validate-chunks`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    max_chunk_size: parseInt(maxSize)
                })
            });

            if (response.ok) {
                const result = await response.json();

                if (result.oversized_count === 0) {
                    alert(`All chunks are within ${maxSize} characters! ✓\n\nTotal chunks: ${result.total_chunks}`);
                } else {
                    const message = `Found ${result.oversized_count} oversized chunks out of ${result.total_chunks} total.\n\n` +
                                  `Would you like to automatically rechunk them?`;

                    if (confirm(message)) {
                        await this.autoRechunk(parseInt(maxSize));
                    } else {
                        // Show details
                        const details = result.oversized_chunks.map(chunk =>
                            `${chunk.chapter_name} - Chunk ${chunk.chunk_id}: ${chunk.chunk_size} chars`
                        ).join('\n');
                        alert('Oversized chunks:\n\n' + details);
                    }
                }
            } else {
                throw new Error('Failed to validate chunks');
            }
        } catch (error) {
            console.error('Error validating chunks:', error);
            alert('Error validating chunks: ' + error.message);
        }
    }

    async autoRechunk(maxSize) {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/auto-rechunk`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    max_chunk_size: maxSize
                })
            });

            if (response.ok) {
                const result = await response.json();
                alert(result.message);
                await this.loadChapters();
            } else {
                throw new Error('Failed to rechunk');
            }
        } catch (error) {
            console.error('Error rechunking:', error);
            alert('Error rechunking: ' + error.message);
        }
    }

    async editChapterName(chapterId) {
        const chapter = this.chapters.find(ch => ch.id === chapterId);
        if (!chapter) return;

        const newName = prompt('Enter new chapter name:', chapter.name || '');
        if (!newName || newName === chapter.name) return;

        chapter.name = newName;

        // Save to project (this would need a new endpoint or update existing one)
        await this.saveChapterChanges();
        this.displayChapters();
    }

    async splitChapter(chapterId) {
        const chapter = this.chapters.find(ch => ch.id === chapterId);
        if (!chapter) return;

        const fullText = chapter.chunks.map(c => c.text).join('\n\n');
        const position = prompt(`Split chapter at character position (0-${fullText.length}):`,
                               Math.floor(fullText.length / 2).toString());

        if (position === null) return;

        const splitPos = parseInt(position);
        if (isNaN(splitPos) || splitPos < 0 || splitPos > fullText.length) {
            alert('Invalid position');
            return;
        }

        try {
            const response = await fetch(`${SERVER_URL}/api/project/split-chapter`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    chapter_id: chapterId,
                    split_position: splitPos
                })
            });

            if (response.ok) {
                alert('Chapter split successfully!');
                await this.loadChapters();
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to split chapter');
            }
        } catch (error) {
            console.error('Error splitting chapter:', error);
            alert('Error splitting chapter: ' + error.message);
        }
    }

    async mergeWithNext(chapterId) {
        const chapterIndex = this.chapters.findIndex(ch => ch.id === chapterId);
        if (chapterIndex === -1 || chapterIndex >= this.chapters.length - 1) return;

        const chapter1 = this.chapters[chapterIndex];
        const chapter2 = this.chapters[chapterIndex + 1];

        const confirmation = confirm(`Merge "${chapter1.name}" with "${chapter2.name}"?`);
        if (!confirmation) return;

        try {
            const response = await fetch(`${SERVER_URL}/api/project/merge-chapters`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    chapter_id1: chapter1.id,
                    chapter_id2: chapter2.id
                })
            });

            if (response.ok) {
                alert('Chapters merged successfully!');
                await this.loadChapters();
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to merge chapters');
            }
        } catch (error) {
            console.error('Error merging chapters:', error);
            alert('Error merging chapters: ' + error.message);
        }
    }

    async saveChapterChanges() {
        // This would save the current chapters back to the project
        // For now, just reload to reflect any server-side changes
        await this.loadChapters();
    }

    showAnnotator() {
        alert('Annotator functionality will be integrated here.\n\n' +
              'This will allow you to add footnotes and annotations to the text ' +
              'that will be ignored by the TTS module.');
    }

    generateId() {
        return 'ch_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

// Global instance
const gutenbergTab = new GutenbergTab();

// Export functions for HTML onclick handlers
window.loadGutenbergText = () => gutenbergTab.loadGutenbergUrl();
window.uploadTextFile = () => gutenbergTab.uploadTextFile();
window.processText = () => gutenbergTab.processText();
window.validateChunks = () => gutenbergTab.validateChunks();
window.showAnnotator = () => gutenbergTab.showAnnotator();
