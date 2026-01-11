/**
 * Gutenberg Processing Tab Logic
 * Handles raw text loading, XML editing, chapter processing, and validation
 */

class GutenbergTab {
    constructor() {
        this.rawText = '';
        this.xmlContent = '';
        this.chapters = [];
    }

    async init() {
        // Load raw text from project if available
        await this.loadRawText();
        // Load XML content from project
        await this.loadXML();
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

    async loadXML() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/get-text-files`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.chapters = data.chapters || [];
                this.xmlContent = data.content_xml || '';

                console.log('[LOAD XML] Loaded chapters:', this.chapters.length);
                console.log('[LOAD XML] XML length:', this.xmlContent.length);

                // If we have chapters but no XML, generate XML from chapters
                if (this.chapters.length > 0 && !this.xmlContent) {
                    this.xmlContent = this.chaptersToXML(this.chapters);
                }

                this.displayXML();
            }
        } catch (error) {
            console.error('Error loading XML:', error);
        }
    }

    displayXML() {
        const editor = document.getElementById('pseudoXmlEditor');
        const placeholder = document.getElementById('xmlPlaceholder');

        console.log('[DISPLAY XML] Called');
        console.log('[DISPLAY XML] XML length:', this.xmlContent.length);

        if (!this.xmlContent) {
            editor.style.display = 'none';
            placeholder.style.display = 'block';
            return;
        }

        editor.style.display = 'block';
        placeholder.style.display = 'none';
        editor.value = this.xmlContent;
    }

    chaptersToXML(chapters) {
        // Convert chapters array to pseudo-XML format
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n<book>\n';

        for (const chapter of chapters) {
            const title = chapter.title || chapter.name || 'Untitled';
            xml += `  <chapter title="${this.escapeXML(title)}">\n`;

            if (chapter.chunks) {
                for (const chunk of chapter.chunks) {
                    if (chunk.type === 'pause') {
                        xml += `    <pause duration="${chunk.duration || 1.0}"/>\n`;
                    } else if (chunk.type === 'common_file') {
                        xml += `    <common_file path="${this.escapeXML(chunk.path || '')}"/>\n`;
                    } else {
                        const text = chunk.text || '';
                        xml += `    <chunk>${this.escapeXML(text)}</chunk>\n`;
                    }
                }
            }

            xml += '  </chapter>\n';
        }

        xml += '</book>';
        return xml;
    }

    escapeXML(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&apos;');
    }

    unescapeXML(text) {
        return text
            .replace(/&apos;/g, "'")
            .replace(/&quot;/g, '"')
            .replace(/&gt;/g, '>')
            .replace(/&lt;/g, '<')
            .replace(/&amp;/g, '&');
    }

    async saveXML() {
        try {
            const editor = document.getElementById('pseudoXmlEditor');
            const xmlContent = editor.value;

            // TODO: Parse XML and convert to chapters format
            // For now, just save the XML content
            const response = await fetch(`${SERVER_URL}/api/project/save-xml`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    xml_content: xmlContent
                })
            });

            if (response.ok) {
                alert('XML saved successfully!');
                this.xmlContent = xmlContent;
                await this.loadXML();  // Reload to sync
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to save XML');
            }
        } catch (error) {
            console.error('Error saving XML:', error);
            alert('Error saving XML: ' + error.message);
        }
    }

    insertTag(tagType) {
        const editor = document.getElementById('pseudoXmlEditor');
        const cursorPos = editor.selectionStart;
        const textBefore = editor.value.substring(0, cursorPos);
        const textAfter = editor.value.substring(cursorPos);

        let insertText = '';
        if (tagType === 'chapter') {
            insertText = '\n  <chapter title="New Chapter">\n    <chunk>Chapter text goes here</chunk>\n  </chapter>\n';
        } else if (tagType === 'chunk') {
            insertText = '\n    <chunk>Chunk text goes here</chunk>\n';
        } else if (tagType === 'pause') {
            insertText = '\n    <pause duration="1.0"/>\n';
        }

        editor.value = textBefore + insertText + textAfter;
        editor.selectionStart = editor.selectionEnd = cursorPos + insertText.length;
        editor.focus();
    }

    wrapSelection(tagType) {
        const editor = document.getElementById('pseudoXmlEditor');
        const start = editor.selectionStart;
        const end = editor.selectionEnd;

        if (start === end) {
            alert('Please select some text first');
            return;
        }

        const selectedText = editor.value.substring(start, end);
        const textBefore = editor.value.substring(0, start);
        const textAfter = editor.value.substring(end);

        let wrappedText = '';
        if (tagType === 'footnote') {
            wrappedText = `<footnote>${selectedText}</footnote>`;
        }

        editor.value = textBefore + wrappedText + textAfter;
        editor.selectionStart = start;
        editor.selectionEnd = start + wrappedText.length;
        editor.focus();
    }

    async validateChunks() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/validate-chunks`, {
                headers: {
                    'X-API-Key': API_KEY
                }
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
                await this.loadXML();  // Reload XML
            } else {
                throw new Error('Failed to auto-rechunk');
            }
        } catch (error) {
            console.error('Error auto-rechunking:', error);
            alert('Error auto-rechunking: ' + error.message);
        }
    }

    async loadGutenbergUrl() {
        const url = prompt('Enter Project Gutenberg URL:');
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

                // Reload both raw text and XML
                console.log('[GUTENBERG] Reloading raw text...');
                await this.loadRawText();
                console.log('[GUTENBERG] Raw text loaded. Length:', this.rawText.length);

                console.log('[GUTENBERG] Reloading XML...');
                await this.loadXML();
                console.log('[GUTENBERG] XML loaded. Count:', this.chapters.length);

                console.log('[GUTENBERG] Displaying content...');
                this.displayRawText();
                this.displayXML();
                console.log('[GUTENBERG] Complete!');

                alert(`Successfully loaded! ${result.chapters?.length || 0} chapters created.`);
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
                await this.loadXML();
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
