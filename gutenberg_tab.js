/**
 * Gutenberg Processing Tab Logic
 * Handles raw text loading, XML editing, chapter processing, and validation
 */

class GutenbergTab {
    constructor() {
        this.rawText = '';
        this.xmlContent = '';
        this.chapters = [];
        this.defaultGutenbergUrl = '';
    }

    async init() {
        // Load default Gutenberg URL from config
        await this.loadDefaultUrl();
        // Load raw text from project if available
        await this.loadRawText();
        // Load XML content from project
        await this.loadXML();
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

    /**
     * Apply syntax highlighting to XML text with visual boundaries for chapters and chunks
     */
    highlightXML(xmlText) {
        if (!xmlText) return '';

        // Escape HTML entities
        let highlighted = xmlText
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // First, wrap entire chunk elements as blocks (including content)
        // Match <chunk>content</chunk> and wrap in a block div
        highlighted = highlighted.replace(
            /(&lt;chunk&gt;)([\s\S]*?)(&lt;\/chunk&gt;)/g,
            '<div class="xml-chunk-block">$1$2$3</div>'
        );

        // Wrap chapter opening tags
        highlighted = highlighted.replace(
            /(&lt;chapter\s+[^&]*&gt;)/g,
            '<div class="xml-chapter-boundary">$1</div>'
        );

        // Wrap chapter closing tags
        highlighted = highlighted.replace(
            /(&lt;\/chapter&gt;)/g,
            '<div class="xml-chapter-end">$1</div>'
        );

        // Wrap non-voiced opening tags
        highlighted = highlighted.replace(
            /(&lt;non-voiced\s+[^&]*&gt;)/g,
            '<div class="xml-non-voiced-boundary">$1</div>'
        );

        // Wrap non-voiced closing tags
        highlighted = highlighted.replace(
            /(&lt;\/non-voiced&gt;)/g,
            '<div class="xml-non-voiced-end">$1</div>'
        );

        // Highlight XML tags with attributes
        highlighted = highlighted.replace(
            /&lt;(\/?)([\w-]+)((?:\s+[\w-]+="[^"]*")*)\s*(\/??)&gt;/g,
            (match, slash1, tagName, attributes, slash2) => {
                let result = '<span class="xml-bracket">&lt;</span>';
                result += slash1;
                result += `<span class="xml-tag">${tagName}</span>`;

                // Highlight attributes
                if (attributes) {
                    result += attributes.replace(
                        /([\w-]+)="([^"]*)"/g,
                        '<span class="xml-attribute">$1</span>=<span class="xml-value">"$2"</span>'
                    );
                }

                result += slash2;
                result += '<span class="xml-bracket">&gt;</span>';
                return result;
            }
        );

        return highlighted;
    }

    displayXML() {
        const editor = document.getElementById('pseudoXmlEditor');

        console.log('[DISPLAY XML] Called');
        console.log('[DISPLAY XML] XML length:', this.xmlContent.length);

        if (!this.xmlContent) {
            editor.innerHTML = '';
            return;
        }

        // Filter out <p> and </p> tags for cleaner display
        let filteredXML = this.xmlContent
            .replace(/<p>/gi, '')
            .replace(/<\/p>/gi, '');

        // Clean up Gutenberg-style formatting:
        // 1. Remove all leading indentation (spaces/tabs at start of lines)
        // 2. Collapse line breaks within text content (keep breaks around tags)
        filteredXML = this.cleanGutenbergWhitespace(filteredXML);

        // Apply syntax highlighting
        editor.innerHTML = this.highlightXML(filteredXML);
    }

    /**
     * Clean up whitespace from Gutenberg source:
     * - Remove all indentation (leading spaces/tabs)
     * - Collapse line breaks that don't coincide with XML tags into single spaces
     */
    cleanGutenbergWhitespace(xml) {
        // First, remove all leading whitespace from each line
        let lines = xml.split('\n');
        lines = lines.map(line => line.trimStart());

        // Join lines back together
        let cleaned = lines.join('\n');

        // Now handle line breaks within chunk content:
        // Line breaks that are NOT adjacent to XML tags should become spaces
        // We process the text between tags separately

        // Match text content between > and < (the actual content, not tags)
        cleaned = cleaned.replace(/>([^<]+)</g, (match, content) => {
            // Collapse multiple whitespace (including newlines) into single spaces
            // but preserve intentional paragraph breaks (double newlines)
            const processed = content
                // First, normalize line breaks
                .replace(/\r\n/g, '\n')
                // Collapse single newlines (not preceded/followed by newlines) into spaces
                .replace(/(?<!\n)\n(?!\n)/g, ' ')
                // Collapse multiple spaces into single space
                .replace(/ +/g, ' ');

            return '>' + processed + '<';
        });

        // Clean up extra whitespace around tags
        // Remove blank lines between tags
        cleaned = cleaned.replace(/>\s*\n\s*\n+\s*</g, '>\n<');

        // Preserve structure by keeping newlines around block-level elements
        cleaned = cleaned
            .replace(/(<\/chunk>)\s*/g, '$1\n')
            .replace(/\s*(<chunk>)/g, '\n$1')
            .replace(/(<\/chapter>)\s*/g, '$1\n')
            .replace(/\s*(<chapter\s)/g, '\n$1')
            .replace(/(<\/non-voiced>)\s*/g, '$1\n')
            .replace(/\s*(<non-voiced\s)/g, '\n$1')
            .replace(/(<\/book>)\s*/g, '$1\n')
            .replace(/\s*(<book>)/g, '\n$1');

        // Clean up multiple consecutive newlines
        cleaned = cleaned.replace(/\n{3,}/g, '\n\n');

        return cleaned.trim();
    }

    chaptersToXML(chapters) {
        // Convert chapters array to pseudo-XML format
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n<book>\n';

        for (const chapter of chapters) {
            const title = chapter.title || chapter.name || 'Untitled';
            const isNonVoiced = chapter.non_voiced || false;

            if (isNonVoiced) {
                xml += `  <non-voiced title="${this.escapeXML(title)}">\n`;
            } else {
                xml += `  <chapter title="${this.escapeXML(title)}">\n`;
            }

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

            if (isNonVoiced) {
                xml += '  </non-voiced>\n';
            } else {
                xml += '  </chapter>\n';
            }
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
            const xmlContent = editor.textContent || editor.innerText;

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

        let insertText = '';
        if (tagType === 'chapter') {
            insertText = '\n  <chapter title="New Chapter">\n    <chunk>Chapter text goes here</chunk>\n  </chapter>\n';
        } else if (tagType === 'chunk') {
            insertText = '\n    <chunk>Chunk text goes here</chunk>\n';
        } else if (tagType === 'pause') {
            insertText = '\n    <pause duration="1.0"/>\n';
        }

        // Insert at cursor position in contenteditable div
        document.execCommand('insertText', false, insertText);
        editor.focus();
    }

    wrapSelection(tagType) {
        const selection = window.getSelection();

        if (selection.rangeCount === 0 || selection.toString().length === 0) {
            alert('Please select some text first');
            return;
        }

        const selectedText = selection.toString();

        let wrappedText = '';
        if (tagType === 'footnote') {
            wrappedText = `<footnote>${selectedText}</footnote>`;
        }

        // Replace selected text with wrapped version
        document.execCommand('insertText', false, wrappedText);
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
