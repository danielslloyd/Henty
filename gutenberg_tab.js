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
                console.log('[LOAD XML] content_xml length:', this.xmlContent.length);
                console.log('[LOAD CODE] content_xml has <chunk>:', this.xmlContent.includes('<chunk>'));
                console.log('[LOAD CODE] content_xml has <chapter:', this.xmlContent.includes('<chapter'));
                console.log('[LOAD CODE] content_xml first 500 chars:', this.xmlContent.substring(0, 500));

                // Always regenerate code from chapters to ensure chunks are included
                // The chapters array is the source of truth and includes all chunk data
                if (this.chapters.length > 0) {
                    console.log('[LOAD CODE] Regenerating code from chapters to include chunks...');
                    this.xmlContent = this.chaptersToXML(this.chapters);
                    console.log('[LOAD CODE] Generated code first 500 chars:', this.xmlContent.substring(0, 500));
                }

                this.displayXML();
            }
        } catch (error) {
            console.error('Error loading XML:', error);
        }
    }

    /**
     * Display XML text as plain text with all tags visible
     * - Escape HTML entities so XML tags display as text
     * - Convert <p> tags to line breaks for readability
     */
    highlightXML(xmlText) {
        if (!xmlText) return '';

        // Escape HTML entities so XML displays as text
        let html = xmlText
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Convert <p> and </p> tags to line breaks (they represent paragraph breaks)
        html = html.replace(/&lt;p&gt;/gi, '\n');
        html = html.replace(/&lt;\/p&gt;/gi, '');

        return html;
    }

    displayXML() {
        const editor = document.getElementById('pseudoXmlEditor');

        console.log('[DISPLAY XML] Called');
        console.log('[DISPLAY XML] xmlContent length:', this.xmlContent.length);

        if (!this.xmlContent) {
            editor.innerHTML = '<div class="xml-empty-state">No content. Load a Project Gutenberg text or upload a file to begin.</div>';
            return;
        }

        // DEBUG: Check if xmlContent has chunk tags before processing
        console.log('[DISPLAY XML] Before clean - has <chunk>:', this.xmlContent.includes('<chunk>'));
        console.log('[DISPLAY XML] Before clean - has <chapter:', this.xmlContent.includes('<chapter'));

        // Clean up Gutenberg-style formatting first
        let cleanedXML = this.cleanGutenbergWhitespace(this.xmlContent);

        // DEBUG: Check if cleanedXML still has chunk tags after processing
        console.log('[DISPLAY XML] After clean - has <chunk>:', cleanedXML.includes('<chunk>'));
        console.log('[DISPLAY XML] After clean - has <chapter:', cleanedXML.includes('<chapter'));
        console.log('[DISPLAY XML] Cleaned first 500 chars:', cleanedXML.substring(0, 500));

        // Apply escaping and p-tag conversion
        const finalHTML = this.highlightXML(cleanedXML);

        // DEBUG: Check final output
        console.log('[DISPLAY XML] Final HTML has &lt;chunk&gt;:', finalHTML.includes('&lt;chunk&gt;'));
        console.log('[DISPLAY XML] Final HTML first 500 chars:', finalHTML.substring(0, 500));

        editor.innerHTML = finalHTML;
    }

    /**
     * Clean up whitespace from Gutenberg source:
     * - Remove all indentation (leading spaces/tabs)
     * - Collapse line breaks INSIDE text content into spaces (Gutenberg source formatting)
     * - Preserve line breaks that are part of XML structure (adjacent to tags)
     */
    cleanGutenbergWhitespace(xml) {
        // Remove all leading whitespace from each line (indentation)
        let lines = xml.split('\n');
        lines = lines.map(line => line.trimStart());
        let cleaned = lines.join('\n');

        // Now we need to collapse line breaks that are INSIDE chunk content
        // (i.e., between > and < but not adjacent to tags)
        // These are Gutenberg source code line breaks, not real content breaks

        // Process text content between tags: collapse internal newlines to spaces
        // Match content between > and < (text nodes)
        cleaned = cleaned.replace(/>([^<]+)</g, (match, content) => {
            // Replace newlines (and surrounding whitespace) with single space
            // But preserve if the content is just whitespace (between tags)
            if (content.trim() === '') {
                return '><';  // Remove whitespace-only text nodes
            }
            // Collapse internal newlines to spaces
            const processed = content
                .replace(/\s*\n\s*/g, ' ')  // Replace newlines with spaces
                .replace(/\s+/g, ' ')        // Collapse multiple spaces
                .trim();
            return '>' + processed + '<';
        });

        // Now add proper line breaks for XML structure readability
        // Add newline before chapter/non-voiced opening tags
        cleaned = cleaned.replace(/<(chapter|non-voiced)\s/g, '\n\n<$1 ');

        // Add newline after chapter/non-voiced closing tags
        cleaned = cleaned.replace(/<\/(chapter|non-voiced)>/g, '</$1>\n\n');

        // Add newline between chunks for readability
        cleaned = cleaned.replace(/<\/chunk><chunk>/g, '</chunk>\n<chunk>');

        // Add newline after book opening and before book closing
        cleaned = cleaned.replace(/<book>/g, '<book>\n');
        cleaned = cleaned.replace(/<\/book>/g, '\n</book>');

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

    async saveCode() {
        try {
            const editor = document.getElementById('pseudoXmlEditor');
            const codeContent = editor.textContent || editor.innerText;

            // Save the code content to the server
            const response = await fetch(`${SERVER_URL}/api/project/save-xml`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    xml_content: codeContent
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.xmlContent = codeContent;
                await this.loadXML();  // Reload to sync chapters

                // Refresh TTS tab to pick up chunk text changes
                if (typeof ttsTab !== 'undefined' && ttsTab.refreshChapters) {
                    console.log('[CODE EDITOR] Refreshing TTS tab after code save...');
                    await ttsTab.refreshChapters();
                    // Reload current chapter if one is selected
                    if (ttsTab.currentChapterIndex !== null) {
                        await ttsTab.loadChapter(ttsTab.currentChapterIndex);
                    }
                }

                showToast('Code saved successfully!', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to save code');
            }
        } catch (error) {
            console.error('Error saving code:', error);
            showToast('Error saving code: ' + error.message, 'error');
        }
    }

    // Alias for backwards compatibility
    async saveXML() {
        return this.saveCode();
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
