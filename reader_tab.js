/**
 * Reader Tab Logic
 * Handles displaying content and playing best takes with text highlighting
 */

class ReaderTab {
    constructor() {
        this.chapters = [];
        this.currentAudio = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.currentChunkIndex = 0;
    }

    async init() {
        await this.loadChapters();
        this.renderContent();
    }

    async loadChapters() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/info`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const projectInfo = await response.json();
                this.chapters = projectInfo.chapters || [];
            }
        } catch (error) {
            console.error('Error loading chapters:', error);
        }
    }

    renderContent() {
        const container = document.getElementById('readerContent');

        if (this.chapters.length === 0) {
            container.innerHTML = '<div class="loading">No content available. Generate audio in the TTS Module first.</div>';
            return;
        }

        // Render all chapters with their chunks
        const html = this.chapters.map((chapter, chapterIndex) => {
            const chunks = chapter.chunks || [];

            return `
                <div class="chapter-section" style="margin-bottom: 40px;">
                    <h2 style="color: #667eea; margin-bottom: 20px; font-size: 1.8em;">
                        ${chapter.name || `Chapter ${chapterIndex + 1}`}
                    </h2>
                    ${chunks.map((chunk, chunkIndex) => {
                        const hasBestTake = chunk.generated_audios?.some(a => a.is_best_take);
                        return `
                            <p class="chunk-text"
                               data-chapter="${chapterIndex}"
                               data-chunk="${chunkIndex}"
                               style="margin-bottom: 1.5em; line-height: 1.8; ${!hasBestTake ? 'color: #ccc;' : ''}">
                                ${chunk.text}
                            </p>
                        `;
                    }).join('')}
                </div>
            `;
        }).join('');

        container.innerHTML = html || '<div class="loading">No content available.</div>';
    }

    async playAllBestTakes() {
        if (this.isPlaying) {
            this.stopPlayback();
            return;
        }

        // Build audio queue from all best takes
        this.audioQueue = [];
        this.chapters.forEach((chapter, chapterIndex) => {
            const chunks = chapter.chunks || [];
            chunks.forEach((chunk, chunkIndex) => {
                const bestTake = chunk.generated_audios?.find(a => a.is_best_take);
                if (bestTake) {
                    this.audioQueue.push({
                        chapterIndex,
                        chunkIndex,
                        audioUrl: bestTake.audio_url,
                        text: chunk.text
                    });
                }
            });
        });

        if (this.audioQueue.length === 0) {
            alert('No best takes available. Generate and set best takes in the TTS Module first.');
            return;
        }

        this.isPlaying = true;
        this.currentChunkIndex = 0;

        // Update button
        const playBtn = document.querySelector('.play-btn');
        if (playBtn) {
            playBtn.innerHTML = '<span class="material-symbols-outlined">stop</span> Stop';
        }

        this.playNext();
    }

    playNext() {
        if (!this.isPlaying || this.currentChunkIndex >= this.audioQueue.length) {
            this.stopPlayback();
            return;
        }

        const item = this.audioQueue[this.currentChunkIndex];

        // Highlight the current chunk
        this.highlightChunk(item.chapterIndex, item.chunkIndex);

        // Create and play audio
        this.currentAudio = new Audio(SERVER_URL + item.audioUrl);

        this.currentAudio.onended = () => {
            this.currentChunkIndex++;
            this.playNext();
        };

        this.currentAudio.onerror = (error) => {
            console.error('Error playing audio:', error);
            this.currentChunkIndex++;
            this.playNext();
        };

        this.currentAudio.play().catch(err => {
            console.error('Playback error:', err);
            this.currentChunkIndex++;
            this.playNext();
        });
    }

    stopPlayback() {
        this.isPlaying = false;

        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
        }

        // Remove all highlights
        document.querySelectorAll('.chunk-text').forEach(el => {
            el.classList.remove('highlight');
        });

        // Update button
        const playBtn = document.querySelector('.play-btn');
        if (playBtn) {
            playBtn.innerHTML = '<span class="material-symbols-outlined">play_arrow</span> Play All Best Takes';
        }
    }

    highlightChunk(chapterIndex, chunkIndex) {
        // Remove previous highlights
        document.querySelectorAll('.chunk-text').forEach(el => {
            el.classList.remove('highlight');
        });

        // Add highlight to current chunk
        const chunk = document.querySelector(
            `.chunk-text[data-chapter="${chapterIndex}"][data-chunk="${chunkIndex}"]`
        );

        if (chunk) {
            chunk.classList.add('highlight');
            chunk.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    adjustFontSize(delta) {
        const sizeSpan = document.getElementById('readerFontSize');
        const readerContent = document.getElementById('readerContent');
        let currentSize = parseInt(sizeSpan.textContent);
        currentSize = Math.max(12, Math.min(32, currentSize + delta));
        sizeSpan.textContent = currentSize;
        readerContent.style.fontSize = currentSize + 'px';
    }
}

// Global instance
const readerTab = new ReaderTab();

// Export functions for HTML onclick handlers
window.playAllBestTakes = function() {
    readerTab.playAllBestTakes();
};

window.adjustReaderFontSize = function(delta) {
    readerTab.adjustFontSize(delta);
};
