/**
 * TTS Module Tab Logic
 * Handles audio generation, voice settings, and take management
 */

/**
 * Global audio manager to handle play/pause state and ensure only one audio plays at a time
 */
class AudioManager {
    constructor() {
        this.currentAudio = null;
        this.currentButton = null;
    }

    stop() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
            this.currentAudio = null;
        }
        if (this.currentButton) {
            this.updateButtonIcon(this.currentButton, false);
            this.currentButton = null;
        }
    }

    play(audioUrl, buttonElement) {
        // If clicking the same button, toggle pause
        if (this.currentButton === buttonElement && this.currentAudio && !this.currentAudio.paused) {
            this.currentAudio.pause();
            this.updateButtonIcon(buttonElement, false);
            return;
        }

        // Stop any currently playing audio
        this.stop();

        // Create and play new audio
        const audio = new Audio(`${SERVER_URL}${audioUrl}`);
        this.currentAudio = audio;
        this.currentButton = buttonElement;

        audio.play().catch(err => {
            console.error('Error playing audio:', err);
            this.stop();
        });

        // Update button to show pause icon
        this.updateButtonIcon(buttonElement, true);

        // When audio ends, reset button to play icon
        audio.addEventListener('ended', () => {
            this.updateButtonIcon(buttonElement, false);
            this.currentAudio = null;
            this.currentButton = null;
        });

        // Handle audio errors
        audio.addEventListener('error', () => {
            this.updateButtonIcon(buttonElement, false);
            this.currentAudio = null;
            this.currentButton = null;
        });
    }

    updateButtonIcon(buttonElement, isPlaying) {
        if (!buttonElement) return;
        const icon = buttonElement.querySelector('.material-symbols-outlined');
        if (icon) {
            icon.textContent = isPlaying ? 'pause' : 'play_arrow';
        }
    }
}

class TTSTab {
    constructor() {
        this.currentChapter = null;
        this.currentChapterIndex = null;
        this.voiceSamples = [];
        this.projectDefaults = {
            exaggeration: 0.6,
            cfg_weight: 0.4,
            voice_sample: '',
            temperature: 0.8
        };
        this.activeGenerations = {};
        this.selectedChunkId = null;
        this.audioManager = new AudioManager();
    }

    async init() {
        await this.loadVoiceSamples();
        await this.loadProjectDefaults();
        await this.refreshChapters();
        this.setupProgressUpdater();
    }

    async loadVoiceSamples() {
        try {
            const response = await fetch(`${SERVER_URL}/api/voice-samples`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.voiceSamples = data.samples || [];
                console.log('[TTS TAB] Loaded', this.voiceSamples.length, 'voice samples');
            }
        } catch (error) {
            console.error('Error loading voice samples:', error);
            this.voiceSamples = []; // Ensure it's always an array
        }
    }

    async loadProjectDefaults() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/info`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const projectInfo = await response.json();
                this.projectDefaults = projectInfo.default_audio_settings || this.projectDefaults;
            }
        } catch (error) {
            console.error('Error loading project defaults:', error);
        }
    }

    async refreshChapters() {
        try {
            console.log('[TTS TAB] Refreshing chapters...');
            const response = await fetch(`${SERVER_URL}/api/project/info`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const projectInfo = await response.json();
                const chapters = projectInfo.metadata?.chapters || projectInfo.chapters || [];
                console.log('[TTS TAB] Found', chapters.length, 'chapters');

                // Populate the dropdown
                const select = document.getElementById('ttsChapterSelect');
                if (select) {
                    select.innerHTML = '<option value="">-- Select a chapter --</option>' +
                        chapters.map((chapter, index) =>
                            `<option value="${index}">${chapter.title || chapter.name || `Chapter ${index + 1}`}</option>`
                        ).join('');

                    // Auto-load first chapter if chapters exist
                    if (chapters.length > 0) {
                        console.log('[TTS TAB] Auto-loading first chapter');
                        select.value = '0';
                        await this.loadChapter(0);
                    }
                }
            }
        } catch (error) {
            console.error('Error refreshing chapters:', error);
        }
    }

    async loadChapter(chapterIndex) {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/info`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (response.ok) {
                const projectInfo = await response.json();
                const chapters = projectInfo.metadata?.chapters || projectInfo.chapters || [];
                this.currentChapter = chapters[chapterIndex];

                if (this.currentChapter) {
                    this.renderChapter();
                }
            }
        } catch (error) {
            console.error('Error loading chapter:', error);
        }
    }

    renderChapter() {
        const container = document.getElementById('ttsContent');
        if (!container) {
            console.error('[TTS TAB] Container ttsContent not found');
            return;
        }

        if (!this.currentChapter) {
            container.innerHTML = '<div style="color: #999; text-align: center; padding: 40px;">Select a chapter to view TTS controls</div>';
            return;
        }

        const chunks = this.currentChapter.chunks || [];

        container.innerHTML = `
            <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h2 style="margin: 0;">${this.currentChapter.title || this.currentChapter.name || 'Untitled Chapter'}</h2>
                    <div style="display: flex; gap: 10px;">
                        <button class="pane-btn" onclick="ttsTab.generateAllChunks()">
                            Generate All
                        </button>
                        <button class="pane-btn secondary" onclick="ttsTab.stitchBestTakes()">
                            Stitch Best Takes
                        </button>
                    </div>
                </div>

                <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                    <h3 style="font-size: 14px; margin-bottom: 12px; color: #555;">Audio Settings</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div>
                            <label style="font-size: 12px; color: #666; display: block; margin-bottom: 4px;">Voice Sample</label>
                            <select id="voiceSampleSelect" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" onchange="ttsTab.updateDefaultVoice(this.value)">
                                ${this.renderVoiceSampleOptions()}
                            </select>
                        </div>
                        <div>
                            <label style="font-size: 12px; color: #666; display: block; margin-bottom: 4px;">
                                Exaggeration: <span id="exaggerationValue">${this.projectDefaults.exaggeration}</span>
                            </label>
                            <input type="range" id="exaggerationSlider" min="0" max="1" step="0.1"
                                   value="${this.projectDefaults.exaggeration}"
                                   style="width: 100%;"
                                   oninput="document.getElementById('exaggerationValue').textContent = this.value; ttsTab.projectDefaults.exaggeration = parseFloat(this.value);">
                        </div>
                        <div>
                            <label style="font-size: 12px; color: #666; display: block; margin-bottom: 4px;">
                                CFG Weight: <span id="cfgWeightValue">${this.projectDefaults.cfg_weight}</span>
                            </label>
                            <input type="range" id="cfgWeightSlider" min="0" max="1" step="0.1"
                                   value="${this.projectDefaults.cfg_weight}"
                                   style="width: 100%;"
                                   oninput="document.getElementById('cfgWeightValue').textContent = this.value; ttsTab.projectDefaults.cfg_weight = parseFloat(this.value);">
                        </div>
                        <div>
                            <label style="font-size: 12px; color: #666; display: block; margin-bottom: 4px;">
                                Temperature: <span id="temperatureValue">${this.projectDefaults.temperature}</span>
                            </label>
                            <input type="range" id="temperatureSlider" min="0" max="1" step="0.1"
                                   value="${this.projectDefaults.temperature}"
                                   style="width: 100%;"
                                   oninput="document.getElementById('temperatureValue').textContent = this.value; ttsTab.projectDefaults.temperature = parseFloat(this.value);">
                        </div>
                    </div>
                </div>

                <div id="chunksContainer">
                    ${chunks.map((chunk, index) => this.renderChunk(chunk, index)).join('')}
                </div>
            </div>
        `;
    }

    renderVoiceSampleOptions() {
        return this.voiceSamples.map(sample => `
            <option value="${sample.name}" ${sample.name === this.projectDefaults.voice_sample ? 'selected' : ''}>
                ${sample.name}
            </option>
        `).join('');
    }

    renderChunk(chunk, index) {
        const audios = chunk.generated_audios || [];
        const bestTake = audios.find(a => a.is_best_take);
        const hasAudio = audios.length > 0;

        return `
            <div class="chunk-container" style="background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: #333; margin-bottom: 5px;">
                            Chunk ${chunk.id + 1}
                            ${chunk.dirty ? '<span style="color: #ff9800; font-size: 12px;">● Modified</span>' : ''}
                        </div>
                        <div style="font-size: 13px; color: #666; line-height: 1.5;">
                            ${chunk.text.substring(0, 200)}${chunk.text.length > 200 ? '...' : ''}
                        </div>
                        <div style="font-size: 12px; color: #999; margin-top: 5px;">
                            ${chunk.text.length} characters
                        </div>
                    </div>
                    <button class="pane-btn" style="margin-left: 10px;"
                            onclick="ttsTab.generateChunk(${index})"
                            id="generateBtn_${index}">
                        Generate
                    </button>
                </div>

                <div id="takesContainer_${index}" style="margin-top: 15px;">
                    ${hasAudio ? this.renderTakes(chunk, index) : '<div style="color: #999; font-size: 13px; text-align: center; padding: 10px;">No audio generated yet</div>'}
                </div>

                <div id="generationProgress_${index}" style="display: none; margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 6px;">
                    <div style="font-size: 13px; color: #666; margin-bottom: 5px;">Generating...</div>
                    <div style="background: #ddd; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div id="progressBar_${index}" style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: 0%; transition: width 0.3s;"></div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTakes(chunk, chunkIndex) {
        const audios = chunk.generated_audios || [];

        return `
            <div style="border-top: 1px solid #e0e0e0; padding-top: 10px;">
                <div style="font-size: 13px; font-weight: 600; color: #666; margin-bottom: 8px;">
                    Takes (${audios.length})
                </div>
                ${audios.map((audio, takeIndex) => this.renderTake(audio, chunkIndex, takeIndex)).join('')}
            </div>
        `;
    }

    renderTake(audio, chunkIndex, takeIndex) {
        const isBest = audio.is_best_take;
        const timestamp = new Date(audio.timestamp).toLocaleString();

        return `
            <div style="display: flex; align-items: center; gap: 10px; padding: 8px; background: ${isBest ? '#e8eaf6' : '#f8f9fa'}; border-radius: 6px; margin-bottom: 6px;">
                <button onclick="ttsTab.playAudio('${audio.audio_url}')"
                        style="padding: 6px 12px; background: #667eea; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 12px;">
                    ▶ Play
                </button>
                <div style="flex: 1; font-size: 12px; color: #666;">
                    ${isBest ? '<strong style="color: #667eea;">★ Best Take</strong> · ' : ''}
                    ${timestamp}
                </div>
                ${!isBest ? `
                    <button onclick="ttsTab.setBestTake(${chunkIndex}, ${takeIndex})"
                            style="padding: 4px 8px; background: #f0f0f0; border: none; border-radius: 4px; cursor: pointer; font-size: 11px;">
                        Set as Best
                    </button>
                ` : ''}
                <button onclick="ttsTab.deleteTake(${chunkIndex}, ${takeIndex})"
                        style="padding: 4px 8px; background: #fee; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; color: #c00;">
                    Delete
                </button>
            </div>
        `;
    }

    async generateChunk(chunkIndex) {
        const chunk = this.currentChapter.chunks[chunkIndex];
        const btnId = `generateBtn_${chunkIndex}`;
        const progressId = `generationProgress_${chunkIndex}`;
        const progressBarId = `progressBar_${chunkIndex}`;

        const btn = document.getElementById(btnId);
        const progress = document.getElementById(progressId);

        if (btn) btn.disabled = true;
        if (progress) progress.style.display = 'block';

        try {
            const response = await fetch(`${SERVER_URL}/api/generate-chunk`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    text: chunk.text,
                    text_file_id: this.currentChapter.id,
                    chunk_id: chunk.id,
                    exaggeration: this.projectDefaults.exaggeration,
                    cfg_weight: this.projectDefaults.cfg_weight,
                    temperature: this.projectDefaults.temperature,
                    voice_sample: this.projectDefaults.voice_sample,
                    language_id: 'en'
                })
            });

            if (response.ok) {
                const result = await response.json();

                // Add audio to chunk
                if (!chunk.generated_audios) {
                    chunk.generated_audios = [];
                }

                chunk.generated_audios.push({
                    timestamp: Date.now(),
                    audio_file: result.audio_file,
                    audio_url: result.audio_url,
                    is_best_take: chunk.generated_audios.length === 0,
                    ...this.projectDefaults
                });

                // Re-render chunk
                this.renderChapter();

                this.showStatus('Audio generated successfully!', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to generate audio');
            }
        } catch (error) {
            console.error('Error generating audio:', error);
            this.showStatus('Error: ' + error.message, 'error');
        } finally {
            if (btn) btn.disabled = false;
            if (progress) progress.style.display = 'none';
        }
    }

    async generateAllChunks() {
        if (!confirm('Generate audio for all chunks in this chapter?')) return;

        const chunks = this.currentChapter.chunks || [];

        for (let i = 0; i < chunks.length; i++) {
            await this.generateChunk(i);
        }

        this.showStatus('All chunks generated!', 'success');
    }

    async setBestTake(chunkIndex, takeIndex) {
        const chunk = this.currentChapter.chunks[chunkIndex];
        const audios = chunk.generated_audios || [];

        // Unset all best takes
        audios.forEach(a => a.is_best_take = false);

        // Set the selected one
        audios[takeIndex].is_best_take = true;

        try {
            // Save to backend
            await fetch(`${SERVER_URL}/api/project/set-chunk-best-take`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    text_file_id: this.currentChapter.id,
                    chunk_id: chunk.id,
                    audio_file: audios[takeIndex].audio_file
                })
            });

            this.renderChapter();
            this.showStatus('Best take updated', 'success');
        } catch (error) {
            console.error('Error setting best take:', error);
            this.showStatus('Error setting best take', 'error');
        }
    }

    async deleteTake(chunkIndex, takeIndex) {
        if (!confirm('Delete this take?')) return;

        const chunk = this.currentChapter.chunks[chunkIndex];
        const audios = chunk.generated_audios || [];
        const audioToDelete = audios[takeIndex];

        try {
            await fetch(`${SERVER_URL}/api/delete-take`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    audio_file: audioToDelete.audio_file
                })
            });

            audios.splice(takeIndex, 1);
            this.renderChapter();
            this.showStatus('Take deleted', 'success');
        } catch (error) {
            console.error('Error deleting take:', error);
            this.showStatus('Error deleting take', 'error');
        }
    }

    async stitchBestTakes() {
        if (!this.currentChapter) return;

        this.showStatus('Stitching best takes...', 'info');

        try {
            const response = await fetch(`${SERVER_URL}/api/project/stitch-best-takes`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    text_file_id: this.currentChapter.id
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.showStitchedAudioModal(result);
                this.showStatus('Stitching complete!', 'success');
            } else {
                throw new Error('Failed to stitch audio');
            }
        } catch (error) {
            console.error('Error stitching:', error);
            this.showStatus('Error: ' + error.message, 'error');
        }
    }

    showStitchedAudioModal(result) {
        const modal = document.createElement('div');
        modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center;';
        modal.innerHTML = `
            <div style="background: white; padding: 30px; border-radius: 15px; max-width: 500px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);">
                <h3 style="margin-bottom: 15px; color: #667eea;">✅ Stitched Audio Ready!</h3>
                <p style="margin-bottom: 15px; color: #666;">
                    Successfully combined ${result.metadata?.chunk_count || 'all'} chunks into one audio file.
                </p>
                <audio controls style="width: 100%; margin-bottom: 15px;">
                    <source src="${SERVER_URL}${result.audio_url}" type="audio/wav">
                </audio>
                <a href="${SERVER_URL}${result.audio_url}" download="${result.audio_file}"
                   class="pane-btn" style="width: 100%; display: block; text-align: center; text-decoration: none; margin-bottom: 10px;">
                    Download
                </a>
                <button class="pane-btn secondary" onclick="this.closest('div[style*=\"position: fixed\"]').remove()"
                        style="width: 100%;">
                    Close
                </button>
            </div>
        `;
        modal.onclick = (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        };
        document.body.appendChild(modal);
    }

    playAudio(url) {
        const audio = new Audio(SERVER_URL + url);
        audio.play().catch(err => console.error('Error playing audio:', err));
    }

    updateDefaultVoice(voiceSample) {
        this.projectDefaults.voice_sample = voiceSample;
    }

    showStatus(message, type) {
        // Simple status display (could be enhanced with a proper toast notification)
        console.log(`[${type.toUpperCase()}] ${message}`);
        alert(message);
    }

    setupProgressUpdater() {
        // Update progress bars periodically for active generations
        setInterval(() => {
            Object.keys(this.activeGenerations).forEach(key => {
                this.updateProgress(key);
            });
        }, 500);
    }

    updateProgress(key) {
        const [chunkIndex] = key.split('_');
        const progressBar = document.getElementById(`progressBar_${chunkIndex}`);
        if (progressBar) {
            // Simulate progress (in reality, you'd get this from the backend)
            const currentWidth = parseFloat(progressBar.style.width) || 0;
            if (currentWidth < 90) {
                progressBar.style.width = (currentWidth + 2) + '%';
            }
        }
    }
}

// Global instance
const ttsTab = new TTSTab();

// Export function for HTML dropdown handler
window.loadTTSChapter = function() {
    const select = document.getElementById('ttsChapterSelect');
    const chapterIndex = select.value;

    if (chapterIndex !== '') {
        ttsTab.loadChapter(parseInt(chapterIndex));
    }
};
