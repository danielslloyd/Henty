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
        this.selectedChunkElement = null; // Cache selected chunk DOM element
        this.audioManager = new AudioManager();
    }

    async init() {
        await this.loadVoiceSamples();
        await this.loadDefaultVoice();
        await this.loadProjectDefaults();
        await this.refreshChapters();
        this.setupProgressUpdater();
    }

    async loadDefaultVoice() {
        try {
            const response = await fetch(`${SERVER_URL}/api/config`);
            if (response.ok) {
                const config = await response.json();
                const defaultVoice = config.default_voice || 'Stoker Extended';

                // Find matching voice sample (file extension agnostic)
                const matchingVoice = this.voiceSamples.find(v => {
                    const voiceName = v.name.replace(/\.[^/.]+$/, ''); // Remove extension
                    return voiceName === defaultVoice || v.name === defaultVoice;
                });

                if (matchingVoice) {
                    this.projectDefaults.voice_sample = matchingVoice.name;
                    console.log('[TTS TAB] Default voice set to:', matchingVoice.name);
                } else {
                    // NEVER use default voice - throw error if voice sample not found
                    const errorMsg = `Voice sample "${defaultVoice}" not found in voice_samples directory`;
                    console.error('[TTS TAB]', errorMsg);
                    showToast(errorMsg, 'error');
                    throw new Error(errorMsg);
                }
            }
        } catch (error) {
            console.error('Error loading default voice:', error);
            throw error;
        }
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
                this.currentChapterIndex = chapterIndex;

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
            ${this.renderProjectSettings()}

            <div class="tts-section">
                <h3>All Takes</h3>
                <div class="all-takes-container" id="allTakesDisplay">
                    ${this.renderChunkDetails()}
                </div>
            </div>
        `;
    }

    renderProjectSettings() {
        return `
            <div class="project-settings-box">
                <div class="project-settings-header" onclick="ttsTab.toggleProjectSettings()">
                    <span class="material-symbols-outlined">settings</span>
                    <span>Project Defaults</span>
                </div>
                <div class="project-settings-content" id="projectSettingsContent">
                    <div class="project-settings-grid">
                        <div class="project-setting-item">
                            <label class="project-setting-label">Voice Sample</label>
                            <div class="project-setting-control">
                                <select id="projectDefaultVoice" onchange="ttsTab.updateProjectDefault('voice_sample', this.value)">
                                    ${this.voiceSamples.map(s => `<option value="${s.name}" ${this.projectDefaults.voice_sample === s.name ? 'selected' : ''}>${s.name}</option>`).join('')}
                                </select>
                            </div>
                        </div>
                        <div class="project-setting-item">
                            <label class="project-setting-label">Exaggeration</label>
                            <div class="project-setting-control">
                                <input type="range" id="projectDefaultExaggeration" min="0" max="1" step="0.1" value="${this.projectDefaults.exaggeration}"
                                       oninput="ttsTab.updateProjectDefault('exaggeration', parseFloat(this.value)); document.getElementById('projectExagValue').textContent = this.value">
                                <span class="project-setting-value" id="projectExagValue">${this.projectDefaults.exaggeration}</span>
                            </div>
                        </div>
                        <div class="project-setting-item">
                            <label class="project-setting-label">CFG Weight</label>
                            <div class="project-setting-control">
                                <input type="range" id="projectDefaultCfgWeight" min="0" max="1" step="0.1" value="${this.projectDefaults.cfg_weight}"
                                       oninput="ttsTab.updateProjectDefault('cfg_weight', parseFloat(this.value)); document.getElementById('projectCfgValue').textContent = this.value">
                                <span class="project-setting-value" id="projectCfgValue">${this.projectDefaults.cfg_weight}</span>
                            </div>
                        </div>
                        <div class="project-setting-item">
                            <label class="project-setting-label">Temperature</label>
                            <div class="project-setting-control">
                                <input type="range" id="projectDefaultTemperature" min="0" max="1" step="0.1" value="${this.projectDefaults.temperature}"
                                       oninput="ttsTab.updateProjectDefault('temperature', parseFloat(this.value)); document.getElementById('projectTempValue').textContent = this.value">
                                <span class="project-setting-value" id="projectTempValue">${this.projectDefaults.temperature}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    toggleProjectSettings() {
        const content = document.getElementById('projectSettingsContent');
        if (content) {
            content.classList.toggle('expanded');
        }
    }

    updateProjectDefault(key, value) {
        this.projectDefaults[key] = value;
        console.log('[TTS TAB] Updated project default:', key, '=', value);
    }

    renderChunkBubbles() {
        if (!this.currentChapter) return '';

        const chunks = this.currentChapter.chunks || [];
        if (chunks.length === 0) {
            return '<p style="color: #999; text-align: center;">No chunks available</p>';
        }

        let html = '';
        chunks.forEach((chunk, index) => {
            const audioList = chunk.generated_audios || [];
            const hasAudio = audioList.length > 0;
            const isSelected = this.selectedChunkId === chunk.id;

            const classes = ['chunk-text'];
            if (hasAudio) classes.push('has-audio');
            if (isSelected) classes.push('selected');

            html += `<span class="${classes.join(' ')}"
                          id="chunkText_${chunk.id}"
                          data-chunk-id="${chunk.id}"
                          onclick="ttsTab.selectChunk(${chunk.id})"
                          onmouseenter="ttsTab.highlightChunk(${chunk.id})"
                          onmouseleave="ttsTab.unhighlightChunk(${chunk.id})">${chunk.text}</span>`;

            // Add space between chunks
            if (index < chunks.length - 1) {
                html += ' ';
            }
        });

        return html;
    }

    renderChunkDetails() {
        if (!this.currentChapter) return '';

        const chunks = this.currentChapter.chunks || [];
        if (chunks.length === 0) {
            return '<div style="padding: 40px 20px; text-align: center; color: #94a3b8;">No chunks available</div>';
        }

        let allTakesHtml = '';
        let takeCounter = 0;

        chunks.forEach((chunk) => {
            const audioList = chunk.generated_audios || [];
            const chunkPreview = chunk.text.substring(0, 48) + (chunk.text.length > 48 ? '...' : '');
            const chunkPreviewId = `chunk_preview_${chunk.id}`;

            if (audioList.length > 0) {
                // Sort takes: best take first, then by timestamp
                const sortedAudio = [...audioList].sort((a, b) => {
                    if (a.is_best_take && !b.is_best_take) return -1;
                    if (!a.is_best_take && b.is_best_take) return 1;
                    return (a.timestamp || 0) - (b.timestamp || 0);
                });

                // First take: chunk preview + icons
                const firstTake = sortedAudio[0];
                const isBest = firstTake.is_best_take;
                const takeId = `take_${chunk.id}_${takeCounter++}`;

                allTakesHtml += `
                    <div class="chunk-take-row ${isBest ? 'best-take' : 'non-best-take'}"
                         data-chunk-id="${chunk.id}">
                        <div class="chunk-take-header"
                             onclick="ttsTab.highlightChunk(${chunk.id}, true)"
                             onmouseenter="ttsTab.highlightChunk(${chunk.id})"
                             onmouseleave="ttsTab.unhighlightChunk(${chunk.id})">
                            <div class="chunk-left">
                                <button class="chunk-icon tune" onclick="event.stopPropagation(); ttsTab.selectChunk(${chunk.id}); setTimeout(() => ttsTab.toggleTakeSettings('${chunkPreviewId}_settings'), 100)" title="Tune settings">
                                    <span class="material-symbols-outlined">tune</span>
                                </button>
                                <button class="chunk-icon add" onclick="event.stopPropagation(); ttsTab.selectChunk(${chunk.id}); setTimeout(() => ttsTab.generateChunkAudio(${chunk.id}), 100)" title="Generate take">
                                    <span class="material-symbols-outlined">add</span>
                                </button>
                                <span class="chunk-preview-text">${chunkPreview}</span>
                            </div>
                            <div class="take-icons">
                                <button class="take-icon check-circle ${isBest ? 'best' : ''}"
                                        onclick='event.stopPropagation(); ttsTab.setBestTake(${chunk.id}, "${firstTake.audio_file}")'
                                        title="${isBest ? 'Best take' : 'Set as best take'}">
                                    <span class="material-symbols-outlined">${isBest ? 'check_circle' : 'radio_button_unchecked'}</span>
                                </button>
                                <button class="take-icon settings" data-settings-for="${takeId}" onclick="event.stopPropagation(); ttsTab.toggleTakeSettings('${takeId}')" title="View settings">
                                    <span class="material-symbols-outlined">settings</span>
                                </button>
                                <button class="take-icon play" onclick="event.stopPropagation(); ttsTab.playTakeAudio('${firstTake.audio_url}', event)" title="Play">
                                    <span class="material-symbols-outlined">play_arrow</span>
                                </button>
                                <button class="take-icon delete" onclick='event.stopPropagation(); ttsTab.deleteTake(${chunk.id}, "${firstTake.audio_file}")' title="Delete">
                                    <span class="material-symbols-outlined">delete</span>
                                </button>
                            </div>
                        </div>
                        <div class="take-settings" id="${takeId}">
                            <div class="setting-row">
                                <span class="setting-label">Voice</span>
                                <span class="setting-value">${firstTake.voice_sample || 'Default'}</span>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">Exaggeration</span>
                                <span class="setting-value">${firstTake.exaggeration}</span>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">CFG Weight</span>
                                <span class="setting-value">${firstTake.cfg_weight}</span>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">Temperature</span>
                                <span class="setting-value">${firstTake.temperature || this.projectDefaults.temperature}</span>
                            </div>
                            ${firstTake.audio_duration_seconds ? `
                            <div class="setting-row">
                                <span class="setting-label">Duration</span>
                                <span class="setting-value">${firstTake.audio_duration_seconds}s</span>
                            </div>
                            ` : ''}
                            ${firstTake.possibly_truncated ? `
                            <div class="setting-row warning-row">
                                <span class="material-symbols-outlined warning-icon">warning</span>
                                <span class="warning-text">May be truncated (at 40s TTS limit)</span>
                            </div>
                            ` : ''}
                        </div>
                        <div class="take-settings" id="${chunkPreviewId}_settings">
                            <div class="setting-row">
                                <span class="setting-label">Voice</span>
                                <div class="setting-control">
                                    <select id="voice_chunk_${chunk.id}">
                                        ${this.voiceSamples.map(s => `<option value="${s.name}" ${this.projectDefaults.voice_sample === s.name ? 'selected' : ''}>${s.name}</option>`).join('')}
                                    </select>
                                </div>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">Exaggeration</span>
                                <div class="setting-control">
                                    <input type="range" id="exaggeration_chunk_${chunk.id}" min="0" max="1" step="0.1" value="${this.projectDefaults.exaggeration}"
                                           oninput="document.getElementById('exag_val_${chunkPreviewId}').textContent = this.value">
                                    <span class="setting-value" id="exag_val_${chunkPreviewId}">${this.projectDefaults.exaggeration}</span>
                                </div>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">CFG Weight</span>
                                <div class="setting-control">
                                    <input type="range" id="cfg_weight_chunk_${chunk.id}" min="0" max="1" step="0.1" value="${this.projectDefaults.cfg_weight}"
                                           oninput="document.getElementById('cfg_val_${chunkPreviewId}').textContent = this.value">
                                    <span class="setting-value" id="cfg_val_${chunkPreviewId}">${this.projectDefaults.cfg_weight}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                // Additional takes: only icons on right
                for (let i = 1; i < sortedAudio.length; i++) {
                    const take = sortedAudio[i];
                    const isBest = take.is_best_take;
                    const takeId = `take_${chunk.id}_${takeCounter++}`;

                    allTakesHtml += `
                        <div class="chunk-take-row additional-take ${isBest ? 'best-take' : 'non-best-take'}"
                             data-chunk-id="${chunk.id}">
                            <div class="chunk-take-header"
                                 onclick="ttsTab.highlightChunk(${chunk.id}, true)"
                                 onmouseenter="ttsTab.highlightChunk(${chunk.id})"
                                 onmouseleave="ttsTab.unhighlightChunk(${chunk.id})">
                                <div class="chunk-left"></div>
                                <div class="take-icons">
                                    <button class="take-icon check-circle ${isBest ? 'best' : ''}"
                                            onclick='event.stopPropagation(); ttsTab.setBestTake(${chunk.id}, "${take.audio_file}")'
                                            title="${isBest ? 'Best take' : 'Set as best take'}">
                                        <span class="material-symbols-outlined">${isBest ? 'check_circle' : 'radio_button_unchecked'}</span>
                                    </button>
                                    <button class="take-icon settings" data-settings-for="${takeId}" onclick="event.stopPropagation(); ttsTab.toggleTakeSettings('${takeId}')" title="View settings">
                                        <span class="material-symbols-outlined">settings</span>
                                    </button>
                                    <button class="take-icon play" onclick="event.stopPropagation(); ttsTab.playTakeAudio('${take.audio_url}', event)" title="Play">
                                        <span class="material-symbols-outlined">play_arrow</span>
                                    </button>
                                    <button class="take-icon delete" onclick='event.stopPropagation(); ttsTab.deleteTake(${chunk.id}, "${take.audio_file}")' title="Delete">
                                        <span class="material-symbols-outlined">delete</span>
                                    </button>
                                </div>
                            </div>
                            <div class="take-settings" id="${takeId}">
                                <div class="setting-row">
                                    <span class="setting-label">Voice</span>
                                    <span class="setting-value">${take.voice_sample || 'Default'}</span>
                                </div>
                                <div class="setting-row">
                                    <span class="setting-label">Exaggeration</span>
                                    <span class="setting-value">${take.exaggeration}</span>
                                </div>
                                <div class="setting-row">
                                    <span class="setting-label">CFG Weight</span>
                                    <span class="setting-value">${take.cfg_weight}</span>
                                </div>
                                <div class="setting-row">
                                    <span class="setting-label">Temperature</span>
                                    <span class="setting-value">${take.temperature || this.projectDefaults.temperature}</span>
                                </div>
                                ${take.audio_duration_seconds ? `
                                <div class="setting-row">
                                    <span class="setting-label">Duration</span>
                                    <span class="setting-value">${take.audio_duration_seconds}s</span>
                                </div>
                                ` : ''}
                                ${take.possibly_truncated ? `
                                <div class="setting-row warning-row">
                                    <span class="material-symbols-outlined warning-icon">warning</span>
                                    <span class="warning-text">May be truncated (at 40s TTS limit)</span>
                                </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                }
            } else {
                // No takes: just chunk preview with tune/add icons
                allTakesHtml += `
                    <div class="chunk-take-row no-takes"
                         data-chunk-id="${chunk.id}">
                        <div class="chunk-take-header"
                             onclick="ttsTab.highlightChunk(${chunk.id}, true)"
                             onmouseenter="ttsTab.highlightChunk(${chunk.id})"
                             onmouseleave="ttsTab.unhighlightChunk(${chunk.id})">
                            <div class="chunk-left">
                                <button class="chunk-icon tune" onclick="event.stopPropagation(); ttsTab.selectChunk(${chunk.id}); setTimeout(() => ttsTab.toggleTakeSettings('${chunkPreviewId}_settings'), 100)" title="Tune settings">
                                    <span class="material-symbols-outlined">tune</span>
                                </button>
                                <button class="chunk-icon add" onclick="event.stopPropagation(); ttsTab.selectChunk(${chunk.id}); setTimeout(() => ttsTab.generateChunkAudio(${chunk.id}), 100)" title="Generate take">
                                    <span class="material-symbols-outlined">add</span>
                                </button>
                                <span class="chunk-preview-text">${chunkPreview}</span>
                            </div>
                            <div class="take-icons"></div>
                        </div>
                        <div class="take-settings" id="${chunkPreviewId}_settings">
                            <div class="setting-row">
                                <span class="setting-label">Voice</span>
                                <div class="setting-control">
                                    <select id="voice_chunk_${chunk.id}">
                                        ${this.voiceSamples.map(s => `<option value="${s.name}" ${this.projectDefaults.voice_sample === s.name ? 'selected' : ''}>${s.name}</option>`).join('')}
                                    </select>
                                </div>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">Exaggeration</span>
                                <div class="setting-control">
                                    <input type="range" id="exaggeration_chunk_${chunk.id}" min="0" max="1" step="0.1" value="${this.projectDefaults.exaggeration}"
                                           oninput="document.getElementById('exag_val_${chunkPreviewId}').textContent = this.value">
                                    <span class="setting-value" id="exag_val_${chunkPreviewId}">${this.projectDefaults.exaggeration}</span>
                                </div>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">CFG Weight</span>
                                <div class="setting-control">
                                    <input type="range" id="cfg_weight_chunk_${chunk.id}" min="0" max="1" step="0.1" value="${this.projectDefaults.cfg_weight}"
                                           oninput="document.getElementById('cfg_val_${chunkPreviewId}').textContent = this.value">
                                    <span class="setting-value" id="cfg_val_${chunkPreviewId}">${this.projectDefaults.cfg_weight}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
        });

        return allTakesHtml || '<div style="padding: 40px 20px; text-align: center; color: #94a3b8;">No takes available</div>';
    }

    renderVoiceSampleOptions() {
        return this.voiceSamples.map(sample => `
            <option value="${sample.name}" ${sample.name === this.projectDefaults.voice_sample ? 'selected' : ''}>
                ${sample.name}
            </option>
        `).join('');
    }

    // Interaction methods
    selectChunk(chunkId) {
        this.selectedChunkId = chunkId;

        // Remove 'selected' class from previously selected chunk (cached reference)
        if (this.selectedChunkElement) {
            this.selectedChunkElement.classList.remove('selected');
        }

        // Get and cache the new selected chunk element
        const selectedChunk = document.getElementById(`chunkText_${chunkId}`);
        if (selectedChunk) {
            selectedChunk.classList.add('selected');
            this.selectedChunkElement = selectedChunk; // Cache the reference
        }

        // Scroll to corresponding takes
        this.scrollToTakes(chunkId);
    }

    highlightChunk(chunkId, scroll = false) {
        const chunkElement = document.getElementById(`chunkText_${chunkId}`);
        if (chunkElement) {
            chunkElement.classList.add('highlighted');
        }

        // Highlight corresponding takes
        const takeElements = document.querySelectorAll(`[data-chunk-id="${chunkId}"]`);
        takeElements.forEach(el => el.classList.add('highlighted'));

        // Scroll to chunk if requested (on click)
        if (scroll) {
            if (chunkElement) {
                chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }

    unhighlightChunk(chunkId) {
        const chunkElement = document.getElementById(`chunkText_${chunkId}`);
        if (chunkElement) {
            chunkElement.classList.remove('highlighted');
        }

        // Remove highlight from takes
        const takeElements = document.querySelectorAll(`[data-chunk-id="${chunkId}"]`);
        takeElements.forEach(el => el.classList.remove('highlighted'));
    }

    scrollToTakes(chunkId) {
        const takeElements = document.querySelectorAll(`[data-chunk-id="${chunkId}"]`);
        if (takeElements.length > 0) {
            takeElements[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    playTakeAudio(audioUrl, event) {
        const buttonElement = event ? event.currentTarget : null;
        this.audioManager.play(audioUrl, buttonElement);
    }

    toggleTakeSettings(settingsId) {
        const settingsEl = document.getElementById(settingsId);
        if (settingsEl) {
            const wasExpanded = settingsEl.classList.contains('expanded');

            // Close all other settings panels
            document.querySelectorAll('.take-settings.expanded').forEach(el => {
                el.classList.remove('expanded');
            });

            // Toggle the settings icons
            document.querySelectorAll('.take-icon.settings.expanded').forEach(el => {
                el.classList.remove('expanded');
            });

            // Toggle this one
            if (!wasExpanded) {
                settingsEl.classList.add('expanded');
                // Find and mark the corresponding settings button as expanded
                const settingsBtn = document.querySelector(`[data-settings-for="${settingsId}"]`);
                if (settingsBtn) {
                    settingsBtn.classList.add('expanded');
                }
            }
        }
    }

    async generateChunkAudio(chunkId) {
        const chunk = this.currentChapter.chunks.find(c => c.id === chunkId);
        if (!chunk) {
            console.error('Chunk not found:', chunkId);
            return;
        }

        const voice = document.getElementById(`voice_chunk_${chunkId}`)?.value || this.projectDefaults.voice_sample;
        const exaggeration = document.getElementById(`exaggeration_chunk_${chunkId}`)?.value || this.projectDefaults.exaggeration;
        const cfgWeight = document.getElementById(`cfg_weight_chunk_${chunkId}`)?.value || this.projectDefaults.cfg_weight;

        // NEVER allow generation with missing voice sample
        if (!voice) {
            const errorMsg = 'No voice sample specified';
            showToast(errorMsg, 'error');
            this.showErrorPlaceholder(chunkId, errorMsg);
            return;
        }

        // Validate voice sample exists
        const voiceExists = this.voiceSamples.some(v => {
            const voiceName = v.name.replace(/\.[^/.]+$/, ''); // Remove extension
            const checkVoice = voice.replace(/\.[^/.]+$/, '');
            return voiceName === checkVoice || v.name === voice;
        });

        if (!voiceExists) {
            const errorMsg = `Voice sample "${voice}" not found`;
            showToast(errorMsg, 'error');
            this.showErrorPlaceholder(chunkId, errorMsg);
            return;
        }

        console.log('[TTS TAB] Generating audio for chunk:', chunkId);

        // Check if chunk already has takes
        const hasExistingTakes = chunk.generated_audios && chunk.generated_audios.length > 0;

        // Immediately hide settings and show generating state
        const settingsEl = document.querySelector(`[data-chunk-id="${chunkId}"] .take-settings.expanded`);
        if (settingsEl) {
            settingsEl.classList.remove('expanded');
        }

        // Add generating class to take rows only
        const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
        rows.forEach(row => row.classList.add('generating'));

        // Track this generation
        this.activeGenerations[chunkId] = {
            hasExistingTakes: hasExistingTakes,
            progressInterval: null
        };

        // If chunk already has takes, create a placeholder for the new take
        if (hasExistingTakes) {
            this.showGeneratingPlaceholder(chunkId);
        }

        // Show progress bar
        this.showProgressBar(chunkId);
        const progressInterval = this.startProgressPolling(chunkId);
        this.activeGenerations[chunkId].progressInterval = progressInterval;

        try {
            const response = await fetch(`${SERVER_URL}/api/project/generate-chunk-audio`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    text_file_id: this.currentChapter.id,
                    chunk_id: chunkId,
                    chunk_text: chunk.text,
                    voice_sample: voice,
                    exaggeration: parseFloat(exaggeration),
                    cfg_weight: parseFloat(cfgWeight),
                    temperature: this.projectDefaults.temperature
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error: ${response.status}`);
            }

            console.log('[TTS TAB] Audio generated successfully');

            // Clear progress polling for this chunk
            clearInterval(progressInterval);
            this.hideProgressBar(chunkId);
            this.hideGeneratingPlaceholder(chunkId);

            // Remove this chunk from active generations
            delete this.activeGenerations[chunkId];

            // Preserve audio player state before reload
            const wasPlaying = this.audioManager.currentAudio && !this.audioManager.currentAudio.paused;
            const playingUrl = wasPlaying ? this.audioManager.currentAudio.src : null;
            const playbackTime = wasPlaying ? this.audioManager.currentAudio.currentTime : 0;

            // Preserve other generating chunks' state
            const otherGeneratingChunks = Object.keys(this.activeGenerations).map(id => parseInt(id));

            // Reload chapter to get updated takes
            await this.loadChapter(this.currentChapterIndex);

            // Restore generating state for other chunks that are still generating
            console.log(`[TTS TAB] Restoring ${otherGeneratingChunks.length} other generating chunks`);
            for (const otherChunkId of otherGeneratingChunks) {
                const genInfo = this.activeGenerations[otherChunkId];

                // Skip if this chunk has already finished (race condition)
                if (!genInfo) {
                    console.log(`[TTS TAB] Skipping chunk ${otherChunkId} - already finished`);
                    continue;
                }

                console.log(`[TTS TAB] Restoring chunk ${otherChunkId} with genInfo:`, genInfo);

                const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${otherChunkId}"]`);
                console.log(`[TTS TAB] Found ${rows.length} take rows for chunk ${otherChunkId}`);
                rows.forEach(row => row.classList.add('generating'));

                if (genInfo.hasExistingTakes) {
                    this.showGeneratingPlaceholder(otherChunkId);
                }
                this.showProgressBar(otherChunkId);
            }

            // Restore audio playback if it was playing
            if (wasPlaying && playingUrl) {
                const audioUrl = playingUrl.replace(SERVER_URL, '');
                const buttonElement = document.querySelector(`[onclick*="${audioUrl}"]`);
                if (buttonElement) {
                    // Create new audio and restore playback
                    const audio = new Audio(playingUrl);
                    this.audioManager.currentAudio = audio;
                    this.audioManager.currentButton = buttonElement;
                    audio.currentTime = playbackTime;
                    audio.play().catch(err => console.error('Error restoring audio:', err));
                    this.audioManager.updateButtonIcon(buttonElement, true);

                    audio.addEventListener('ended', () => {
                        this.audioManager.updateButtonIcon(buttonElement, false);
                        this.audioManager.currentAudio = null;
                        this.audioManager.currentButton = null;
                    });

                    audio.addEventListener('error', () => {
                        this.audioManager.updateButtonIcon(buttonElement, false);
                        this.audioManager.currentAudio = null;
                        this.audioManager.currentButton = null;
                    });
                }
            }
        } catch (error) {
            console.error('Error generating chunk audio:', error);
            clearInterval(progressInterval);
            this.hideProgressBar(chunkId);
            this.hideGeneratingPlaceholder(chunkId);

            // Remove this chunk from active generations
            delete this.activeGenerations[chunkId];

            // Remove generating class from take rows
            const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
            rows.forEach(row => row.classList.remove('generating'));

            const errorMsg = error.message || 'Unknown error';
            showToast('Failed to generate audio: ' + errorMsg, 'error');
            this.showErrorPlaceholder(chunkId, errorMsg);
        }
    }

    showErrorPlaceholder(chunkId, errorMessage) {
        const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
        rows.forEach(row => {
            // Remove any existing error placeholders
            const existingError = row.querySelector('.chunk-error-placeholder');
            if (existingError) existingError.remove();

            // Create error placeholder
            const errorPlaceholder = document.createElement('div');
            errorPlaceholder.className = 'chunk-error-placeholder';
            errorPlaceholder.innerHTML = `
                <span class="material-symbols-outlined error-icon">close</span>
                <span class="error-text">${this.escapeHtml(errorMessage)}</span>
            `;
            row.appendChild(errorPlaceholder);
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showProgressBar(chunkId) {
        // Only select take rows, not text preview spans
        const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
        console.log(`[TTS TAB] showProgressBar for chunk ${chunkId}: found ${rows.length} take rows`);

        rows.forEach(row => {
            // Check if progress bar already exists
            if (row.querySelector('.chunk-progress-bar')) {
                console.log(`[TTS TAB] Progress bar already exists for chunk ${chunkId}`);
                return;
            }

            const progressBar = document.createElement('div');
            progressBar.className = 'chunk-progress-bar';
            progressBar.innerHTML = `
                <div class="chunk-progress-fill" data-chunk-id="${chunkId}"></div>
            `;
            row.appendChild(progressBar);
            console.log(`[TTS TAB] Created progress bar for chunk ${chunkId}`);
        });
    }

    hideProgressBar(chunkId) {
        const progressBars = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"] .chunk-progress-bar`);
        progressBars.forEach(bar => bar.remove());
    }

    showGeneratingPlaceholder(chunkId) {
        // Find the last take row for this chunk
        const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
        if (rows.length === 0) return;

        const lastRow = rows[rows.length - 1];

        // Remove any existing placeholder
        this.hideGeneratingPlaceholder(chunkId);

        // Create a placeholder row for the generating take
        const placeholder = document.createElement('div');
        placeholder.className = 'chunk-take-row additional-take generating-placeholder';
        placeholder.setAttribute('data-chunk-id', chunkId);
        placeholder.innerHTML = `
            <div class="chunk-take-header">
                <div class="chunk-left"></div>
                <div class="take-icons">
                    <span class="generating-text">Generating take...</span>
                </div>
            </div>
        `;

        // Insert after the last take row
        lastRow.parentNode.insertBefore(placeholder, lastRow.nextSibling);
    }

    hideGeneratingPlaceholder(chunkId) {
        const placeholders = document.querySelectorAll(`[data-chunk-id="${chunkId}"].generating-placeholder`);
        placeholders.forEach(p => p.remove());
    }

    startProgressPolling(chunkId) {
        return setInterval(async () => {
            try {
                const response = await fetch(`${SERVER_URL}/api/generation-progress`, {
                    headers: { 'X-API-Key': API_KEY }
                });
                const data = await response.json();

                if (data.in_progress) {
                    const progress = data.progress_percent || 0;
                    const fills = document.querySelectorAll(`.chunk-progress-fill[data-chunk-id="${chunkId}"]`);
                    fills.forEach(fill => {
                        fill.style.width = `${Math.min(progress, 95)}%`;
                    });
                }
            } catch (error) {
                console.error('Error polling progress:', error);
            }
        }, 1000); // Poll every 1000ms (1 second) - reduced from 200ms for better performance
    }

    async setBestTake(chunkId, audioFile) {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/set-chunk-best-take`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    text_file_id: this.currentChapter.id,
                    chunk_id: chunkId,
                    audio_filename: audioFile
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to set best take');
            }

            console.log('[TTS TAB] Best take updated');

            // Reload chapter to reflect changes
            await this.loadChapter(this.currentChapterIndex);
        } catch (error) {
            console.error('Error setting best take:', error);
            showToast('Failed to set best take: ' + error.message, 'error');
        }
    }

    async deleteTake(chunkId, audioFile) {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/delete-audio`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY
                },
                body: JSON.stringify({
                    text_file_id: this.currentChapter.id,
                    chunk_id: chunkId,
                    audio_file: audioFile
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to delete take');
            }

            console.log('[TTS TAB] Take deleted');

            // Reload chapter to reflect changes
            await this.loadChapter(this.currentChapterIndex);
        } catch (error) {
            console.error('Error deleting take:', error);
            showToast('Failed to delete take: ' + error.message, 'error');
        }
    }

    async generateAllChunks() {
        const chunks = this.currentChapter.chunks || [];

        // Generate all chunks in parallel using Promise.all() for better performance
        try {
            await Promise.all(chunks.map(chunk => this.generateChunkAudio(chunk.id)));
            showToast('All chunks generated successfully!', 'success');
        } catch (error) {
            console.error('Error generating chunks:', error);
            showToast('Some chunks failed to generate. Check console for details.', 'error');
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
        // Display toast notification
        console.log(`[${type.toUpperCase()}] ${message}`);
        showToast(message, type);
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
