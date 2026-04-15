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
            temperature: 0.8,
            tts_model: 'chatterbox'  // 'chatterbox' or 'chatterbox_turbo'
        };
        this.activeGenerations = {};
        this.generationQueue = [];
        this.maxParallelGenerations = 3; // Will be loaded from server
        this.selectedChunkId = null;
        this.selectedChunkElement = null; // Cache selected chunk DOM element
        this.audioManager = new AudioManager();
        this.generationStatusToast = null; // Track the single generation status toast

        // Sequential playback state
        this.sequentialPlayback = {
            isPlaying: false,
            currentChunkIndex: 0,
            audio: null
        };

        // Flags: map of chapter_id -> chunk_id -> flag[]
        this.flagsByChapter = {};
    }

    async init() {
        await this.loadVoiceSamples();
        await this.loadDefaultVoice();
        await this.loadProjectDefaults();
        await this.loadDeviceStatus();
        await this.refreshChapters();
        this.setupProgressUpdater();
    }

    async loadDeviceStatus() {
        try {
            const response = await fetch(`${SERVER_URL}/api/device-status`, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (response.ok) {
                const status = await response.json();
                this.maxParallelGenerations = status.max_parallel_generations || 3;
                console.log(`[TTS TAB] Max parallel generations: ${this.maxParallelGenerations}`);
            }
        } catch (error) {
            console.error('Error loading device status:', error);
        }
    }

    async loadDefaultVoice() {
        try {
            const response = await fetch(`${SERVER_URL}/api/config`);
            if (response.ok) {
                const config = await response.json();
                const defaultVoice = config.default_voice || 'Stoker Extended';

                // Load default TTS model from config
                if (config.default_tts_model) {
                    this.projectDefaults.tts_model = config.default_tts_model;
                    console.log('[TTS TAB] Default TTS model:', config.default_tts_model);
                }

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
                    await this.loadProjectFlags();
                    this.renderChapter();
                }
            }
        } catch (error) {
            console.error('Error loading chapter:', error);
        }
    }

    async loadProjectFlags() {
        try {
            const response = await fetch(`${SERVER_URL}/api/project/flags`, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (!response.ok) return;
            const data = await response.json();
            // Index by chapter_id -> chunk_id -> flag[]
            this.flagsByChapter = {};
            for (const flag of (data.flags || [])) {
                if (flag.resolved) continue;
                if (!this.flagsByChapter[flag.chapter_id]) this.flagsByChapter[flag.chapter_id] = {};
                const byChunk = this.flagsByChapter[flag.chapter_id];
                if (!byChunk[flag.chunk_id]) byChunk[flag.chunk_id] = [];
                byChunk[flag.chunk_id].push(flag);
            }
        } catch (e) {
            console.warn('Could not load flags:', e);
        }
    }

    getFlagsForChunk(chunkId) {
        if (!this.currentChapter) return [];
        return this.flagsByChapter[this.currentChapter.id]?.[chunkId] || [];
    }

    async resolveFlag(flagId) {
        await fetch(`${SERVER_URL}/api/project/resolve-flag`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
            body: JSON.stringify({ flag_id: flagId }),
        });
        await this.loadProjectFlags();
        this.renderChapter();
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
                        <div class="project-setting-item">
                            <label class="project-setting-label">TTS Model</label>
                            <div class="project-setting-control">
                                <select id="projectDefaultModel" onchange="ttsTab.updateProjectDefault('tts_model', this.value)">
                                    <option value="chatterbox" ${this.projectDefaults.tts_model === 'chatterbox' ? 'selected' : ''}>Chatterbox (Standard)</option>
                                    <option value="chatterbox_turbo" ${this.projectDefaults.tts_model === 'chatterbox_turbo' ? 'selected' : ''}>Chatterbox Turbo</option>
                                </select>
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

    async updateProjectDefault(key, value) {
        this.projectDefaults[key] = value;
        console.log('[TTS TAB] Updated project default:', key, '=', value);

        // Update all visible per-chunk DOM controls for this setting
        if (key === 'exaggeration') {
            document.querySelectorAll('[id^="exaggeration_chunk_"]').forEach(el => {
                el.value = value;
                const chunkId = el.id.replace('exaggeration_chunk_', '');
                const span = document.getElementById(`exag_val_chunk_preview_${chunkId}`);
                if (span) span.textContent = value;
            });
        } else if (key === 'cfg_weight') {
            document.querySelectorAll('[id^="cfg_weight_chunk_"]').forEach(el => {
                el.value = value;
                const chunkId = el.id.replace('cfg_weight_chunk_', '');
                const span = document.getElementById(`cfg_val_chunk_preview_${chunkId}`);
                if (span) span.textContent = value;
            });
        } else if (key === 'voice_sample') {
            document.querySelectorAll('[id^="voice_chunk_"]').forEach(el => {
                el.value = value;
            });
        } else if (key === 'tts_model') {
            document.querySelectorAll('[id^="model_chunk_"]').forEach(el => {
                el.value = value;
            });
        }

        // Persist to server
        try {
            const resp = await fetch(`${SERVER_URL}/api/project/update-defaults`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({
                    exaggeration: this.projectDefaults.exaggeration,
                    cfg_weight: this.projectDefaults.cfg_weight,
                    voice_sample: this.projectDefaults.voice_sample,
                    temperature: this.projectDefaults.temperature,
                    seed: this.projectDefaults.seed || 0,
                    ref_vad_trimming: this.projectDefaults.ref_vad_trimming || false,
                    tts_model: this.projectDefaults.tts_model || 'chatterbox'
                })
            });
            if (!resp.ok) console.warn('[TTS TAB] Failed to persist project default:', key);
        } catch (e) {
            console.warn('[TTS TAB] Error persisting project default:', e);
        }
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
                          onmouseleave="ttsTab.unhighlightChunk(${chunk.id})">${chunk.text.replace(/\{([^|}]+)\|[^}]*\}/g, '$1')}</span>`;

            // Add space between chunks
            if (index < chunks.length - 1) {
                html += ' ';
            }
        });

        return html;
    }

    /**
     * Check if a take's input text matches the current chunk text
     * Returns true if the take is outdated (text has changed since generation)
     */
    isTakeOutdated(take, currentChunkText) {
        // If no input_text stored (older takes), we can't tell - assume current
        if (!take.input_text) return false;
        // Compare stored input text with current chunk text
        return take.input_text !== currentChunkText;
    }

    renderSingleChunk(chunk, takeCounter = 0) {
        // Renders a single chunk's take rows
        let html = '';

        const audioList = chunk.generated_audios || [];
        const chunkDisplayText = chunk.text.replace(/\{([^|}]+)\|[^}]*\}/g, '$1');
        const chunkPreview = chunkDisplayText.substring(0, 48) + (chunkDisplayText.length > 48 ? '...' : '');
        const chunkPreviewId = `chunk_preview_${chunk.id}`;

        if (audioList.length > 0) {
                // Sort takes by timestamp only (keep generation order)
                const sortedAudio = [...audioList].sort((a, b) => {
                    return (a.timestamp || 0) - (b.timestamp || 0);
                });

                // First take: chunk preview + icons
                const firstTake = sortedAudio[0];
                const isBest = firstTake.is_best_take;
                const isOutdated = this.isTakeOutdated(firstTake, chunk.text);
                const takeId = `take_${chunk.id}_${takeCounter++}`;

                html += `
                    <div class="chunk-take-row ${isBest ? 'best-take' : 'non-best-take'}${isOutdated ? ' outdated-take' : ''}"
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
                                    <span class="material-symbols-outlined">text_to_speech</span>
                                </button>
                                <span class="chunk-preview-text">${chunkPreview}</span>
                                ${(() => {
                                    const flags = this.getFlagsForChunk(chunk.id);
                                    return flags.length ? `<button class="chunk-flag-badge" onclick="event.stopPropagation(); ttsTab.toggleFlagPanel('flagpanel_${chunk.id}')" title="${flags.length} flag${flags.length > 1 ? 's' : ''}">🚩 ${flags.length}</button>` : '';
                                })()}
                            </div>
                            <div class="take-icons">
                                ${isOutdated ? '<span class="material-symbols-outlined outdated-icon" title="Text changed since this take was generated">edit_off</span>' : ''}
                                <button class="take-icon check-circle ${isBest ? 'best' : ''}"
                                        onclick='event.stopPropagation(); ttsTab.setBestTake(${chunk.id}, "${firstTake.audio_file}")'
                                        title="${isBest ? 'Best take' : 'Set as best take'}">
                                    <span class="material-symbols-outlined">${isBest ? 'check_circle' : 'radio_button_unchecked'}</span>
                                </button>
                                <button class="take-icon settings" data-settings-for="${takeId}" onclick="event.stopPropagation(); ttsTab.toggleTakeSettings('${takeId}')" title="View settings">
                                    <span class="material-symbols-outlined">settings</span>
                                </button>
                                <button class="take-icon diff${firstTake.similarity_score !== undefined && firstTake.similarity_score < 85 ? (firstTake.similarity_score >= 60 ? ' diff-warn' : ' diff-error') : ''}" data-diff-for="diff_${takeId}" onclick="event.stopPropagation(); ttsTab.toggleDiffPanel('diff_${takeId}')" title="Compare transcription" ${firstTake.transcription ? '' : 'disabled'}>
                                    <span class="material-symbols-outlined">speaker_notes</span>
                                </button>
                                <button class="take-icon play" onclick="event.stopPropagation(); ttsTab.playTakeAudio('${firstTake.audio_url}', event)" title="Play">
                                    <span class="material-symbols-outlined">play_arrow</span>
                                </button>
                                <button class="take-icon delete" onclick='event.stopPropagation(); ttsTab.deleteTake(${chunk.id}, "${firstTake.audio_file}")' title="Delete">
                                    <span class="material-symbols-outlined">delete</span>
                                </button>
                            </div>
                        </div>
                        <div class="take-diff" id="diff_${takeId}">${this.renderDiffPanel(firstTake)}</div>
                        <div class="take-settings" id="${takeId}">
                            ${isOutdated ? `
                            <div class="setting-row warning-row outdated-warning">
                                <span class="material-symbols-outlined warning-icon">edit_off</span>
                                <span class="warning-text">Outdated: chunk text has changed since this take was generated</span>
                            </div>
                            ` : ''}
                            <div class="setting-row">
                                <span class="setting-label">Voice</span>
                                <span class="setting-value">${firstTake.voice_sample || 'Default'}</span>
                            </div>
                            <div class="setting-row">
                                <span class="setting-label">Model</span>
                                <span class="setting-value take-model-value">${firstTake.tts_model === 'chatterbox_turbo' ? 'Turbo' : 'Standard'}${firstTake.tts_model_emotion_forced ? ' <span class="model-forced-badge" title="Forced by paralinguistic tags">(auto)</span>' : ''}</span>
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
                            ${firstTake.generation_time_ms ? `
                            <div class="setting-row">
                                <span class="setting-label">Gen Time</span>
                                <span class="setting-value">${(firstTake.generation_time_ms / 1000).toFixed(1)}s</span>
                            </div>
                            ` : ''}
                            ${firstTake.similarity_score !== undefined ? `
                            <div class="setting-row">
                                <span class="setting-label">Similarity</span>
                                <span class="setting-value take-score-badge score-${firstTake.similarity_score >= 85 ? 'high' : firstTake.similarity_score >= 60 ? 'mid' : 'low'}">${firstTake.similarity_score}%</span>
                            </div>
                            ` : ''}
                            ${firstTake.possibly_truncated ? `
                            <div class="setting-row warning-row">
                                <span class="material-symbols-outlined warning-icon">warning</span>
                                <span class="warning-text">May be truncated (at 40s TTS limit)</span>
                            </div>
                            ` : ''}
                        </div>
                        ${(() => {
                            const flags = this.getFlagsForChunk(chunk.id);
                            if (!flags.length) return '';
                            const rows = flags.map(f => `
                                <div class="flag-panel-row">
                                    <span class="flag-panel-time">${f.playback_seconds != null ? Math.floor(f.playback_seconds/60)+':'+String(Math.floor(f.playback_seconds%60)).padStart(2,'0') : ''}</span>
                                    <span class="flag-panel-nick">${(f.chunk_nickname || 'Chunk '+f.chunk_id).substring(0,60)}</span>
                                    <button class="flag-panel-resolve" onclick="event.stopPropagation(); ttsTab.resolveFlag('${f.id}')">Resolve</button>
                                </div>`).join('');
                            return `<div class="chunk-flag-panel" id="flagpanel_${chunk.id}">${rows}</div>`;
                        })()}
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
                            <div class="setting-row">
                                <span class="setting-label">TTS Model</span>
                                <div class="setting-control">
                                    <select id="model_chunk_${chunk.id}">
                                        <option value="chatterbox" ${this.projectDefaults.tts_model === 'chatterbox' ? 'selected' : ''}>Chatterbox</option>
                                        <option value="chatterbox_turbo" ${this.projectDefaults.tts_model === 'chatterbox_turbo' ? 'selected' : ''}>Turbo</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>
                `;

                // Additional takes: only icons on right
                for (let i = 1; i < sortedAudio.length; i++) {
                    const take = sortedAudio[i];
                    const isBest = take.is_best_take;
                    const isOutdated = this.isTakeOutdated(take, chunk.text);
                    const takeId = `take_${chunk.id}_${takeCounter++}`;

                    html += `
                        <div class="chunk-take-row additional-take ${isBest ? 'best-take' : 'non-best-take'}${isOutdated ? ' outdated-take' : ''}"
                             data-chunk-id="${chunk.id}">
                            <div class="chunk-take-header"
                                 onclick="ttsTab.highlightChunk(${chunk.id}, true)"
                                 onmouseenter="ttsTab.highlightChunk(${chunk.id})"
                                 onmouseleave="ttsTab.unhighlightChunk(${chunk.id})">
                                <div class="chunk-left"></div>
                                <div class="take-icons">
                                    ${isOutdated ? '<span class="material-symbols-outlined outdated-icon" title="Text changed since this take was generated">edit_off</span>' : ''}
                                    <button class="take-icon check-circle ${isBest ? 'best' : ''}"
                                            onclick='event.stopPropagation(); ttsTab.setBestTake(${chunk.id}, "${take.audio_file}")'
                                            title="${isBest ? 'Best take' : 'Set as best take'}">
                                        <span class="material-symbols-outlined">${isBest ? 'check_circle' : 'radio_button_unchecked'}</span>
                                    </button>
                                    <button class="take-icon settings" data-settings-for="${takeId}" onclick="event.stopPropagation(); ttsTab.toggleTakeSettings('${takeId}')" title="View settings">
                                        <span class="material-symbols-outlined">settings</span>
                                    </button>
                                    <button class="take-icon diff${take.similarity_score !== undefined && take.similarity_score < 85 ? (take.similarity_score >= 60 ? ' diff-warn' : ' diff-error') : ''}" data-diff-for="diff_${takeId}" onclick="event.stopPropagation(); ttsTab.toggleDiffPanel('diff_${takeId}')" title="Compare transcription" ${take.transcription ? '' : 'disabled'}>
                                        <span class="material-symbols-outlined">speaker_notes</span>
                                    </button>
                                    <button class="take-icon play" onclick="event.stopPropagation(); ttsTab.playTakeAudio('${take.audio_url}', event)" title="Play">
                                        <span class="material-symbols-outlined">play_arrow</span>
                                    </button>
                                    <button class="take-icon delete" onclick='event.stopPropagation(); ttsTab.deleteTake(${chunk.id}, "${take.audio_file}")' title="Delete">
                                        <span class="material-symbols-outlined">delete</span>
                                    </button>
                                </div>
                            </div>
                            <div class="take-diff" id="diff_${takeId}">${this.renderDiffPanel(take)}</div>
                            <div class="take-settings" id="${takeId}">
                                ${isOutdated ? `
                                <div class="setting-row warning-row outdated-warning">
                                    <span class="material-symbols-outlined warning-icon">edit_off</span>
                                    <span class="warning-text">Outdated: chunk text has changed since this take was generated</span>
                                </div>
                                ` : ''}
                                <div class="setting-row">
                                    <span class="setting-label">Voice</span>
                                    <span class="setting-value">${take.voice_sample || 'Default'}</span>
                                </div>
                                <div class="setting-row">
                                    <span class="setting-label">Model</span>
                                    <span class="setting-value take-model-value">${take.tts_model === 'chatterbox_turbo' ? 'Turbo' : 'Standard'}${take.tts_model_emotion_forced ? ' <span class="model-forced-badge" title="Forced by paralinguistic tags">(auto)</span>' : ''}</span>
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
                                ${take.generation_time_ms ? `
                                <div class="setting-row">
                                    <span class="setting-label">Gen Time</span>
                                    <span class="setting-value">${(take.generation_time_ms / 1000).toFixed(1)}s</span>
                                </div>
                                ` : ''}
                                ${take.similarity_score !== undefined ? `
                                <div class="setting-row">
                                    <span class="setting-label">Similarity</span>
                                    <span class="setting-value take-score-badge score-${take.similarity_score >= 85 ? 'high' : take.similarity_score >= 60 ? 'mid' : 'low'}">${take.similarity_score}%</span>
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
            html += `
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
                                    <span class="material-symbols-outlined">text_to_speech</span>
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
                            <div class="setting-row">
                                <span class="setting-label">TTS Model</span>
                                <div class="setting-control">
                                    <select id="model_chunk_${chunk.id}">
                                        <option value="chatterbox" ${this.projectDefaults.tts_model === 'chatterbox' ? 'selected' : ''}>Chatterbox</option>
                                        <option value="chatterbox_turbo" ${this.projectDefaults.tts_model === 'chatterbox_turbo' ? 'selected' : ''}>Turbo</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
        }

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
            allTakesHtml += this.renderSingleChunk(chunk, takeCounter);
            takeCounter += (chunk.generated_audios || []).length;
        });

        return allTakesHtml || '<div style="padding: 40px 20px; text-align: center; color: #94a3b8;">No takes available</div>';
    }

    async reloadSingleChunk(chunkId) {
        // Reloads just one chunk's UI after generation completes
        // This preserves progress bars of other generating chunks
        try {
            console.log(`[TTS TAB] Reloading UI for chunk ${chunkId}`);

            // Fetch updated chapter data
            const response = await fetch(`${SERVER_URL}/api/project/info`, {
                headers: {
                    'X-API-Key': API_KEY
                }
            });

            if (!response.ok) {
                console.error('[TTS TAB] Failed to fetch project info');
                return;
            }

            const projectInfo = await response.json();
            const chapters = projectInfo.metadata?.chapters || projectInfo.chapters || [];
            const updatedChapter = chapters[this.currentChapterIndex];

            if (!updatedChapter) {
                console.error('[TTS TAB] Updated chapter not found');
                return;
            }

            // Update the current chapter data
            this.currentChapter = updatedChapter;

            // Find the updated chunk
            const updatedChunk = updatedChapter.chunks.find(c => c.id === chunkId);
            if (!updatedChunk) {
                console.error(`[TTS TAB] Chunk ${chunkId} not found in updated data`);
                return;
            }

            // Render the updated chunk HTML
            const newChunkHtml = this.renderSingleChunk(updatedChunk, 0);

            // Find and replace the old chunk rows in the DOM
            const oldRows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);

            if (oldRows.length === 0) {
                console.warn(`[TTS TAB] No existing rows found for chunk ${chunkId}`);
                return;
            }

            // Create a temporary container to parse the new HTML
            const tempContainer = document.createElement('div');
            tempContainer.innerHTML = newChunkHtml;
            const newRows = tempContainer.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);

            // Replace old rows with new ones
            const firstOldRow = oldRows[0];
            const parent = firstOldRow.parentNode;

            // Insert new rows before the first old row
            newRows.forEach(newRow => {
                parent.insertBefore(newRow, firstOldRow);
            });

            // Remove all old rows
            oldRows.forEach(row => row.remove());

            console.log(`[TTS TAB] Successfully reloaded UI for chunk ${chunkId}`);

        } catch (error) {
            console.error(`[TTS TAB] Error reloading chunk ${chunkId}:`, error);
        }
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

            // Close all settings and diff panels
            document.querySelectorAll('.take-settings.expanded, .take-diff.expanded').forEach(el => {
                el.classList.remove('expanded');
            });
            document.querySelectorAll('.take-icon.settings.expanded, .take-icon.diff.expanded').forEach(el => {
                el.classList.remove('expanded');
            });

            // Toggle this one
            if (!wasExpanded) {
                settingsEl.classList.add('expanded');
                const settingsBtn = document.querySelector(`[data-settings-for="${settingsId}"]`);
                if (settingsBtn) settingsBtn.classList.add('expanded');
            }
        }
    }

    toggleDiffPanel(panelId) {
        const panelEl = document.getElementById(panelId);
        if (panelEl) {
            const wasExpanded = panelEl.classList.contains('expanded');

            // Close all settings and diff panels
            document.querySelectorAll('.take-settings.expanded, .take-diff.expanded').forEach(el => {
                el.classList.remove('expanded');
            });
            document.querySelectorAll('.take-icon.settings.expanded, .take-icon.diff.expanded').forEach(el => {
                el.classList.remove('expanded');
            });

            if (!wasExpanded) {
                panelEl.classList.add('expanded');
                const diffBtn = document.querySelector(`[data-diff-for="${panelId}"]`);
                if (diffBtn) diffBtn.classList.add('expanded');
            }
        }
    }

    toggleFlagPanel(panelId) {
        const panelEl = document.getElementById(panelId);
        if (!panelEl) return;
        const wasOpen = panelEl.classList.contains('open');
        // Close all flag panels first
        document.querySelectorAll('.chunk-flag-panel.open').forEach(el => el.classList.remove('open'));
        if (!wasOpen) panelEl.classList.add('open');
    }

    buildDiffHtml(original, transcription) {
        const norm = t => t.toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ').trim();
        const origWords = norm(original).split(' ').filter(Boolean);
        const transWords = transcription.split(/\s+/).filter(Boolean);
        const transNorm = transWords.map(w => norm(w));

        // LCS DP
        const m = origWords.length, n = transNorm.length;
        const dp = Array.from({length: m + 1}, () => new Array(n + 1).fill(0));
        for (let i = 1; i <= m; i++)
            for (let j = 1; j <= n; j++)
                dp[i][j] = origWords[i-1] === transNorm[j-1]
                    ? dp[i-1][j-1] + 1
                    : Math.max(dp[i-1][j], dp[i][j-1]);

        // Backtrack to mark matched positions in transcription
        const matched = new Set();
        let i = m, j = n;
        while (i > 0 && j > 0) {
            if (origWords[i-1] === transNorm[j-1]) { matched.add(j - 1); i--; j--; }
            else if (dp[i-1][j] >= dp[i][j-1]) i--;
            else j--;
        }

        return transWords.map((word, idx) => {
            const e = word.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return matched.has(idx) ? `<span class="diff-match">${e}</span>` : `<span class="diff-diff">${e}</span>`;
        }).join(' ');
    }

    renderDiffPanel(take) {
        if (!take.transcription || !take.input_text) return '';
        const score = take.similarity_score ?? 0;
        const cls = score >= 85 ? 'score-high' : score >= 60 ? 'score-mid' : 'score-low';
        const diffHtml = this.buildDiffHtml(take.input_text, take.transcription);
        const orig = take.input_text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `<div class="diff-panel-header">
                    <span class="diff-col-label">Original</span>
                    <span class="diff-divider"></span>
                    <span class="diff-col-label">Transcription <span class="score-badge ${cls}">${score}%</span></span>
                </div>
                <div class="diff-cols">
                    <div class="diff-col diff-original">${orig}</div>
                    <div class="diff-col diff-transcription">${diffHtml}</div>
                </div>`;
    }

    async generateChunkAudio(chunkId) {
        // Auto-save editor if it has unsaved changes (ensures pronunciation edits are committed)
        if (typeof gutenbergTab !== 'undefined') {
            const editor = document.getElementById('pseudoXmlEditor');
            if (editor) {
                const editorText = editor.textContent || editor.innerText;
                if (editorText !== gutenbergTab.markdownContent && editorText.length > 0 && !gutenbergTab.cleanViewActive) {
                    console.log('[TTS TAB] Auto-saving editor changes before generation...');
                    await gutenbergTab.saveCode();
                    // Reload chapter to pick up saved changes
                    if (this.currentChapterIndex !== null) {
                        await this.loadChapter(this.currentChapterIndex);
                    }
                }
            }
        }

        const chunk = this.currentChapter.chunks.find(c => c.id === chunkId);
        if (!chunk) {
            console.error('Chunk not found:', chunkId);
            return;
        }

        const voice = document.getElementById(`voice_chunk_${chunkId}`)?.value || this.projectDefaults.voice_sample;
        const exaggeration = document.getElementById(`exaggeration_chunk_${chunkId}`)?.value || this.projectDefaults.exaggeration;
        const cfgWeight = document.getElementById(`cfg_weight_chunk_${chunkId}`)?.value || this.projectDefaults.cfg_weight;
        let ttsModel = document.getElementById(`model_chunk_${chunkId}`)?.value || this.projectDefaults.tts_model || 'chatterbox';

        // Auto-detect paralinguistic tags in chunk text → force turbo
        const hasParalinguisticTags = /\{[^}]*\|\s*\[(?:laugh|chuckle|cough|sigh|gasp|groan|sniff|clear throat|shush)\]\s*\}/.test(chunk.text);
        if (hasParalinguisticTags) {
            ttsModel = 'chatterbox_turbo';
            console.log(`[TTS TAB] Paralinguistic tags detected in chunk ${chunkId}, using Turbo model`);
        }

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

        // Check if we're at max parallel generations
        const activeCount = Object.keys(this.activeGenerations).length;
        if (activeCount >= this.maxParallelGenerations) {
            console.log(`[TTS TAB] Max parallel generations (${this.maxParallelGenerations}) reached, queueing chunk ${chunkId}`);
            this.generationQueue.push({
                chunkId,
                voice,
                exaggeration,
                cfgWeight,
                ttsModel,
                chapterId: this.currentChapter.id,
                chunkText: chunk.text,
                temperature: this.projectDefaults.temperature,
                projectPath: window.currentProjectPath
            });
            this.updateGenerationStatusToast();
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
        const modelLabel = ttsModel === 'chatterbox_turbo' ? 'Turbo' : 'Standard';
        const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
        rows.forEach(row => {
            row.classList.add('generating');
            // Inject model label into the take-icons area for no-takes rows
            const takeIcons = row.querySelector('.take-icons');
            if (takeIcons && row.classList.contains('no-takes')) {
                takeIcons.innerHTML = `<span class="generating-text">Generating... <span class="generating-model-badge">${modelLabel}</span></span>`;
            }
        });

        // Track this generation
        this.activeGenerations[chunkId] = {
            hasExistingTakes: hasExistingTakes,
            progressInterval: null
        };

        // Update generation status toast
        this.updateGenerationStatusToast();

        // If chunk already has takes, create a placeholder for the new take
        if (hasExistingTakes) {
            this.showGeneratingPlaceholder(chunkId, ttsModel);
        }

        // Show progress bar (only on placeholder if regenerating)
        this.showProgressBar(chunkId, hasExistingTakes);
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
                    temperature: this.projectDefaults.temperature,
                    tts_model: ttsModel
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error: ${response.status}`);
            }

            console.log(`[TTS TAB] Audio generated successfully (model: ${ttsModel})`);

            // Capture new audio file for transcription (fire-and-forget after UI reload)
            const newAudioFile = data.audio_file;
            const chunkTextForTranscription = chunk ? chunk.text : null;

            // Clear progress polling for this chunk
            clearInterval(progressInterval);
            this.hideProgressBar(chunkId);
            this.hideGeneratingPlaceholder(chunkId);

            // Remove this chunk from active generations
            delete this.activeGenerations[chunkId];

            // Process next item in queue if any
            this.processNextInQueue();

            // Update generation status toast
            this.updateGenerationStatusToast();

            // Check if other chunks are still generating
            const otherGeneratingChunks = Object.keys(this.activeGenerations);
            const hasOthersGenerating = otherGeneratingChunks.length > 0;

            console.log(`[TTS TAB] Other chunks still generating: ${otherGeneratingChunks.join(', ')}`);

            // Preserve audio player state before reload
            const wasPlaying = this.audioManager.currentAudio && !this.audioManager.currentAudio.paused;
            const playingUrl = wasPlaying ? this.audioManager.currentAudio.src : null;
            const playbackTime = wasPlaying ? this.audioManager.currentAudio.currentTime : 0;

            // Only reload chapter if no other chunks are generating
            // This prevents progress bars from being destroyed while other generations are active
            if (!hasOthersGenerating) {
                console.log('[TTS TAB] No other generations active, reloading chapter');
                await this.loadChapter(this.currentChapterIndex);
            } else {
                console.log('[TTS TAB] Other generations still active, reloading just this chunk');
                // Reload just this chunk's UI to show the new take
                await this.reloadSingleChunk(chunkId);
            }

            // Auto-transcribe the new take (non-blocking)
            if (newAudioFile && chunkTextForTranscription) {
                this.transcribeTake(chunkId, newAudioFile, chunkTextForTranscription);
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

            // Process next item in queue if any
            this.processNextInQueue();

            // Update generation status toast
            this.updateGenerationStatusToast();

            // Remove generating class from take rows
            const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
            rows.forEach(row => row.classList.remove('generating'));

            const errorMsg = error.message || 'Unknown error';
            showToast('Failed to generate audio: ' + errorMsg, 'error');
            this.showErrorPlaceholder(chunkId, errorMsg);
        }
    }

    async processNextInQueue() {
        if (this.generationQueue.length === 0) {
            return;
        }

        const activeCount = Object.keys(this.activeGenerations).length;
        if (activeCount >= this.maxParallelGenerations) {
            return;
        }

        // Find the next item for the current project
        const currentProject = window.currentProjectPath || null;
        const nextIndex = this.generationQueue.findIndex(item =>
            !item.projectPath || item.projectPath === currentProject
        );

        if (nextIndex === -1) {
            // No items for current project in queue
            console.log(`[TTS TAB] No queued items for current project. Queue has ${this.generationQueue.length} items from other projects.`);
            return;
        }

        // Remove the item from the queue
        const nextGen = this.generationQueue.splice(nextIndex, 1)[0];
        console.log(`[TTS TAB] Processing next in queue: chunk ${nextGen.chunkId} from project ${nextGen.projectPath || 'current'}`);

        // If the queued item has stored context (multi-project queue), use it directly
        if (nextGen.chapterId && nextGen.chunkText) {
            // Check if this chunk is still in the current chapter (user might have switched chapters)
            const isCurrentChapter = this.currentChapter && this.currentChapter.id === nextGen.chapterId;

            if (isCurrentChapter) {
                // UI is showing this chapter, can use normal path
                this.generateChunkAudio(nextGen.chunkId);
            } else {
                // Chapter not visible, use direct generation
                await this.generateChunkDirect(nextGen);
            }
        } else {
            // Legacy path: restore UI values and call generateChunkAudio
            const voiceSelect = document.getElementById(`voice_chunk_${nextGen.chunkId}`);
            const exagSlider = document.getElementById(`exaggeration_chunk_${nextGen.chunkId}`);
            const cfgSlider = document.getElementById(`cfg_weight_chunk_${nextGen.chunkId}`);

            if (voiceSelect) voiceSelect.value = nextGen.voice;
            if (exagSlider) exagSlider.value = nextGen.exaggeration;
            if (cfgSlider) cfgSlider.value = nextGen.cfgWeight;

            this.generateChunkAudio(nextGen.chunkId);
        }
    }

    async generateChunkDirect(genContext) {
        const { chunkId, voice, exaggeration, cfgWeight, chapterId, chunkText, temperature, projectPath, ttsModel } = genContext;

        console.log('[TTS TAB] Generating audio directly for chunk:', chunkId, 'from queued project');

        // Track this generation
        this.activeGenerations[chunkId] = {
            hasExistingTakes: false,
            progressInterval: null,
            projectPath: projectPath,
            chapterId: chapterId
        };

        // Update generation status toast
        this.updateGenerationStatusToast();

        // Start progress polling
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
                    text_file_id: chapterId,
                    chunk_id: chunkId,
                    chunk_text: chunkText,
                    voice_sample: voice,
                    exaggeration: parseFloat(exaggeration),
                    cfg_weight: parseFloat(cfgWeight),
                    temperature: temperature,
                    tts_model: ttsModel || 'chatterbox'
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error: ${response.status}`);
            }

            console.log('[TTS TAB] Audio generated successfully (direct)');

            // Clear progress polling
            clearInterval(progressInterval);

            // Remove this chunk from active generations
            delete this.activeGenerations[chunkId];

            // Process next item in queue if any
            this.processNextInQueue();

            // Update generation status toast
            this.updateGenerationStatusToast();

            // Show success toast
            const projectName = projectPath ? projectPath.split('/').pop() : 'current';
            showToast(`Audio generated for queued chunk from "${projectName}"`, 'success');

        } catch (error) {
            console.error('[TTS TAB] Error generating audio:', error);

            // Clear progress polling
            clearInterval(progressInterval);

            // Remove from active generations
            delete this.activeGenerations[chunkId];

            // Process next item in queue
            this.processNextInQueue();

            // Update generation status toast
            this.updateGenerationStatusToast();

            showToast('Generation failed: ' + error.message, 'error');
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

    showProgressBar(chunkId, onlyPlaceholder = false) {
        // If regenerating, only add progress bar to placeholder
        // Otherwise, add to all rows with this chunk-id
        let selector = `.chunk-take-row[data-chunk-id="${chunkId}"]`;
        if (onlyPlaceholder) {
            selector += '.generating-placeholder';
        }

        const rows = document.querySelectorAll(selector);

        rows.forEach(row => {
            // Check if progress bar already exists
            if (row.querySelector('.chunk-progress-bar')) {
                return;
            }

            const progressBar = document.createElement('div');
            progressBar.className = 'chunk-progress-bar';
            progressBar.innerHTML = `
                <div class="chunk-progress-fill" data-chunk-id="${chunkId}"></div>
            `;
            row.appendChild(progressBar);
        });
    }

    hideProgressBar(chunkId) {
        const progressBars = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"] .chunk-progress-bar`);
        progressBars.forEach(bar => bar.remove());
    }

    showGeneratingPlaceholder(chunkId, ttsModel = 'chatterbox') {
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
        const modelLabel = ttsModel === 'chatterbox_turbo' ? 'Turbo' : 'Standard';
        placeholder.innerHTML = `
            <div class="chunk-take-header">
                <div class="chunk-left"></div>
                <div class="take-icons">
                    <span class="generating-text">Generating... <span class="generating-model-badge">${modelLabel}</span></span>
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

        // Skip chunks that already have takes; only generate the new ones.
        const pending = chunks.filter(c => !c.generated_audios || c.generated_audios.length === 0);

        if (pending.length === 0) {
            showToast('All chunks already have takes.', 'info');
            return;
        }

        // Kick off up to maxParallelGenerations immediately; the rest will be queued
        // and processed automatically as each generation completes.
        for (const chunk of pending) {
            this.generateChunkAudio(chunk.id);
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
                const skipped = result.segments_skipped || 0;
                const status = skipped > 0
                    ? `Stitching complete! (${skipped} chunk${skipped > 1 ? 's' : ''} skipped — see modal for details)`
                    : 'Stitching complete!';
                this.showStatus(status, skipped > 0 ? 'warning' : 'success');
            } else {
                const errBody = await response.json().catch(() => ({}));
                throw new Error(errBody.error || `Server error ${response.status}`);
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
                <h3 style="margin-bottom: 15px; color: #1e293b;">
                    <span class="material-symbols-outlined" style="vertical-align: middle; color: #10b981;">check_circle</span>
                    Stitched Audio Ready!
                </h3>
                <p style="margin-bottom: 15px; color: #666;">
                    Stitched ${result.segments_included || result.metadata?.chunk_count || 'all'} segment(s) into one audio file${result.segments_skipped ? ` (${result.segments_skipped} skipped)` : ''}.
                </p>
                ${result.warnings && result.warnings.length > 0 ? `
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 10px 14px; margin-bottom: 15px;">
                    <strong style="color: #92400e;">⚠ Skipped chunks (no audio yet):</strong>
                    <ul style="margin: 6px 0 0 0; padding-left: 18px; color: #92400e; font-size: 13px;">
                        ${result.warnings.map(w => `<li>${w}</li>`).join('')}
                    </ul>
                </div>` : ''}
                <audio controls style="width: 100%; margin-bottom: 15px;">
                    <source src="${SERVER_URL}${result.audio_url}" type="audio/wav">
                </audio>
                <a href="${SERVER_URL}${result.audio_url}" download="${result.audio_file}"
                   class="pane-btn" style="width: 100%; display: block; text-align: center; text-decoration: none; margin-bottom: 10px;">
                    Download
                </a>
                <a href="${SERVER_URL}/listen" target="_blank"
                   class="pane-btn" style="width: 100%; display: block; text-align: center; text-decoration: none; margin-bottom: 10px; background: #334155;">
                    🎧 Open Mobile Listener
                </a>
                <button class="pane-btn secondary" id="closeStitchModal"
                        style="width: 100%; background: #64748b;">
                    Close
                </button>
            </div>
        `;

        // Attach close button handler via JavaScript instead of inline onclick
        const closeBtn = modal.querySelector('#closeStitchModal');
        closeBtn.onclick = () => modal.remove();

        // Click outside to close
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

    updateGenerationStatusToast() {
        const activeCount = Object.keys(this.activeGenerations).length;
        const queuedCount = this.generationQueue.length;

        if (activeCount === 0 && queuedCount === 0) {
            // No generations - remove toast if it exists
            if (this.generationStatusToast) {
                this.generationStatusToast.classList.add('fade-out');
                setTimeout(() => {
                    if (this.generationStatusToast && this.generationStatusToast.parentNode) {
                        this.generationStatusToast.remove();
                    }
                    this.generationStatusToast = null;
                }, 300);
            }
            return;
        }

        // Build status message
        let message = `Generating: ${activeCount} active`;
        if (queuedCount > 0) {
            // Count items from other projects
            const currentProject = window.currentProjectPath || null;
            const otherProjectCount = this.generationQueue.filter(item =>
                item.projectPath && item.projectPath !== currentProject
            ).length;

            message += `, ${queuedCount} queued`;
            if (otherProjectCount > 0) {
                message += ` (${otherProjectCount} from other projects)`;
            }
        }

        if (!this.generationStatusToast || !this.generationStatusToast.parentNode) {
            // Create new toast
            const container = document.getElementById('toastContainer') || this.createToastContainer();

            const toast = document.createElement('div');
            toast.className = 'toast info generation-status-toast';
            toast.innerHTML = `
                <span class="toast-icon material-symbols-outlined">sync</span>
                <div class="toast-content">
                    <div class="toast-message">${message}</div>
                </div>
            `;

            container.appendChild(toast);
            this.generationStatusToast = toast;
        } else {
            // Update existing toast
            const messageEl = this.generationStatusToast.querySelector('.toast-message');
            if (messageEl) {
                messageEl.textContent = message;
            }
        }
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        document.body.appendChild(container);
        return container;
    }

    togglePlayAllBestTakes() {
        if (this.sequentialPlayback.isPlaying) {
            // Pause playback
            this.pauseSequentialPlayback();
        } else {
            // Start or resume playback
            this.startSequentialPlayback();
        }
    }

    startSequentialPlayback() {
        if (!this.currentChapter || !this.currentChapter.chunks) {
            showToast('No chapter loaded', 'error');
            return;
        }

        this.sequentialPlayback.isPlaying = true;
        this.updatePlayAllButton();

        // If no chunk index set, start from beginning
        if (this.sequentialPlayback.currentChunkIndex === undefined ||
            this.sequentialPlayback.currentChunkIndex >= this.currentChapter.chunks.length) {
            this.sequentialPlayback.currentChunkIndex = 0;
        }

        this.playNextBestTake();
    }

    pauseSequentialPlayback() {
        this.sequentialPlayback.isPlaying = false;

        if (this.sequentialPlayback.audio) {
            this.sequentialPlayback.audio.pause();
        }

        this.updatePlayAllButton();
    }

    playNextBestTake() {
        if (!this.sequentialPlayback.isPlaying) {
            return;
        }

        const chunks = this.currentChapter.chunks || [];

        // Find next chunk with a best take
        while (this.sequentialPlayback.currentChunkIndex < chunks.length) {
            const chunk = chunks[this.sequentialPlayback.currentChunkIndex];
            const bestTake = (chunk.generated_audios || []).find(audio => audio.is_best_take);

            if (bestTake) {
                // Found a best take, play it
                console.log(`[TTS TAB] Playing best take for chunk ${this.sequentialPlayback.currentChunkIndex}`);

                const audio = new Audio(SERVER_URL + bestTake.audio_url);
                this.sequentialPlayback.audio = audio;

                audio.addEventListener('ended', () => {
                    this.sequentialPlayback.currentChunkIndex++;
                    this.playNextBestTake();
                });

                audio.addEventListener('error', (e) => {
                    console.error('[TTS TAB] Error playing audio:', e);
                    showToast('Error playing audio, skipping to next', 'error');
                    this.sequentialPlayback.currentChunkIndex++;
                    this.playNextBestTake();
                });

                audio.play().catch(err => {
                    console.error('[TTS TAB] Error starting playback:', err);
                    showToast('Error starting playback', 'error');
                    this.pauseSequentialPlayback();
                });

                return;
            }

            // No best take for this chunk, move to next
            this.sequentialPlayback.currentChunkIndex++;
        }

        // Reached end of chapter
        console.log('[TTS TAB] Sequential playback complete');
        this.sequentialPlayback.currentChunkIndex = 0;
        this.sequentialPlayback.isPlaying = false;
        this.updatePlayAllButton();
        showToast('Playback complete', 'success');
    }

    updatePlayAllButton() {
        const btn = document.getElementById('playAllBestTakesBtn');
        if (!btn) return;

        const icon = btn.querySelector('.material-symbols-outlined');
        if (this.sequentialPlayback.isPlaying) {
            icon.textContent = 'pause';
            btn.innerHTML = icon.outerHTML + ' Pause';
        } else {
            icon.textContent = 'play_arrow';
            btn.innerHTML = icon.outerHTML + ' Play All';
        }
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

    // --- Transcription scoring ---

    renderTranscriptionBar(take) {
        if (take.transcription) {
            const score = take.similarity_score ?? 0;
            const cls = score >= 85 ? 'score-high' : score >= 60 ? 'score-mid' : 'score-low';
            const escaped = take.transcription.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<div class="take-transcription">
                <span class="score-badge ${cls}">${score}%</span>
                <span class="transcription-text" title="${escaped}">${escaped}</span>
            </div>`;
        }
        if (take._transcribing) {
            return `<div class="take-transcription transcribing">
                <span class="material-symbols-outlined" style="font-size:14px;animation:spin 1s linear infinite">progress_activity</span>
                <span>Transcribing...</span>
            </div>`;
        }
        return '';
    }

    async transcribeTake(chunkId, audioFile, chunkText) {
        // Show "transcribing" indicator on the row immediately
        const rowId = `transcription_bar_${chunkId}_${audioFile}`;
        const el = document.getElementById(rowId);
        if (el) {
            el.outerHTML = `<div class="take-transcription transcribing" id="${rowId}">
                <span class="material-symbols-outlined" style="font-size:14px;animation:spin 1s linear infinite">progress_activity</span>
                <span>Transcribing...</span>
            </div>`;
        }

        try {
            const response = await fetch(`${SERVER_URL}/api/project/transcribe-take`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
                body: JSON.stringify({
                    audio_file: audioFile,
                    chunk_text: chunkText,
                    text_file_id: this.currentChapter.id,
                    chunk_id: chunkId
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Transcription failed');

            const score = data.similarity_score ?? 0;
            const cls = score >= 85 ? 'score-high' : score >= 60 ? 'score-mid' : 'score-low';
            const escaped = data.transcription.replace(/</g, '&lt;').replace(/>/g, '&gt;');

            const updated = document.getElementById(rowId);
            if (updated) {
                updated.className = 'take-transcription';
                updated.innerHTML = `<span class="score-badge ${cls}">${score}%</span>
                    <span class="transcription-text" title="${escaped}">${escaped}</span>`;
            }

            // Persist in local chapter data then re-render the chunk row
            const chunk = this.currentChapter.chunks.find(c => c.id === chunkId);
            if (chunk) {
                const entry = (chunk.generated_audios || []).find(a => a.audio_file === audioFile);
                if (entry) {
                    entry.transcription = data.transcription;
                    entry.similarity_score = score;
                    entry.input_text = entry.input_text || chunkText;
                }
                // Re-render the whole chunk row so score badge + diff panel are current
                const oldRows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
                if (oldRows.length > 0) {
                    const tempContainer = document.createElement('div');
                    tempContainer.innerHTML = this.renderSingleChunk(chunk, 0);
                    const newRows = tempContainer.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
                    const parent = oldRows[0].parentNode;
                    newRows.forEach(newRow => parent.insertBefore(newRow, oldRows[0]));
                    oldRows.forEach(row => row.remove());
                }
            }
        } catch (err) {
            console.error('[TTS TAB] Transcription failed:', err);
            const errMsg = err.message || 'Transcription failed';

            // Store error on take entry and re-render the chunk row
            const errChunk = this.currentChapter?.chunks.find(c => c.id === chunkId);
            if (errChunk) {
                const errEntry = (errChunk.generated_audios || []).find(a => a.audio_file === audioFile);
                if (errEntry) {
                    errEntry.transcription_error = errMsg;
                    errEntry.similarity_score = undefined;
                }
                const oldRows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
                if (oldRows.length > 0) {
                    const tempContainer = document.createElement('div');
                    tempContainer.innerHTML = this.renderSingleChunk(errChunk, 0);
                    const newRows = tempContainer.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunkId}"]`);
                    const parent = oldRows[0].parentNode;
                    newRows.forEach(newRow => parent.insertBefore(newRow, oldRows[0]));
                    oldRows.forEach(row => row.remove());
                }
            }
        }
    }
    /**
     * Update dirty indicators on take rows without full re-render.
     * Called by gutenberg_tab after auto-save to instantly show red "text changed" icons.
     */
    updateDirtyIndicators(updatedChapters) {
        if (!this.currentChapter || this.currentChapterIndex === null) return;

        const updatedChapter = updatedChapters[this.currentChapterIndex];
        if (!updatedChapter) return;

        const updatedChunks = updatedChapter.chunks || [];

        for (const chunk of updatedChunks) {
            const rows = document.querySelectorAll(`.chunk-take-row[data-chunk-id="${chunk.id}"]`);
            if (rows.length === 0) continue;

            const audios = chunk.generated_audios || [];
            const hasOutdated = audios.some(a => a.input_text && a.input_text !== chunk.text);

            rows.forEach(row => {
                // Toggle outdated-take class; per-take icons are set on full re-render
                if (hasOutdated) {
                    row.classList.add('outdated-take');
                } else {
                    row.classList.remove('outdated-take');
                }
            });
        }

        // Update local chapter data
        this.currentChapter = updatedChapter;
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
