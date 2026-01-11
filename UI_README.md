# Henty UI Redesign

## Overview

The Henty UI has been completely overhauled with a new modern interface featuring a landing page and a three-tab application structure.

## File Structure

### Entry Points
- **index.html** - Landing page (choose to create new project or load existing)
- **app.html** - Main application with three tabs
- **index_old.html** - Original single-page interface (backup)
- **reader.html** - Original standalone reader (still available)

### JavaScript Modules
- **gutenberg_tab.js** - Gutenberg Processing tab logic
- **tts_tab.js** - TTS Module tab logic
- **reader_tab.js** - Reader tab logic

### Backend Files (Updated)
- **server.py** - Added new API endpoints:
  - `/api/project/recent` - Get recent projects
  - `/api/project/raw-text` - Get/save raw text
  - `/api/project/validate-chunks` - Validate chunk sizes
  - `/api/project/auto-rechunk` - Auto-rechunk oversized chunks
  - `/api/project/split-chapter` - Split chapter
  - `/api/project/merge-chapters` - Merge adjacent chapters

## Application Flow

### 1. Landing Page (index.html)
- **Create New Project**: Set project name and location
- **Load Existing Project**: Browse to project directory
- **Recent Projects**: Quick access to recently used projects

### 2. Main Application (app.html)

#### Tab 1: Gutenberg Processing
- **Left Pane**: Raw text display
  - Load from Project Gutenberg URL
  - Upload text file
- **Right Pane**: Chapter structure
  - View chapters and chunks
  - Edit pseudo-XML structure
  - Split/merge chapters
  - Validate chunk sizes
  - Auto-rechunk oversized chunks
  - Annotate text (footnotes for TTS to ignore)

#### Tab 2: TTS Module
- **Chapter Selection**: Dropdown to select chapter
- **Audio Settings**:
  - Voice sample selection
  - Exaggeration slider
  - CFG Weight slider
  - Temperature slider
- **Chunk Display**: View all chunks with:
  - Text preview
  - Character count
  - Generated takes
  - Play, set best take, delete takes
- **Batch Operations**:
  - Generate all chunks
  - Stitch best takes

#### Tab 3: Reader
- **Display**: Shows all chapters with text
- **Playback Controls**:
  - Play all best takes sequentially
  - Text highlighting during playback
  - Font size adjustment
- **Features**:
  - Chunks without best takes shown in gray
  - Auto-scroll to playing chunk
  - Stop/resume playback

## Key Features

### Chunk Validation & Auto-Rechunking
The system can now:
- Validate all chunks against a maximum size
- Identify oversized chunks
- Automatically rechunk them with minimal disruption
- Preserve existing audio for unchanged chunks

### Chapter Management
- Split chapters at any position
- Merge adjacent chapters
- Rename chapters
- View chapter statistics (chunk count, character count)

### Pseudo-XML Format
Chapters are displayed in an editable pseudo-XML format:
```xml
<chapter id="uuid" name="Chapter Name">
  <chunk id="0" start="0" end="500">
    Chunk text here...
  </chunk>
</chapter>
```

### Annotator Integration
- Add footnotes to text
- Annotations ignored by TTS module
- Future: Full annotation UI integration

## Migration Notes

### From Old UI
The old UI (index_old.html) remains available but the new UI provides:
- Better organization with separate concerns
- Easier navigation between tasks
- Project-centric workflow
- Better chunk management tools

### Preserving Old Workflows
All existing functionality is preserved:
- Audio generation works the same way
- Voice samples managed identically
- Project structure unchanged
- All API endpoints backward compatible

## Development Notes

### Adding New Features
Each tab is modular:
- **gutenberg_tab.js** - Text processing features
- **tts_tab.js** - Audio generation features
- **reader_tab.js** - Reading and playback features

### API Endpoints
New endpoints follow REST conventions:
- All require API key authentication
- Return JSON responses
- Include error handling
- Update project.json automatically

## Future Enhancements

### Planned Features
1. Full annotator UI in Tab 1
2. Annotation display in Reader
3. Export annotated text
4. Batch chapter processing
5. Audio waveform visualization
6. Generation queue management
7. Project templates
8. Import/export project settings

### Technical Improvements
1. WebSocket for real-time generation updates
2. Service worker for offline capability
3. IndexedDB for local caching
4. React/Vue component migration
5. TypeScript conversion
6. Unit tests for each module

## Troubleshooting

### Landing Page Not Showing Recent Projects
- Check that projects have a valid `project.json`
- Ensure server has access to projects directory
- Check API key configuration

### Tab Not Loading Content
- Verify project is properly loaded
- Check browser console for errors
- Ensure all JavaScript files are loaded
- Check API connectivity

### Audio Playback Issues
- Verify best takes are set
- Check audio file paths in project.json
- Ensure server can serve audio files
- Check browser audio permissions

## Support

For issues, feature requests, or contributions:
- GitHub: https://github.com/danielslloyd/Henty
- File an issue with the label "ui-redesign"
