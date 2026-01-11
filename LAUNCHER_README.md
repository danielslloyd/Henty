# Henty Launcher Scripts

Quick start scripts to launch the Henty server and open the landing page in your browser.

## Quick Start

### Linux/Mac

**Option 1: Shell Script (Recommended)**
```bash
./start_henty.sh
```

**Option 2: Python Script**
```bash
python3 start_henty.py
```

### Windows

**Option 1: Batch File (Recommended)**
Double-click `start_henty.bat` or run from command prompt:
```cmd
start_henty.bat
```

**Option 2: Python Script**
```cmd
python start_henty.py
```

## What the Launchers Do

1. ✅ Check if Python is installed
2. ✅ Check if server.py exists
3. 🚀 Start the Flask server on http://localhost:5000
4. ⏳ Wait a few seconds for server initialization
5. 🌐 Open your default browser to the landing page
6. 📊 Display server information and URLs

## Stopping the Server

### Linux/Mac (Shell Script)
- Press `Ctrl+C` in the terminal

### Windows (Batch File)
- Close the "Henty Server" window that opens

### Python Script (All Platforms)
- **Linux/Mac**: Press `Ctrl+C`
- **Windows**: Close the server window or press `Ctrl+C`

## Troubleshooting

### "Python is not installed or not in PATH"
- Install Python 3.8 or higher from https://python.org
- Make sure Python is added to your system PATH

### "server.py not found"
- Make sure you're running the script from the Henty directory
- Navigate to the Henty folder first: `cd /path/to/Henty`

### Browser doesn't open automatically
- Manually open your browser and go to: http://localhost:5000/index.html

### Port 5000 is already in use
- Stop any existing server: `pkill -f "python.*server.py"` (Linux/Mac)
- Or use Task Manager to end the Python process (Windows)

### Permission denied (Linux/Mac)
Make the scripts executable:
```bash
chmod +x start_henty.sh
chmod +x start_henty.py
```

## Manual Start (Alternative)

If the launchers don't work, you can start manually:

1. **Start the server:**
   ```bash
   python3 server.py
   ```

2. **Open browser to:**
   ```
   http://localhost:5000/index.html
   ```

## Configuration

The launchers use default settings:
- **Server URL**: http://localhost:5000
- **Browser delay**: 3 seconds (to allow server startup)

To customize, edit the `.env` file in the Henty directory.

## Files

- `start_henty.sh` - Linux/Mac shell script
- `start_henty.bat` - Windows batch file
- `start_henty.py` - Cross-platform Python script
- `server.py` - The main Flask server
- `index.html` - Landing page
- `app.html` - Main application

## Support

For issues or questions:
- Check the main README.md
- Check UI_README.md for UI-specific documentation
- Open an issue on GitHub
