# Python Setup Guide - Install 64-bit Python 3.11

## Current Issue
You have **Python 3.14 (32-bit)** installed, but PyTorch and Chatterbox TTS require **Python 3.10 or 3.11 (64-bit)**.

## Solution: Install Python 3.11 (64-bit)

### Step 1: Download Python 3.11 (64-bit)

1. Go to: https://www.python.org/downloads/release/python-31110/
2. Scroll down to "Files"
3. Download: **Windows installer (64-bit)**
   - File name: `python-3.11.10-amd64.exe`
   - DO NOT download the one that says "x86" (that's 32-bit)

### Step 2: Install Python 3.11

1. Run the downloaded installer
2. **IMPORTANT:** Check these boxes:
   - ✅ **Add Python 3.11 to PATH**
   - ✅ Install launcher for all users
3. Click "Customize installation"
4. Make sure all Optional Features are checked:
   - ✅ pip
   - ✅ tcl/tk and IDLE
   - ✅ Python test suite
   - ✅ py launcher
   - ✅ for all users
5. Click "Next"
6. Advanced Options - Check these:
   - ✅ Install for all users
   - ✅ Associate files with Python
   - ✅ Create shortcuts
   - ✅ Add Python to environment variables
   - ✅ Precompile standard library
7. Click "Install"

### Step 3: Verify Installation

Open a **NEW** Command Prompt or PowerShell window and run:

```bash
python --version
```
Should show: `Python 3.11.10` (or similar)

Check if it's 64-bit:
```bash
python -c "import struct; print(struct.calcsize('P') * 8, 'bit')"
```
Should show: `64 bit`

If you see 32-bit or Python 3.14, you need to adjust your PATH or uninstall the 32-bit version.

### Step 4: Install Dependencies

Once you have 64-bit Python 3.11 installed:

```bash
cd C:\Users\danie\Desktop\Git\Henty
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This should now work without errors!

---

## Alternative: Use Conda (If you prefer)

If you have Anaconda or Miniconda installed:

1. Open Anaconda Prompt
2. Create a new environment:
   ```bash
   conda create -n tts python=3.11 -y
   conda activate tts
   ```

3. Install dependencies:
   ```bash
   cd C:\Users\danie\Desktop\Git\Henty
   pip install -r requirements.txt
   ```

4. Always activate this environment before running the app:
   ```bash
   conda activate tts
   python server.py
   ```

---

## Troubleshooting

### "Python 3.14 still shows up"

Your PATH might have multiple Python installations. To fix:

1. Search Windows for "Environment Variables"
2. Click "Environment Variables"
3. In "System variables", find "Path"
4. Click "Edit"
5. Move the Python 3.11 entries to the TOP of the list
6. Remove or move down Python 3.14 entries
7. Click OK and restart your terminal

### "How do I uninstall Python 3.14?"

1. Go to Settings > Apps > Installed Apps
2. Search for "Python 3.14"
3. Click the three dots > Uninstall

---

## After Installing Python 3.11 (64-bit)

Run these commands to get your app working:

```bash
# Navigate to project folder
cd C:\Users\danie\Desktop\Git\Henty

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (this will take a few minutes)
python -m pip install -r requirements.txt

# Start the server
python server.py
```

Then open `index.html` in your browser and enjoy!
