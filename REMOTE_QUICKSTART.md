# Remote Access Quick Start

Get your Henty server running for remote collaboration in 5 minutes!

## For the Server Owner (GPU Machine)

### 1. Run the Setup Script

```bash
python setup_remote.py
```

This will:
- ✅ Check dependencies
- ✅ Generate a secure API key
- ✅ Create `.env` configuration
- ✅ Create `client_config.js` for your collaborator
- ✅ Display your IP address

### 2. Configure Firewall

**Linux:**
```bash
sudo ufw allow 5000/tcp
```

**Windows (PowerShell as Admin):**
```powershell
New-NetFirewallRule -DisplayName "Henty Server" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

### 3. Start the Server

```bash
python server.py
```

Look for this output:
```
API Key: xK9mP2vL8nQ4wR7sT3yU6zA1bC5dE0fG8hJ2kM4pN6qR9sV
Server address: http://0.0.0.0:5000
Authentication: ENABLED
```

### 4. Share with Collaborator

Send your collaborator:
1. The `client_config.js` file (contains server URL and API key)
2. The `index.html` file
3. Instructions to open `index.html` in their browser

**⚠️ Security:** Share the API key through secure channels (encrypted messaging, password manager, etc.)

---

## For the Collaborator (No GPU Needed)

### 1. Get Files from Server Owner

You need:
- `index.html`
- `client_config.js` (configured with server URL and API key)

### 2. Open in Browser

**Option A: Direct File Access**
```
Simply open index.html in your web browser
```

**Option B: Local HTTP Server**
```bash
# If you have Python installed
python -m http.server 8080

# Then open: http://localhost:8080
```

### 3. Start Using Henty!

You can now:
- ✅ Create and load projects
- ✅ Add text files
- ✅ Generate audio (runs on the server's GPU)
- ✅ Listen to generated audio
- ✅ Manage voice samples
- ✅ Export final audio files

All audio generation happens on the server - you just need a web browser!

---

## Troubleshooting

### Can't Connect?

1. **Check server is running** - Ask server owner to verify
2. **Verify IP address** - Make sure `SERVER_URL` in `client_config.js` is correct
3. **Check firewall** - Port 5000 must be open on server
4. **Try local connection first** - Server owner should test with `http://localhost:5000`

### "API key required" Error?

1. **Check the API key** - Make sure it matches exactly (no extra spaces)
2. **Verify it's set** - Open `client_config.js` and check `API_KEY` field

### Slow Performance?

- Audio generation is CPU/GPU intensive and takes time
- Check your network connection
- Large text files take longer to process
- Watch the progress bar for estimated time

---

## Network Options

### Local Network (LAN) - Recommended

**Best for:**
- Same WiFi network
- Same office/home

**Server URL example:**
```
http://192.168.1.100:5000
```

**Pros:** Fast, secure, low latency

### Internet (WAN)

**Best for:**
- Remote locations
- Different networks

**Server URL example:**
```
http://123.45.67.89:5000
```

**Requirements:**
- Port forwarding on router
- Consider using HTTPS (see REMOTE_SETUP.md)

---

## Need More Help?

- **Detailed Setup:** See `REMOTE_SETUP.md` for comprehensive instructions
- **Server Config:** Check `.env` file for server settings
- **Client Config:** Check `client_config.js` for connection settings
- **Security:** See security section in `REMOTE_SETUP.md` for HTTPS and VPN setup

---

## Features Available Remotely

All Henty features work remotely:

- ✅ **Project Management:** Create, load, and manage projects
- ✅ **Text Processing:** Add files, chunk text, edit chunks
- ✅ **Audio Generation:** Generate audio with full parameter control
- ✅ **Voice Cloning:** Upload and use custom voice samples
- ✅ **Multi-Take Recording:** Generate multiple takes, select best
- ✅ **Audio Stitching:** Combine chunks into final output
- ✅ **Real-time Progress:** WebSocket updates show generation progress
- ✅ **Project Gutenberg:** Process books directly from URLs

Everything runs on the server's GPU - collaborators just need a web browser! 🚀
