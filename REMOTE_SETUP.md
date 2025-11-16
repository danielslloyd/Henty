# Remote Server Setup Guide

This guide explains how to set up Henty for remote collaboration, allowing users without a GPU to connect to your server and generate audio.

## Overview

The remote setup consists of:
- **Server (GPU machine)**: Runs the Flask API and handles audio generation
- **Client (collaborator's machine)**: Accesses the web UI and interacts with projects

## Table of Contents

1. [Server Setup (GPU Machine)](#server-setup-gpu-machine)
2. [Client Setup (Collaborator's Machine)](#client-setup-collaborators-machine)
3. [Network Configuration](#network-configuration)
4. [Security Considerations](#security-considerations)
5. [Troubleshooting](#troubleshooting)

---

## Server Setup (GPU Machine)

### 1. Install Dependencies

First, install the new dependencies for remote access:

```bash
pip install -r requirements.txt
```

This installs:
- `flask-socketio` - WebSocket support for real-time updates
- `python-socketio` - SocketIO client library
- `python-dotenv` - Environment variable management

### 2. Configure the Server

Create a `.env` file in the Henty directory:

```bash
# Copy the template
cat > .env << 'EOF'
# Henty Server Configuration

# Server Network Settings
SERVER_HOST=0.0.0.0  # Listen on all network interfaces
SERVER_PORT=5000
SERVER_DEBUG=False

# Authentication (REQUIRED for remote access)
REQUIRE_AUTH=True
API_KEY=your-secret-api-key-here

# CORS - Allowed Origins
# Use * for any origin (less secure) or specify allowed origins
ALLOWED_ORIGINS=*

# WebSocket Support
ENABLE_WEBSOCKET=True

# Max Upload Size (bytes) - 100MB default
MAX_UPLOAD_SIZE=104857600
EOF
```

### 3. Generate a Secure API Key

Generate a strong API key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output and set it as your `API_KEY` in the `.env` file.

**Example:**
```bash
API_KEY=xK9mP2vL8nQ4wR7sT3yU6zA1bC5dE0fG8hJ2kM4pN6qR9sV
```

### 4. Configure Firewall

Allow incoming connections on port 5000:

**Linux (ufw):**
```bash
sudo ufw allow 5000/tcp
```

**Linux (iptables):**
```bash
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
```

**Windows Firewall:**
```powershell
New-NetFirewallRule -DisplayName "Henty Server" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

**macOS:**
```bash
# Add rule in System Preferences > Security & Privacy > Firewall > Firewall Options
```

### 5. Start the Server

```bash
python server.py
```

You should see output like:

```
🔐 NEW API KEY GENERATED
================================================================================

API Key: xK9mP2vL8nQ4wR7sT3yU6zA1bC5dE0fG8hJ2kM4pN6qR9sV

To reuse this key on server restart, set the environment variable:
  export API_KEY='xK9mP2vL8nQ4wR7sT3yU6zA1bC5dE0fG8hJ2kM4pN6qR9sV'

Or add to a .env file:
  API_KEY=xK9mP2vL8nQ4wR7sT3yU6zA1bC5dE0fG8hJ2kM4pN6qR9sV

Share this key securely with your collaborator.
================================================================================

Starting Text to Audio Converter API...
Using device: cuda
Server address: http://0.0.0.0:5000
Authentication: ENABLED
API Key required for protected endpoints
WebSocket support: ENABLED

Allowed CORS origins: *

Ready for connections!
```

**Important:** Share the API key securely with your collaborator (use encrypted messaging, password manager, etc.).

### 6. Find Your Server IP Address

Your collaborator needs your machine's IP address to connect.

**Local Network (LAN):**
```bash
# Linux/macOS
hostname -I | awk '{print $1}'
# or
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig | findstr IPv4
```

**Example output:** `192.168.1.100`

**Public IP (for internet access):**
```bash
curl ifconfig.me
```

⚠️ **Warning:** Exposing your server to the internet requires additional security measures. See [Security Considerations](#security-considerations).

---

## Client Setup (Collaborator's Machine)

### Option 1: Direct File Access (Recommended for LAN)

If your collaborator can access the `index.html` file directly (via shared folder or USB):

1. **Copy the Henty directory** to their machine (without Python dependencies)
2. **Create `client_config.js`** in the same directory as `index.html`:

```javascript
const HentyConfig = {
    // Replace with your server's IP address
    SERVER_URL: 'http://192.168.1.100:5000',

    // API key provided by server administrator
    API_KEY: 'xK9mP2vL8nQ4wR7sT3yU6zA1bC5dE0fG8hJ2kM4pN6qR9sV',

    WEBSOCKET_URL: null,  // Auto-generated
    TIMEOUT: 300000,
    DEBUG: false
};
```

3. **Update `index.html`** to include the config file. Add this line in the `<head>` section, before any other scripts:

```html
<script src="client_config.js"></script>
```

4. **Open `index.html`** in a web browser

The UI will now connect to your remote server!

### Option 2: Simple HTTP Server

If the collaborator wants to host the UI locally:

1. **Copy only the frontend files:**
   - `index.html`
   - `client_config.js` (configured with your server URL and API key)

2. **Start a simple HTTP server:**

```bash
# Python 3
python -m http.server 8080

# Python 2
python -m SimpleHTTPServer 8080

# Node.js (if installed)
npx http-server -p 8080
```

3. **Open browser:** `http://localhost:8080`

---

## Network Configuration

### Local Network (LAN) Setup

**Advantages:**
- Fast connection
- More secure (not exposed to internet)
- Lower latency

**Requirements:**
- Both machines on the same network
- Server IP address (e.g., `192.168.1.100`)
- Port 5000 open on server firewall

**Server URL format:**
```
http://[SERVER_IP]:5000
```

**Example:**
```
http://192.168.1.100:5000
```

### Internet Setup (WAN)

**Advantages:**
- Access from anywhere
- Works across different networks

**Requirements:**
- Port forwarding on router
- Public IP address or dynamic DNS
- Strong authentication
- HTTPS recommended (see below)

**Port Forwarding:**
1. Log into your router admin panel
2. Forward external port 5000 to your server's internal IP (192.168.x.x)
3. Use your public IP for connection

**Dynamic DNS (if you don't have a static IP):**
- Use services like No-IP, DuckDNS, or Dynu
- Automatically updates your domain when IP changes

**Server URL format:**
```
http://[PUBLIC_IP_OR_DOMAIN]:5000
```

**Example:**
```
http://123.45.67.89:5000
http://my-henty-server.duckdns.org:5000
```

---

## Security Considerations

### Essential Security Measures

1. **Use Strong API Keys**
   - Generate with `secrets.token_urlsafe(32)` or similar
   - Minimum 32 characters
   - Change regularly

2. **Limit CORS Origins**
   - Instead of `*`, specify allowed origins:
   ```bash
   ALLOWED_ORIGINS=http://192.168.1.101:8080,http://collaborator-pc:8080
   ```

3. **Firewall Configuration**
   - Only allow necessary ports
   - Use IP whitelisting if possible

4. **Monitor Access**
   - Check server logs regularly
   - Watch for unusual activity

### Recommended for Internet Access

1. **Use HTTPS (SSL/TLS)**
   - Encrypt traffic between client and server
   - Prevents API key interception
   - Use Let's Encrypt for free SSL certificates
   - Set up reverse proxy with nginx or Apache

2. **VPN**
   - Create a VPN for secure remote access
   - Use WireGuard, OpenVPN, or Tailscale
   - Keeps traffic encrypted without exposing ports

3. **Rate Limiting**
   - Add rate limiting to prevent abuse
   - Use Flask-Limiter or nginx rate limiting

4. **Access Logs**
   - Monitor who connects and when
   - Set up log rotation

### Example: Setting up HTTPS with nginx

**Install nginx and certbot:**
```bash
sudo apt install nginx certbot python3-certbot-nginx
```

**Configure nginx reverse proxy:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # WebSocket support
    location /socket.io {
        proxy_pass http://localhost:5000/socket.io;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```

**Get SSL certificate:**
```bash
sudo certbot --nginx -d your-domain.com
```

**Update client config to use HTTPS:**
```javascript
SERVER_URL: 'https://your-domain.com'
```

---

## Troubleshooting

### Connection Issues

**Problem:** "Failed to connect to server"

**Solutions:**
1. Verify server is running: `curl http://localhost:5000/api/status`
2. Check firewall rules
3. Verify IP address is correct
4. Test from server machine first
5. Check if port 5000 is already in use: `netstat -an | grep 5000`

**Problem:** "API key required" or "Invalid API key"

**Solutions:**
1. Verify API key in `client_config.js` matches `.env` file
2. Check for extra spaces or quotes
3. Ensure `REQUIRE_AUTH=True` in server `.env`

**Problem:** WebSocket not connecting

**Solutions:**
1. Check browser console for errors
2. Verify `ENABLE_WEBSOCKET=True` in server config
3. Check CORS settings allow the client origin
4. Some corporate firewalls block WebSockets

### Performance Issues

**Problem:** Slow audio generation

**Solutions:**
1. Check GPU is being used: Server should show "Using device: cuda"
2. Monitor GPU memory with `nvidia-smi`
3. Ensure no other heavy processes running
4. Check network bandwidth for large file transfers

**Problem:** Upload failures

**Solutions:**
1. Check file size against `MAX_UPLOAD_SIZE` setting
2. Increase timeout in `client_config.js`
3. Verify disk space on server

### Permission Issues

**Problem:** Can't create projects or save files

**Solutions:**
1. Check file permissions on server
2. Ensure server has write access to project directories
3. Run server with appropriate user permissions

**Problem:** "Address already in use"

**Solutions:**
1. Port 5000 is already taken
2. Kill existing process: `lsof -ti:5000 | xargs kill -9`
3. Or change port in `.env`: `SERVER_PORT=5001`

---

## Testing the Setup

### 1. Test from Server Machine

```bash
# Test API
curl http://localhost:5000/api/status

# Should return: {"status":"running","device":"cuda","model_loaded":false}

# Test with API key
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:5000/api/voice-samples
```

### 2. Test from Client Machine

Open browser console (F12) and run:

```javascript
HentyConfig.testConnection()
    .then(data => console.log('✅ Connection successful:', data))
    .catch(err => console.error('❌ Connection failed:', err));
```

### 3. Test Full Workflow

1. **Create a project** on the client
2. **Add a text file**
3. **Generate audio** for a chunk
4. **Listen to the result**
5. **Check server logs** for generation stats

---

## Usage Tips

### For the Server Owner

1. **Keep the server running**
   - Use `screen` or `tmux` to run in background:
     ```bash
     screen -S henty
     python server.py
     # Press Ctrl+A, then D to detach
     # Reattach with: screen -r henty
     ```

2. **Monitor resources**
   - Watch GPU usage: `watch -n 1 nvidia-smi`
   - Check disk space: `df -h`
   - Monitor logs: `tail -f server.log`

3. **Backup projects**
   - Regularly backup the project directories
   - Projects are portable folders with all metadata

### For the Collaborator

1. **Save the API key securely**
   - Don't share it publicly
   - Store in password manager

2. **Be patient with generation**
   - Large texts take time
   - Watch the progress bar
   - WebSocket shows real-time updates

3. **Communicate with server owner**
   - Coordinate heavy workloads
   - Report any issues promptly
   - Respect server resources

---

## Advanced Configuration

### Custom Port

Change the port in `.env`:
```bash
SERVER_PORT=8080
```

Update firewall and client config accordingly.

### Multiple Collaborators

Each collaborator can use the same API key, or you can implement per-user keys by modifying `auth.py`.

### Project Sharing

Projects are stored as folders. Share projects by:
1. Compressing the project folder
2. Transferring via cloud storage or network share
3. Client loads project using "Load Project" button

---

## Getting Help

If you encounter issues not covered here:

1. Check server logs for error messages
2. Enable debug mode: `DEBUG=True` in `.env`
3. Check browser console for client-side errors
4. Verify all prerequisites are installed
5. Test with minimal configuration first

---

## Summary

**Quick Setup Checklist:**

**Server:**
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create `.env` with API key
- [ ] Configure firewall
- [ ] Start server: `python server.py`
- [ ] Note IP address and API key

**Client:**
- [ ] Copy `index.html` and create `client_config.js`
- [ ] Set `SERVER_URL` and `API_KEY` in config
- [ ] Include config in HTML
- [ ] Open in browser
- [ ] Test connection

**Done!** Your collaborator can now use your GPU for audio generation! 🎉
