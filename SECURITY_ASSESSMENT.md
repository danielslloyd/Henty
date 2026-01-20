# Security Assessment - Henty Project
**Date:** 2026-01-20
**Status:** 🚨 CRITICAL EXPOSURE DETECTED

## Executive Summary

Your Gradio annotation interface is **currently exposed to your entire network without authentication**, making it vulnerable to all identified Gradio CVEs. This is a **HIGH RISK** configuration.

---

## Current Configuration Analysis

### What I Found in Your Code

#### 1. Flask Server (server.py) - Port 5000
```python
# From .env file:
SERVER_HOST=127.0.0.1  # ✅ SAFE - Localhost only
REQUIRE_AUTH=False     # ⚠️ Disabled (but not exposed)
```
- **Status:** Currently safe (localhost only)
- **Risk:** LOW (not accessible externally)

#### 2. Gradio Annotator UI (annotator_ui.py) - Port 7860
```python
# From annotator_ui.py line 437:
demo.launch(
    share=False,
    server_name="0.0.0.0",  # 🚨 EXPOSED to all interfaces
    server_port=7860,
    show_error=True
)
```
- **Status:** 🚨 **EXPOSED WITHOUT AUTHENTICATION**
- **Risk:** **CRITICAL** - Vulnerable to all Gradio CVEs

---

## Your Specific Risk Level: 🔴 HIGH

### Why This Is Critical for You

**Gradio is accessible to anyone who can reach your machine's network:**
- ✅ If someone is on your WiFi → They can access it
- ✅ If you're on a corporate/shared network → Coworkers can access it
- ✅ If you have port forwarding setup → Internet can access it
- ✅ If you're on cloud/VPS → Entire internet can access it
- ✅ If you're on university network → Anyone on campus can access it

### Active Vulnerabilities on Port 7860

Since your Gradio UI is on `0.0.0.0:7860` with no authentication, anyone who can reach your network can:

1. **Read Arbitrary Files** (CVE-2024-1561, CVE-2023-51449)
   - Steal files from your system
   - Read environment variables if running with them
   - Access your SSH keys, browser cookies, etc.

2. **Execute Local File Inclusion** (CVE-2024-1728)
   - Access any file the Python process can read
   - Potentially access your trained voice models
   - Read your personal documents

3. **Abuse Your GPU**
   - Use your GPU for their own TTS generation
   - Run arbitrary text processing
   - Consume your compute resources

4. **Data Exfiltration**
   - Download your voice samples
   - Steal your annotated texts
   - Copy your Gutenberg library

---

## Test Your Exposure RIGHT NOW

### Step 1: Check What's Listening
```bash
# Run this command to see what's exposed:
netstat -tuln | grep -E ':(5000|7860)'

# Look for:
# 0.0.0.0:7860 = EXPOSED to all networks (DANGEROUS)
# 127.0.0.1:7860 = Only localhost (SAFE)
```

### Step 2: Find Your IP Address
```bash
# On Linux/Mac:
ip addr show | grep "inet " | grep -v 127.0.0.1

# On Windows:
ipconfig | findstr IPv4
```

### Step 3: Test External Access
From another device on your network:
```
http://[YOUR_IP]:7860
```
**If this loads, you're exposed.**

---

## Which Vulnerabilities Apply to You

### 🔴 CRITICAL - Active Right Now

| CVE | Vulnerability | Can Attacker... | Requires |
|-----|--------------|-----------------|----------|
| **CVE-2024-1561** | Arbitrary File Read | Steal `/proc/self/environ`, API keys, configs | Network access to port 7860 |
| **CVE-2023-51449** | Path Traversal | Read any file on system | Network access to port 7860 |
| **CVE-2024-1728** | Local File Inclusion | Access sensitive documents | Network access to port 7860 |
| **CVE-2024-0964** | Path Traversal (JSON) | Read files via API | Network access to port 7860 |
| **CVE-2024-8021** | Open Redirect | Phish users who trust your server | Network access to port 7860 |

### 🟡 MEDIUM - If You Change Configuration

| CVE | Vulnerability | Matters When... |
|-----|--------------|-----------------|
| Flask-CORS issues | CORS bypasses | You change `SERVER_HOST=0.0.0.0` |

---

## Immediate Actions Required

### Option 1: Make Gradio Localhost Only (Recommended)

Edit `annotator_ui.py` line 437:
```python
demo.launch(
    share=False,
    server_name="127.0.0.1",  # Changed from 0.0.0.0
    server_port=7860,
    show_error=True
)
```

### Option 2: Add Gradio Authentication

Edit `annotator_ui.py` line 435:
```python
demo.launch(
    share=False,
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    auth=("username", "password")  # Add this line
)
```

### Option 3: Update Gradio (Best Practice)

Update to Gradio 4.31.3+ which patches these vulnerabilities:
```bash
pip install 'gradio>=4.31.3'
```

---

## Recommended Actions by Priority

### 🚨 DO IMMEDIATELY (Today)

1. **Test your exposure** (5 minutes)
   - Run the netstat command above
   - Try accessing from another device

2. **Lock down Gradio** (2 minutes)
   - Change `server_name="0.0.0.0"` to `"127.0.0.1"` in `annotator_ui.py`
   - Restart the server

3. **Update dependencies** (10-15 minutes)
   ```bash
   pip install -r requirements-updated.txt
   ```

### 🔴 DO THIS WEEK

4. **Enable Flask authentication** for when you need remote access
   - Edit `.env`: Change `REQUIRE_AUTH=False` to `REQUIRE_AUTH=True`
   - Keep `SERVER_HOST=127.0.0.1` unless you need remote access

5. **Review network exposure**
   - Ensure no port forwarding to 7860 or 5000
   - Check firewall rules
   - Verify router configuration

### 🟠 ONGOING

6. **Keep dependencies updated**
   - Review DEPENDENCY_AUDIT.md quarterly
   - Subscribe to security advisories for Gradio and Flask

---

## How to Safely Allow Remote Access

If you need to let external users access your GPU, here's the secure way:

### For Flask Server (TTS API)

1. **Enable authentication in .env:**
   ```
   REQUIRE_AUTH=True
   SERVER_HOST=0.0.0.0
   ```

2. **Share the API key with trusted users only**

3. **Use a reverse proxy with SSL** (recommended for internet exposure)

### For Gradio UI

1. **Add authentication:**
   ```python
   demo.launch(
       auth=("username", "strong-password"),
       server_name="0.0.0.0",
       server_port=7860
   )
   ```

2. **Or use a VPN** (better):
   - WireGuard, Tailscale, or ZeroTier
   - Keeps Gradio on localhost
   - VPN handles authentication and encryption

---

## Why You Might Not Have Noticed

You probably haven't been attacked yet because:
1. Attackers need to know your IP address
2. Port 7860 isn't commonly scanned (unlike 22, 80, 443)
3. You might be behind a router NAT (but still vulnerable to local network users)
4. The vulnerabilities require specific exploit techniques

**But if someone on your network is malicious or you're on a shared network, you're exposed.**

---

## Questions to Determine Your Risk

Please check:

1. **What network are you on?**
   - [ ] Home WiFi (just you)
   - [ ] Home WiFi (shared with others)
   - [ ] Corporate/office network
   - [ ] University/school network
   - [ ] Public WiFi
   - [ ] Cloud server (AWS, DigitalOcean, etc.)

2. **Do you have port forwarding set up?**
   ```bash
   # Check your router for port forwards to:
   # - Port 5000
   # - Port 7860
   ```

3. **How do external users currently access your GPU?**
   - Describe your setup so I can assess the risk

---

## Additional Security Findings

### API Key in .env File
Your `.env` contains:
```
API_KEY=tq01SBP8SYLpWIbOsmNx8vYRlVMFRxvfMAKXQynCn7o
```

While this key isn't currently being used (auth disabled), ensure:
- Don't commit `.env` to git (already in .gitignore ✅)
- Don't share this key publicly
- Regenerate if you've ever shared it

### No Anthropic/OpenAI Keys Found
Good news: I didn't find API keys for cloud services in your .env file. If you use these services, ensure:
- Keys are in environment variables, not files
- Keys have spending limits set
- Keys are rotated periodically

---

## Summary: What You Need to Do

**Severity: 🔴 HIGH** - You have critical exposure on port 7860

**Immediate Actions:**
1. Test your exposure with netstat
2. Change Gradio to `server_name="127.0.0.1"` OR add auth
3. Update to `gradio>=4.31.3`

**This Week:**
4. Enable Flask authentication in .env
5. Review network security settings

**Tell me:**
- What network are you on?
- Can you run `netstat -tuln | grep -E ':(5000|7860)'` and share the output?
- How do external users currently connect?

Then I can give you more specific guidance.
