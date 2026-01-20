# Dependency Audit Report
**Generated:** 2026-01-20

## Executive Summary

This audit analyzed the Python dependencies in `requirements.txt` for outdated packages, security vulnerabilities, and unnecessary bloat. **Critical security vulnerabilities were found** in `gradio>=4.0.0` and `flask-cors>=4.0.0` that require immediate attention.

### Key Findings
- ⚠️ **2 CRITICAL security vulnerabilities** requiring immediate updates
- 📦 **6 packages significantly outdated** (2+ major/minor versions behind)
- ✅ **All dependencies are necessary** (no bloat identified)
- 🔒 **4 high-severity CVEs** affecting current dependency versions

---

## Security Vulnerabilities

### 🚨 CRITICAL: Gradio 4.0.0 (IMMEDIATE UPDATE REQUIRED)

**Current:** `gradio>=4.0.0`
**Recommended:** `gradio>=4.31.3`
**Severity:** CRITICAL

#### Vulnerabilities:
1. **CVE-2023-51449** - Path Traversal (High Severity)
   - Affects: Gradio 4.0 – 4.10
   - Risk: Allows reading arbitrary files from the server

2. **CVE-2024-1561** - Arbitrary File Read (High Severity)
   - Affects: Gradio 3.47 to 4.12
   - Risk: Can access secrets in environment variables via /proc/self/environ

3. **CVE-2024-0964** - Path Traversal
   - Risk: Remote local file inclusion via vulnerable JSON API requests

4. **CVE-2024-1728** - Local File Inclusion
   - Affects: Up to Gradio 4.25
   - Fixed in: 4.31.3

5. **CVE-2024-8021** - Open Redirect
   - Risk: URL encoding allows redirects to malicious websites

6. **CVE-2024-47871** - Insecure Communication
   - Risk: share=True uses unencrypted connections, allowing file interception

**Impact:** Attackers could read sensitive files, steal API keys, and access environment variables.

---

### ⚠️ HIGH: Flask-CORS 4.0.0 (UPDATE REQUIRED)

**Current:** `flask-cors>=4.0.0`
**Recommended:** `flask-cors>=6.0.2`
**Severity:** HIGH

#### Vulnerabilities:
1. **CVE-2024-1681** - Log Injection
   - Affects: flask-cors 4.0.0
   - Risk: CRLF injection in logs via crafted GET requests

2. **CVE-2024-6221** - Improper Access Control
   - Risk: Access-Control-Allow-Private-Network set to true by default
   - Impact: Exposes private network resources to external access

3. **CVE-2024-6844** - Inconsistent CORS Matching
   - Risk: '+' character handling causes unauthorized cross-origin access

4. **CVE-2024-6866** - Case Sensitivity Handling
   - Risk: Case-insensitive path matching causes security bypasses

**Impact:** CORS security bypasses and unauthorized access to protected resources.

---

## Outdated Packages

### Major Updates Required

| Package | Current Version | Latest Version | Status | Priority |
|---------|----------------|----------------|--------|----------|
| **gradio** | ≥4.0.0 | 6.3.0 | 2 major versions behind | 🔴 CRITICAL |
| **flask-cors** | ≥4.0.0 | 6.0.2 | 2 major versions behind | 🔴 CRITICAL |
| **anthropic** | ≥0.39.0 | 0.76.0 | ~37 versions behind | 🟠 HIGH |
| **torch** | ≥2.0.0 | 2.9.1 | 9 minor versions behind | 🟠 HIGH |
| **torchaudio** | ≥2.0.0 | 2.9.1 | 9 minor versions behind | 🟠 HIGH |
| **scipy** | ≥1.11.0 | 1.17.0 | 6 minor versions behind | 🟡 MEDIUM |

### Minor Updates Available

| Package | Current Version | Latest Version | Notes |
|---------|----------------|----------------|-------|
| **numpy** | ≥2.0.0 | 2.4.1 | Minor updates available |
| **flask** | ≥3.0.0 | 3.1.2 | Minor updates available |
| **flask-socketio** | ≥5.3.0 | 5.6.0 | Minor updates available |
| **python-socketio** | ≥5.9.0 | 5.16.0 | 7 patch versions behind |
| **python-dotenv** | ≥1.0.0 | 1.2.1 | Minor updates available |
| **ollama** | ≥0.4.0 | 0.6.1 | Minor updates available |

### Up-to-Date Packages ✅

- **requests** (2.31.0 → 2.32.5) - Current, only patch update
- **pydub** (0.25.1) - Current
- **chatterbox-tts** (no version pinned, latest: 0.1.6)

---

## Dependency Analysis

### All Dependencies Are Necessary ✅

After analyzing the codebase, **all dependencies in requirements.txt are actively used**:

#### Core Web Server Dependencies
- `flask`, `flask-cors`, `flask-socketio`, `python-socketio` - Used in `server.py` for the main web application

#### TTS/Audio Processing Dependencies
- `torch`, `torchaudio`, `chatterbox-tts` - Used for text-to-speech functionality
- `numpy`, `scipy`, `pydub` - Audio processing and manipulation

#### UI Dependencies
- `gradio` - Used in `annotator_ui.py` (main interface) and `app.py`

#### Utility Dependencies
- `requests` - Used in `scripts/gutenberg_processor.py` for downloading texts
- `python-dotenv` - Configuration management in `config.py`

#### Optional LLM Dependencies
- `anthropic` - Used in `scripts/text_annotator.py` (optional, when using Anthropic backend)
- `ollama` - Used in `scripts/text_annotator.py` (optional, when using Ollama backend)

**Verdict:** No unnecessary bloat detected. All packages serve specific functionality.

---

## Recommendations

### Immediate Actions (Critical Priority)

1. **Update Gradio** - Critical security vulnerabilities
   ```txt
   gradio>=4.31.3
   ```

2. **Update Flask-CORS** - Multiple CORS security issues
   ```txt
   flask-cors>=6.0.2
   ```

### High Priority Updates

3. **Update Anthropic SDK** - Significant API improvements and bug fixes
   ```txt
   anthropic>=0.76.0
   ```

4. **Update PyTorch Stack** - Performance improvements and bug fixes
   ```txt
   torch>=2.9.1
   torchaudio>=2.9.1
   ```

### Medium Priority Updates

5. **Update Scientific Computing Stack**
   ```txt
   scipy>=1.17.0
   numpy>=2.4.1
   ```

6. **Update Flask Stack**
   ```txt
   flask>=3.1.2
   flask-socketio>=5.6.0
   python-socketio>=5.16.0
   ```

7. **Update Utilities**
   ```txt
   python-dotenv>=1.2.1
   ollama>=0.6.1
   requests>=2.32.5
   ```

### No Changes Needed

- `pydub>=0.25.1` - Already current
- `chatterbox-tts` - No version constraints (appropriate for this package)

---

## Updated requirements.txt

```txt
# Core Web Framework
flask>=3.1.2
flask-cors>=6.0.2
flask-socketio>=5.6.0
python-socketio>=5.16.0

# Machine Learning / TTS
torch>=2.9.1
torchaudio>=2.9.1
chatterbox-tts

# Scientific Computing
numpy>=2.4.1
scipy>=1.17.0

# Audio Processing
pydub>=0.25.1

# UI Framework
gradio>=4.31.3

# HTTP Client
requests>=2.32.5

# Utilities
python-dotenv>=1.2.1

# LLM Providers (optional)
anthropic>=0.76.0
ollama>=0.6.1
```

---

## Testing Recommendations

After updating dependencies:

1. **Test the Gradio UI** (`annotator_ui.py`)
   - Verify Gutenberg download functionality
   - Test text annotation with both Ollama and Anthropic backends
   - Check file upload functionality

2. **Test the TTS Server** (`server.py`)
   - Verify audio generation works
   - Test WebSocket connections
   - Check CORS configuration

3. **Test the Reader** (`reader.html`)
   - Verify annotations display correctly
   - Test responsive design
   - Check audio playback

4. **Verify Breaking Changes**
   - Review Gradio 6.x migration guide (major version jump)
   - Review Flask-CORS 6.x changelog
   - Test PyTorch model compatibility with 2.9.x

---

## Sources

### Security Vulnerabilities
- [CVE-2024-1681: Flask-CORS Log Injection](https://www.cvedetails.com/cve/CVE-2024-1681/)
- [CVE-2024-6221: Flask-CORS Access Control](https://github.com/advisories/ghsa-hxwh-jpp2-84pm)
- [CVE-2024-6844: Flask-CORS Inconsistent Matching](https://advisories.gitlab.com/pkg/pypi/flask-cors/CVE-2024-6844/)
- [CVE-2024-6866: Flask-CORS Case Sensitivity](https://advisories.gitlab.com/pkg/pypi/flask-cors/CVE-2024-6866/)
- [Flask-CORS Vulnerabilities (Snyk)](https://security.snyk.io/package/pip/Flask-Cors)
- [CVE-2023-51449: Gradio Path Traversal](https://github.com/advisories/GHSA-f3h9-8phc-6gvh)
- [CVE-2024-1561: Gradio Arbitrary File Read](https://blog.certcube.com/gradio-arbitrary-file-read-cve-2024-1561/)
- [CVE-2024-1728: Gradio Local File Inclusion](https://vulert.com/vuln-db/CVE-2024-1728)
- [CVE-2024-8021: Gradio Open Redirect](https://github.com/advisories/GHSA-7v2w-h4gh-w5cv)
- [CVE-2024-47871: Gradio Insecure Communication](https://advisories.gitlab.com/pkg/pypi/gradio/CVE-2024-47871/)
- [Gradio Vulnerabilities (Snyk)](https://security.snyk.io/package/pip/gradio)
- [Exploiting Gradio File Read Vulnerabilities (Horizon3)](https://horizon3.ai/attack-research/disclosures/exploiting-file-read-vulnerabilities-in-gradio-to-steal-secrets-from-hugging-face-spaces/)

### Package Information
- [Anthropic Python SDK Releases](https://github.com/anthropics/anthropic-sdk-python/releases)
- [Anthropic SDK Security Analysis](https://secure.software/pypi/packages/anthropic/0.39.0)

---

**Audit completed by Claude on 2026-01-20**
