import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Check if precheck should be skipped (after first successful run)
if os.environ.get('SKIP_PRECHECK', '0') in ('1', 'true', 'True'):
    print("[PRECHECK] Skipped (SKIP_PRECHECK=1)")
    sys.exit(0)

"""
Henty pre-flight dependency checker.
Runs before server.py to detect and repair broken packages.
All import checks run in fresh subprocesses to avoid Python module cache issues.
"""
import sys
import subprocess

def run_check(code):
    """Run a Python snippet in a fresh subprocess. Returns (ok, output)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()

def pip(*args):
    print("    Running: pip install " + " ".join(str(a) for a in args))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *args],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("    pip error: " + result.stderr[-600:])
    return result.returncode == 0

# ── 1. torch / torchvision ────────────────────────────────────────────────────

TORCHVISION_CHECK = """
import torchvision
_ = torchvision.transforms.InterpolationMode
print("ok")
"""

TORCH_CUDA_CHECK = """
import torch, sys
cuda = getattr(torch.version, 'cuda', None) or ''
print(cuda)
"""

def get_torch_index():
    ok, out = run_check(TORCH_CUDA_CHECK)
    cuda = out.strip() if ok else ""
    print(f"  Detected torch CUDA version: {repr(cuda)}")
    if cuda.startswith("12.4"):
        return "https://download.pytorch.org/whl/cu124"
    elif cuda.startswith("12.1"):
        return "https://download.pytorch.org/whl/cu121"
    elif cuda.startswith("11.8"):
        return "https://download.pytorch.org/whl/cu118"
    elif cuda:
        # Unknown CUDA — cu121 is a safe default for most modern cards
        print(f"  Unknown CUDA version {cuda!r}, defaulting to cu121 index")
        return "https://download.pytorch.org/whl/cu121"
    else:
        print("  No CUDA detected — using CPU index")
        return "https://download.pytorch.org/whl/cpu"

def check_torchvision():
    ok, out = run_check(TORCHVISION_CHECK)
    return ok

def fix_torch():
    index = get_torch_index()
    print(f"  [FIX] Force-reinstalling torch/torchvision/torchaudio from {index}")
    ok = pip("--force-reinstall", "torch", "torchvision", "torchaudio",
             "--index-url", index)
    if ok:
        print("  [OK ] Reinstalled")
    else:
        print("  [ERR] pip install failed")
    return ok

# ── 2. numpy / numba compatibility ───────────────────────────────────────────

NUMBA_CHECK = """
import numba
print("ok")
"""

def check_numba():
    ok, _ = run_check(NUMBA_CHECK)
    return ok

def fix_numpy():
    print("  [FIX] Downgrading numpy to <2.3 (numba requirement)...")
    ok = pip("numpy<2.3", "--quiet")
    if ok:
        print("  [OK ] numpy downgraded")
    else:
        print("  [ERR] pip install failed")
    return ok

# ── 3. transformers / LlamaModel ─────────────────────────────────────────────

LLAMA_CHECK = """
import transformers
_ = transformers.LlamaModel
print("ok")
"""

def check_transformers():
    ok, _ = run_check(LLAMA_CHECK)
    return ok

def fix_transformers():
    print("  [FIX] Downgrading transformers to <4.52.0 ...")
    ok = pip("transformers<4.52.0", "--quiet")
    if ok:
        print("  [OK ] transformers downgraded")
    else:
        print("  [ERR] pip install failed")
    return ok

# ── 3. chatterbox end-to-end ─────────────────────────────────────────────────

CHATTERBOX_CHECK = """
from chatterbox.tts import ChatterboxTTS
print("ok")
"""

def check_chatterbox():
    ok, out = run_check(CHATTERBOX_CHECK)
    return ok, out

# ── GPU detection ─────────────────────────────────────────────────────────────

def check_gpu_available():
    """Check if CUDA GPU is available on this system."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except:
        return False

def check_torch_cuda():
    """Check if torch has CUDA support enabled."""
    ok, out = run_check("import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
    return 'cuda' in out.lower()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("")
    print("=" * 60)
    print("  Henty pre-flight check")
    print("  Python: " + sys.executable)
    print("=" * 60)

    # GPU check
    print("")
    print("[GPU] Checking for CUDA support...")
    gpu_available = check_gpu_available()
    torch_has_cuda = check_torch_cuda()
    if gpu_available and not torch_has_cuda:
        print(f"  GPU detected but PyTorch is CPU version")
        print(f"  To enable GPU:")
        print(f"    pip uninstall torch torchvision torchaudio -y")
        print(f"    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        print()
    elif gpu_available and torch_has_cuda:
        print("  GPU available and enabled")
    else:
        print("  Running on CPU")
    print()

    # 1. numpy / numba
    print("")
    print("[1/4] numpy / numba compatibility...")
    if check_numba():
        print("  [OK ]")
    else:
        if not fix_numpy():
            print("")
            print("FAILED: could not downgrade numpy.")
            sys.exit(1)
        if not check_numba():
            print("  [ERR] Still broken after numpy downgrade.")
            sys.exit(1)

    # 2. torch + torchvision
    print("")
    print("[2/4] torch + torchvision compatibility...")
    if check_torchvision():
        print("  [OK ]")
    else:
        if not fix_torch():
            print("")
            print("FAILED: could not reinstall torch.")
            print("Try running the launcher as Administrator.")
            sys.exit(1)
        if not check_torchvision():
            print("  [ERR] Still broken after reinstall.")
            print("  This may require running the launcher as Administrator,")
            print("  or manually running:")
            print("    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            sys.exit(1)

    # 3. transformers LlamaModel
    print("")
    print("[3/4] transformers / LlamaModel...")
    if check_transformers():
        print("  [OK ]")
    else:
        if not fix_transformers():
            print("")
            print("FAILED: could not fix transformers.")
            sys.exit(1)
        if not check_transformers():
            print("  [ERR] Still broken after downgrade.")
            sys.exit(1)

    # 4. chatterbox end-to-end
    print("")
    print("[4/4] chatterbox.tts...")
    ok, err = check_chatterbox()
    if ok:
        print("  [OK ]")
    else:
        print("  [ERR] " + err[-400:])
        print("")
        print("FAILED: chatterbox cannot be imported.")
        sys.exit(1)

    print("")
    print("=" * 60)
    print("  All checks passed -- starting server")
    print("=" * 60)
    print("")
    sys.exit(0)

if __name__ == "__main__":
    main()
