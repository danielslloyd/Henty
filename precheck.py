"""
Henty pre-flight dependency checker.
Runs before server.py to detect and repair broken packages.
Always uses sys.executable so it operates on the correct Python environment.
"""
import sys
import subprocess
import importlib

def pip(*args):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *args],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("    pip error: " + result.stderr[-400:])
    return result.returncode == 0

def try_import(module):
    try:
        return importlib.import_module(module), None
    except Exception as e:
        return None, str(e)

def get_torch_index():
    """Return the correct PyTorch index URL based on installed CUDA version."""
    torch, err = try_import("torch")
    if torch is None:
        return "https://download.pytorch.org/whl/cpu"
    cuda = getattr(torch.version, "cuda", None) or ""
    if cuda.startswith("12.1"):
        return "https://download.pytorch.org/whl/cu121"
    elif cuda.startswith("12.4"):
        return "https://download.pytorch.org/whl/cu124"
    elif cuda.startswith("11.8"):
        return "https://download.pytorch.org/whl/cu118"
    elif cuda:
        # Unknown CUDA version — try cu121 as a reasonable default
        return "https://download.pytorch.org/whl/cu121"
    return "https://download.pytorch.org/whl/cpu"

def check_torchvision():
    """Return True if torchvision loads without error."""
    try:
        import torchvision
        _ = torchvision.transforms.InterpolationMode
        return True
    except Exception:
        return False

def fix_torch():
    print("  [FIX] Reinstalling torch/torchvision/torchaudio with matching binaries...")
    index = get_torch_index()
    print(f"  [FIX] Using index: {index}")
    ok = pip("--force-reinstall", "torch", "torchvision", "torchaudio",
             "--index-url", index)
    if ok:
        print("  [OK ] torch/torchvision reinstalled")
    else:
        print("  [ERR] Could not reinstall torch — try running as administrator")
    return ok

def check_transformers_llama():
    """Return True if transformers can load LlamaModel."""
    try:
        import importlib
        t = importlib.import_module("transformers")
        _ = t.LlamaModel
        return True
    except Exception:
        return False

def fix_transformers():
    print("  [FIX] Pinning transformers to compatible version...")
    ok = pip("transformers<4.52.0")
    if ok:
        print("  [OK ] transformers pinned")
    else:
        print("  [ERR] Could not fix transformers")
    return ok

def check_chatterbox():
    try:
        # Force fresh import in case precheck re-installed things
        if "chatterbox" in sys.modules:
            del sys.modules["chatterbox"]
        import chatterbox.tts  # noqa
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("")
    print("=" * 60)
    print("  Henty pre-flight check")
    print("  Python: " + sys.executable)
    print("=" * 60)

    # --- Step 1: torchvision/torch compatibility ---
    print("")
    print("[1/3] torch + torchvision compatibility...")
    if check_torchvision():
        print("  [OK ]")
    else:
        if not fix_torch():
            print("")
            print("Pre-flight FAILED: could not fix torch/torchvision.")
            print("Try running this terminal as Administrator, then restart the server.")
            sys.exit(1)
        if not check_torchvision():
            print("  [ERR] Still broken after reinstall — environment may need manual repair")
            sys.exit(1)

    # --- Step 2: transformers / LlamaModel ---
    print("")
    print("[2/3] transformers LlamaModel import...")
    if check_transformers_llama():
        print("  [OK ]")
    else:
        if not fix_transformers():
            print("")
            print("Pre-flight FAILED: could not fix transformers.")
            sys.exit(1)
        if not check_transformers_llama():
            print("  [ERR] Still broken after downgrade")
            sys.exit(1)

    # --- Step 3: chatterbox end-to-end ---
    print("")
    print("[3/3] chatterbox.tts import...")
    ok, err = check_chatterbox()
    if ok:
        print("  [OK ]")
    else:
        print(f"  [ERR] {err}")
        print("")
        print("Pre-flight FAILED: chatterbox cannot be imported.")
        print("Check the error above; you may need to reinstall chatterbox-tts.")
        sys.exit(1)

    print("")
    print("=" * 60)
    print("  All checks passed — starting server")
    print("=" * 60)
    print("")
    sys.exit(0)

if __name__ == "__main__":
    main()
