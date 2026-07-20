import sys
# Force UTF-8 output on Windows (prevents crashes from Unicode chars in print statements)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    HAS_TURBO = True
except ImportError:
    HAS_TURBO = False
    print("WARNING: chatterbox.tts_turbo not available. Emotion tags will be converted to display text.")
import torch
import numpy as np

# Turbo dtype guard: ChatterboxTurboTTS.norm_loudness multiplies the float32 reference
# wav by a numpy float64 loudness gain, upcasting the whole array to float64. That then
# crashes downstream ("RNN input dtype torch.float64 ... weight dtype torch.float32" and
# "expected scalar type Float but found Double"). Force the result back to float32.
if HAS_TURBO:
    try:
        _orig_norm_loudness = ChatterboxTurboTTS.norm_loudness
        def _norm_loudness_f32(self, wav, sr, target_lufs=-27):
            out = _orig_norm_loudness(self, wav, sr, target_lufs)
            if hasattr(out, 'astype'):
                out = out.astype(np.float32, copy=False)
            return out
        ChatterboxTurboTTS.norm_loudness = _norm_loudness_f32
        print("[PATCH] ChatterboxTurboTTS.norm_loudness float32 guard applied")
    except Exception as _patch_err:
        print(f"[PATCH] could not patch norm_loudness: {_patch_err}")

# Chatterbox's S3 tokenizer crashes when the reference (voice prompt) audio is loaded as
# float64 — "expected scalar type Float but found Double" inside log_mel_spectrogram. This
# happens for the Turbo voice prompt on some systems. Coerce the audio to float32 there.
try:
    from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer as _S3Tok
    _orig_log_mel = _S3Tok.log_mel_spectrogram
    def _log_mel_float32(self, audio, padding=0):
        if torch.is_tensor(audio):
            if audio.dtype != torch.float32:
                audio = audio.float()
        else:
            audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        return _orig_log_mel(self, audio, padding)
    _S3Tok.log_mel_spectrogram = _log_mel_float32
    print("[PATCH] s3tokenizer.log_mel_spectrogram float32 guard applied")
except Exception as _patch_err:
    print(f"[PATCH] could not patch s3tokenizer for float32: {_patch_err}")

from scipy.io import wavfile
import re
from pydub import AudioSegment
import json
import time
import threading
import uuid
from datetime import datetime
from config import config
from auth import AuthManager

app = Flask(__name__)

# Lazy-loaded Whisper model for transcription scoring
_whisper_model = None
_whisper_model_lock = threading.Lock()

def get_whisper_model():
    global _whisper_model
    with _whisper_model_lock:
        if _whisper_model is None:
            try:
                import whisper
            except ImportError:
                print("openai-whisper not found, installing...")
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
                import whisper
            print("Loading Whisper base.en model for transcription scoring...")
            _whisper_model = whisper.load_model("base.en")
            print("Whisper model loaded.")
        return _whisper_model

def compute_similarity(text_a, text_b):
    """Compute normalized similarity score between two strings (0.0–1.0)."""
    import difflib
    import re
    def normalize(t):
        t = t.lower()
        t = re.sub(r"[^\w\s]", "", t)
        return " ".join(t.split())
    a = normalize(text_a)
    b = normalize(text_b)
    return difflib.SequenceMatcher(None, a, b).ratio()


LOG_PREVIEW_LIMIT = 300


def preview_for_log(value, limit=LOG_PREVIEW_LIMIT):
    """Return a single-line preview capped for safe server logging."""
    if value is None:
        return ""
    text = str(value).replace('\r', '\\r').replace('\n', '\\n')
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text)} chars total]"


def preview_json_for_log(value, limit=LOG_PREVIEW_LIMIT):
    """Serialize JSON-ish values and cap the output for logs."""
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except Exception:
        serialized = str(value)
    return preview_for_log(serialized, limit=limit)







# Configure CORS for remote access
# Note: When using wildcard (*), we can't use credentials
if config.ALLOWED_ORIGINS == ['*'] or config.ALLOWED_ORIGINS == '*':
    cors_config = {
        'origins': '*',
        'supports_credentials': False,
        'allow_headers': ['Content-Type', 'X-API-Key'],
        'expose_headers': ['Content-Type']
    }
else:
    cors_config = {
        'origins': config.ALLOWED_ORIGINS,
        'supports_credentials': True,
        'allow_headers': ['Content-Type', 'X-API-Key'],
        'expose_headers': ['Content-Type']
    }
CORS(app, resources={r"/*": cors_config})

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins=config.ALLOWED_ORIGINS, async_mode='threading')

# Initialize authentication
auth_manager = AuthManager(api_key=config.API_KEY, require_auth=config.REQUIRE_AUTH)

class TextToAudioConverter:
    def __init__(self):
        self.model = None
        self.turbo_model = None

        # Detailed CUDA diagnostics
        print("\n" + "="*80)
        print("GPU/CUDA INITIALIZATION")
        print("="*80)

        # Show which Python is running
        import sys
        print(f"Python executable: {sys.executable}")
        print(f"Python version: {sys.version.split()[0]}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch location: {torch.__file__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # GPU-only: Henty requires CUDA. Refuse to run on CPU.
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            import sys
            msg = (
                "CUDA is not available — Henty requires a CUDA GPU and will not run on CPU.\n"
                f"This Python has PyTorch WITHOUT CUDA support at: {torch.__file__}\n"
                "To fix, reinstall the CUDA build of PyTorch:\n"
                f"  {sys.executable} -m pip uninstall torch torchvision torchaudio -y\n"
                f"  {sys.executable} -m pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu130\n"
                "Then restart the server."
            )
            print("\n" + msg + "\n")
            raise RuntimeError(msg)

        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self.device = 'cuda'
        print(f"\nUsing device: {self.device}")
        print("="*80 + "\n")

        self.audio_dir = "generated_audio"
        self.voice_samples_dir = "voice_samples"
        self.common_files_dir = config.COMMON_FILES_DIR
        self.projects_dir = config.DEFAULT_PROJECT_DIR
        self.stats_file = "generation_stats.json"
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.voice_samples_dir, exist_ok=True)
        os.makedirs(self.common_files_dir, exist_ok=True)
        os.makedirs(self.projects_dir, exist_ok=True)

        # Generation tracking
        self.current_generation = None
        self.generation_lock = threading.Lock()

        # Load existing stats or initialize
        self.generation_stats = self.load_stats()

        # Project management
        self.current_project_path = None
        self.current_project_metadata = None
        self.undo_stack = []  # snapshots of chapters before each edit (max 20)

        # Performance optimization: cached lookup dictionaries
        self._chapter_lookup_cache = None
        self._text_file_lookup_cache = None
        self._chunk_lookup_cache = None

    def _invalidate_lookup_caches(self):
        """Invalidate lookup caches when project metadata changes"""
        self._chapter_lookup_cache = None
        self._text_file_lookup_cache = None
        self._chunk_lookup_cache = None

    def get_chapter_and_chunk_lookups(self):
        """
        Build O(1) lookup dictionaries for chapters, text_files, and their chunks.
        Returns: (chapter_map, text_file_map, chunk_maps)
        where chunk_maps is a dict mapping container_id -> {chunk_id: chunk}
        """
        # Return cached lookups if available
        if (self._chapter_lookup_cache is not None and
            self._text_file_lookup_cache is not None and
            self._chunk_lookup_cache is not None):
            return self._chapter_lookup_cache, self._text_file_lookup_cache, self._chunk_lookup_cache

        if not self.current_project_metadata:
            return {}, {}, {}

        chapters = self.current_project_metadata.get('chapters', [])
        text_files = self.current_project_metadata.get('text_files', [])

        # Build chapter and text_file lookups - O(n)
        chapter_map = {ch['id']: ch for ch in chapters}
        text_file_map = {tf['id']: tf for tf in text_files}

        # Build chunk lookups for each container - O(n*m) once, then O(1) per access
        chunk_maps = {}
        for container_id, container in {**chapter_map, **text_file_map}.items():
            chunks = container.get('chunks', [])
            chunk_maps[container_id] = {c['id']: c for c in chunks}

        # Cache the lookups
        self._chapter_lookup_cache = chapter_map
        self._text_file_lookup_cache = text_file_map
        self._chunk_lookup_cache = chunk_maps

        return chapter_map, text_file_map, chunk_maps

    def get_relative_path(self, absolute_path, base_path=None):
        """Convert absolute path to relative path within project"""
        if base_path is None:
            base_path = self.current_project_path or os.getcwd()
        try:
            return os.path.relpath(absolute_path, base_path)
        except ValueError:
            # On Windows, paths on different drives can't be made relative
            return absolute_path

    def load_model(self):
        """Load the Chatterbox TTS model"""
        if self.model is None:
            print(f"\n{'='*80}")
            print(f"Loading Chatterbox TTS model...")
            print(f"Device: {self.device}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA Version: {torch.version.cuda}")
            print(f"{'='*80}\n")

            self.model = ChatterboxTTS.from_pretrained(device=self.device)

            # Verify model is on correct device
            print(f"\n{'='*80}")
            print(f"Model loaded successfully!")
            print(f"Expected device: {self.device}")

            # Try to get actual device from model (ChatterboxTTS may not have standard parameters())
            try:
                if hasattr(self.model, 'parameters'):
                    model_device = next(self.model.parameters()).device
                    print(f"Model device: {model_device}")
                    if str(model_device) != self.device and not (self.device == "cuda" and "cuda" in str(model_device)):
                        print(f"WARNING: Model device ({model_device}) does not match expected device ({self.device})")
                    else:
                        print(f"✓ Model confirmed on {model_device}")
                else:
                    print(f"✓ Model loaded (device verification not available for this model type)")
            except Exception as e:
                print(f"Note: Could not verify model device: {e}")

            print(f"{'='*80}\n")
        return self.model

    def load_turbo_model(self):
        """Load the Chatterbox Turbo TTS model"""
        if not HAS_TURBO:
            raise RuntimeError("Chatterbox Turbo not available. Install chatterbox-tts >= 0.1.6")
        if self.turbo_model is None:
            print(f"\n{'='*80}")
            print(f"Loading Chatterbox Turbo TTS model...")
            print(f"Device: {self.device}")
            print(f"{'='*80}\n")

            self.turbo_model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            print(f"Chatterbox Turbo model loaded successfully on {self.device}")
            print(f"{'='*80}\n")
        return self.turbo_model

    def load_stats(self):
        """Load generation statistics from file"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading stats: {e}")
                return []
        return []

    def save_stats(self):
        """Save generation statistics to file"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.generation_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")

    def get_gpu_usage(self):
        """Get current GPU memory usage and temperature"""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                gpu_utilization = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else None

                # Try to get GPU temperature using nvidia-smi
                gpu_temp = None
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                          capture_output=True, text=True, timeout=1)
                    if result.returncode == 0:
                        gpu_temp = int(result.stdout.strip())
                except Exception:
                    pass

                return {
                    'memory_allocated_gb': round(gpu_mem_allocated, 2),
                    'memory_reserved_gb': round(gpu_mem_reserved, 2),
                    'utilization_percent': gpu_utilization,
                    'temperature_c': gpu_temp
                }
            except Exception as e:
                print(f"Error getting GPU stats: {e}")
                return None
        return None

    def estimate_generation_time(self, char_count):
        """Estimate generation time based on historical data"""
        if not self.generation_stats:
            return None

        # Calculate average ms per character from recent generations (last 20)
        recent_stats = self.generation_stats[-20:]
        if not recent_stats:
            return None

        total_chars = sum(s['char_count'] for s in recent_stats)
        total_time = sum(s['generation_time_ms'] for s in recent_stats)

        if total_chars == 0:
            return None

        avg_ms_per_char = total_time / total_chars
        estimated_ms = char_count * avg_ms_per_char

        return {
            'estimated_ms': round(estimated_ms),
            'estimated_seconds': round(estimated_ms / 1000, 1),
            'avg_ms_per_char': round(avg_ms_per_char, 2),
            'based_on_samples': len(recent_stats)
        }

    def log_generation(self, char_count, audio_duration_sec, generation_time_ms, gpu_stats_before, gpu_stats_after):
        """Log a generation event with all metrics"""
        log_entry = {
            'timestamp': int(time.time() * 1000),
            'char_count': char_count,
            'audio_duration_sec': round(audio_duration_sec, 2),
            'generation_time_ms': generation_time_ms,
            'chars_per_second': round(char_count / (generation_time_ms / 1000), 2),
            'gpu_before': gpu_stats_before,
            'gpu_after': gpu_stats_after
        }

        self.generation_stats.append(log_entry)

        # Keep only last 100 entries to prevent file from growing too large
        if len(self.generation_stats) > 100:
            self.generation_stats = self.generation_stats[-100:]

        self.save_stats()

        print(f"\n=== Generation Stats ===")
        print(f"Characters: {char_count}")
        print(f"Audio Duration: {audio_duration_sec:.2f}s")
        print(f"Generation Time: {generation_time_ms}ms ({generation_time_ms/1000:.2f}s)")
        print(f"Speed: {log_entry['chars_per_second']:.2f} chars/sec")
        if gpu_stats_after:
            print(f"GPU Memory: {gpu_stats_after['memory_allocated_gb']:.2f} GB allocated")
        print("========================\n")

    def find_txt_files(self, directory):
        """Find all .txt files in the given directory"""
        if not directory or not os.path.isdir(directory):
            return []

        txt_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, directory)
                    txt_files.append({
                        'name': relative_path,
                        'path': full_path
                    })

        return sorted(txt_files, key=lambda x: x['name'])

    def read_text_file(self, file_path):
        """Read the content of a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    # Words that end in a period but do NOT end a sentence. Lower-cased, no trailing dot.
    _ABBREV = {
        'dr', 'mr', 'mrs', 'ms', 'messrs', 'mmes', 'st', 'prof', 'rev', 'hon',
        'sr', 'jr', 'capt', 'col', 'gen', 'lt', 'maj', 'sgt', 'cmdr', 'gov',
        'pres', 'sen', 'rep', 'supt', 'fr', 'pr', 'vs', 'etc', 'al', 'inc',
        'ltd', 'co', 'corp', 'no', 'vol', 'pp', 'ph', 'mt', 'ft', 'esq',
    }

    def _split_sentences(self, text):
        """Split into sentences on . ! ? followed by whitespace, but never break
        immediately after a known abbreviation (e.g. "Dr." / "Mr.")."""
        parts = []
        last = 0
        for m in re.finditer(r'[.!?]+["\'\)\]]?\s+', text):
            prefix = text[last:m.start()]
            wm = re.search(r'([A-Za-z]+)$', prefix)
            if wm and wm.group(1).lower() in self._ABBREV:
                continue  # abbreviation — keep the sentence going
            seg = text[last:m.end()].strip()
            if seg:
                parts.append(seg)
            last = m.end()
        tail = text[last:].strip()
        if tail:
            parts.append(tail)
        return parts

    def smart_chunk_text(self, text, max_chunk_size=None):
        """
        Chunk text with newline-forced boundaries + sentence splitting for long lines.
        Rules:
        1. Split on newlines first — each non-empty line is at least one chunk.
        2. Consecutive non-empty lines are NOT merged (newline forces boundary).
        3. If a line exceeds max_chunk_size, split by sentences.
        4. max_chunk_size is a HARD limit — any text still over it is split at the
           nearest word boundary (or hard-cut if no space exists).
        """
        if max_chunk_size is None:
            max_chunk_size = config.MAX_CHUNK_SIZE

        chunks = []
        chunk_id = 0

        def add_chunk(t):
            nonlocal chunk_id
            t = t.strip()
            if not t:
                return
            # Hard-limit enforcement: split at word boundary until within limit
            while len(t) > max_chunk_size:
                split_at = t.rfind(' ', 0, max_chunk_size)
                if split_at <= 0:
                    split_at = max_chunk_size  # no space found — hard cut
                part = t[:split_at].strip()
                if part:
                    nick = part[:50] + ('...' if len(part) > 50 else '')
                    chunks.append({'id': chunk_id, 'text': part, 'nickname': nick})
                    chunk_id += 1
                t = t[split_at:].strip()
            if t:
                nick = t[:50] + ('...' if len(t) > 50 else '')
                chunks.append({'id': chunk_id, 'text': t, 'nickname': nick})
                chunk_id += 1

        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) <= max_chunk_size:
                add_chunk(stripped)
            else:
                # Split by sentences, accumulate up to max_chunk_size
                sentences = self._split_sentences(stripped)
                current = ''
                for sentence in sentences:
                    if current and len(current) + 1 + len(sentence) > max_chunk_size:
                        add_chunk(current)
                        current = sentence
                    else:
                        current = (current + ' ' + sentence).strip() if current else sentence
                if current:
                    add_chunk(current)

        return chunks

    def stitch_audio_files(self, audio_paths, output_path):
        """
        Stitch multiple audio files together into a single file.
        Supports:
        - File paths (string): loads audio from file (WAV, MP3, etc.)
        - Pause tuples ('pause', duration_ms): generates silence

        Returns the path to the stitched audio file.
        """
        try:
            if not audio_paths:
                raise ValueError("No audio files provided for stitching")

            print(f"=== Starting audio stitching ===")
            print(f"Total items to stitch: {len(audio_paths)}")

            # Load all audio files
            combined = None
            for i, item in enumerate(audio_paths):
                # Check if this is a pause tuple
                if isinstance(item, tuple) and item[0] == 'pause':
                    duration_ms = item[1]
                    print(f"Item {i+1}/{len(audio_paths)}: Generating {duration_ms}ms pause")
                    audio_segment = AudioSegment.silent(duration=duration_ms)
                else:
                    # Regular audio file
                    audio_path = item
                    if not os.path.exists(audio_path):
                        print(f"Warning: Audio file not found: {audio_path}")
                        continue

                    print(f"Loading audio file {i+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")

                    # Detect file type and load accordingly
                    file_ext = os.path.splitext(audio_path)[1].lower()
                    if file_ext == '.wav':
                        audio_segment = AudioSegment.from_wav(audio_path)
                    elif file_ext == '.mp3':
                        audio_segment = AudioSegment.from_mp3(audio_path)
                    elif file_ext in ['.ogg', '.oga']:
                        audio_segment = AudioSegment.from_ogg(audio_path)
                    elif file_ext == '.flac':
                        audio_segment = AudioSegment.from_file(audio_path, format='flac')
                    elif file_ext == '.m4a':
                        audio_segment = AudioSegment.from_file(audio_path, format='m4a')
                    else:
                        # Try to load as generic file
                        audio_segment = AudioSegment.from_file(audio_path)

                    print(f"  - Duration: {len(audio_segment)}ms, Sample rate: {audio_segment.frame_rate}Hz, Channels: {audio_segment.channels}")

                if combined is None:
                    combined = audio_segment
                else:
                    # Add a small pause between chunks (100ms)
                    silence = AudioSegment.silent(duration=100)
                    combined = combined + silence + audio_segment

            if combined is None:
                raise ValueError("No valid audio files found to stitch")

            print(f"Final combined audio duration: {len(combined)}ms ({len(combined)/1000:.2f} seconds)")

            # Export the combined audio
            combined.export(output_path, format="wav")
            print(f"Successfully stitched {len(audio_paths)} items to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error stitching audio files: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def ensure_valid_wav_format(self, wav_path):
        """Ensure a WAV file is in a format librosa can read"""
        try:
            import librosa
            # Try to load it - if this fails, we need to convert
            test_audio, test_sr = librosa.load(wav_path, sr=None, mono=False, duration=0.1)
            print(f"Voice sample format validation: OK ({test_sr} Hz)")
            return True
        except Exception as e:
            print(f"Voice sample format validation: FAILED - {str(e)}")
            print(f"Attempting to convert voice sample to proper format...")
            temp_path = None
            try:
                # Read the existing WAV file with scipy
                from scipy.io import wavfile as scipy_wavfile
                sample_rate, audio_data = scipy_wavfile.read(wav_path)

                # Create a temp backup
                temp_path = wav_path + ".temp"
                os.rename(wav_path, temp_path)

                # Normalize to float
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32767.0
                elif audio_data.dtype == np.int32:
                    audio_float = audio_data.astype(np.float32) / 2147483647.0
                else:
                    audio_float = audio_data.astype(np.float32)

                # Ensure proper shape and convert back to int16
                if audio_float.ndim == 1:
                    audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
                else:
                    audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)

                # Write as proper WAV
                scipy_wavfile.write(wav_path, sample_rate, audio_int16)
                os.remove(temp_path)

                # Validate the conversion worked
                test_audio, test_sr = librosa.load(wav_path, sr=None, mono=False, duration=0.1)
                print(f"Voice sample converted successfully! ({test_sr} Hz)")
                return True

            except Exception as conv_error:
                print(f"Failed to convert voice sample: {str(conv_error)}")
                import traceback
                print(traceback.format_exc())
                # Restore original file if conversion failed
                if temp_path and os.path.exists(temp_path):
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    os.rename(temp_path, wav_path)
                return False

    def get_audio_file_info(self, audio_path):
        """Get information about an audio file (duration, sample rate, etc.)"""
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            return {
                'duration_ms': len(audio_segment),
                'duration_sec': len(audio_segment) / 1000.0,
                'sample_rate': audio_segment.frame_rate,
                'channels': audio_segment.channels,
                'file_size_bytes': os.path.getsize(audio_path)
            }
        except Exception as e:
            print(f"Error getting audio file info: {str(e)}")
            return None

    def list_common_files(self):
        """List all audio files in the common files directory"""
        try:
            if not os.path.exists(self.common_files_dir):
                return []

            common_files = []
            for filename in os.listdir(self.common_files_dir):
                file_path = os.path.join(self.common_files_dir, filename)
                # Check if it's an audio file
                if os.path.isfile(file_path) and filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    info = self.get_audio_file_info(file_path)
                    if info:
                        common_files.append({
                            'filename': filename,
                            'path': file_path,
                            'relative_path': os.path.relpath(file_path, os.getcwd()),
                            'duration_ms': info['duration_ms'],
                            'duration_sec': info['duration_sec'],
                            'file_size_bytes': info['file_size_bytes']
                        })

            return sorted(common_files, key=lambda x: x['filename'])
        except Exception as e:
            print(f"Error listing common files: {str(e)}")
            return []


    def strip_xml_tags(self, text):
        """Strip all XML-like tags from text for TTS processing"""
        # Remove all XML tags: <tag>content</tag> or <tag/>
        cleaned = re.sub(r'<[^>]+>', '', text)
        return cleaned

    def process_pronunciation_markup(self, text, use_turbo=True):
        """Replace {display|spoken} pronunciation markup with the spoken form for TTS.
        Supports paralinguistic tags: {display|[laugh]} or {|[cough]} — extracts just
        the [tag] part for Chatterbox Turbo.
        If use_turbo is False and the spoken form is an emotion tag, use display form instead.
        If no pipe is present inside braces, returns the text unchanged.
        """
        def replacer(m):
            parts = m.group(1).split('|', 1)
            if len(parts) == 2:
                spoken = parts[1].strip()
                # If using Standard model and the "spoken" part is an emotion tag, use display instead
                if not use_turbo and re.match(r'^\[(?:laugh|chuckle|cough|sigh|gasp|groan|sniff|clear throat|shush)\]$', spoken):
                    return parts[0].strip()  # display form
                return spoken  # spoken form (may be [laugh], [cough], etc.)
            return m.group(0)    # no pipe = not a pronunciation marker, leave as-is
        return re.sub(r'\{([^}]+)\}', replacer, text)

    def text_has_paralinguistic_tags(self, text):
        """Check if text contains Chatterbox Turbo paralinguistic tags, e.g. {ha ha|[laugh]}."""
        paralinguistic_tags = ['laugh', 'chuckle', 'cough', 'sigh', 'gasp', 'groan', 'sniff', 'clear throat', 'shush']
        pattern = r'\{[^}]*\|\s*\[(?:' + '|'.join(re.escape(t) for t in paralinguistic_tags) + r')\]\s*\}'
        return bool(re.search(pattern, text))

    def extract_display_text(self, text):
        """Replace {display|spoken} pronunciation markup with just the display form.
        Used for reader display and clean editor view.
        """
        return re.sub(r'\{([^|}]+)\|[^}]*\}', r'\1', text)


    def generate_audio(self, text, output_path, audio_prompt_path=None, language_id="en", exaggeration=0.6, cfg_weight=0.4, tts_model="chatterbox", temperature=0.8, seed=0):
        """Generate audio from text using Chatterbox TTS or Chatterbox Turbo"""
        try:
            # Start timing and capture initial GPU stats
            start_time = time.time()
            gpu_stats_before = self.get_gpu_usage()
            char_count = len(text)

            # Set current generation info for progress tracking
            with self.generation_lock:
                self.current_generation = {
                    'char_count': char_count,
                    'start_time': start_time,
                    'estimated_time': self.estimate_generation_time(char_count),
                    'status': 'generating'
                }

            # Emit progress update via WebSocket
            if config.ENABLE_WEBSOCKET:
                socketio.emit('generation_started', {
                    'char_count': char_count,
                    'estimated_time': self.current_generation['estimated_time']
                })

            use_turbo = (tts_model == 'chatterbox_turbo')
            print(f"\n[TTS MODEL LOADING]")
            print(f"  Model: {tts_model}, HAS_TURBO={HAS_TURBO}")
            if use_turbo:
                if not HAS_TURBO:
                    print(f"  ⚠️  Turbo selected but not available — loading Standard instead")
                    model = self.load_model()
                    print(f"  ✓ Loaded: {type(model).__name__} (Standard fallback)")
                else:
                    print(f"  → Loading Turbo model...")
                    model = self.load_turbo_model()
                    print(f"  ✓ Loaded: {type(model).__name__} (Turbo)")
            else:
                print(f"  → Loading Standard model...")
                model = self.load_model()
                print(f"  ✓ Loaded: {type(model).__name__} (Standard)")

            # Prepare generation parameters with explicit float32 conversion
            gen_params = {
                "exaggeration": torch.tensor(exaggeration, dtype=torch.float32).item(),
                "cfg_weight": torch.tensor(cfg_weight, dtype=torch.float32).item(),
                "temperature": torch.tensor(temperature, dtype=torch.float32).item()
            }

            # Add audio prompt if provided
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                print(f"\n=== Validating voice sample: {audio_prompt_path} ===")
                if self.ensure_valid_wav_format(audio_prompt_path):
                    gen_params["audio_prompt_path"] = audio_prompt_path
                    print(f"Voice sample ready for use")
                else:
                    print(f"WARNING: Could not validate/convert voice sample, proceeding without voice cloning")

            # Add language ID if not English
            if language_id != "en":
                gen_params["language_id"] = language_id

            # Get actual model device (safely)
            device_name = "GPU" if self.device == "cuda" else "CPU"

            print(f"Generating with parameters: {gen_params}")
            print(f"Text length: {char_count} characters")
            print(f"Device: {device_name}")

            # Ensure float32 throughout — numpy <2.3 can produce float64 tensors
            # which cause "expected scalar type Double but found Float" errors
            torch.set_default_dtype(torch.float32)
            try:
                model = model.float()
            except (AttributeError, RuntimeError):
                try:
                    model = model.to(dtype=torch.float32)
                except (AttributeError, RuntimeError):
                    pass

            # Seed for reproducible generation when a non-zero seed is provided
            # (seed=0 keeps the previous non-deterministic behavior).
            if seed:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            wav = model.generate(text, **gen_params)
            print(f"Generated wav type: {type(wav)}, shape: {wav.shape if hasattr(wav, 'shape') else 'N/A'}")

            # Convert tensor to numpy array
            if torch.is_tensor(wav):
                wav_numpy = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
            else:
                wav_numpy = np.array(wav)

            print(f"Converted to numpy, shape: {wav_numpy.shape}, dtype: {wav_numpy.dtype}")

            # Ensure 1D array for mono audio
            if wav_numpy.ndim == 2:
                # If stereo or multiple channels, take first channel
                if wav_numpy.shape[0] == 2 or wav_numpy.shape[0] < wav_numpy.shape[1]:
                    wav_numpy = wav_numpy[0] if wav_numpy.shape[0] < wav_numpy.shape[1] else wav_numpy[:, 0]
                else:
                    wav_numpy = wav_numpy.flatten()

            print(f"After flattening, shape: {wav_numpy.shape}")

            # Normalize to int16 range for WAV file
            wav_numpy = np.clip(wav_numpy, -1.0, 1.0)
            wav_int16 = (wav_numpy * 32767).astype(np.int16)

            print(f"Final audio shape: {wav_int16.shape}, Sample rate: {model.sr}, dtype: {wav_int16.dtype}")

            # Calculate audio duration
            audio_duration_sec = len(wav_int16) / model.sr

            # Use scipy.io.wavfile which is more reliable on Windows
            try:
                wavfile.write(output_path, model.sr, wav_int16)
                print(f"Successfully wrote WAV file to: {output_path}")
            except Exception as write_error:
                print(f"Error writing WAV file: {str(write_error)}")
                raise

            # End timing and capture final GPU stats
            end_time = time.time()
            generation_time_ms = int((end_time - start_time) * 1000)
            gpu_stats_after = self.get_gpu_usage()

            # Log the generation
            self.log_generation(
                char_count=char_count,
                audio_duration_sec=audio_duration_sec,
                generation_time_ms=generation_time_ms,
                gpu_stats_before=gpu_stats_before,
                gpu_stats_after=gpu_stats_after
            )

            # Emit completion via WebSocket
            if config.ENABLE_WEBSOCKET:
                socketio.emit('generation_completed', {
                    'char_count': char_count,
                    'audio_duration_sec': audio_duration_sec,
                    'generation_time_ms': generation_time_ms,
                    'gpu_stats': gpu_stats_after
                })

            # Clear current generation
            with self.generation_lock:
                self.current_generation = None

            return {
                'path': output_path,
                'duration_seconds': audio_duration_sec,
                'generation_time_ms': generation_time_ms
            }
        except Exception as e:
            # Emit error via WebSocket
            if config.ENABLE_WEBSOCKET:
                socketio.emit('generation_error', {
                    'error': str(e)
                })
            # Clear current generation on error
            with self.generation_lock:
                self.current_generation = None
            print(f"Error generating audio: {str(e)}")
            raise

    def get_or_generate_audio(self, txt_file_path):
        """Get existing audio or generate new one"""
        # Create audio filename based on text filename
        txt_filename = os.path.basename(txt_file_path)
        audio_filename = os.path.splitext(txt_filename)[0] + ".wav"
        audio_path = os.path.join(self.audio_dir, audio_filename)

        # If audio doesn't exist, generate it
        if not os.path.exists(audio_path):
            print(f"Generating audio for {txt_filename}...")
            text = self.read_text_file(txt_file_path)
            if not text or text.startswith("Error"):
                raise Exception("Failed to read text file")

            # Limit text length for demo purposes (adjust as needed)
            if len(text) > 1000:
                text = text[:1000] + "..."

            result = self.generate_audio(text, audio_path)
            audio_path = result['path']

        return audio_path

# Create converter instance
converter = TextToAudioConverter()

@app.route('/')
def index():
    """Serve the HTML file"""
    return send_file('index.html')






@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    try:
        return send_from_directory(converter.audio_dir, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/project/audio/<path:filename>')
def serve_project_audio(filename):
    """Serve audio files from the current project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        audio_dir = os.path.join(converter.current_project_path, 'audio')
        print(f"Serving project audio: {filename} from {audio_dir}")
        return send_from_directory(audio_dir, filename)
    except Exception as e:
        print(f"Error serving project audio: {str(e)}")
        return jsonify({'error': str(e)}), 404

@app.route('/api/published/<path:filename>')
def serve_published(filename):
    """Serve a published (stitched) chapter audio file from the current project."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400
        published_dir = os.path.join(converter.current_project_path, 'published')
        return send_from_directory(published_dir, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/status')
def status():
    """Check server status"""
    return jsonify({
        'status': 'running',
        'device': converter.device,
        'model_loaded': converter.model is not None
    })

@app.route('/api/config')
def get_config():
    """Get client configuration (public endpoint)"""
    return jsonify(config.get_client_config())

@app.route('/api/device-status', methods=['GET'])
@auth_manager.require_api_key
def get_device_status():
    """Report GPU status (GPU-only; no device switching)."""
    return jsonify({
        'current_device': converter.device,
        'cuda_available': converter.cuda_available,
        'model_loaded': converter.model is not None,
        'max_parallel_generations': config.MAX_PARALLEL_GENERATIONS
    })

@app.route('/api/voice-samples', methods=['GET'])
@auth_manager.require_api_key
def list_voice_samples():
    """List all voice samples"""
    try:
        samples = []
        for filename in os.listdir(converter.voice_samples_dir):
            if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a')):
                file_path = os.path.join(converter.voice_samples_dir, filename)
                samples.append({
                    'name': filename,
                    'path': file_path,
                    'url': f'/api/voice-samples/{filename}'
                })
        return jsonify({'samples': samples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-samples/<filename>')
def serve_voice_sample(filename):
    """Serve voice sample files"""
    try:
        return send_from_directory(converter.voice_samples_dir, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/voice-samples/upload', methods=['POST'])
def upload_voice_sample():
    """Upload a voice sample"""
    try:
        print(f"\n=== Received voice sample upload ===")

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filename = request.form.get('filename', file.filename)

        # Validate file type
        allowed_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'Invalid audio file type'}), 400

        # Save temporary file first
        temp_path = os.path.join(converter.voice_samples_dir, f"temp_{filename}")
        file.save(temp_path)
        print(f"Temporary file saved: {temp_path}")

        # Convert to proper WAV format using scipy
        try:
            # Try to read with scipy first
            import librosa
            import soundfile as sf_temp

            # Load audio (librosa can handle many formats)
            print(f"Converting audio to proper WAV format...")
            audio_data, sample_rate = librosa.load(temp_path, sr=None, mono=False)

            # Ensure WAV extension
            if not filename.lower().endswith('.wav'):
                filename = os.path.splitext(filename)[0] + '.wav'

            file_path = os.path.join(converter.voice_samples_dir, filename)

            # Save as proper WAV using scipy.wavfile
            if audio_data.ndim == 1:
                audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                # Multi-channel audio
                audio_int16 = (np.clip(audio_data.T, -1.0, 1.0) * 32767).astype(np.int16)

            wavfile.write(file_path, sample_rate, audio_int16)
            print(f"Voice sample converted and saved: {file_path}")

            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as conv_error:
            print(f"Conversion error: {str(conv_error)}")
            import traceback
            print(f"Full traceback:")
            print(traceback.format_exc())
            # If conversion fails, just use the original file
            file_path = os.path.join(converter.voice_samples_dir, filename)
            if os.path.exists(temp_path):
                os.rename(temp_path, file_path)
            print(f"Using original file format: {file_path}")

        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/api/voice-samples/{filename}',
            'path': file_path
        })
    except Exception as e:
        import traceback
        print(f"\n=== ERROR ===")
        print(f"Exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Common Files API Endpoints
@app.route('/api/common-files', methods=['GET'])
@auth_manager.require_api_key
def list_common_files():
    """List all audio files in the common files directory"""
    try:
        common_files = converter.list_common_files()
        return jsonify({'common_files': common_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/common-files/<filename>')
def serve_common_file(filename):
    """Serve a common file"""
    try:
        return send_from_directory(converter.common_files_dir, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/common-files/upload', methods=['POST'])
@auth_manager.require_api_key
def upload_common_file():
    """Upload a common audio file (intro, outro, etc.)"""
    try:
        print(f"\n=== Received common file upload ===")

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filename = request.form.get('filename', file.filename)

        # Validate file type - support various audio formats
        allowed_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'Invalid audio file type. Allowed: WAV, MP3, OGG, FLAC, M4A'}), 400

        # Save file
        file_path = os.path.join(converter.common_files_dir, filename)
        file.save(file_path)
        print(f"Common file saved: {file_path}")

        # Get file info
        info = converter.get_audio_file_info(file_path)

        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/api/common-files/{filename}',
            'path': file_path,
            'relative_path': os.path.relpath(file_path, os.getcwd()),
            'duration_ms': info['duration_ms'] if info else None,
            'duration_sec': info['duration_sec'] if info else None,
            'file_size_bytes': info['file_size_bytes'] if info else None
        })
    except Exception as e:
        import traceback
        print(f"\n=== ERROR uploading common file ===")
        print(f"Exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/common-files/info/<filename>', methods=['GET'])
@auth_manager.require_api_key
def get_common_file_info(filename):
    """Get detailed information about a specific common file"""
    try:
        file_path = os.path.join(converter.common_files_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        info = converter.get_audio_file_info(file_path)
        if not info:
            return jsonify({'error': 'Could not read audio file information'}), 500

        return jsonify({
            'filename': filename,
            'path': file_path,
            'relative_path': os.path.relpath(file_path, os.getcwd()),
            'duration_ms': info['duration_ms'],
            'duration_sec': info['duration_sec'],
            'sample_rate': info['sample_rate'],
            'channels': info['channels'],
            'file_size_bytes': info['file_size_bytes']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generated-audio/<txt_filename>', methods=['GET'])
def list_generated_audio(txt_filename):
    """List all generated audio files for a specific text file"""
    try:
        import json
        base_name = os.path.splitext(txt_filename)[0]
        generated_audio = []

        # Look for all audio and metadata files matching this text file
        for filename in os.listdir(converter.audio_dir):
            if filename.startswith(base_name) and filename.endswith('.json'):
                metadata_path = os.path.join(converter.audio_dir, filename)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    generated_audio.append({
                        'audio_url': f'/api/audio/{metadata["audio_file"]}',
                        'audio_file': metadata['audio_file'],
                        'metadata': metadata
                    })

        # Sort by timestamp (newest first)
        generated_audio.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)

        return jsonify({'generated_audio': generated_audio})
    except Exception as e:
        import traceback
        print(f"Error listing generated audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ===== PROJECT MANAGEMENT ENDPOINTS =====


@app.route('/api/project/load', methods=['POST'])
@auth_manager.require_api_key
def load_project():
    """Load an existing project from a folder"""
    try:
        import json
        from datetime import datetime

        print('\n━━━ [LOAD PROJECT API] START ━━━')
        data = request.json
        print(f'[LOAD PROJECT API] Request data: {data}')

        project_path = data.get('project_path')
        print(f'[LOAD PROJECT API] Project path: {project_path}')

        if not project_path:
            print('[LOAD PROJECT API] ✗ ERROR: project_path is required')
            return jsonify({'error': 'project_path is required'}), 400

        print(f'[LOAD PROJECT API] Checking if path exists: {project_path}')
        if not os.path.exists(project_path):
            print(f'[LOAD PROJECT API] ✗ ERROR: Path does not exist')
            return jsonify({'error': 'Project path does not exist'}), 404

        # Load project metadata
        project_file = os.path.join(project_path, 'project.json')
        print(f'[LOAD PROJECT API] Looking for project file: {project_file}')

        if not os.path.exists(project_file):
            print(f'[LOAD PROJECT API] ✗ ERROR: project.json not found')
            return jsonify({'error': 'Not a valid project folder (project.json not found)'}), 400

        print(f'[LOAD PROJECT API] Reading project.json...')
        with open(project_file, 'r', encoding='utf-8') as f:
            project_metadata = json.load(f)
        print(f'[LOAD PROJECT API] Project metadata: {preview_json_for_log(project_metadata)}')

        # Add default audio settings if not present (backwards compatibility)
        if 'default_audio_settings' not in project_metadata:
            project_metadata['default_audio_settings'] = {
                'exaggeration': 0.6,
                'cfg_weight': 0.4,
                'voice_sample': 'none',
                'seed': 0,
                'temperature': 0.8,
                'ref_vad_trimming': False
            }

        # Update last modified
        project_metadata['last_modified'] = datetime.now().isoformat()
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_metadata, f, indent=2, ensure_ascii=False)

        # Update converter paths
        converter.current_project_path = project_path
        converter.current_project_metadata = project_metadata
        converter.undo_stack = []  # fresh undo history per loaded project
        converter.audio_dir = os.path.join(project_path, 'audio')
        # Keep voice_samples_dir pointing to main folder (not project-specific)

        # Get list of text files
        texts_dir = os.path.join(project_path, 'texts')
        text_files = []
        if os.path.exists(texts_dir):
            text_files = [f for f in os.listdir(texts_dir) if f.endswith('.txt')]

        print(f'[LOAD PROJECT API] ✓ Project loaded successfully')
        print(f'[LOAD PROJECT API] Text files found: {len(text_files)}')
        print('━━━ [LOAD PROJECT API] END ━━━\n')

        return jsonify({
            'success': True,
            'project_path': project_path,
            'metadata': project_metadata,
            'text_files': text_files
        })

    except Exception as e:
        import traceback
        print(f"[LOAD PROJECT API] ✗ ERROR: {str(e)}")
        print(traceback.format_exc())
        print('━━━ [LOAD PROJECT API] END (ERROR) ━━━\n')
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/info', methods=['GET'])
def get_project_info():
    """Get current project information"""
    try:
        if converter.current_project_path is None:
            return jsonify({'has_project': False})

        return jsonify({
            'has_project': True,
            'project_path': converter.current_project_path,
            'metadata': converter.current_project_metadata
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/update-defaults', methods=['POST'])
def update_project_defaults():
    """Update project default audio settings"""
    try:
        from datetime import datetime

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        exaggeration = float(data.get('exaggeration', 0.5))
        cfg_weight = float(data.get('cfg_weight', 0.5))
        voice_sample = data.get('voice_sample', 'none')
        seed = int(data.get('seed', 0))
        temperature = float(data.get('temperature', 0.8))
        ref_vad_trimming = bool(data.get('ref_vad_trimming', False))

        # Update metadata
        if 'default_audio_settings' not in converter.current_project_metadata:
            converter.current_project_metadata['default_audio_settings'] = {}

        converter.current_project_metadata['default_audio_settings'] = {
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'voice_sample': voice_sample,
            'seed': seed,
            'temperature': temperature,
            'ref_vad_trimming': ref_vad_trimming
        }
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'default_audio_settings': converter.current_project_metadata['default_audio_settings']
        })

    except Exception as e:
        import traceback
        print(f"Error updating project defaults: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/get-text-files', methods=['GET'])
def get_project_text_files():
    """Get all text files/chapters in the current project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        # Return both chapters (new) and text_files (old) for backwards compatibility
        chapters = converter.current_project_metadata.get('chapters', [])
        text_files = converter.current_project_metadata.get('text_files', [])

        return jsonify({
            'success': True,
            'chapters': chapters,
            'text_files': text_files,  # Keep for backwards compatibility
            'has_chapters': len(chapters) > 0,
            'content_xml': converter.current_project_metadata.get('content_xml', None),
            'gutenberg': {
                'book_id': converter.current_project_metadata.get('gutenberg_book_id'),
                'source_url': converter.current_project_metadata.get('gutenberg_source_url'),
                'txt_url': converter.current_project_metadata.get('gutenberg_txt_url'),
                'epub_url': converter.current_project_metadata.get('gutenberg_epub_url'),
            }
        })

    except Exception as e:
        import traceback
        print(f"Error getting project text files: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/update-chunk-text', methods=['POST'])
def update_chunk_text():
    """Update the text of a specific chunk and mark as dirty. Auto-splits if text is too long."""
    try:
        import json
        from datetime import datetime

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        text_file_id = data.get('text_file_id')
        chunk_id = int(data.get('chunk_id'))
        new_text = data.get('new_text')
        new_nickname = data.get('new_nickname')

        # Try to find in chapters first (new format), then text_files (old format)
        # Use O(1) lookup dictionaries instead of O(n) linear search
        chapter_map, text_file_map, chunk_maps = converter.get_chapter_and_chunk_lookups()

        container = chapter_map.get(text_file_id) or text_file_map.get(text_file_id)

        if not container:
            return jsonify({'error': 'Text file not found'}), 404

        # Find the chunk
        chunk = chunk_maps.get(text_file_id, {}).get(chunk_id)

        if not chunk:
            return jsonify({'error': 'Chunk not found'}), 404

        # Length is measured on the text actually sent to the TTS engine (pronunciation
        # markup resolved, tags stripped) — NOT the displayed text with {display|spoken}
        # or [tag] annotations. If it's too long we refuse the edit and say so explicitly
        # rather than silently auto-splitting (which could drop content).
        max_chunk_size = config.MAX_CHUNK_SIZE
        new_text = new_text or ''
        resolved = converter.strip_xml_tags(converter.process_pronunciation_markup(new_text))
        if len(resolved) > max_chunk_size:
            print(f"[UPDATE CHUNK] Rejected chunk {chunk_id}: {len(resolved)} TTS chars "
                  f"(raw {len(new_text)}) exceeds max {max_chunk_size}")
            return jsonify({
                'error': (f'Chunk too long for the TTS engine: {len(resolved)} characters after '
                          f'resolving markup (limit {max_chunk_size}). Shorten it or split the chunk.'),
                'too_long': True,
                'resolved_length': len(resolved),
                'raw_length': len(new_text),
                'max_length': max_chunk_size,
            }), 400

        push_undo()
        chunk['text'] = new_text
        if new_nickname is not None:
            chunk['nickname'] = new_nickname
        else:
            chunk['nickname'] = new_text[:50].strip() + ('...' if len(new_text) > 50 else '')
        if len(chunk.get('generated_audios', [])) > 0:
            chunk['dirty'] = True

        _save_current_project()
        return jsonify({'success': True, 'chunk': chunk, 'was_split': False})

    except Exception as e:
        import traceback
        print(f"Error updating chunk text: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/dismiss-dirty-flag', methods=['POST'])
def dismiss_dirty_flag():
    """Dismiss the dirty flag for a chunk"""
    try:
        import json
        from datetime import datetime

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        text_file_id = data.get('text_file_id')
        chunk_id = int(data.get('chunk_id'))

        # Find the text file
        text_files = converter.current_project_metadata.get('text_files', [])
        text_file = next((tf for tf in text_files if tf['id'] == text_file_id), None)

        if not text_file:
            return jsonify({'error': 'Text file not found'}), 404

        # Find the chunk
        chunk = next((c for c in text_file['chunks'] if c['id'] == chunk_id), None)

        if not chunk:
            return jsonify({'error': 'Chunk not found'}), 404

        # Dismiss dirty flag
        chunk['dirty'] = False

        # Update project metadata
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'chunk': chunk
        })

    except Exception as e:
        import traceback
        print(f"Error dismissing dirty flag: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/insert-chunk', methods=['POST'])
def insert_chunk():
    """Insert a new chunk (text, pause, or common_file) at a specific position"""
    try:
        import json
        from datetime import datetime

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        push_undo()
        chapter_id = data.get('chapter_id') or data.get('text_file_id')  # Support both old and new formats
        insert_after_id = data.get('insert_after_id')  # Insert after this chunk ID (or -1 to insert at beginning)
        chunk_type = data.get('type', 'text')  # 'text', 'pause', or 'common_file'

        # Find the chapter (try new structure first, then fall back to old)
        chapters = converter.current_project_metadata.get('chapters', [])
        text_files = converter.current_project_metadata.get('text_files', [])

        chapter = next((ch for ch in chapters if ch['id'] == chapter_id), None)
        if not chapter:
            # Fall back to old text_files structure
            chapter = next((tf for tf in text_files if tf['id'] == chapter_id), None)

        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404

        # Create new chunk based on type
        if chunk_type == 'pause':
            duration_ms = int(data.get('duration_ms', 1000))
            new_chunk = {
                'id': 0,  # Will be set correctly below
                'type': 'pause',
                'duration_ms': duration_ms,
                'text': f'[Pause: {duration_ms}ms]',
                'nickname': f'Pause ({duration_ms}ms)',
                'start_pos': 0,
                'end_pos': 0,
                'dirty': False,
                'generated_audios': []
            }
        elif chunk_type == 'common_file':
            # Common file chunk
            common_file_path = data.get('common_file_path')
            if not common_file_path:
                return jsonify({'error': 'common_file_path is required for common_file chunks'}), 400

            # Get file info
            filename = os.path.basename(common_file_path)
            full_path = os.path.join(converter.common_files_dir, filename)

            if not os.path.exists(full_path):
                return jsonify({'error': f'Common file not found: {filename}'}), 404

            # Get audio file info
            info = converter.get_audio_file_info(full_path)

            new_chunk = {
                'id': 0,  # Will be set correctly below
                'type': 'common_file',
                'common_file_path': os.path.relpath(full_path, os.getcwd()),
                'filename': filename,
                'duration_ms': info['duration_ms'] if info else 0,
                'text': f'[Common File: {filename}]',
                'nickname': f'{filename} ({info["duration_sec"]:.1f}s)' if info else filename,
                'start_pos': 0,
                'end_pos': 0,
                'dirty': False,
                'generated_audios': []  # Common files don't have generated audios, they're used as-is
            }
        else:  # text chunk
            text = data.get('text', '')
            new_chunk = {
                'id': 0,  # Will be set correctly below
                'type': 'text',
                'text': text,
                'nickname': text[:50].strip() + ('...' if len(text) > 50 else ''),
                'start_pos': 0,
                'end_pos': len(text),
                'dirty': False,
                'generated_audios': []
            }

        # Determine insertion position
        chunks = chapter.get('chunks', [])
        insert_position = insert_after_id + 1  # Insert after the specified chunk

        # Insert the new chunk
        chunks.insert(insert_position, new_chunk)

        # Renumber all chunk IDs
        for i, chunk in enumerate(chunks):
            chunk['id'] = i

        # Update project metadata
        chapter['chunks'] = chunks
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'chunks': chunks,
            'inserted_chunk': new_chunk
        })

    except Exception as e:
        import traceback
        print(f"Error inserting chunk: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/set-pause-duration', methods=['POST'])
@auth_manager.require_api_key
def set_pause_duration():
    """Update a pause chunk's duration. Body: { chapter_id, chunk_id, duration_ms }."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400
        data = request.json or {}
        chapter_id = data.get('chapter_id') or data.get('text_file_id')
        chunk_id = data.get('chunk_id')
        duration_ms = int(data.get('duration_ms', 500))
        duration_ms = max(0, min(duration_ms, 60000))
        push_undo()

        chapter = _find_chapter(chapter_id)
        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404
        chunk = next((c for c in chapter.get('chunks', []) if str(c.get('id')) == str(chunk_id)), None)
        if not chunk:
            return jsonify({'error': 'Chunk not found'}), 404
        if chunk.get('type') != 'pause':
            return jsonify({'error': 'Chunk is not a pause'}), 400

        chunk['duration_ms'] = duration_ms
        chunk['text'] = f'[Pause: {duration_ms}ms]'
        chunk['nickname'] = f'Pause ({duration_ms}ms)'
        _save_current_project()
        return jsonify({'success': True, 'chunk_id': chunk_id, 'duration_ms': duration_ms})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/delete-chunk', methods=['POST'])
@auth_manager.require_api_key
def delete_chunk():
    """Delete a chunk (e.g. a pause). Body: { chapter_id, chunk_id }."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400
        data = request.json or {}
        chapter_id = data.get('chapter_id') or data.get('text_file_id')
        chunk_id = data.get('chunk_id')
        push_undo()

        chapter = _find_chapter(chapter_id)
        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404
        chunks = chapter.get('chunks', [])
        idx = next((i for i, c in enumerate(chunks) if str(c.get('id')) == str(chunk_id)), None)
        if idx is None:
            return jsonify({'error': 'Chunk not found'}), 404
        del chunks[idx]
        _save_current_project()
        return jsonify({'success': True})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/add-audio-to-chunk', methods=['POST'])
def add_audio_to_chunk():
    """Add generated audio metadata to a chunk in the project"""
    try:
        import json
        from datetime import datetime

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        text_file_id = data.get('text_file_id')
        chunk_id = int(data.get('chunk_id'))
        audio_metadata = data.get('audio_metadata')

        if not text_file_id or chunk_id is None or not audio_metadata:
            return jsonify({'error': 'text_file_id, chunk_id, and audio_metadata are required'}), 400

        # Try to find in chapters first (new format), then text_files (old format)
        # Use O(1) lookup dictionaries instead of O(n) linear search
        chapter_map, text_file_map, chunk_maps = converter.get_chapter_and_chunk_lookups()

        container = chapter_map.get(text_file_id) or text_file_map.get(text_file_id)

        if not container:
            return jsonify({'error': 'Text file not found'}), 404

        # Find the chunk
        chunk = chunk_maps.get(text_file_id, {}).get(chunk_id)

        if not chunk:
            return jsonify({'error': 'Chunk not found'}), 404

        # Initialize generated_audios if not present
        if 'generated_audios' not in chunk:
            chunk['generated_audios'] = []

        # Add audio metadata to chunk
        chunk['generated_audios'].append(audio_metadata)

        # Update project metadata
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'chunk': chunk
        })

    except Exception as e:
        import traceback
        print(f"Error adding audio to chunk: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/set-chunk-best-take', methods=['POST'])
def set_chunk_best_take():
    """Set the best take for a chunk in the project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        push_undo()
        text_file_id = data.get('text_file_id')
        chunk_id = data.get('chunk_id')
        audio_filename = data.get('audio_filename')

        if not text_file_id or chunk_id is None or not audio_filename:
            return jsonify({'error': 'text_file_id, chunk_id, and audio_filename are required'}), 400

        # Try to find in chapters first (new format), then text_files (old format)
        # Use O(1) lookup dictionaries instead of O(n) linear search
        chapter_map, text_file_map, chunk_maps = converter.get_chapter_and_chunk_lookups()

        container = chapter_map.get(text_file_id) or text_file_map.get(text_file_id)

        if not container:
            return jsonify({'error': 'Text file not found'}), 404

        # Find the chunk
        chunk = chunk_maps.get(text_file_id, {}).get(chunk_id)

        if not chunk:
            return jsonify({'error': 'Chunk not found'}), 404

        # Update all audio items - set is_best_take to False
        for audio in chunk.get('generated_audios', []):
            audio['is_best_take'] = False

        # Find and mark the selected audio as best take
        selected_audio = next((a for a in chunk.get('generated_audios', [])
                              if a['audio_file'] == audio_filename), None)

        if not selected_audio:
            return jsonify({'error': 'Audio file not found in chunk'}), 404

        selected_audio['is_best_take'] = True

        # Update project metadata
        from datetime import datetime
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'chunk': chunk
        })

    except Exception as e:
        import traceback
        print(f"Error setting chunk best take: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/delete-audio', methods=['POST'])
def delete_project_audio():
    """Delete an audio file from a chunk in the project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        push_undo()
        text_file_id = data.get('text_file_id')
        chunk_id = data.get('chunk_id')
        audio_filename = data.get('audio_file') or data.get('audio_filename')

        if not text_file_id or chunk_id is None or not audio_filename:
            return jsonify({'error': 'text_file_id, chunk_id, and audio_file are required'}), 400

        # Try to find in chapters first (new format), then text_files (old format)
        # Use O(1) lookup dictionaries instead of O(n) linear search
        chapter_map, text_file_map, chunk_maps = converter.get_chapter_and_chunk_lookups()

        container = chapter_map.get(text_file_id) or text_file_map.get(text_file_id)

        if not container:
            return jsonify({'error': 'Text file not found'}), 404

        # Find the chunk
        chunk = chunk_maps.get(text_file_id, {}).get(chunk_id)

        if not chunk:
            return jsonify({'error': 'Chunk not found'}), 404

        # Find and remove the audio from generated_audios list
        generated_audios = chunk.get('generated_audios', [])
        audio_to_remove = next((a for a in generated_audios if a['audio_file'] == audio_filename), None)

        if not audio_to_remove:
            return jsonify({'error': 'Audio file not found in chunk'}), 404

        # Remove from list
        generated_audios.remove(audio_to_remove)

        # If this was the best take and there are other audios, make the first one the best take
        if audio_to_remove.get('is_best_take') and len(generated_audios) > 0:
            generated_audios[0]['is_best_take'] = True

        # Delete the actual audio file
        audio_path = os.path.join(converter.audio_dir, audio_filename)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Deleted audio file: {audio_path}")

        # Update project metadata
        from datetime import datetime
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'chunk': chunk
        })

    except Exception as e:
        import traceback
        print(f"Error deleting audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/generate-chunk-audio', methods=['POST'])
def generate_project_chunk_audio():
    """Generate audio for a chunk within the current project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        text_file_id = data.get('text_file_id')
        chunk_id = data.get('chunk_id')
        chunk_text = data.get('chunk_text')

        if not text_file_id or chunk_id is None or not chunk_text:
            return jsonify({'error': 'text_file_id, chunk_id, and chunk_text are required'}), 400

        # Get generation parameters
        voice_sample = data.get('voice_sample')
        exaggeration = float(data.get('exaggeration', 0.6))
        cfg_weight = float(data.get('cfg_weight', 0.4))
        temperature = float(data.get('temperature', 0.8))
        seed = int(data.get('seed', 0))
        language_id = data.get('language_id', 'en')
        tts_model = data.get('tts_model', config.DEFAULT_TTS_MODEL if hasattr(config, 'DEFAULT_TTS_MODEL') else 'chatterbox')

        # Verbose model selection logging
        requested_model = tts_model
        has_paralinguistic_tags = converter.text_has_paralinguistic_tags(chunk_text)
        print(f"\n[TTS MODEL SELECTION]")
        print(f"  Requested: {requested_model}")
        print(f"  HAS_TURBO available: {HAS_TURBO}")
        print(f"  Paralinguistic tags detected: {has_paralinguistic_tags}")

        # Note: Chatterbox Turbo expects [laugh], [cough], etc. embedded directly in text.
        # process_pronunciation_markup strips the {display|[tag]} wrapper, leaving [tag] in the text.

        # Auto-detect paralinguistic tags → force turbo
        if has_paralinguistic_tags:
            tts_model = 'chatterbox_turbo'
            print(f"  → Emotion tags found, requesting Turbo model")
        else:
            print(f"  → No emotion tags, using requested model: {requested_model}")

        if tts_model == 'chatterbox_turbo' and not HAS_TURBO:
            print(f"  ⚠️  Turbo requested but NOT INSTALLED (HAS_TURBO=False)")
            print(f"  ⚠️  Falling back to Standard model (emotion tags will be stripped)")
            tts_model = 'chatterbox'

        print(f"  ✓ Final model to use: {tts_model}")

        # Resolve voice sample — never fall back to the Chatterbox default voice.
        audio_prompt_path = resolve_voice_sample_path(voice_sample)
        if audio_prompt_path is None:
            audio_prompt_path = resolve_voice_sample_path(getattr(config, 'DEFAULT_VOICE', None))
        if audio_prompt_path is None:
            return jsonify({
                'error': 'No usable voice sample found. Refusing to generate with the '
                         'Chatterbox default voice. Set a project voice or configure DEFAULT_VOICE.',
                'requested_voice': voice_sample,
            }), 400
        voice_sample = os.path.basename(audio_prompt_path)

        # Check device before generation
        if converter.model is not None:
            model_device = "GPU" if converter.device == "cuda" else "CPU"
        elif torch.cuda.is_available():
            model_device = "GPU (will load)"
        else:
            model_device = "CPU"

        print(f"\n=== Generating audio for chunk {chunk_id} on {model_device} ===")
        print(f"Text file ID: {text_file_id}")
        print(f"Voice: {voice_sample}")
        print(f"Model: {tts_model} (requested={requested_model}, emotion_tags={has_paralinguistic_tags}, turbo_available={HAS_TURBO})")
        print(f"Parameters: Exaggeration={exaggeration}, CFG={cfg_weight}, Temp={temperature}")

        # Generate audio filename with timestamp and chunk ID
        timestamp = int(time.time() * 1000)
        audio_filename = f"chunk{chunk_id}_{timestamp}.wav"
        audio_path = os.path.join(converter.audio_dir, audio_filename)

        # Process pronunciation markup: {display|spoken} → spoken form for TTS
        # If using Standard model and text has emotion tags, strip them (Standard doesn't support them)
        use_turbo = (tts_model == 'chatterbox_turbo')
        tts_text = converter.process_pronunciation_markup(chunk_text, use_turbo=use_turbo)
        if tts_text != chunk_text:
            print(f"[TTS] Pronunciation markup applied: {len(chunk_text)} → {len(tts_text)} chars")
            if not use_turbo:
                print(f"[TTS] Using Standard model: emotion tags will be converted to display text")

        # Generate the audio (returns dict with path and duration)
        generation_result = converter.generate_audio(
            text=tts_text,
            output_path=audio_path,
            audio_prompt_path=audio_prompt_path,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            tts_model=tts_model,
            temperature=temperature,
            seed=seed
        )

        # Extract path, duration, and generation time from result
        generated_path = generation_result['path']
        audio_duration = generation_result['duration_seconds']
        generation_time_ms = generation_result['generation_time_ms']

        # Extract just the filename from the path
        audio_file = os.path.basename(generated_path)
        audio_url = f"/api/audio/{audio_file}"

        # Check if audio is at or near the Chatterbox TTS 40-second limit
        # Flag if duration >= 39.5 seconds (might be truncated)
        CHATTERBOX_MAX_DURATION = 40.0
        possibly_truncated = audio_duration >= (CHATTERBOX_MAX_DURATION - 0.5)

        # Update project metadata with the new audio
        # Use O(1) lookup dictionaries instead of O(n) linear search
        chapter_map, text_file_map, chunk_maps = converter.get_chapter_and_chunk_lookups()

        container = chapter_map.get(text_file_id) or text_file_map.get(text_file_id)

        if container:
            chunk = chunk_maps.get(text_file_id, {}).get(chunk_id)
            if chunk:
                if 'generated_audios' not in chunk:
                    chunk['generated_audios'] = []

                # Add the new audio
                audio_entry = {
                    'timestamp': int(time.time() * 1000),
                    'audio_file': audio_file,
                    'audio_url': audio_url,
                    'is_best_take': len(chunk['generated_audios']) == 0,  # First take is best by default
                    'voice_sample': voice_sample,
                    'exaggeration': exaggeration,
                    'cfg_weight': cfg_weight,
                    'temperature': temperature,
                    'seed': seed,
                    'audio_duration_seconds': round(audio_duration, 2),
                    'possibly_truncated': possibly_truncated,
                    'generation_time_ms': generation_time_ms,
                    'input_text': chunk_text,  # Original text with markup (for detecting outdated takes)
                    'tts_input_text': tts_text,  # Exact text sent to TTS engine (for debugging)
                    'tts_model': tts_model,
                    'tts_model_requested': requested_model,
                    'tts_model_emotion_forced': has_paralinguistic_tags and requested_model != tts_model
                }
                chunk['generated_audios'].append(audio_entry)

                # Log what was saved
                print(f"\n[TTS METADATA SAVED]")
                print(f"  Model saved as: {tts_model}")
                print(f"  Model requested: {requested_model}")
                print(f"  Emotion tags in input: {has_paralinguistic_tags}")
                if audio_entry['tts_model_emotion_forced']:
                    print(f"  ⚠️  Model was forced due to emotion tags (requested {requested_model}, saved {tts_model})")

                # Mark chunk as not dirty
                chunk['dirty'] = False

                # Save project metadata
                from datetime import datetime
                converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

                project_file = os.path.join(converter.current_project_path, 'project.json')
                with open(project_file, 'w', encoding='utf-8') as f:
                    json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

                # Invalidate lookup caches after modifying metadata
                converter._invalidate_lookup_caches()

                return jsonify({
                    'success': True,
                    'audio_file': audio_file,
                    'audio_url': audio_url,
                    'chunk': chunk
                })

        return jsonify({'error': 'Chunk not found in project metadata'}), 404

    except Exception as e:
        import traceback
        print(f"Error generating chunk audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/transcribe-take', methods=['POST'])
def transcribe_take():
    """Transcribe a generated take with Whisper and score it against the original chunk text."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        audio_file = data.get('audio_file')
        chunk_text = data.get('chunk_text', '')
        text_file_id = data.get('text_file_id')
        chunk_id = data.get('chunk_id')

        if not audio_file:
            return jsonify({'error': 'audio_file is required'}), 400

        audio_path = os.path.join(converter.audio_dir, audio_file)
        audio_path = os.path.abspath(audio_path)
        if not os.path.exists(audio_path):
            return jsonify({'error': f'Audio file not found: {audio_file}'}), 404

        print(f"[Transcription] Loading audio from: {audio_path}")

        # Load model and transcribe
        model = get_whisper_model()
        result = model.transcribe(str(audio_path), language='en', fp16=False)
        transcription = result['text'].strip()

        # Compute similarity against original chunk text
        similarity = 0.0
        if chunk_text:
            similarity = compute_similarity(chunk_text, transcription)
        similarity_score = round(similarity * 100, 1)

        print(f"[Transcription] {audio_file}: score={similarity_score}%")

        # Persist into project.json if chunk reference provided
        if text_file_id is not None and chunk_id is not None:
            chapter_map, text_file_map, chunk_maps = converter.get_chapter_and_chunk_lookups()
            container = chapter_map.get(text_file_id) or text_file_map.get(text_file_id)
            if container:
                chunk = chunk_maps.get(text_file_id, {}).get(chunk_id)
                if chunk:
                    for entry in chunk.get('generated_audios', []):
                        if entry.get('audio_file') == audio_file:
                            entry['transcription'] = transcription
                            entry['similarity_score'] = similarity_score
                            break
                    from datetime import datetime
                    converter.current_project_metadata['last_modified'] = datetime.now().isoformat()
                    project_file = os.path.join(converter.current_project_path, 'project.json')
                    with open(project_file, 'w', encoding='utf-8') as f:
                        json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)
                    converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'transcription': transcription,
            'similarity_score': similarity_score
        })

    except Exception as e:
        import traceback
        print(f"Error transcribing take: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/whisper', methods=['GET'])
def debug_whisper():
    """
    Diagnostic endpoint: checks ffmpeg availability and tests Whisper on a
    voice sample so we can isolate path vs. model vs. ffmpeg issues.
    """
    import subprocess
    import shutil
    import traceback

    result = {
        'ffmpeg_in_path': False,
        'ffmpeg_version': None,
        'whisper_model_loaded': False,
        'voice_sample_path': None,
        'voice_sample_exists': False,
        'transcription_test': None,
        'error': None,
    }

    # 1. Check ffmpeg
    ffmpeg_exe = shutil.which('ffmpeg')
    result['ffmpeg_in_path'] = ffmpeg_exe is not None
    if ffmpeg_exe:
        try:
            ver = subprocess.check_output(
                [ffmpeg_exe, '-version'], stderr=subprocess.STDOUT, timeout=5
            ).decode(errors='replace').split('\n')[0]
            result['ffmpeg_version'] = ver
        except Exception as fe:
            result['ffmpeg_version'] = f'error: {fe}'

    # 2. Find a voice sample to test with
    vs_dir = os.path.abspath(converter.voice_samples_dir)
    sample_path = None
    if os.path.isdir(vs_dir):
        for fn in os.listdir(vs_dir):
            if fn.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                sample_path = os.path.join(vs_dir, fn)
                break
    result['voice_sample_path'] = sample_path
    result['voice_sample_exists'] = sample_path is not None and os.path.exists(sample_path)

    # 3. Try to load the model
    try:
        model = get_whisper_model()
        result['whisper_model_loaded'] = True
    except Exception as me:
        result['error'] = f'Model load failed: {me}'
        return jsonify(result)

    # 4. Try transcribing the voice sample
    if sample_path and os.path.exists(sample_path):
        try:
            tr = model.transcribe(str(sample_path), language='en', fp16=False)
            result['transcription_test'] = tr['text'].strip()[:200]
        except Exception as te:
            result['transcription_test'] = f'FAILED: {te}'
            result['error'] = traceback.format_exc()
    else:
        result['transcription_test'] = 'skipped — no voice sample found'

    return jsonify(result)

@app.route('/api/project/stitch-best-takes', methods=['POST'])
def stitch_project_best_takes():
    """Stitch together the best takes from all chunks in a chapter/text file within the current project"""
    try:
        import json
        import time

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        chapter_id = data.get('chapter_id') or data.get('text_file_id')  # Support both old and new formats

        if not chapter_id:
            return jsonify({'error': 'chapter_id (or text_file_id) is required'}), 400

        # Find the chapter (try new structure first, then fall back to old)
        # Use O(1) lookup dictionaries instead of O(n) linear search
        chapter_map, text_file_map, chunk_maps = converter.get_chapter_and_chunk_lookups()

        chapter = chapter_map.get(chapter_id)
        is_new_structure = chapter is not None

        if not chapter:
            # Fall back to old text_files structure
            chapter = text_file_map.get(chapter_id)

        if not chapter:
            return jsonify({'error': 'Chapter/text file not found in project'}), 404

        # Get chunks sorted by ID
        chunks = sorted(chapter.get('chunks', []), key=lambda c: c['id'])

        if not chunks:
            return jsonify({'error': 'No chunks found in chapter'}), 400

        # Collect audio paths for best takes
        audio_paths = []
        segment_entries = []  # parallel list for segment map generation
        skipped_chunks = []   # chunks with missing/invalid audio (warnings)
        audio_dir = os.path.join(converter.current_project_path, 'audio')

        for chunk in chunks:
            chunk_type = chunk.get('type', 'text')

            if chunk_type == 'common_file':
                # For common_file chunks, use the common file directly
                common_file_path = chunk.get('common_file_path')
                if not common_file_path:
                    msg = f'Chunk {chunk["id"]} is missing common_file_path'
                    print(f'WARNING: {msg} — skipping')
                    skipped_chunks.append({'chunk_id': chunk['id'], 'reason': msg})
                    continue

                # Resolve relative path
                if not os.path.isabs(common_file_path):
                    common_file_path = os.path.join(os.getcwd(), common_file_path)

                if not os.path.exists(common_file_path):
                    msg = f'Common file not found: {common_file_path}'
                    print(f'WARNING: {msg} — skipping chunk {chunk["id"]}')
                    skipped_chunks.append({'chunk_id': chunk['id'], 'reason': msg})
                    continue

                print(f"Chunk {chunk['id']}: Using common file: {common_file_path}")
                audio_paths.append(common_file_path)
                segment_entries.append({'type': 'common_file', 'chunk_id': chunk['id'], 'chunk_nickname': chunk.get('nickname', ''), 'take_timestamp': None, 'audio_path': common_file_path})

            elif chunk_type == 'pause':
                # For pause chunks, generate silence
                duration_ms = chunk.get('duration_ms', 1000)
                print(f"Chunk {chunk['id']}: Generating {duration_ms}ms pause")
                # We'll handle pause chunks in the stitching function
                audio_paths.append(('pause', duration_ms))
                segment_entries.append({'type': 'pause', 'chunk_id': chunk['id'], 'chunk_nickname': chunk.get('nickname', ''), 'take_timestamp': None, 'duration_ms': duration_ms})

            else:
                # For text chunks, use generated audio
                generated_audios = chunk.get('generated_audios', [])

                if not generated_audios:
                    msg = f'No audio generated for chunk {chunk["id"]}'
                    print(f'WARNING: {msg} — skipping')
                    skipped_chunks.append({'chunk_id': chunk['id'], 'reason': msg})
                    continue

                # Find best take
                best_audio = None
                for audio in generated_audios:
                    if audio.get('is_best_take', False):
                        best_audio = audio
                        break

                # If no best take marked, use the most recent
                if not best_audio:
                    best_audio = max(generated_audios, key=lambda a: a.get('timestamp', 0))
                    print(f"Chunk {chunk['id']}: No best take marked, using most recent (timestamp: {best_audio.get('timestamp', 0)})")
                else:
                    print(f"Chunk {chunk['id']}: Using best take (timestamp: {best_audio.get('timestamp', 0)})")

                # Build audio file path
                # Handle both 'audio_file' (filename) and 'audio_url' (URL path)
                if 'audio_file' in best_audio:
                    audio_file = best_audio['audio_file']
                elif 'audio_url' in best_audio:
                    # Extract filename from URL like '/api/audio/filename.wav'
                    audio_file = best_audio['audio_url'].split('/')[-1]
                else:
                    msg = f'Chunk {chunk["id"]} has invalid audio metadata (no audio_file or audio_url)'
                    print(f'WARNING: {msg} — skipping. Audio object: {best_audio}')
                    skipped_chunks.append({'chunk_id': chunk['id'], 'reason': msg})
                    continue
                audio_path = os.path.join(audio_dir, audio_file)

                print(f"Chunk {chunk['id']}: Selected audio file: {audio_file}")
                print(f"Chunk {chunk['id']}: Full path: {audio_path}")
                print(f"Chunk {chunk['id']}: File exists: {os.path.exists(audio_path)}")

                if not os.path.exists(audio_path):
                    msg = f'Audio file not found for chunk {chunk["id"]}: {audio_file}'
                    print(f'WARNING: {msg} — skipping')
                    skipped_chunks.append({'chunk_id': chunk['id'], 'reason': msg})
                    continue

                audio_paths.append(audio_path)
                segment_entries.append({'type': 'text', 'chunk_id': chunk['id'], 'chunk_nickname': chunk.get('nickname', ''), 'take_timestamp': best_audio.get('timestamp'), 'audio_path': audio_path})

        if not audio_paths:
            skip_reasons = '; '.join(s['reason'] for s in skipped_chunks)
            return jsonify({'error': f'No audio available to stitch. Skipped chunks: {skip_reasons}'}), 400

        if skipped_chunks:
            print(f"WARNING: Stitching with {len(audio_paths)} segments; skipped {len(skipped_chunks)} chunk(s): {[s['chunk_id'] for s in skipped_chunks]}")

        # Create stitched audio filename
        timestamp = int(time.time() * 1000)
        if is_new_structure:
            chapter_title = chapter.get('title', 'output')
            base_name = chapter_title.replace(' ', '_').replace('/', '_')[:50]
        else:
            original_filename = chapter.get('original_filename', 'output')
            base_name = os.path.splitext(original_filename)[0]
        stitched_filename = f"{base_name}_stitched_{timestamp}.wav"
        stitched_path = os.path.join(audio_dir, stitched_filename)

        # Stitch the audio files
        print(f"Stitching {len(audio_paths)} audio files...")
        converter.stitch_audio_files(audio_paths, stitched_path)
        print(f"Stitched audio saved to: {stitched_path}")

        # Build segment timing map (mirrors stitch_audio_files inter-chunk silence)
        INTER_CHUNK_MS = 100
        cursor_ms = 0
        segments = []
        for entry in segment_entries:
            if entry['type'] == 'pause':
                dur_ms = entry['duration_ms']
            else:
                try:
                    seg = AudioSegment.from_file(entry['audio_path'])
                    dur_ms = len(seg)
                except Exception:
                    dur_ms = 0
            segments.append({
                'chunk_id': entry['chunk_id'],
                'chunk_nickname': entry.get('chunk_nickname', ''),
                'take_timestamp': entry.get('take_timestamp'),
                'type': entry['type'],
                'start_seconds': round(cursor_ms / 1000, 3),
                'end_seconds': round((cursor_ms + dur_ms) / 1000, 3),
            })
            cursor_ms += dur_ms + INTER_CHUNK_MS
        total_duration_seconds = round(cursor_ms / 1000, 3)

        # Save segment map JSON alongside stitched file
        map_filename = f"{base_name}_map_{timestamp}.json"
        map_path = os.path.join(audio_dir, map_filename)
        segment_map = {
            'chapter_id': chapter_id,
            'chapter_title': chapter.get('title', ''),
            'stitched_file': stitched_filename,
            'total_duration_seconds': total_duration_seconds,
            'created_at': datetime.now().isoformat(),
            'segments': segments,
        }
        with open(map_path, 'w') as f:
            json.dump(segment_map, f, indent=2)
        print(f"Segment map saved: {map_filename} ({len(segments)} segments)")

        # Create metadata for stitched audio (backward compat)
        stitched_metadata = {
            'audio_file': stitched_filename,
            'timestamp': timestamp,
            'is_stitched': True,
            'chunk_count': len(chunks),
            'text_file_id': chapter_id
        }
        metadata_filename = f"{base_name}_stitched_{timestamp}.json"
        metadata_path = os.path.join(audio_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(stitched_metadata, f, indent=2)

        # Update chapter's audio_output and persist project.json
        audio_url = f'/api/project/audio/{stitched_filename}'
        map_url = f'/api/project/audio/{map_filename}'

        if is_new_structure:
            chapter['audio_output'] = {
                'audio_url': audio_url,
                'audio_file': stitched_filename,
                'map_url': map_url,
                'map_file': map_filename,
                'total_duration_seconds': total_duration_seconds,
                'segment_count': len(segments),
                'stitched_at': datetime.now().isoformat(),
            }
            project_file = os.path.join(converter.current_project_path, 'project.json')
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)
            converter._invalidate_lookup_caches()

        response_data = {
            'success': True,
            'audio_url': audio_url,
            'audio_file': stitched_filename,
            'audio_path': stitched_path,
            'metadata': stitched_metadata,
            'segment_map': segment_map,
            'segments_included': len(audio_paths),
            'segments_skipped': len(skipped_chunks),
        }
        if skipped_chunks:
            response_data['warnings'] = [s['reason'] for s in skipped_chunks]
        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"Error stitching project audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/set-best-take', methods=['POST'])
def set_best_take():
    """Set or unset an audio file as the best take for a text file"""
    try:
        import json
        data = request.json
        txt_filename = data.get('txt_filename')
        audio_filename = data.get('audio_filename')
        chunk_id = data.get('chunk_id', None)

        if not txt_filename or not audio_filename:
            return jsonify({'error': 'txt_filename and audio_filename are required'}), 400

        base_name = os.path.splitext(txt_filename)[0]
        updated = False

        # Update all metadata files for this text file (and chunk if specified)
        for filename in os.listdir(converter.audio_dir):
            if filename.startswith(base_name) and filename.endswith('.json'):
                metadata_path = os.path.join(converter.audio_dir, filename)

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # If chunk_id is specified, only update files for that chunk
                if chunk_id is not None and metadata.get('chunk_id') != chunk_id:
                    continue

                # Set is_best_take based on whether this is the selected audio file
                if metadata['audio_file'] == audio_filename:
                    metadata['is_best_take'] = True
                    updated = True
                else:
                    metadata['is_best_take'] = False

                # Write updated metadata back
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

        if not updated:
            return jsonify({'error': 'Audio file not found'}), 404

        return jsonify({'success': True, 'message': 'Best take updated successfully'})

    except Exception as e:
        import traceback
        print(f"Error setting best take: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-take', methods=['POST'])
def delete_take():
    """Delete an audio file and its metadata"""
    try:
        import json
        from datetime import datetime

        data = request.json
        txt_filename = data.get('txt_filename')
        audio_filename = data.get('audio_filename')
        text_file_id = data.get('text_file_id')  # Optional: for project-based deletion
        chunk_id = data.get('chunk_id')  # Optional: for chunk-based deletion

        if not txt_filename or not audio_filename:
            return jsonify({'error': 'txt_filename and audio_filename are required'}), 400

        # Delete the audio file
        audio_path = os.path.join(converter.audio_dir, audio_filename)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Deleted audio file: {audio_path}")
        else:
            print(f"Audio file not found: {audio_path}")

        # Delete the metadata file
        metadata_filename = os.path.splitext(audio_filename)[0] + '.json'
        metadata_path = os.path.join(converter.audio_dir, metadata_filename)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"Deleted metadata file: {metadata_path}")
        else:
            print(f"Metadata file not found: {metadata_path}")

        # Update project metadata if project is loaded and file/chunk info provided
        if converter.current_project_path and converter.current_project_metadata:
            try:
                # If text_file_id and chunk_id are provided, remove from that specific chunk
                if text_file_id is not None and chunk_id is not None:
                    text_files = converter.current_project_metadata.get('text_files', [])
                    text_file = next((tf for tf in text_files if tf['id'] == text_file_id), None)

                    if text_file:
                        chunks = text_file.get('chunks', [])
                        chunk = next((c for c in chunks if c['id'] == chunk_id), None)

                        if chunk:
                            # Remove the audio from generated_audios
                            chunk['generated_audios'] = [
                                audio for audio in chunk.get('generated_audios', [])
                                if audio.get('audio_file') != audio_filename
                            ]
                            print(f"Removed take from project metadata: {audio_filename}")
                else:
                    # Search through all chunks to find and remove the audio
                    text_files = converter.current_project_metadata.get('text_files', [])
                    for text_file in text_files:
                        for chunk in text_file.get('chunks', []):
                            original_count = len(chunk.get('generated_audios', []))
                            chunk['generated_audios'] = [
                                audio for audio in chunk.get('generated_audios', [])
                                if audio.get('audio_file') != audio_filename
                            ]
                            if len(chunk['generated_audios']) < original_count:
                                print(f"Removed take from project metadata: {audio_filename}")

                # Update last modified
                converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

                # Save project metadata
                project_file = os.path.join(converter.current_project_path, 'project.json')
                with open(project_file, 'w', encoding='utf-8') as f:
                    json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

            except Exception as meta_error:
                print(f"Warning: Could not update project metadata: {str(meta_error)}")
                # Continue anyway since the files were deleted

        return jsonify({'success': True, 'message': 'Take deleted successfully'})

    except Exception as e:
        import traceback
        print(f"Error deleting take: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



@app.route('/api/generation-stats', methods=['GET'])
def get_generation_stats():
    """Get generation statistics and averages"""
    try:
        stats = converter.generation_stats[-50:]  # Last 50 generations

        if not stats:
            return jsonify({
                'stats': [],
                'averages': None
            })

        # Calculate averages
        total_chars = sum(s['char_count'] for s in stats)
        total_time = sum(s['generation_time_ms'] for s in stats)
        total_audio = sum(s['audio_duration_sec'] for s in stats)

        averages = {
            'avg_chars': round(total_chars / len(stats)),
            'avg_time_ms': round(total_time / len(stats)),
            'avg_audio_duration': round(total_audio / len(stats), 2),
            'avg_ms_per_char': round(total_time / total_chars, 2) if total_chars > 0 else 0,
            'sample_count': len(stats)
        }

        return jsonify({
            'stats': stats,
            'averages': averages
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generation-progress', methods=['GET'])
def get_generation_progress():
    """Get current generation progress"""
    try:
        with converter.generation_lock:
            if converter.current_generation is None:
                return jsonify({
                    'in_progress': False,
                    'gpu_stats': converter.get_gpu_usage()
                })

            current = converter.current_generation.copy()

        elapsed_time = time.time() - current['start_time']
        estimated = current.get('estimated_time')

        progress_data = {
            'in_progress': True,
            'char_count': current['char_count'],
            'elapsed_ms': int(elapsed_time * 1000),
            'elapsed_seconds': round(elapsed_time, 1),
            'gpu_stats': converter.get_gpu_usage()
        }

        if estimated:
            progress_percent = min(100, (elapsed_time * 1000 / estimated['estimated_ms']) * 100)
            remaining_ms = max(0, estimated['estimated_ms'] - (elapsed_time * 1000))

            progress_data.update({
                'estimated_total_ms': estimated['estimated_ms'],
                'estimated_total_seconds': estimated['estimated_seconds'],
                'remaining_ms': int(remaining_ms),
                'remaining_seconds': round(remaining_ms / 1000, 1),
                'progress_percent': round(progress_percent, 1),
                'based_on_samples': estimated['based_on_samples']
            })

        return jsonify(progress_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500








def _get_newline_action(n_newlines, rules):
    """
    Look up the action for a run of n_newlines consecutive newlines.
    Actions: 'chapter' | 'paragraph' | 'trailing' | 'ignore'
    """
    key = str(n_newlines)
    if key in rules:
        return rules[key]
    # Try descending specific keys first, then fall back to '4+'
    for k in sorted((k for k in rules if k.isdigit()), key=int, reverse=True):
        if n_newlines >= int(k):
            return rules[k]
    return rules.get('4+', 'chapter')






def _carry_over_takes(old_chapters, new_chapters):
    """Move already-generated takes from a prior project's chapters onto freshly-built ones.

    Re-importing rebuilds chapters deterministically from book.json (chapter ids are new
    UUIDs, but chunk ids are positional and the text is stable for a given variant). We
    match on (chapter order, chunk id) and only carry takes when the chunk text is byte-for-
    byte identical, so a variant change (different spoken text) correctly drops stale takes.

    Returns the number of chunks whose takes were preserved.
    """
    old_by_pos = {}
    for ci, ch in enumerate(old_chapters or []):
        for chunk in ch.get('chunks', []):
            old_by_pos[(ci, chunk.get('id'))] = chunk
    carried = 0
    for ci, ch in enumerate(new_chapters):
        for chunk in ch.get('chunks', []):
            old = old_by_pos.get((ci, chunk.get('id')))
            if not old:
                continue
            takes = old.get('generated_audios') or []
            if takes and (old.get('text') or '') == (chunk.get('text') or ''):
                chunk['generated_audios'] = takes
                if 'dirty' in old:
                    chunk['dirty'] = old['dirty']
                carried += 1
    return carried


def build_chapters_from_rewriter_blocks(blocks, variant='original'):
    """Convert Rewriter book.json `blocks` into Henty chapter/chunk structures.

    `variant`:
      - 'original': speak the source `text` of each block
      - 'rewrite' : speak the `rewrite` field (falls back to `text` when empty)

    Heading blocks start a new chapter (their text becomes the chapter title and the
    first spoken chunk). Content appearing before the first heading is gathered into a
    leading 'Front Matter' chapter. `para`/`verse` blocks become text chunks; oversized
    paragraphs are split with the converter's smart_chunk_text. Image/caption metadata
    from the block is preserved on the chunk for later reader use.
    """
    import uuid as _uuid
    from datetime import datetime

    def block_text(b):
        if variant == 'rewrite':
            rw = b.get('rewrite')
            if rw is not None and str(rw).strip():
                return str(rw).strip()
        if b.get('type') == 'verse':
            return '\n'.join(b.get('lines', [])).strip()
        return str(b.get('text') or '').strip()

    chapters = []
    current = None

    def start_chapter(title):
        nonlocal current
        title = (title or 'Untitled').strip() or 'Untitled'
        current = {
            'id': str(_uuid.uuid4()),
            'title': title,
            'name': title,
            'order': len(chapters),
            'non_voiced': False,
            'source': 'rewriter_json',
            'added_at': datetime.now().isoformat(),
            'chunks': [{
                'id': 0,
                'type': 'text',
                'text': title,
                'original_text': title,
                'nickname': title[:50].strip() + ('...' if len(title) > 50 else ''),
                'dirty': False,
                'generated_audios': []
            }],
        }
        chapters.append(current)

    def add_content(text, block=None):
        nonlocal current
        if current is None:
            start_chapter('Front Matter')
        if not text:
            return
        # Split oversized paragraphs; most blocks become a single chunk.
        pieces = converter.smart_chunk_text(text)
        for piece in pieces:
            ptext = piece['text']
            chunk = {
                'id': len(current['chunks']),
                'type': 'text',
                'text': ptext,
                'original_text': text,   # the full un-edited source paragraph
                'nickname': ptext[:50].strip() + ('...' if len(ptext) > 50 else ''),
                'dirty': False,
                'generated_audios': []
            }
            if block is not None:
                if block.get('image_prompt'):
                    chunk['image_prompt'] = block['image_prompt']
                if block.get('image'):
                    chunk['image'] = block['image']
                if block.get('caption'):
                    chunk['caption'] = block['caption']
                # Enrichment authored upstream in Cowork: footnote/sidenote/gloss notes
                # and an optional enriched-markdown rendering. Carried through verbatim;
                # Henty only renders them (see reader_tab.js). Absent => renders as today.
                if block.get('notes'):
                    chunk['notes'] = block['notes']
                enriched = block.get('enriched') or block.get('enriched_text')
                if enriched and str(enriched).strip():
                    chunk['enriched_text'] = str(enriched).strip()
                chunk['source_block_id'] = block.get('id')
            current['chunks'].append(chunk)

    for block in blocks:
        btype = block.get('type')
        if btype == 'heading':
            start_chapter(block_text(block))
        else:
            add_content(block_text(block), block)

    # Drop chapters that ended up with only a title and no body (e.g. stray headings)
    return [ch for ch in chapters if len(ch['chunks']) > 0]


@app.route('/api/books', methods=['GET'])
@auth_manager.require_api_key
def list_books():
    """List book folders (each containing a book.json) under the configured BOOKS_DIR."""
    try:
        import json as _json
        books_dir = config.BOOKS_DIR
        books = []
        if books_dir and os.path.isdir(books_dir):
            for item in os.listdir(books_dir):
                folder = os.path.join(books_dir, item)
                book_json = os.path.join(folder, 'book.json')
                if not os.path.isfile(book_json):
                    continue
                title = item
                block_count = 0
                try:
                    with open(book_json, 'r', encoding='utf-8') as f:
                        data = _json.load(f)
                    title = data.get('title') or item
                    block_count = len(data.get('blocks', []))
                except Exception as e:
                    print(f"[IMPORT] Error reading {book_json}: {e}")
                project_json = os.path.join(folder, 'project.json')
                # "Most recent" = the latest of the project.json / book.json mtimes, so
                # recently-opened or freshly-produced books float to the top.
                mtime = os.path.getmtime(book_json)
                if os.path.isfile(project_json):
                    mtime = max(mtime, os.path.getmtime(project_json))
                books.append({
                    'folder': item,
                    'title': title,
                    'path': os.path.abspath(folder),
                    'block_count': block_count,
                    'has_project': os.path.isfile(project_json),
                    'mtime': mtime,
                })
        books.sort(key=lambda b: b.get('mtime', 0), reverse=True)
        return jsonify({
            'books_dir': os.path.abspath(books_dir) if books_dir else None,
            'books': books
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Seed for the priority queue — the books to process first, in order. Persisted to a
# JSON file in BOOKS_DIR so the curated order survives restarts.
DEFAULT_PRIORITY_QUEUE = [
    'treasure_island',
    'the_count_of_monte_cristo',
    'king_solomon_s_mines',
    'a_knight_of_the_white_cross_a_tale_of_the_siege_of_rhodes',
    'the_three_musketeers',
    'dracula',
    'the_sign_of_the_four',
]


def _queue_file_path():
    return os.path.join(config.BOOKS_DIR, '.henty_priority_queue.json') if config.BOOKS_DIR else None


@app.route('/api/queue', methods=['GET'])
@auth_manager.require_api_key
def get_priority_queue():
    """Return the ordered priority queue (list of book folder names).

    On first run (no file yet) the queue is seeded from DEFAULT_PRIORITY_QUEUE and written
    to disk so it becomes editable.
    """
    try:
        import json
        path = _queue_file_path()
        queue = list(DEFAULT_PRIORITY_QUEUE)
        if path and os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                queue = json.load(f).get('queue', queue)
        elif path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump({'queue': queue}, f, indent=2)
            except Exception as e:
                print(f"[QUEUE] Could not seed queue file: {e}")
        return jsonify({'queue': queue})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/queue', methods=['PUT'])
@auth_manager.require_api_key
def set_priority_queue():
    """Persist the ordered priority queue. Body: { queue: [folder, ...] }."""
    try:
        import json
        path = _queue_file_path()
        if not path:
            return jsonify({'error': 'BOOKS_DIR is not configured'}), 400
        data = request.json or {}
        queue = data.get('queue', [])
        if not isinstance(queue, list):
            return jsonify({'error': 'queue must be a list'}), 400
        queue = [str(f) for f in queue]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'queue': queue}, f, indent=2)
        return jsonify({'queue': queue})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/import-book', methods=['POST'])
@auth_manager.require_api_key
def import_book():
    """Create/refresh a Henty project from a book.json.

    The book's own folder becomes the project directory: project.json and the audio/
    folder are saved alongside book.json. Chapters are built directly from the blocks
    and locked (no parsing needed).

    Body: { folder | book_path, variant: 'original'|'rewrite' }
    """
    try:
        import json
        from datetime import datetime

        data = request.json or {}
        variant = data.get('variant', 'original')
        if variant not in ('original', 'rewrite'):
            variant = 'original'

        book_path = data.get('book_path')
        if not book_path:
            folder = data.get('folder') or data.get('book')
            if not folder:
                return jsonify({'error': 'book_path or folder is required'}), 400
            if not config.BOOKS_DIR:
                return jsonify({'error': 'BOOKS_DIR is not configured'}), 400
            book_path = os.path.join(config.BOOKS_DIR, folder)

        book_path = os.path.abspath(book_path)
        book_json = os.path.join(book_path, 'book.json')
        if not os.path.isfile(book_json):
            return jsonify({'error': f'book.json not found in {book_path}'}), 404

        with open(book_json, 'r', encoding='utf-8') as f:
            book = json.load(f)

        blocks = book.get('blocks', [])
        if not blocks:
            return jsonify({'error': 'book.json contains no blocks'}), 400

        title = book.get('title') or os.path.basename(book_path)
        chapters = build_chapters_from_rewriter_blocks(blocks, variant)
        if not chapters:
            return jsonify({'error': 'No chapters could be built from book.json'}), 400

        project_path = book_path
        keep_audio = bool(data.get('preserve_audio', True))
        audio_dir = os.path.join(project_path, 'audio')
        texts_dir = os.path.join(project_path, 'texts')
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(texts_dir, exist_ok=True)

        # Preserve an existing project.json (audio settings, prior takes) where present.
        project_file = os.path.join(project_path, 'project.json')
        if os.path.isfile(project_file):
            with open(project_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'name': title,
                'created_at': datetime.now().isoformat(),
                'default_audio_settings': {
                    'exaggeration': 0.6, 'cfg_weight': 0.4, 'voice_sample': 'none',
                    'seed': 0, 'temperature': 0.8, 'ref_vad_trimming': False
                },
            }

        metadata['name'] = metadata.get('name') or title
        metadata['last_modified'] = datetime.now().isoformat()
        metadata['version'] = '3.0'
        metadata['source'] = 'book_json'
        metadata['book_path'] = project_path
        metadata['text_variant'] = variant
        metadata['original_filename'] = f"{title}.json"
        # Re-import rebuilds chapters from book.json, which would otherwise discard any
        # takes already generated (e.g. by the Run Queue before the book was ever opened
        # in the UI). Carry takes across when the rebuilt chunk text is identical.
        carried = 0
        if keep_audio and metadata.get('chapters'):
            carried = _carry_over_takes(metadata['chapters'], chapters)
        metadata['chapters'] = chapters
        metadata['chapters_locked'] = True

        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Locked-text snapshot for parity with the Gutenberg lock workflow.
        try:
            parts = []
            for ch in chapters:
                parts.append(f"## {ch['title']}")
                for c in ch['chunks'][1:]:
                    if c.get('type') == 'text':
                        parts.append(c['text'])
            with open(os.path.join(project_path, 'chapters_original.txt'), 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(parts))
        except Exception as e:
            print(f"[IMPORT] Could not write chapters_original.txt: {e}")

        # Activate as the current project.
        converter.current_project_path = project_path
        converter.current_project_metadata = metadata
        converter.undo_stack = []
        converter.audio_dir = audio_dir

        total_chunks = sum(
            len([c for c in ch['chunks'] if c.get('type') == 'text']) for ch in chapters
        )
        print(f"[IMPORT] Imported '{title}' ({variant}): "
              f"{len(chapters)} chapters, {total_chunks} chunks → {project_path}"
              + (f" (preserved takes on {carried} chunks)" if carried else ""))

        return jsonify({
            'success': True,
            'project_path': project_path,
            'title': title,
            'variant': variant,
            'chapter_count': len(chapters),
            'chunk_count': total_chunks,
            'preserved_takes': carried,
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def _safe_filename_part(s, maxlen=30):
    """Sanitize a string for safe use inside a filename on Windows/macOS/Linux.

    Critically removes ':' and other characters that, on Windows, would otherwise
    be interpreted as an NTFS alternate-data-stream separator and silently write
    a zero-byte file (the cause of blank Chapter_I / Chapter_II files).
    """
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s or '')
    s = s.replace(' ', '_')
    return s[:maxlen].strip('_') or 'chunk'


def resolve_voice_sample_path(name):
    """Resolve a voice sample name (with or without extension) to a real file path.

    Returns an existing absolute path, or None. Used to enforce the hard rule that
    the Chatterbox built-in default voice must never be used: callers treat a None
    return as a fatal condition rather than generating without a voice prompt.
    """
    if not name or name == 'none':
        return None
    vs_dir = converter.voice_samples_dir
    direct = os.path.join(vs_dir, name)
    if os.path.isfile(direct):
        return os.path.abspath(direct)
    for ext in ('.wav', '.mp3', '.ogg', '.flac', '.m4a'):
        cand = os.path.join(vs_dir, name + ext)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    # Case-insensitive stem match against the directory contents.
    stem = os.path.splitext(name)[0].lower()
    try:
        for f in os.listdir(vs_dir):
            if os.path.splitext(f)[0].lower() == stem:
                return os.path.abspath(os.path.join(vs_dir, f))
    except FileNotFoundError:
        pass
    return None


@app.route('/api/project/generate-entire-book', methods=['POST'])
@auth_manager.require_api_key
def generate_entire_book():
    """Generate audio for every text chunk across all chapters using default settings.

    Skips chunks that already have a take (unless they are dirty, or force=true).
    The first take generated for a chunk becomes its best take.
    """
    try:
        import json
        import time
        from datetime import datetime

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json or {}
        force = bool(data.get('force', False))
        defaults = converter.current_project_metadata.get('default_audio_settings', {})
        language_id = data.get('language_id') or defaults.get('language_id', 'en')
        exaggeration = data.get('exaggeration') or defaults.get('exaggeration', 0.6)
        cfg_weight = data.get('cfg_weight') or defaults.get('cfg_weight', 0.4)
        # --- Resolve the voice sample. NEVER fall back to the Chatterbox default voice. ---
        requested_voice = data.get('voice_sample') or defaults.get('voice_sample')
        audio_prompt_path = resolve_voice_sample_path(requested_voice)
        resolved_from = requested_voice
        if audio_prompt_path is None:
            # Try the server-configured default voice (still a real sample file on disk).
            cfg_default = getattr(config, 'DEFAULT_VOICE', None)
            audio_prompt_path = resolve_voice_sample_path(cfg_default)
            resolved_from = cfg_default
        if audio_prompt_path is None:
            return jsonify({
                'error': 'No usable voice sample found. Refusing to generate with the '
                         'Chatterbox default voice. Set a project voice or configure '
                         'DEFAULT_VOICE to a file in the voice_samples directory.',
                'requested_voice': requested_voice,
                'config_default_voice': getattr(config, 'DEFAULT_VOICE', None),
            }), 400
        voice_sample = os.path.basename(audio_prompt_path)
        print(f"[GEN-BOOK] Voice: {voice_sample} (requested '{resolved_from}') → {audio_prompt_path}",
              flush=True)

        audio_dir = os.path.join(converter.current_project_path, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

        chapters = converter.current_project_metadata.get('chapters', [])
        generated = 0
        skipped = 0
        errors = 0
        error_detail = []

        print(f"\n=== Generating entire book ({len(chapters)} chapters) ===")

        project_file = os.path.join(converter.current_project_path, 'project.json')

        def _persist():
            converter.current_project_metadata['last_modified'] = datetime.now().isoformat()
            tmp = project_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)
            os.replace(tmp, project_file)   # atomic: never leave a half-written project.json

        for chapter in chapters:
            if chapter.get('non_voiced'):
                continue
            for chunk in chapter.get('chunks', []):
                if chunk.get('type', 'text') != 'text':
                    continue

                has_audio = len(chunk.get('generated_audios', [])) > 0
                if has_audio and not force and not chunk.get('dirty'):
                    skipped += 1
                    continue

                try:
                    clean_text = converter.process_pronunciation_markup(chunk['text'])
                    clean_text = converter.strip_xml_tags(clean_text)
                    if not clean_text.strip():
                        skipped += 1
                        continue

                    timestamp = int(time.time() * 1000)
                    safe = _safe_filename_part(chapter['title'])
                    audio_filename = f"{safe}_chunk{chunk['id']}_{timestamp}.wav"
                    audio_path = os.path.join(audio_dir, audio_filename)

                    preview = clean_text[:40].replace('\n', ' ')
                    print(f"[GEN] {chapter['title'][:25]} c{chunk['id']}: "
                          f"\"{preview}…\" ({len(clean_text)}c)", flush=True)

                    converter.generate_audio(
                        clean_text,
                        audio_path,
                        audio_prompt_path=audio_prompt_path,
                        language_id=language_id,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight
                    )

                    audio_metadata = {
                        'audio_file': audio_filename,
                        'audio_url': f"/api/audio/{audio_filename}",
                        'timestamp': timestamp,
                        'language_id': language_id,
                        'exaggeration': exaggeration,
                        'cfg_weight': cfg_weight,
                        'voice_sample': voice_sample,
                        'text_preview': chunk['text'][:200],
                        'input_text': chunk['text'],
                        'is_best_take': True
                    }
                    if 'generated_audios' not in chunk:
                        chunk['generated_audios'] = []
                    # New take becomes best; demote any earlier takes.
                    for t in chunk['generated_audios']:
                        t['is_best_take'] = False
                    chunk['generated_audios'].append(audio_metadata)
                    chunk['dirty'] = False
                    generated += 1
                    print(f"      ✓ done c{chunk['id']}", flush=True)

                    # Persist incrementally so an interruption (or overnight restart) never
                    # strands generated WAVs that project.json doesn't reference.
                    if generated % 10 == 0:
                        _persist()

                except Exception as ce:
                    errors += 1
                    error_detail.append({
                        'chapter': chapter.get('title'),
                        'chunk_id': chunk.get('id'),
                        'error': str(ce)
                    })
                    print(f"✗ Error chunk {chunk.get('id')}: {ce}")

        _persist()

        print(f"=== Book generation done: {generated} generated, "
              f"{skipped} skipped, {errors} errors ===")

        return jsonify({
            'success': True,
            'generated': generated,
            'skipped': skipped,
            'errors': errors,
            'error_detail': error_detail[:20],
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


















@app.route('/api/project/delete-non-best-takes', methods=['POST'])
@auth_manager.require_api_key
def delete_non_best_takes():
    """Delete every take that is not the best take, freeing disk space after review.

    Scope: whole book by default, or one chapter when `chapter_id` is given. A chunk with
    takes but none flagged best keeps its most recent take (so nothing useful is lost).
    Deletes the WAV files and prunes generated_audios. Returns counts. Undoable.
    """
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400
        data = request.get_json() or {}
        chapter_id = data.get('chapter_id')

        audio_dir = os.path.join(converter.current_project_path, 'audio')
        chapters = converter.current_project_metadata.get('chapters', [])
        if chapter_id is not None:
            chapters = [ch for ch in chapters if str(ch.get('id')) == str(chapter_id)]
            if not chapters:
                return jsonify({'error': 'Chapter not found'}), 404

        push_undo()
        removed_files = removed_takes = files_missing = 0
        for ch in chapters:
            for chunk in ch.get('chunks', []):
                takes = chunk.get('generated_audios') or []
                if len(takes) <= 1:
                    continue
                # Keep the best take; if none is flagged, keep the most recent.
                keep = next((t for t in takes if t.get('is_best_take')), None)
                if keep is None:
                    keep = max(takes, key=lambda t: t.get('timestamp', 0))
                    keep['is_best_take'] = True
                doomed = [t for t in takes if t is not keep]
                for t in doomed:
                    fn = t.get('audio_file')
                    if fn:
                        path = os.path.join(audio_dir, fn)
                        try:
                            if os.path.exists(path):
                                os.remove(path); removed_files += 1
                            else:
                                files_missing += 1
                        except Exception as fe:
                            print(f"[CLEAN] could not delete {path}: {fe}")
                    removed_takes += 1
                chunk['generated_audios'] = [keep]

        _save_current_project()
        print(f"[CLEAN] Removed {removed_takes} non-best takes "
              f"({removed_files} files deleted, {files_missing} already gone)", flush=True)
        return jsonify({'success': True, 'removed_takes': removed_takes,
                        'removed_files': removed_files, 'files_missing': files_missing})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/recover-takes', methods=['POST'])
@auth_manager.require_api_key
def recover_takes():
    """Rebuild generated_audios references from WAVs on disk for the current project.

    Audio filenames are `{safe_chapter_title}_chunk{chunk_id}_{timestamp}.wav`, so a project
    whose project.json lost its take references (e.g. an interrupted generate-entire-book)
    can be reconstructed from the files alone. Non-destructive: only fills chunks that
    currently have no takes (unless force=true). Newest take per chunk becomes best.
    """
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400
        force = bool((request.get_json() or {}).get('force', False))

        audio_dir = os.path.join(converter.current_project_path, 'audio')
        if not os.path.isdir(audio_dir):
            return jsonify({'error': 'No audio directory'}), 400

        fre = re.compile(r'^(.*)_chunk(\d+)_(\d+)\.wav$')
        pool = {}
        for f in os.listdir(audio_dir):
            if not f.lower().endswith('.wav'):
                continue
            mt = fre.match(f)
            if not mt:
                continue
            pool.setdefault((mt.group(1), int(mt.group(2))), []).append((int(mt.group(3)), f))

        meta = converter.current_project_metadata
        defs = meta.get('default_audio_settings', {})
        chunks_recovered = takes_assigned = 0
        for ch in meta.get('chapters', []):
            pfx = _safe_filename_part(ch.get('title') or ch.get('name') or 'chunk')
            for c in ch.get('chunks', []):
                if c.get('type', 'text') != 'text':
                    continue
                if c.get('generated_audios') and not force:
                    continue
                takes = pool.get((pfx, c.get('id')))
                if not takes:
                    continue
                takes.sort()
                gas = []
                for i, (ts, fn) in enumerate(takes):
                    gas.append({
                        'audio_file': fn,
                        'audio_url': f"/api/audio/{fn}",
                        'timestamp': ts,
                        'voice_sample': defs.get('voice_sample'),
                        'exaggeration': defs.get('exaggeration'),
                        'cfg_weight': defs.get('cfg_weight'),
                        'language_id': defs.get('language_id', 'en'),
                        'input_text': c.get('text'),
                        'text_preview': (c.get('text') or '')[:200],
                        'is_best_take': i == len(takes) - 1,
                        'recovered': True,
                    })
                c['generated_audios'] = gas
                c['dirty'] = False
                chunks_recovered += 1
                takes_assigned += len(gas)

        if chunks_recovered:
            _save_current_project()
        print(f"[RECOVER] {chunks_recovered} chunks, {takes_assigned} takes from {audio_dir}", flush=True)
        return jsonify({'success': True, 'chunks_recovered': chunks_recovered,
                        'takes_assigned': takes_assigned})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/chapter/publish', methods=['POST'])
@auth_manager.require_api_key
def publish_chapter():
    """Compile a chapter's best takes (and pauses) into one WAV and publish it.

    Walks the chapter's chunks in order: each text chunk contributes its best take
    (falling back to its most recent take), and each pause chunk contributes silence.
    The result is stitched into `<project>/published/<chapter>.wav` and exposed at
    /api/published/<file>. A future external preview server can pick the file up from
    there; for now it is served locally.

    Body: { chapter_id }
    """
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400
        data = request.get_json() or {}
        chapter_id = data.get('chapter_id')
        if chapter_id is None:
            return jsonify({'error': 'chapter_id is required'}), 400

        chapter = _find_chapter(chapter_id)
        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404

        audio_dir = os.path.join(converter.current_project_path, 'audio')
        items = []          # paths / pause tuples for stitch_audio_files
        missing = []        # text chunks with no usable take
        timeline = []       # per-chunk [start_ms, end_ms) into the stitched WAV
        offset_ms = 0       # running position as we walk the chunks in order
        for chunk in chapter.get('chunks', []):
            ctype = chunk.get('type', 'text')
            if ctype == 'pause':
                ms = chunk.get('duration_ms')
                if ms is None:
                    ms = int(round(chunk.get('duration', 0.5) * 1000))
                ms = max(0, int(ms))
                items.append(('pause', ms))
                timeline.append({'chunk_id': chunk.get('id'), 'type': 'pause',
                                 'start_ms': offset_ms, 'end_ms': offset_ms + ms,
                                 'duration_ms': ms})
                offset_ms += ms
                continue
            if ctype != 'text':
                continue
            takes = chunk.get('generated_audios') or []
            if not takes:
                missing.append(chunk.get('id'))
                continue
            best = next((t for t in takes if t.get('is_best_take')), takes[-1])
            items.append(os.path.join(audio_dir, best.get('audio_file', '')))
            dur_ms = int(round((best.get('audio_duration_seconds') or 0) * 1000))
            timeline.append({'chunk_id': chunk.get('id'), 'type': 'text',
                             'start_ms': offset_ms, 'end_ms': offset_ms + dur_ms,
                             'duration_ms': dur_ms})
            offset_ms += dur_ms

        if not any(not isinstance(it, tuple) for it in items):
            return jsonify({'error': 'No takes to publish in this chapter. Generate audio first.',
                            'missing_chunk_ids': missing}), 400

        published_dir = os.path.join(converter.current_project_path, 'published')
        os.makedirs(published_dir, exist_ok=True)
        out_name = f"{_safe_filename_part(chapter.get('title') or 'chapter', maxlen=60)}.wav"
        out_path = os.path.join(published_dir, out_name)
        converter.stitch_audio_files(items, out_path)

        # Bake the chunk-level timeline into the project so a passage can be mapped to a
        # position in the published audio (drives reader scroll-sync). Offsets are derived
        # from the stitch order above, so they match the published WAV exactly. The shape
        # is forward-compatible: a future forced-alignment pass can add `words` per entry.
        from datetime import datetime as _dt
        chapter['timeline'] = {
            'published_file': out_name,
            'total_ms': offset_ms,
            'generated_at': _dt.now().isoformat(),
            'chunks': timeline,
        }
        _save_current_project()

        size = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
        print(f"[PUBLISH] Chapter '{chapter.get('title')}' → {out_path} "
              f"({size} bytes, {len(missing)} chunks missing takes)", flush=True)
        return jsonify({
            'success': True,
            'file': out_name,
            'url': f"/api/published/{out_name}",
            'bytes': size,
            'missing_chunk_ids': missing,
            'timeline': chapter['timeline'],
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def _find_chapter(chapter_id):
    """Return the chapter dict with the given id, or None."""
    for ch in converter.current_project_metadata.get('chapters', []):
        if str(ch.get('id')) == str(chapter_id):
            return ch
    return None


def _save_current_project():
    from datetime import datetime
    converter.current_project_metadata['last_modified'] = datetime.now().isoformat()
    project_file = os.path.join(converter.current_project_path, 'project.json')
    with open(project_file, 'w', encoding='utf-8') as f:
        json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)
    converter._invalidate_lookup_caches()


def push_undo():
    """Snapshot the current chapters before a mutating edit. Keeps up to 20 moves."""
    try:
        if converter.current_project_metadata is None:
            return
        snap = json.loads(json.dumps(converter.current_project_metadata.get('chapters', [])))
        converter.undo_stack.append(snap)
        if len(converter.undo_stack) > 20:
            del converter.undo_stack[0]
    except Exception as e:
        print(f"[UNDO] push failed: {e}")


@app.route('/api/project/undo', methods=['POST'])
@auth_manager.require_api_key
def undo_edit():
    """Restore the chapters snapshot from before the last edit (up to 20 moves)."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400
        if not converter.undo_stack:
            return jsonify({'success': False, 'empty': True})
        snap = converter.undo_stack.pop()
        converter.current_project_metadata['chapters'] = snap
        _save_current_project()
        return jsonify({'success': True, 'remaining': len(converter.undo_stack)})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/split-chunk', methods=['POST'])
@auth_manager.require_api_key
def split_chunk():
    """Split a text chunk into two, operating directly on project.json chapters.

    Body: { "chapter_id": "...", "chunk_id": 3, "char_offset": 145 }
    The first part keeps the original chunk id (its takes are marked dirty since the
    text changed); the second part becomes a new chunk inserted immediately after.
    """
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.get_json() or {}
        chapter_id = data.get('chapter_id')
        chunk_id = data.get('chunk_id')
        char_offset = int(data.get('char_offset', 0))
        if chapter_id is None or chunk_id is None:
            return jsonify({'error': 'chapter_id and chunk_id are required'}), 400
        push_undo()

        chapter = _find_chapter(chapter_id)
        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404
        chunks = chapter.get('chunks', [])
        idx = next((i for i, c in enumerate(chunks) if str(c.get('id')) == str(chunk_id)), None)
        if idx is None:
            return jsonify({'error': 'Chunk not found'}), 404
        chunk = chunks[idx]
        if chunk.get('type', 'text') != 'text':
            return jsonify({'error': 'Only text chunks can be split'}), 400

        text = chunk.get('text', '')
        char_offset = max(0, min(char_offset, len(text)))

        # Prefer a sentence boundary near the offset, else a word boundary.
        split_at = char_offset
        best_sentence = None
        for sm in re.finditer(r'[.!?]\s+', text, re.DOTALL):
            if sm.end() <= char_offset:
                best_sentence = sm.end()
        if best_sentence and best_sentence > max(0, char_offset - 60):
            split_at = best_sentence
        else:
            space_pos = text.rfind(' ', 0, char_offset)
            if space_pos > 0:
                split_at = space_pos + 1

        part_a = text[:split_at].strip()
        part_b = text[split_at:].strip()
        if not part_a or not part_b:
            return jsonify({'error': 'Cannot split: one part would be empty'}), 400

        had_audio = len(chunk.get('generated_audios', [])) > 0
        chunk['text'] = part_a
        chunk['nickname'] = part_a[:50].strip() + ('...' if len(part_a) > 50 else '')
        if had_audio:
            chunk['dirty'] = True

        new_id = max([c.get('id', 0) for c in chunks if isinstance(c.get('id'), int)] + [0]) + 1
        new_chunk = {
            'id': new_id,
            'type': 'text',
            'text': part_b,
            'original_text': chunk.get('original_text', ''),  # both halves keep the source paragraph
            'nickname': part_b[:50].strip() + ('...' if len(part_b) > 50 else ''),
            'dirty': False,
            'generated_audios': []
        }
        chunks.insert(idx + 1, new_chunk)

        _save_current_project()
        return jsonify({'success': True, 'chapter_id': chapter_id,
                        'chunk_id': chunk['id'], 'new_chunk_id': new_id})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/merge-chunk', methods=['POST'])
@auth_manager.require_api_key
def merge_chunk():
    """Merge a text chunk with its neighbour, operating on project.json chapters.

    Body: { "chapter_id": "...", "chunk_id": 3, "direction": "prev"|"next" }
    The surviving chunk keeps the earlier chunk's id; its takes are marked dirty
    because the text changed.
    """
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.get_json() or {}
        chapter_id = data.get('chapter_id')
        chunk_id = data.get('chunk_id')
        direction = data.get('direction', 'prev')
        if chapter_id is None or chunk_id is None:
            return jsonify({'error': 'chapter_id and chunk_id are required'}), 400
        push_undo()
        if direction not in ('prev', 'next'):
            return jsonify({'error': 'direction must be "prev" or "next"'}), 400

        all_chapters = converter.current_project_metadata.get('chapters', [])
        ci = next((i for i, ch in enumerate(all_chapters) if str(ch.get('id')) == str(chapter_id)), None)
        if ci is None:
            return jsonify({'error': 'Chapter not found'}), 404
        chapter = all_chapters[ci]
        chunks = chapter.get('chunks', [])
        idx = next((i for i, c in enumerate(chunks) if str(c.get('id')) == str(chunk_id)), None)
        if idx is None:
            return jsonify({'error': 'Chunk not found'}), 404

        at_boundary = (direction == 'prev' and idx == 0) or (direction == 'next' and idx == len(chunks) - 1)

        # ---- Cross-chapter merge: the chunk is at the chapter boundary ----
        if at_boundary:
            if not data.get('cross_chapter'):
                return jsonify({'error': f'No {direction} chunk to merge with'}), 400
            adj = ci - 1 if direction == 'prev' else ci + 1
            if adj < 0 or adj >= len(all_chapters):
                return jsonify({'error': 'No adjacent chapter to merge into'}), 400

            moving = chunks[idx]
            if moving.get('type', 'text') != 'text':
                return jsonify({'error': 'Only text chunks can be merged across chapters'}), 400
            target_chapter = all_chapters[adj]
            tchunks = target_chapter.get('chunks', [])
            # Target chunk: last text chunk (prev) or first text chunk (next).
            text_idxs = [i for i, c in enumerate(tchunks) if c.get('type', 'text') == 'text']

            if text_idxs:
                tgt = tchunks[text_idxs[-1] if direction == 'prev' else text_idxs[0]]
                if direction == 'prev':
                    merged = (tgt.get('text', '').strip() + '\n' + moving.get('text', '').strip()).strip()
                else:
                    merged = (moving.get('text', '').strip() + '\n' + tgt.get('text', '').strip()).strip()
                tgt['text'] = merged
                tgt['nickname'] = merged[:50].strip() + ('...' if len(merged) > 50 else '')
                if tgt.get('generated_audios') or moving.get('generated_audios'):
                    tgt['dirty'] = True
                survivor_id = tgt['id']
            else:
                # No text chunk in the target — move this chunk over instead of merging.
                new_id = max([c.get('id', 0) for c in tchunks if isinstance(c.get('id'), int)] + [-1]) + 1
                moving = {**moving, 'id': new_id, 'dirty': bool(moving.get('generated_audios'))}
                tchunks.insert(len(tchunks) if direction == 'prev' else 0, moving)
                survivor_id = new_id

            del chunks[idx]
            removed_chapter = False
            if not chunks:                       # chapter is now empty → remove it
                del all_chapters[ci]
                removed_chapter = True

            _save_current_project()
            return jsonify({'success': True, 'cross_chapter': True,
                            'target_chapter_id': target_chapter.get('id'),
                            'chunk_id': survivor_id, 'removed_chapter': removed_chapter})

        # ---- Normal in-chapter merge ----
        a_idx, b_idx = (idx - 1, idx) if direction == 'prev' else (idx, idx + 1)
        a, b = chunks[a_idx], chunks[b_idx]
        if a.get('type', 'text') != 'text' or b.get('type', 'text') != 'text':
            return jsonify({'error': 'Only adjacent text chunks can be merged'}), 400

        merged = (a.get('text', '').strip() + '\n' + b.get('text', '').strip()).strip()
        a['text'] = merged
        a['nickname'] = merged[:50].strip() + ('...' if len(merged) > 50 else '')
        if a.get('generated_audios') or b.get('generated_audios'):
            a['dirty'] = True
        del chunks[b_idx]

        _save_current_project()
        return jsonify({'success': True, 'chapter_id': chapter_id, 'chunk_id': a['id']})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500














@app.route('/api/project/chapter/generate-all', methods=['POST'])
@auth_manager.require_api_key
def generate_all_chapter_chunks():
    """Generate audio for all chunks in a specific chapter"""
    try:
        import json
        import time

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        chapter_id = data.get('chapter_id')

        if not chapter_id:
            return jsonify({'error': 'chapter_id is required'}), 400

        # Get audio generation parameters (use project defaults if not provided)
        language_id = data.get('language_id') or converter.current_project_metadata.get('default_audio_settings', {}).get('language_id', 'en')
        exaggeration = data.get('exaggeration') or converter.current_project_metadata.get('default_audio_settings', {}).get('exaggeration', 0.6)
        cfg_weight = data.get('cfg_weight') or converter.current_project_metadata.get('default_audio_settings', {}).get('cfg_weight', 0.4)
        voice_sample = data.get('voice_sample') or converter.current_project_metadata.get('default_audio_settings', {}).get('voice_sample', 'none')

        # Find the chapter
        chapters = converter.current_project_metadata.get('chapters', [])
        chapter = next((ch for ch in chapters if ch['id'] == chapter_id), None)

        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404

        chunks = chapter.get('chunks', [])
        if not chunks:
            return jsonify({'error': 'No chunks found in chapter'}), 400

        print(f"\n=== Generating audio for all chunks in chapter: {chapter.get('title', 'Unknown')} ===")
        print(f"Total chunks to generate: {len(chunks)}")
        print(f"Parameters - Language: {language_id}, Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}")

        # Resolve voice sample — never fall back to the Chatterbox default voice.
        audio_prompt_path = resolve_voice_sample_path(voice_sample)
        if audio_prompt_path is None:
            audio_prompt_path = resolve_voice_sample_path(getattr(config, 'DEFAULT_VOICE', None))
        if audio_prompt_path is None:
            return jsonify({
                'error': 'No usable voice sample found. Refusing to generate with the '
                         'Chatterbox default voice. Set a project voice or configure DEFAULT_VOICE.',
                'requested_voice': voice_sample,
            }), 400
        voice_sample = os.path.basename(audio_prompt_path)
        print(f"Voice: {voice_sample} → {audio_prompt_path}")

        audio_dir = os.path.join(converter.current_project_path, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

        generated_audio_info = []
        success_count = 0
        error_count = 0

        # Generate audio for each text chunk (skip pause and common_file chunks)
        for chunk in chunks:
            chunk_type = chunk.get('type', 'text')

            if chunk_type != 'text':
                print(f"Skipping chunk {chunk['id']} (type: {chunk_type})")
                continue

            try:
                # Process pronunciation markup and strip XML tags
                clean_text = converter.process_pronunciation_markup(chunk['text'])
                clean_text = converter.strip_xml_tags(clean_text)

                # Generate filename
                timestamp = int(time.time() * 1000)
                chapter_title_safe = _safe_filename_part(chapter['title'])
                audio_filename = f"{chapter_title_safe}_chunk{chunk['id']}_{timestamp}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)

                # Check device
                if converter.model is not None:
                    model_device = "GPU" if converter.device == "cuda" else "CPU"
                elif torch.cuda.is_available():
                    model_device = "GPU (will load)"
                else:
                    model_device = "CPU"

                print(f"Generating audio for chunk {chunk['id']} on {model_device}... ({len(clean_text)} chars)")

                # Generate audio
                result = converter.generate_audio(
                    clean_text,
                    audio_path,
                    audio_prompt_path=audio_prompt_path,
                    language_id=language_id,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )

                # Create metadata
                audio_metadata = {
                    'audio_file': audio_filename,
                    'audio_url': f"/api/audio/{audio_filename}",
                    'timestamp': timestamp,
                    'language_id': language_id,
                    'exaggeration': exaggeration,
                    'cfg_weight': cfg_weight,
                    'voice_sample': voice_sample,
                    'text_preview': chunk['text'][:200],
                    'input_text': chunk['text'],
                    'is_best_take': len(chunk.get('generated_audios', [])) == 0  # First generation is best take
                }

                # Add to chunk's generated_audios
                if 'generated_audios' not in chunk:
                    chunk['generated_audios'] = []
                chunk['generated_audios'].append(audio_metadata)

                generated_audio_info.append({
                    'chunk_id': chunk['id'],
                    'audio_file': audio_filename,
                    'success': True
                })

                success_count += 1
                print(f"✓ Successfully generated audio for chunk {chunk['id']}")

            except Exception as chunk_error:
                print(f"✗ Error generating audio for chunk {chunk['id']}: {str(chunk_error)}")
                generated_audio_info.append({
                    'chunk_id': chunk['id'],
                    'error': str(chunk_error),
                    'success': False
                })
                error_count += 1

        # Save updated project metadata
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        print(f"\n=== Chapter audio generation complete ===")
        print(f"Success: {success_count}, Errors: {error_count}")

        return jsonify({
            'success': True,
            'chapter_id': chapter_id,
            'chapter_title': chapter.get('title'),
            'total_chunks': len(chunks),
            'generated': success_count,
            'errors': error_count,
            'generated_audio': generated_audio_info
        })

    except Exception as e:
        import traceback
        print(f"Error generating chapter audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/recent', methods=['GET'])
@auth_manager.require_api_key
def get_recent_projects():
    """Get list of all project directories"""
    try:
        import os
        from datetime import datetime

        recent_projects = []
        seen_paths = set()

        def add_project(project_path, include_without_json):
            """Append a project entry for project_path if it qualifies."""
            abs_path = os.path.abspath(project_path)
            if abs_path in seen_paths or not os.path.isdir(project_path):
                return
            item = os.path.basename(project_path.rstrip('/\\'))
            project_file = os.path.join(project_path, 'project.json')
            if os.path.exists(project_file):
                seen_paths.add(abs_path)
                try:
                    with open(project_file, 'r', encoding='utf-8', errors='replace') as f:
                        project_data = json.load(f)
                    recent_projects.append({
                        'name': project_data.get('name', item),
                        'path': abs_path,
                        'last_modified': project_data.get('last_modified', ''),
                        'created_at': project_data.get('created_at', '')
                    })
                except Exception as e:
                    print(f"Error reading project {project_path}: {e}")
                    recent_projects.append({
                        'name': item, 'path': abs_path, 'last_modified': '', 'created_at': ''
                    })
            elif include_without_json:
                seen_paths.add(abs_path)
                recent_projects.append({
                    'name': item, 'path': abs_path, 'last_modified': '', 'created_at': ''
                })

        # Default projects folder: list every subdirectory.
        projects_dir = config.DEFAULT_PROJECT_DIR
        if os.path.exists(projects_dir):
            for item in os.listdir(projects_dir):
                add_project(os.path.join(projects_dir, item), include_without_json=True)

        # Rewriter books folder: only include folders already imported (have project.json).
        books_dir = getattr(config, 'BOOKS_DIR', None)
        if books_dir and os.path.isdir(books_dir):
            for item in os.listdir(books_dir):
                add_project(os.path.join(books_dir, item), include_without_json=False)

        # Sort by last_modified (most recent first), then by name
        recent_projects.sort(key=lambda x: (x.get('last_modified', ''), x.get('name', '')), reverse=True)

        return jsonify(recent_projects)

    except Exception as e:
        import traceback
        print(f"Error getting recent projects: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500























# Serve static HTML and JavaScript files
@app.route('/')
def serve_landing():
    """Serve the landing page"""
    return send_file('index.html')

@app.route('/index.html')
def serve_index():
    """Serve the landing page"""
    return send_file('index.html')

@app.route('/app.html')
def serve_app():
    """Serve the main application"""
    return send_file('app.html')

@app.route('/landing.html')
def serve_landing_alt():
    """Serve the landing page (alternative route)"""
    return send_file('landing.html')

@app.route('/reader.html')
def serve_reader():
    """Serve the reader page"""
    return send_file('reader.html')


# Serve JavaScript files
@app.route('/<path:filename>.js')
def serve_js(filename):
    """Serve JavaScript files"""
    js_file = f'{filename}.js'
    if os.path.exists(js_file):
        return send_file(js_file, mimetype='application/javascript')
    return jsonify({'error': 'File not found'}), 404

# Serve CSS files
@app.route('/<path:filename>.css')
def serve_css(filename):
    """Serve CSS files"""
    css_file = f'{filename}.css'
    if os.path.exists(css_file):
        return send_file(css_file, mimetype='text/css')
    return jsonify({'error': 'File not found'}), 404

# ── Mobile listener page ─────────────────────────────────────────────────────

@app.route('/listen')
def serve_listen_page():
    """Serve the mobile podcast listener page (no auth required)."""
    return send_file('listen.html')


# ── Podcast / flag API ────────────────────────────────────────────────────────

@app.route('/api/project/podcast-data', methods=['GET'])
@auth_manager.require_api_key
def get_podcast_data():
    """Return project name + chapters that have stitched audio (with audio_output)."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        meta = converter.current_project_metadata
        project_name = meta.get('name', 'Untitled Project')
        chapters = meta.get('chapters', [])

        ready = [
            {
                'id': ch['id'],
                'title': ch.get('title', f"Chapter {i}"),
                'order': ch.get('order', i),
                'audio_output': ch['audio_output'],
            }
            for i, ch in enumerate(chapters)
            if ch.get('audio_output')
        ]
        ready.sort(key=lambda c: c['order'])

        return jsonify({'project_name': project_name, 'chapters': ready})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/flag-take', methods=['POST'])
@auth_manager.require_api_key
def flag_take():
    """Record a flag at a playback position; look up which chunk/take it maps to."""
    try:
        import uuid as _uuid
        from datetime import datetime as _dt

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json or {}
        chapter_id = data.get('chapter_id')
        playback_seconds = data.get('playback_seconds')
        note = data.get('note', '')

        if chapter_id is None or playback_seconds is None:
            return jsonify({'error': 'chapter_id and playback_seconds are required'}), 400

        chapter_map, _, _ = converter.get_chapter_and_chunk_lookups()
        chapter = chapter_map.get(chapter_id)
        if not chapter:
            return jsonify({'error': 'Chapter not found'}), 404

        audio_output = chapter.get('audio_output')
        if not audio_output or not audio_output.get('map_file'):
            return jsonify({'error': 'Chapter has no segment map — stitch it first'}), 400

        map_path = os.path.join(converter.current_project_path, 'audio', audio_output['map_file'])
        with open(map_path, 'r', encoding='utf-8') as f:
            segment_map = json.load(f)

        matched = None
        for seg in segment_map.get('segments', []):
            if seg['start_seconds'] <= playback_seconds <= seg['end_seconds']:
                matched = seg
                break
        if not matched and segment_map.get('segments'):
            last = segment_map['segments'][-1]
            if playback_seconds <= last['end_seconds'] + 1.0:
                matched = last

        if not matched:
            return jsonify({'error': f'No segment found at {playback_seconds}s'}), 404

        flag = {
            'id': str(_uuid.uuid4()),
            'created_at': _dt.now().isoformat(),
            'chapter_id': chapter_id,
            'chapter_title': chapter.get('title', ''),
            'chunk_id': matched['chunk_id'],
            'chunk_nickname': matched.get('chunk_nickname', ''),
            'take_timestamp': matched.get('take_timestamp'),
            'playback_seconds': round(playback_seconds, 2),
            'note': note,
            'resolved': False,
        }

        meta = converter.current_project_metadata
        if 'flags' not in meta:
            meta['flags'] = []
        meta['flags'].append(flag)

        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"Flag saved: chapter={chapter_id} chunk={matched['chunk_id']} at {playback_seconds}s")
        return jsonify({'success': True, 'flag': flag})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/flags', methods=['GET'])
@auth_manager.require_api_key
def get_flags():
    """Return all flags for the current project."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        flags = converter.current_project_metadata.get('flags', [])
        unresolved = sum(1 for f in flags if not f.get('resolved'))
        return jsonify({'flags': flags, 'unresolved_count': unresolved})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/resolve-flag', methods=['POST'])
@auth_manager.require_api_key
def resolve_flag():
    """Mark a flag as resolved."""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        flag_id = (request.json or {}).get('flag_id')
        if not flag_id:
            return jsonify({'error': 'flag_id is required'}), 400

        meta = converter.current_project_metadata
        flags = meta.get('flags', [])
        for flag in flags:
            if flag.get('id') == flag_id:
                flag['resolved'] = True
                break
        else:
            return jsonify({'error': 'Flag not found'}), 404

        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Starting Text to Audio Converter API...")
    print(f"Using device: {converter.device}")
    print(f"Server address: http://{config.HOST}:{config.PORT}")
    if config.REQUIRE_AUTH:
        print(f"Authentication: ENABLED")
        print(f"API Key required for protected endpoints")
    else:
        print(f"Authentication: DISABLED (not recommended for remote access)")
    print(f"WebSocket support: {'ENABLED' if config.ENABLE_WEBSOCKET else 'DISABLED'}")
    print(f"\nAllowed CORS origins: {', '.join(config.ALLOWED_ORIGINS)}")
    # Debug: confirm build identity so we can verify the right code is running
    try:
        import subprocess as _sp
        _sha = _sp.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        _sha = 'unknown'
    _listen_registered = any(r.rule == '/listen' for r in app.url_map.iter_rules())
    print(f"\n[BUILD] git commit : {_sha}")
    print(f"[BUILD] /listen route registered: {_listen_registered}")
    print(f"\nReady for connections!")

    socketio.run(app, debug=config.DEBUG, port=config.PORT, host=config.HOST, allow_unsafe_werkzeug=True)
