from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
import torch
import numpy as np
from scipy.io import wavfile
import re
from pydub import AudioSegment
import json
import time
import threading
import uuid
from datetime import datetime
from scripts.gutenberg_processor import GutenbergProcessor
from config import config
from auth import AuthManager

app = Flask(__name__)

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

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            import sys
            print("WARNING: CUDA is not available!")
            print("\nThis Python has PyTorch WITHOUT CUDA support.")
            print("The CPU version of PyTorch is installed at:")
            print(f"  {torch.__file__}")
            print("\nTo fix, run these commands:")
            print(f"\n  {sys.executable} -m pip uninstall torch torchvision torchaudio -y")
            print(f"  {sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
            print("\nThen restart the server.")

        # Set device based on preference and availability
        self.cuda_available = torch.cuda.is_available()
        if config.DEVICE_PREFERENCE == 'cpu':
            self.device = 'cpu'
        elif config.DEVICE_PREFERENCE == 'cuda':
            self.device = 'cuda' if self.cuda_available else 'cpu'
        else:  # auto
            self.device = 'cuda' if self.cuda_available else 'cpu'

        print(f"\nDevice preference: {config.DEVICE_PREFERENCE}")
        print(f"Using device: {self.device}")
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

    def switch_device(self, new_device):
        """Switch between GPU and CPU, reloading the model if necessary"""
        if new_device not in ['cuda', 'cpu']:
            raise ValueError(f"Invalid device: {new_device}")

        # Check if CUDA is available when requesting it
        if new_device == 'cuda' and not self.cuda_available:
            raise RuntimeError("CUDA is not available on this system")

        # If device is already set and model is loaded, need to reload
        if self.device != new_device:
            old_device = self.device
            self.device = new_device

            # Clear the model to force reload on next use
            if self.model is not None:
                print(f"Switching device from {old_device} to {new_device}")
                print("Model will be reloaded on next generation...")
                self.model = None

            return True
        return False

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

    def smart_chunk_text(self, text, max_chunk_size=None):
        """
        Smart text chunking that respects paragraph breaks, quotations, and sentence boundaries.
        Returns a list of dicts with chunk metadata.
        """
        if max_chunk_size is None:
            max_chunk_size = config.MAX_CHUNK_SIZE
        if len(text) <= max_chunk_size:
            # Text is short enough, return as single chunk
            return [{
                'id': 0,
                'text': text,
                'nickname': text[:50].strip() + ('...' if len(text) > 50 else ''),
                'start_pos': 0,
                'end_pos': len(text)
            }]

        chunks = []
        chunk_id = 0
        current_pos = 0

        # Split by double newlines (paragraphs) first
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""
        chunk_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding this paragraph would exceed the limit
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    chunk_start = current_pos
            else:
                # Current paragraph is too large, need to finalize current chunk
                if current_chunk:
                    # Save current chunk
                    nickname = current_chunk[:50].strip() + '...'
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk,
                        'nickname': nickname,
                        'start_pos': chunk_start,
                        'end_pos': chunk_start + len(current_chunk)
                    })
                    chunk_id += 1

                # Handle large paragraph that needs to be split
                if len(para) > max_chunk_size:
                    # Common abbreviations that should not end a sentence
                    abbrev_pattern = r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|Rd|etc|vs|i\.e|e\.g|Vol|No|Fig|viz|al|Co|Corp|Inc|Ltd)\.'

                    # Temporarily replace abbreviations with placeholders
                    import uuid
                    placeholder_map = {}
                    temp_para = para
                    for match in re.finditer(abbrev_pattern, para, re.IGNORECASE):
                        placeholder = f"<<ABBREV{len(placeholder_map)}>>"
                        placeholder_map[placeholder] = match.group(0)
                        temp_para = temp_para.replace(match.group(0), placeholder, 1)

                    # Split by sentences (now without false positives from abbreviations)
                    sentences = re.split(r'([.!?]+\s+|[.!?]+$)', temp_para)
                    sentence_chunk = ""
                    sentence_start = current_pos

                    for i in range(0, len(sentences), 2):
                        sentence = sentences[i]
                        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
                        full_sentence = sentence + punctuation

                        # Restore abbreviations
                        for placeholder, orig in placeholder_map.items():
                            full_sentence = full_sentence.replace(placeholder, orig)

                        if len(sentence_chunk) + len(full_sentence) <= max_chunk_size:
                            if not sentence_chunk:
                                sentence_start = current_pos
                            sentence_chunk += full_sentence
                        else:
                            if sentence_chunk:
                                nickname = sentence_chunk[:50].strip() + '...'
                                chunks.append({
                                    'id': chunk_id,
                                    'text': sentence_chunk.strip(),
                                    'nickname': nickname,
                                    'start_pos': sentence_start,
                                    'end_pos': sentence_start + len(sentence_chunk)
                                })
                                chunk_id += 1

                            # Start new chunk
                            sentence_chunk = full_sentence
                            sentence_start = current_pos

                        current_pos += len(full_sentence)

                    # Add remaining sentences
                    if sentence_chunk:
                        nickname = sentence_chunk[:50].strip() + '...'
                        chunks.append({
                            'id': chunk_id,
                            'text': sentence_chunk.strip(),
                            'nickname': nickname,
                            'start_pos': sentence_start,
                            'end_pos': sentence_start + len(sentence_chunk)
                        })
                        chunk_id += 1

                    current_chunk = ""
                else:
                    # Paragraph fits in a new chunk
                    current_chunk = para
                    chunk_start = current_pos

            current_pos += len(para) + 2  # +2 for \n\n

        # Add final chunk if any
        if current_chunk:
            nickname = current_chunk[:50].strip() + ('...' if len(current_chunk) > 50 else '')
            chunks.append({
                'id': chunk_id,
                'text': current_chunk,
                'nickname': nickname,
                'start_pos': chunk_start,
                'end_pos': chunk_start + len(current_chunk)
            })

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

    def detect_chapters(self, text):
        """
        Detect chapters in text using various patterns.
        Returns list of chapter dicts with: id, title, text, start_pos, end_pos
        Also preserves pre-chapter text as non-voiced content.
        """
        import uuid

        # Chapter detection patterns (in order of priority)
        chapter_patterns = [
            # Standard chapter headings with numbers/roman numerals
            r'^(Chapter\s+(?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve)[^\n]*)',
            # All caps chapter headings
            r'^(CHAPTER\s+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)[^\n]*)',
            # Book/Part divisions
            r'^((?:Book|Part|Section)\s+(?:[IVXLCDM]+|\d+)[^\n]*)',
            # Simple "I.", "II.", etc at start of line
            r'^([IVXLCDM]+\.)\s*$',
            # Numbered sections "1.", "2.", etc at start of line (but not in middle of sentence)
            r'^\s*(\d+\.)\s*$'
        ]

        chapters = []
        chapter_positions = []

        # Find all chapter markers
        for pattern in chapter_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                # Find the end of the title line
                title_end = text.find('\n', match.start())
                if title_end == -1:
                    title_end = len(text)

                chapter_positions.append({
                    'pos': match.start(),
                    'title_end': title_end,
                    'title': match.group(1).strip(),
                    'pattern': pattern
                })

        # Also look for section breaks (4+ consecutive newlines) if no chapters found
        if not chapter_positions:
            section_breaks = list(re.finditer(r'\n{4,}', text))
            if section_breaks:
                # Add implicit chapters based on section breaks
                for i, match in enumerate(section_breaks):
                    chapter_positions.append({
                        'pos': match.end(),
                        'title_end': match.end(),
                        'title': f'Section {i+1}',
                        'pattern': 'section_break'
                    })

        # Sort by position
        chapter_positions.sort(key=lambda x: x['pos'])

        # Deduplicate chapters at the same position (multiple patterns matching same chapter)
        seen_positions = set()
        deduplicated_positions = []
        for chapter_info in chapter_positions:
            if chapter_info['pos'] not in seen_positions:
                deduplicated_positions.append(chapter_info)
                seen_positions.add(chapter_info['pos'])
        chapter_positions = deduplicated_positions

        # If we found chapter markers, create chapters
        if chapter_positions:
            # Preserve text before first chapter as non-voiced content
            first_chapter_pos = chapter_positions[0]['pos']
            if first_chapter_pos > 0:
                pre_chapter_text = text[0:first_chapter_pos].strip()
                if pre_chapter_text:
                    chapters.append({
                        'id': str(uuid.uuid4()),
                        'title': '[Non-voiced Preface]',
                        'text': pre_chapter_text,
                        'start_pos': 0,
                        'end_pos': first_chapter_pos,
                        'order': -1,
                        'non_voiced': True
                    })

            for i, chapter_info in enumerate(chapter_positions):
                # Start reading text AFTER the title line
                start_pos = chapter_info['title_end']
                # Skip any newlines immediately after title
                while start_pos < len(text) and text[start_pos] in '\n\r':
                    start_pos += 1

                # Find end position (start of next chapter or end of text)
                end_pos = chapter_positions[i+1]['pos'] if i+1 < len(chapter_positions) else len(text)

                chapter_text = text[start_pos:end_pos].strip()

                chapters.append({
                    'id': str(uuid.uuid4()),
                    'title': chapter_info['title'],
                    'text': chapter_text,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'order': i,
                    'non_voiced': False
                })
        else:
            # No chapters detected, treat entire text as single chapter
            chapters.append({
                'id': str(uuid.uuid4()),
                'title': 'Complete Text',
                'text': text,
                'start_pos': 0,
                'end_pos': len(text),
                'order': 0,
                'non_voiced': False
            })

        return chapters

    def strip_xml_tags(self, text):
        """Strip all XML-like tags from text for TTS processing"""
        # Remove all XML tags: <tag>content</tag> or <tag/>
        cleaned = re.sub(r'<[^>]+>', '', text)
        return cleaned

    def text_to_xml_content(self, text, chapters=None):
        """
        Convert text (with optional detected chapters) to XML-embedded-in-JSON format.
        Returns a string of XML content.
        Wraps non-voiced content in <non-voiced> tags.
        """
        if chapters is None:
            chapters = self.detect_chapters(text)

        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append('<book>')

        for chapter in chapters:
            chapter_title = chapter['title'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            is_non_voiced = chapter.get('non_voiced', False)

            if is_non_voiced:
                # Wrap non-voiced content
                xml_lines.append(f'  <non-voiced title="{chapter_title}">')
            else:
                xml_lines.append(f'  <chapter id="{chapter["id"]}" title="{chapter_title}" order="{chapter["order"]}">')

            # Split chapter text into paragraphs
            paragraphs = [p.strip() for p in chapter['text'].split('\n\n') if p.strip()]

            for para in paragraphs:
                # Escape XML special characters
                para_escaped = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                xml_lines.append(f'    <p>{para_escaped}</p>')

            if is_non_voiced:
                xml_lines.append('  </non-voiced>')
            else:
                xml_lines.append('  </chapter>')

        xml_lines.append('</book>')

        return '\n'.join(xml_lines)

    def generate_audio(self, text, output_path, audio_prompt_path=None, language_id="en", exaggeration=0.6, cfg_weight=0.4):
        """Generate audio from text using Chatterbox TTS"""
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

            model = self.load_model()

            # Prepare generation parameters
            gen_params = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight
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

@app.route('/api/scan', methods=['POST'])
def scan_directory():
    """Scan directory for text files"""
    try:
        data = request.json
        directory = data.get('directory', '')

        if not directory:
            return jsonify({'error': 'Directory path is required'}), 400

        if not os.path.isdir(directory):
            return jsonify({'error': 'Invalid directory path'}), 400

        files = converter.find_txt_files(directory)
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/read', methods=['POST'])
def read_file():
    """Read text file content"""
    try:
        data = request.json
        file_path = data.get('file_path', '')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400

        content = converter.read_text_file(file_path)
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
@auth_manager.require_api_key
def generate_audio():
    """Generate audio for a text file"""
    try:
        data = request.json
        file_path = data.get('file_path', '')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400

        audio_path = converter.get_or_generate_audio(file_path)
        audio_filename = os.path.basename(audio_path)

        return jsonify({
            'audio_url': f'/api/audio/{audio_filename}',
            'audio_path': audio_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-from-upload', methods=['POST'])
def generate_from_upload():
    """Generate audio from uploaded text file"""
    try:
        print(f"\n=== Received upload request ===")
        print(f"Files in request: {list(request.files.keys())}")
        print(f"Form data: {list(request.form.keys())}")

        if 'file' not in request.files:
            error_msg = 'No file provided'
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 400

        file = request.files['file']
        filename = request.form.get('filename', file.filename)
        print(f"Processing file: {filename}")

        if not filename.endswith('.txt'):
            error_msg = 'Only .txt files are supported'
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 400

        # Read file content
        text = file.read().decode('utf-8')
        print(f"File content length: {len(text)} characters")

        # Get optional parameters from form data
        language_id = request.form.get('language_id', 'en')
        exaggeration = float(request.form.get('exaggeration', 0.5))
        cfg_weight = float(request.form.get('cfg_weight', 0.5))
        voice_sample_name = request.form.get('voice_sample', None)

        # Construct full path to voice sample if provided
        audio_prompt_path = None
        if voice_sample_name and voice_sample_name != 'none':
            audio_prompt_path = os.path.join(converter.voice_samples_dir, voice_sample_name)
            if not os.path.exists(audio_prompt_path):
                print(f"Warning: Voice sample not found: {audio_prompt_path}")
                audio_prompt_path = None

        print(f"Parameters - Language: {language_id}, Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}, Voice Sample: {voice_sample_name}")

        # Limit text length for demo purposes
        if len(text) > 1000:
            print(f"Truncating text from {len(text)} to 1000 characters")
            text = text[:1000] + "..."

        # Generate audio filename with timestamp for multiple generations
        import time
        import json
        base_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp = int(time.time() * 1000)
        audio_filename = f"{base_name}_{timestamp}.wav"
        audio_path = os.path.join(converter.audio_dir, audio_filename)

        # Save metadata
        metadata_filename = f"{base_name}_{timestamp}.json"
        metadata_path = os.path.join(converter.audio_dir, metadata_filename)
        metadata = {
            'text_file': filename,
            'timestamp': timestamp,
            'language_id': language_id,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'voice_sample': voice_sample_name,
            'audio_file': audio_filename,
            'text_preview': text[:200],
            'is_best_take': False
        }

        print(f"Audio will be saved to: {audio_path}")

        # Generate audio if it doesn't exist
        if not os.path.exists(audio_path):
            print(f"Generating audio for: {filename}...")
            print(f"Text preview: {text[:100]}...")
            result = converter.generate_audio(
                text,
                audio_path,
                audio_prompt_path=audio_prompt_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            print(f"Audio generated successfully! Duration: {result['duration_seconds']:.2f}s")

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
        else:
            print(f"Audio already exists, using cached version")

        response_data = {
            'audio_url': f'/api/audio/{audio_filename}',
            'audio_path': audio_path,
            'metadata': metadata
        }
        print(f"Returning success response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n=== ERROR ===")
        print(f"Exception: {str(e)}")
        print(f"Traceback:\n{error_trace}")
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500

@app.route('/api/generate-all', methods=['POST'])
def generate_all_audio():
    """Generate audio for all text files in directory"""
    try:
        data = request.json
        directory = data.get('directory', '')

        if not directory or not os.path.isdir(directory):
            return jsonify({'error': 'Invalid directory path'}), 400

        files = converter.find_txt_files(directory)
        results = []

        for file_info in files:
            try:
                audio_path = converter.get_or_generate_audio(file_info['path'])
                results.append({
                    'file': file_info['name'],
                    'status': 'success',
                    'audio_path': audio_path
                })
            except Exception as e:
                results.append({
                    'file': file_info['name'],
                    'status': 'failed',
                    'error': str(e)
                })

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    """Get current device status"""
    return jsonify({
        'current_device': converter.device,
        'cuda_available': converter.cuda_available,
        'model_loaded': converter.model is not None,
        'max_parallel_generations': config.MAX_PARALLEL_GENERATIONS
    })

@app.route('/api/device', methods=['POST'])
@auth_manager.require_api_key
def switch_device():
    """Switch between GPU and CPU"""
    try:
        data = request.json
        new_device = data.get('device')

        if not new_device:
            return jsonify({'error': 'device parameter is required'}), 400

        # Try to switch device
        changed = converter.switch_device(new_device)

        return jsonify({
            'success': True,
            'device': converter.device,
            'changed': changed,
            'message': f"Switched to {new_device}" if changed else f"Already using {new_device}"
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/project/create', methods=['POST'])
@auth_manager.require_api_key
def create_project():
    """Create a new project in a user-selected folder or default location"""
    try:
        import json
        import shutil
        import urllib.request
        from datetime import datetime

        print('\n━━━ [CREATE PROJECT API] START ━━━')
        data = request.json
        print(f'[CREATE PROJECT API] Request data keys: {data.keys() if data else None}')

        # Accept both 'name'/'project_name' and 'path'/'project_path' for flexibility
        project_name = data.get('name') or data.get('project_name', 'Unnamed Project')
        base_path = data.get('path') or data.get('project_path')
        text_source = data.get('text_source')

        print(f'[CREATE PROJECT API] Project name: {project_name}')
        print(f'[CREATE PROJECT API] Base path: {base_path or "(using default)"}')
        print(f'[CREATE PROJECT API] Text source: {text_source.get("type") if text_source else "None"}')

        if not project_name:
            print('[CREATE PROJECT API] ✗ ERROR: Project name is required')
            return jsonify({'error': 'Project name is required'}), 400

        # If no path provided, use default project directory
        if not base_path:
            base_path = config.DEFAULT_PROJECT_DIR
            print(f'[CREATE PROJECT API] Using default path: {base_path}')

        # Construct full project path
        project_path = os.path.join(base_path, project_name)
        print(f'[CREATE PROJECT API] Full project path: {project_path}')

        # Create project directory structure
        print(f'[CREATE PROJECT API] Creating directories...')
        os.makedirs(project_path, exist_ok=True)
        texts_dir = os.path.join(project_path, 'texts')
        audio_dir = os.path.join(project_path, 'audio')

        os.makedirs(texts_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # Create initial project metadata
        project_metadata = {
            'name': project_name,
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'version': '1.0',
            'note': 'All file references use relative paths for portability',
            'default_audio_settings': {
                'exaggeration': 0.6,
                'cfg_weight': 0.4,
                'voice_sample': 'none',
                'seed': 0,
                'temperature': 0.8,
                'ref_vad_trimming': False
            },
            'chapters': []
        }

        # Update converter paths (must be done before processing text)
        converter.current_project_path = project_path
        converter.current_project_metadata = project_metadata
        converter.audio_dir = audio_dir
        # Keep voice_samples_dir pointing to main folder (not project-specific)

        # Handle text source if provided
        if text_source:
            source_type = text_source.get('type')

            if source_type == 'url':
                url = text_source.get('url')
                print(f'[CREATE PROJECT API] Processing URL: {url}')

                # Check if it's a Gutenberg URL
                if 'gutenberg.org' in url.lower():
                    print(f'[CREATE PROJECT API] Using Gutenberg processing for: {url}')
                    try:
                        # Use GutenbergProcessor for special Gutenberg handling
                        processor = GutenbergProcessor(output_dir=project_path)

                        # Download content (supports both .txt and .html)
                        print(f'[CREATE PROJECT API] Downloading content from Gutenberg...')
                        content = processor.download_text(url)

                        # Check if content is HTML
                        is_html = processor.is_html(content)
                        print(f'[CREATE PROJECT API] Content type: {"HTML" if is_html else "Plain text"}')

                        if is_html:
                            # Process HTML content
                            print(f'[CREATE PROJECT API] Processing HTML content...')

                            # Extract title from HTML
                            try:
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(content, 'html.parser')
                                title_tag = soup.find('title')
                                if title_tag:
                                    # Clean up title
                                    title = title_tag.get_text(strip=True)
                                    title = re.sub(r'The Project Gutenberg eBook of\s+', '', title, flags=re.IGNORECASE)
                                    title = re.sub(r'\s+by\s+.+$', '', title)
                                    title = re.sub(r'[^\w\s-]', '', title)
                                    title = re.sub(r'\s+', '_', title)
                                    title = title[:50]
                                else:
                                    title = processor.extract_book_name(url)
                            except:
                                title = processor.extract_book_name(url)

                            print(f'[CREATE PROJECT API] Extracted title: {title}')

                            # Extract chapters from HTML
                            chapters_data = processor.extract_html_chapters(content, title)
                            print(f'[CREATE PROJECT API] Extracted {len(chapters_data)} chapters from HTML')

                            # Save raw HTML
                            raw_html_file = os.path.join(project_path, 'raw_text.html')
                            with open(raw_html_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            print(f'[CREATE PROJECT API] Saved raw HTML to {raw_html_file}')

                            # Create chapters with chunks
                            new_chapters = []
                            for i, (chapter_title, chapter_text) in enumerate(chapters_data):
                                # Chunk the chapter text
                                chunks = converter.smart_chunk_text(chapter_text)

                                # Add chunk structure
                                for chunk in chunks:
                                    chunk['dirty'] = False
                                    chunk['generated_audios'] = []

                                chapter_entry = {
                                    'id': f'chapter_{i}',
                                    'title': chapter_title,
                                    'order': i,
                                    'chunks': chunks,
                                    'audio_output': None,
                                    'added_at': datetime.now().isoformat(),
                                    'source': 'gutenberg_html',
                                    'source_url': url
                                }

                                new_chapters.append(chapter_entry)

                            # Save chapter texts to texts directory
                            for i, (chapter_title, chapter_text) in enumerate(chapters_data):
                                sanitized = re.sub(r'[^\w\s-]', '', chapter_title)
                                sanitized = re.sub(r'\s+', '_', sanitized)
                                sanitized = sanitized.strip('_')[:50]
                                if not sanitized:
                                    sanitized = f"chapter_{i}"

                                text_file_path = os.path.join(texts_dir, f'{sanitized}.txt')
                                with open(text_file_path, 'w', encoding='utf-8') as f:
                                    f.write(f"{chapter_title}\n\n{chapter_text}")

                            # Generate XML content from all chapter texts
                            all_text = '\n\n'.join([chapter_text for _, chapter_text in chapters_data])
                            detected_chapters_for_xml = converter.detect_chapters(all_text)
                            xml_content = converter.text_to_xml_content(all_text, detected_chapters_for_xml)

                            # Update metadata with chapters and XML
                            project_metadata['chapters'] = new_chapters
                            project_metadata['content_xml'] = xml_content
                            project_metadata['original_filename'] = f"{title}.html"
                            project_metadata['version'] = '3.0'

                            print(f'[CREATE PROJECT API] Created {len(new_chapters)} chapters with HTML processing')

                        else:
                            # Process plain text content (original logic)
                            print(f'[CREATE PROJECT API] Processing plain text content...')

                            # Extract title
                            title = processor.extract_title(content)
                            if not title:
                                title = processor.extract_book_name(url)
                            print(f'[CREATE PROJECT API] Extracted title: {title}')

                            # Strip Gutenberg metadata
                            text = processor.strip_gutenberg_metadata(content, title)

                            # Process carriage returns
                            text = processor.process_carriage_returns(text)

                            # Replace SECTION_BREAK markers
                            text = text.replace('<<<SECTION_BREAK>>>', '\n\n')

                            print(f'[CREATE PROJECT API] Processed text length: {len(text)} characters')

                            # Save raw text to raw_text.txt
                            raw_text_file = os.path.join(project_path, 'raw_text.txt')
                            with open(raw_text_file, 'w', encoding='utf-8') as f:
                                f.write(text)
                            print(f'[CREATE PROJECT API] Saved raw text to {raw_text_file}')

                            # Also save to texts directory
                            text_file_path = os.path.join(texts_dir, f'{title}.txt')
                            with open(text_file_path, 'w', encoding='utf-8') as f:
                                f.write(text)

                            # Detect chapters
                            detected_chapters = converter.detect_chapters(text)
                            print(f'[CREATE PROJECT API] Detected {len(detected_chapters)} chapter(s)')

                            # Generate XML content
                            xml_content = converter.text_to_xml_content(text, detected_chapters)

                            # Create chapters with chunks
                            new_chapters = []
                            for detected_chapter in detected_chapters:
                                chunks = converter.smart_chunk_text(detected_chapter['text'])

                                # Add chunk structure
                                for chunk in chunks:
                                    chunk['dirty'] = False
                                    chunk['generated_audios'] = []

                                chapter_entry = {
                                    'id': detected_chapter['id'],
                                    'title': detected_chapter['title'],
                                    'order': detected_chapter['order'],
                                    'chunks': chunks,
                                    'audio_output': None,
                                    'added_at': datetime.now().isoformat(),
                                    'source': 'gutenberg',
                                    'source_url': url
                                }

                                new_chapters.append(chapter_entry)

                            # Update metadata with chapters and XML
                            project_metadata['chapters'] = new_chapters
                            project_metadata['content_xml'] = xml_content
                            project_metadata['original_filename'] = f"{title}.txt"
                            project_metadata['version'] = '3.0'

                            print(f'[CREATE PROJECT API] Created {len(new_chapters)} chapters with Gutenberg processing')

                    except Exception as e:
                        print(f'[CREATE PROJECT API] ✗ ERROR in Gutenberg processing: {str(e)}')
                        import traceback
                        print(traceback.format_exc())
                        return jsonify({'error': f'Failed to process Gutenberg URL: {str(e)}'}), 400
                else:
                    # Regular URL (non-Gutenberg)
                    print(f'[CREATE PROJECT API] Downloading from non-Gutenberg URL')
                    try:
                        with urllib.request.urlopen(url) as response:
                            text_content = response.read().decode('utf-8')

                        # Save raw text
                        raw_text_path = os.path.join(project_path, 'raw_text.txt')
                        with open(raw_text_path, 'w', encoding='utf-8') as f:
                            f.write(text_content)

                        # Also save to texts directory
                        text_file_path = os.path.join(texts_dir, 'source.txt')
                        with open(text_file_path, 'w', encoding='utf-8') as f:
                            f.write(text_content)

                        print(f'[CREATE PROJECT API] Downloaded and saved {len(text_content)} characters')
                    except Exception as e:
                        print(f'[CREATE PROJECT API] ✗ ERROR downloading from URL: {str(e)}')
                        return jsonify({'error': f'Failed to download from URL: {str(e)}'}), 400

            elif source_type == 'file':
                # Uploaded file
                text_content = text_source.get('content')
                source_filename = text_source.get('filename', 'source.txt')
                print(f'[CREATE PROJECT API] Using uploaded file: {source_filename}')

                # Save raw text
                raw_text_path = os.path.join(project_path, 'raw_text.txt')
                with open(raw_text_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)

                # Also save to texts directory
                text_file_path = os.path.join(texts_dir, source_filename)
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)

                print(f'[CREATE PROJECT API] Saved {len(text_content)} characters')

        # Update converter metadata reference
        converter.current_project_metadata = project_metadata

        # Save project metadata to file
        project_file = os.path.join(project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_metadata, f, indent=2, ensure_ascii=False)
        print(f'[CREATE PROJECT API] Saved project metadata to {project_file}')

        print(f'[CREATE PROJECT API] ✓ Project created successfully')
        print(f'[CREATE PROJECT API] Returning: project_path={project_path}')
        print(f'[CREATE PROJECT API] Path repr: {repr(project_path)}')
        print(f'[CREATE PROJECT API] Path bytes: {[hex(ord(c)) for c in project_path]}')
        print('━━━ [CREATE PROJECT API] END ━━━\n')

        result = {
            'success': True,
            'project_path': project_path,
            'metadata': project_metadata
        }
        print(f'[CREATE PROJECT API] JSON being returned: {json.dumps(result, indent=2)}')

        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"[CREATE PROJECT API] ✗ ERROR: {str(e)}")
        print(traceback.format_exc())
        print('━━━ [CREATE PROJECT API] END (ERROR) ━━━\n')
        return jsonify({'error': str(e)}), 500

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
        print(f'[LOAD PROJECT API] Project metadata: {project_metadata}')

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

@app.route('/api/project/add-text-file', methods=['POST'])
@auth_manager.require_api_key
def add_text_file_to_project():
    """Add a text file to the project with chapter detection and chunking"""
    try:
        import json
        from datetime import datetime
        import uuid

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        # Get file from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read text content
        text_content = file.read().decode('utf-8')
        original_filename = file.filename

        print(f"\n=== Adding text file: {original_filename} ===")

        # Detect chapters in the text
        detected_chapters = converter.detect_chapters(text_content)
        print(f"Detected {len(detected_chapters)} chapter(s)")

        # Generate XML content
        xml_content = converter.text_to_xml_content(text_content, detected_chapters)

        # Initialize chapters structure if not present
        if 'chapters' not in converter.current_project_metadata:
            converter.current_project_metadata['chapters'] = []

        # Create chapters with chunks for each detected chapter
        new_chapters = []
        for detected_chapter in detected_chapters:
            # Chunk the chapter text
            chunks = converter.smart_chunk_text(detected_chapter['text'])

            # Add chunk structure with dirty flag and generated_audios
            for chunk in chunks:
                chunk['dirty'] = False
                chunk['generated_audios'] = []

            # Create chapter entry
            chapter_entry = {
                'id': detected_chapter['id'],
                'title': detected_chapter['title'],
                'order': detected_chapter['order'],
                'chunks': chunks,
                'audio_output': None,  # Will be set when chapter is stitched
                'added_at': datetime.now().isoformat()
            }

            new_chapters.append(chapter_entry)
            converter.current_project_metadata['chapters'].append(chapter_entry)

        # Store/update XML content
        converter.current_project_metadata['content_xml'] = xml_content
        converter.current_project_metadata['original_filename'] = original_filename
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()
        converter.current_project_metadata['version'] = '3.0'

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        print(f"Successfully added {len(new_chapters)} chapters to project")

        return jsonify({
            'success': True,
            'chapters': new_chapters,
            'chapter_count': len(new_chapters),
            'content_xml': xml_content
        })

    except Exception as e:
        import traceback
        print(f"Error adding text file to project: {str(e)}")
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
            'content_xml': converter.current_project_metadata.get('content_xml', None)
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

        # Check if the new text exceeds max chunk size and needs splitting
        max_chunk_size = config.MAX_CHUNK_SIZE
        was_split = False
        new_chunks = []

        if len(new_text) > max_chunk_size:
            # Text is too long, need to split it
            print(f"[UPDATE CHUNK] Chunk {chunk_id} text ({len(new_text)} chars) exceeds max ({max_chunk_size}), splitting...")

            # Get split chunks
            split_result = converter.smart_chunk_text(new_text, max_chunk_size)

            if len(split_result) > 1:
                was_split = True

                # Find the chunk's position in the container's chunks list
                chunks_list = container.get('chunks', [])
                chunk_index = next((i for i, c in enumerate(chunks_list) if c['id'] == chunk_id), None)

                if chunk_index is None:
                    return jsonify({'error': 'Chunk position not found'}), 404

                # Update the original chunk with the first split piece
                first_piece = split_result[0]
                chunk['text'] = first_piece['text']
                chunk['nickname'] = first_piece['text'][:50].strip() + ('...' if len(first_piece['text']) > 50 else '')

                # Mark as dirty if there are generated audios
                if len(chunk.get('generated_audios', [])) > 0:
                    chunk['dirty'] = True

                new_chunks.append(chunk)

                # Get the highest chunk ID in this chapter
                max_id = max(c['id'] for c in chunks_list)

                # Insert remaining pieces as new chunks after the original
                for i, piece in enumerate(split_result[1:], 1):
                    new_chunk = {
                        'id': max_id + i,
                        'type': 'text',
                        'text': piece['text'],
                        'nickname': piece['text'][:50].strip() + ('...' if len(piece['text']) > 50 else ''),
                        'start_pos': piece.get('start_pos', 0),
                        'end_pos': piece.get('end_pos', len(piece['text'])),
                        'dirty': False,
                        'generated_audios': []
                    }
                    # Insert at the correct position
                    chunks_list.insert(chunk_index + i, new_chunk)
                    new_chunks.append(new_chunk)

                print(f"[UPDATE CHUNK] Split into {len(split_result)} chunks")
        else:
            # No split needed, just update the chunk normally
            chunk['text'] = new_text
            # Use provided nickname if available, otherwise auto-generate from text
            if new_nickname is not None:
                chunk['nickname'] = new_nickname
            else:
                chunk['nickname'] = new_text[:50].strip() + ('...' if len(new_text) > 50 else '')

            # Mark as dirty if there are generated audios
            if len(chunk.get('generated_audios', [])) > 0:
                chunk['dirty'] = True

            new_chunks.append(chunk)

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
            'chunk': chunk,
            'was_split': was_split,
            'new_chunks': new_chunks,
            'split_count': len(new_chunks) if was_split else 0
        })

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
        language_id = data.get('language_id', 'en')

        # Construct voice sample path
        audio_prompt_path = None
        if voice_sample and voice_sample != 'none':
            audio_prompt_path = os.path.join(converter.voice_samples_dir, voice_sample)
            if not os.path.exists(audio_prompt_path):
                print(f"Warning: Voice sample not found: {audio_prompt_path}")
                audio_prompt_path = None

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
        print(f"Exaggeration: {exaggeration}, CFG: {cfg_weight}, Temp: {temperature}")

        # Generate audio filename with timestamp and chunk ID
        timestamp = int(time.time() * 1000)
        audio_filename = f"chunk{chunk_id}_{timestamp}.wav"
        audio_path = os.path.join(converter.audio_dir, audio_filename)

        # Generate the audio (returns dict with path and duration)
        generation_result = converter.generate_audio(
            text=chunk_text,
            output_path=audio_path,
            audio_prompt_path=audio_prompt_path,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
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
                    'audio_duration_seconds': round(audio_duration, 2),
                    'possibly_truncated': possibly_truncated,
                    'generation_time_ms': generation_time_ms,
                    'input_text': chunk_text  # Store the text used for generation to detect outdated takes
                }
                chunk['generated_audios'].append(audio_entry)

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
        audio_dir = os.path.join(converter.current_project_path, 'audio')

        for chunk in chunks:
            chunk_type = chunk.get('type', 'text')

            if chunk_type == 'common_file':
                # For common_file chunks, use the common file directly
                common_file_path = chunk.get('common_file_path')
                if not common_file_path:
                    return jsonify({'error': f'Chunk {chunk["id"]} is missing common_file_path'}), 400

                # Resolve relative path
                if not os.path.isabs(common_file_path):
                    common_file_path = os.path.join(os.getcwd(), common_file_path)

                if not os.path.exists(common_file_path):
                    return jsonify({'error': f'Common file not found: {common_file_path}'}), 400

                print(f"Chunk {chunk['id']}: Using common file: {common_file_path}")
                audio_paths.append(common_file_path)

            elif chunk_type == 'pause':
                # For pause chunks, generate silence
                duration_ms = chunk.get('duration_ms', 1000)
                print(f"Chunk {chunk['id']}: Generating {duration_ms}ms pause")
                # We'll handle pause chunks in the stitching function
                audio_paths.append(('pause', duration_ms))

            else:
                # For text chunks, use generated audio
                generated_audios = chunk.get('generated_audios', [])

                if not generated_audios:
                    return jsonify({'error': f'No audio generated for chunk {chunk["id"]}'}), 400

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
                    print(f"ERROR: audio object missing 'audio_file' or 'audio_url' key. Audio object: {best_audio}")
                    return jsonify({'error': f'Chunk {chunk["id"]} has invalid audio metadata'}), 400
                audio_path = os.path.join(audio_dir, audio_file)

                print(f"Chunk {chunk['id']}: Selected audio file: {audio_file}")
                print(f"Chunk {chunk['id']}: Full path: {audio_path}")
                print(f"Chunk {chunk['id']}: File exists: {os.path.exists(audio_path)}")

                if not os.path.exists(audio_path):
                    return jsonify({'error': f'Audio file not found: {audio_file}'}), 400

                audio_paths.append(audio_path)

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

        # Create metadata for stitched audio
        stitched_metadata = {
            'audio_file': stitched_filename,
            'timestamp': timestamp,
            'is_stitched': True,
            'chunk_count': len(chunks),
            'text_file_id': chapter_id
        }

        # Optionally save metadata to a JSON file
        metadata_filename = f"{base_name}_stitched_{timestamp}.json"
        metadata_path = os.path.join(audio_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(stitched_metadata, f, indent=2)

        # Return success with audio URL
        audio_url = f'/api/project/audio/{stitched_filename}'

        return jsonify({
            'success': True,
            'audio_url': audio_url,
            'audio_file': stitched_filename,
            'audio_path': stitched_path,
            'metadata': stitched_metadata
        })

    except Exception as e:
        import traceback
        print(f"Error stitching project audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/save-text', methods=['POST'])
def save_text_to_project():
    """Save a text file to the current project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        filename = data.get('filename')
        content = data.get('content')

        if not filename or content is None:
            return jsonify({'error': 'filename and content are required'}), 400

        texts_dir = os.path.join(converter.current_project_path, 'texts')
        file_path = os.path.join(texts_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Return relative path
        relative_path = converter.get_relative_path(file_path)
        return jsonify({'success': True, 'file_path': relative_path})

    except Exception as e:
        import traceback
        print(f"Error saving text: {str(e)}")
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

@app.route('/api/chunk-text', methods=['POST'])
def chunk_text():
    """Chunk text into manageable pieces"""
    try:
        data = request.json
        text = data.get('text', '')
        max_chunk_size = data.get('max_chunk_size', config.MAX_CHUNK_SIZE)

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        chunks = converter.smart_chunk_text(text, max_chunk_size)
        return jsonify({'chunks': chunks})

    except Exception as e:
        import traceback
        print(f"Error chunking text: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-chunk', methods=['POST'])
def generate_chunk():
    """Generate audio for a specific chunk of text"""
    try:
        import json
        import time

        print(f"\n=== Received chunk generation request ===")

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filename = request.form.get('filename', file.filename)
        chunk_id = int(request.form.get('chunk_id', 0))
        chunk_text = request.form.get('chunk_text', '')

        # Get optional parameters
        language_id = request.form.get('language_id', 'en')
        exaggeration = float(request.form.get('exaggeration', 0.5))
        cfg_weight = float(request.form.get('cfg_weight', 0.5))
        voice_sample_name = request.form.get('voice_sample', None)

        # Construct voice sample path
        audio_prompt_path = None
        if voice_sample_name and voice_sample_name != 'none':
            audio_prompt_path = os.path.join(converter.voice_samples_dir, voice_sample_name)
            if not os.path.exists(audio_prompt_path):
                print(f"Warning: Voice sample not found: {audio_prompt_path}")
                audio_prompt_path = None

        print(f"Generating chunk {chunk_id} for: {filename}")
        print(f"Chunk text length: {len(chunk_text)} characters")
        print(f"Parameters - Language: {language_id}, Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}")

        # Strip XML tags from chunk text before TTS processing
        clean_text = converter.strip_xml_tags(chunk_text)
        print(f"Clean text length (after stripping XML tags): {len(clean_text)} characters")

        # Generate audio filename with timestamp and chunk ID
        base_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp = int(time.time() * 1000)
        audio_filename = f"{base_name}_chunk{chunk_id}_{timestamp}.wav"
        audio_path = os.path.join(converter.audio_dir, audio_filename)

        # Save metadata
        metadata_filename = f"{base_name}_chunk{chunk_id}_{timestamp}.json"
        metadata_path = os.path.join(converter.audio_dir, metadata_filename)
        metadata = {
            'text_file': filename,
            'chunk_id': chunk_id,
            'timestamp': timestamp,
            'language_id': language_id,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'voice_sample': voice_sample_name,
            'audio_file': audio_filename,
            'text_preview': chunk_text[:200],  # Keep original text with tags for display
            'is_best_take': False
        }

        # Generate audio using clean text (no XML tags)
        print(f"Generating audio for chunk {chunk_id}...")
        result = converter.generate_audio(
            clean_text,
            audio_path,
            audio_prompt_path=audio_prompt_path,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        print(f"Chunk audio generated successfully!")

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return jsonify({
            'audio_url': f'/api/audio/{audio_filename}',
            'audio_path': audio_path,
            'metadata': metadata
        })

    except Exception as e:
        import traceback
        print(f"Error generating chunk audio: {str(e)}")
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

@app.route('/api/estimate-time', methods=['POST'])
def estimate_generation_time():
    """Estimate generation time for given text"""
    try:
        data = request.json
        text = data.get('text', '')
        char_count = len(text)

        estimate = converter.estimate_generation_time(char_count)

        if estimate is None:
            return jsonify({
                'char_count': char_count,
                'has_estimate': False,
                'message': 'No historical data available yet'
            })

        return jsonify({
            'char_count': char_count,
            'has_estimate': True,
            **estimate
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stitch-audio', methods=['POST'])
def stitch_audio():
    """Stitch together the best takes from all chunks"""
    try:
        import json
        import time

        data = request.json
        txt_filename = data.get('txt_filename')
        chunk_ids = data.get('chunk_ids', [])

        if not txt_filename:
            return jsonify({'error': 'txt_filename is required'}), 400

        base_name = os.path.splitext(txt_filename)[0]
        audio_paths = []

        # Find the best take for each chunk in order
        for chunk_id in chunk_ids:
            best_audio = None
            best_timestamp = 0

            # Search for the best take for this chunk
            for filename in os.listdir(converter.audio_dir):
                if filename.startswith(base_name) and f'_chunk{chunk_id}_' in filename and filename.endswith('.json'):
                    metadata_path = os.path.join(converter.audio_dir, filename)

                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    if metadata.get('is_best_take', False):
                        audio_file = metadata['audio_file']
                        audio_path = os.path.join(converter.audio_dir, audio_file)
                        if os.path.exists(audio_path):
                            best_audio = audio_path
                            break

            # If no best take found, use the most recent one
            if not best_audio:
                for filename in os.listdir(converter.audio_dir):
                    if filename.startswith(base_name) and f'_chunk{chunk_id}_' in filename and filename.endswith('.json'):
                        metadata_path = os.path.join(converter.audio_dir, filename)

                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        if metadata['timestamp'] > best_timestamp:
                            best_timestamp = metadata['timestamp']
                            audio_file = metadata['audio_file']
                            audio_path = os.path.join(converter.audio_dir, audio_file)
                            if os.path.exists(audio_path):
                                best_audio = audio_path

            if best_audio:
                audio_paths.append(best_audio)
            else:
                return jsonify({'error': f'No audio found for chunk {chunk_id}'}), 400

        # Create stitched audio filename
        timestamp = int(time.time() * 1000)
        stitched_filename = f"{base_name}_stitched_{timestamp}.wav"
        stitched_path = os.path.join(converter.audio_dir, stitched_filename)

        # Stitch the audio files
        converter.stitch_audio_files(audio_paths, stitched_path)

        # Save metadata for stitched audio
        metadata_filename = f"{base_name}_stitched_{timestamp}.json"
        metadata_path = os.path.join(converter.audio_dir, metadata_filename)
        metadata = {
            'text_file': txt_filename,
            'timestamp': timestamp,
            'audio_file': stitched_filename,
            'is_stitched': True,
            'chunk_count': len(chunk_ids),
            'chunk_ids': chunk_ids
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return jsonify({
            'audio_url': f'/api/audio/{stitched_filename}',
            'audio_path': stitched_path,
            'metadata': metadata
        })

    except Exception as e:
        import traceback
        print(f"Error stitching audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-gutenberg', methods=['POST'])
def process_gutenberg():
    """Process Project Gutenberg URLs and save to directory"""
    try:
        data = request.json
        output_dir = data.get('output_dir')
        urls = data.get('urls', [])

        if not output_dir:
            return jsonify({'error': 'output_dir is required'}), 400

        if not urls or len(urls) == 0:
            return jsonify({'error': 'At least one URL is required'}), 400

        # Validate URLs
        for url in urls:
            if not url.strip():
                return jsonify({'error': 'Empty URL provided'}), 400
            if 'gutenberg.org' not in url.lower():
                return jsonify({'error': f'URL does not appear to be from Project Gutenberg: {url}'}), 400

        # Create processor and process URLs
        processor = GutenbergProcessor(output_dir)
        results = processor.process_urls(urls)

        # Count successes and failures
        successes = sum(1 for r in results.values() if 'error' not in r)
        failures = sum(1 for r in results.values() if 'error' in r)
        total_chapters = sum(r.get('count', 0) for r in results.values() if 'error' not in r)

        return jsonify({
            'success': True,
            'output_dir': output_dir,
            'processed': len(urls),
            'successes': successes,
            'failures': failures,
            'total_chapters': total_chapters,
            'results': results
        })

    except Exception as e:
        import traceback
        print(f"Error processing Gutenberg URLs: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/add-gutenberg-url', methods=['POST'])
@auth_manager.require_api_key
def add_gutenberg_url_to_project():
    """Add Project Gutenberg content directly to current project with XML chapter structure"""
    try:
        import json
        from datetime import datetime
        import uuid

        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        url = data.get('url')

        if not url:
            return jsonify({'error': 'url is required'}), 400

        if 'gutenberg.org' not in url.lower():
            return jsonify({'error': f'URL does not appear to be from Project Gutenberg: {url}'}), 400

        print(f"\n=== Adding Gutenberg URL to project: {url} ===")

        # Use GutenbergProcessor to download and process the text
        processor = GutenbergProcessor(output_dir=converter.current_project_path)

        # Download text
        print(f"Downloading text from {url}...")
        text = processor.download_text(url)

        # Extract title
        title = processor.extract_title(text)
        if not title:
            title = processor.extract_book_name(url)
        print(f"Extracted title: {title}")

        # Strip Gutenberg metadata
        text = processor.strip_gutenberg_metadata(text, title)

        # Process carriage returns
        text = processor.process_carriage_returns(text)

        # Replace SECTION_BREAK markers with paragraph breaks
        # This ensures they don't appear in the final text
        text = text.replace('<<<SECTION_BREAK>>>', '\n\n')

        print(f"Processed text length: {len(text)} characters")

        # Save raw text to raw_text.txt in project directory
        raw_text_file = os.path.join(converter.current_project_path, 'raw_text.txt')
        with open(raw_text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved raw text to {raw_text_file}")

        # Detect chapters in the processed text
        detected_chapters = converter.detect_chapters(text)
        print(f"Detected {len(detected_chapters)} chapter(s)")

        # Generate XML content
        xml_content = converter.text_to_xml_content(text, detected_chapters)

        # Initialize chapters structure if not present
        if 'chapters' not in converter.current_project_metadata:
            converter.current_project_metadata['chapters'] = []

        # Create chapters with chunks for each detected chapter
        new_chapters = []
        for detected_chapter in detected_chapters:
            # Chunk the chapter text
            chunks = converter.smart_chunk_text(detected_chapter['text'])

            # Add chunk structure with dirty flag and generated_audios
            for chunk in chunks:
                chunk['dirty'] = False
                chunk['generated_audios'] = []

            # Create chapter entry
            chapter_entry = {
                'id': detected_chapter['id'],
                'title': detected_chapter['title'],
                'order': detected_chapter['order'],
                'chunks': chunks,
                'audio_output': None,
                'added_at': datetime.now().isoformat(),
                'source': 'gutenberg',
                'source_url': url
            }

            new_chapters.append(chapter_entry)
            converter.current_project_metadata['chapters'].append(chapter_entry)

        # Store/update XML content
        converter.current_project_metadata['content_xml'] = xml_content
        converter.current_project_metadata['original_filename'] = f"{title}.txt"
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()
        converter.current_project_metadata['version'] = '3.0'

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        print(f"Successfully added {len(new_chapters)} chapters from Gutenberg to project")

        return jsonify({
            'success': True,
            'title': title,
            'url': url,
            'chapters': new_chapters,
            'chapter_count': len(new_chapters),
            'content_xml': xml_content
        })

    except Exception as e:
        import traceback
        print(f"Error adding Gutenberg URL to project: {str(e)}")
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

        # Voice sample path
        audio_prompt_path = None
        if voice_sample and voice_sample != 'none':
            audio_prompt_path = os.path.join(converter.voice_samples_dir, voice_sample)
            if not os.path.exists(audio_prompt_path):
                print(f"Warning: Voice sample not found: {audio_prompt_path}")
                audio_prompt_path = None

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
                # Strip XML tags from chunk text
                clean_text = converter.strip_xml_tags(chunk['text'])

                # Generate filename
                timestamp = int(time.time() * 1000)
                chapter_title_safe = chapter['title'].replace(' ', '_').replace('/', '_')[:30]
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
                    'timestamp': timestamp,
                    'language_id': language_id,
                    'exaggeration': exaggeration,
                    'cfg_weight': cfg_weight,
                    'voice_sample': voice_sample,
                    'text_preview': chunk['text'][:200],
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

        # Get all project directories from the default projects folder
        projects_dir = config.DEFAULT_PROJECT_DIR
        if not os.path.exists(projects_dir):
            return jsonify([])

        recent_projects = []

        # Scan for all subdirectories
        for item in os.listdir(projects_dir):
            project_path = os.path.join(projects_dir, item)

            # Only include directories
            if not os.path.isdir(project_path):
                continue

            project_file = os.path.join(project_path, 'project.json')

            # Try to read project.json if it exists
            if os.path.exists(project_file):
                try:
                    with open(project_file, 'r') as f:
                        project_data = json.load(f)

                    recent_projects.append({
                        'name': project_data.get('name', item),
                        'path': os.path.abspath(project_path),  # Return absolute path
                        'last_modified': project_data.get('last_modified', ''),
                        'created_at': project_data.get('created_at', '')
                    })
                except Exception as e:
                    print(f"Error reading project {project_path}: {e}")
                    # Still include the directory even if project.json is invalid
                    recent_projects.append({
                        'name': item,
                        'path': os.path.abspath(project_path),
                        'last_modified': '',
                        'created_at': ''
                    })
            else:
                # Include directory even without project.json
                recent_projects.append({
                    'name': item,
                    'path': os.path.abspath(project_path),
                    'last_modified': '',
                    'created_at': ''
                })

        # Sort by last_modified (most recent first), then by name
        recent_projects.sort(key=lambda x: (x.get('last_modified', ''), x.get('name', '')), reverse=True)

        return jsonify(recent_projects)

    except Exception as e:
        import traceback
        print(f"Error getting recent projects: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/raw-text', methods=['GET', 'POST'])
@auth_manager.require_api_key
def project_raw_text():
    """Get or save raw text for the current project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        raw_text_file = os.path.join(converter.current_project_path, 'raw_text.txt')

        if request.method == 'POST':
            # Save raw text
            data = request.json
            raw_text = data.get('raw_text', '')

            with open(raw_text_file, 'w', encoding='utf-8') as f:
                f.write(raw_text)

            return jsonify({'success': True})
        else:
            # Get raw text
            if os.path.exists(raw_text_file):
                with open(raw_text_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                return jsonify({'raw_text': raw_text})
            else:
                return jsonify({'raw_text': ''})

    except Exception as e:
        import traceback
        print(f"Error with raw text: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/save-raw-text', methods=['POST'])
@auth_manager.require_api_key
def save_project_raw_text():
    """Save raw text for the current project"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        raw_text = data.get('raw_text', '')

        # Save raw text to file
        raw_text_file = os.path.join(converter.current_project_path, 'raw_text.txt')
        with open(raw_text_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)

        return jsonify({'success': True})

    except Exception as e:
        import traceback
        print(f"Error saving raw text: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/validate-chunks', methods=['POST'])
@auth_manager.require_api_key
def validate_project_chunks():
    """Validate chunk sizes and identify chunks that are too large"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json or {}
        max_chunk_size = data.get('max_chunk_size', config.MAX_CHUNK_SIZE)

        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'r') as f:
            project_data = json.load(f)

        oversized_chunks = []

        for chapter in project_data.get('chapters', []):
            chapter_title = chapter.get('title') or chapter.get('name', '')
            for chunk in chapter.get('chunks', []):
                chunk_text = chunk.get('text', '')
                if len(chunk_text) > max_chunk_size:
                    oversized_chunks.append({
                        'chapter_id': chapter.get('id', ''),
                        'chapter_title': chapter_title,
                        'chunk_id': chunk.get('id', 0),
                        'size': len(chunk_text),
                        'chunk_preview': chunk_text[:50] + '...'
                    })

        return jsonify({
            'success': True,
            'max_chunk_size': max_chunk_size,
            'total_chunks': sum(len(ch.get('chunks', [])) for ch in project_data.get('chapters', [])),
            'oversized_count': len(oversized_chunks),
            'oversized_chunks': oversized_chunks
        })

    except Exception as e:
        import traceback
        print(f"Error validating chunks: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/auto-rechunk', methods=['POST'])
@auth_manager.require_api_key
def auto_rechunk_oversized():
    """Automatically rechunk oversized chunks with minimal disruption"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        max_chunk_size = data.get('max_chunk_size', config.MAX_CHUNK_SIZE)
        chapter_id = data.get('chapter_id')  # Optional: rechunk specific chapter

        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'r') as f:
            project_data = json.load(f)

        rechunked_count = 0

        for chapter in project_data.get('chapters', []):
            # Skip if specific chapter requested and this isn't it
            if chapter_id and chapter['id'] != chapter_id:
                continue

            chunks = chapter.get('chunks', [])
            new_chunks = []
            chunk_id_counter = 0

            for chunk in chunks:
                chunk_text = chunk.get('text', '')

                if len(chunk_text) <= max_chunk_size:
                    # Keep chunk as-is, but update ID to maintain sequence
                    chunk['id'] = chunk_id_counter
                    new_chunks.append(chunk)
                    chunk_id_counter += 1
                else:
                    # Rechunk this oversized chunk
                    rechunked_count += 1
                    sub_chunks = converter.smart_chunk_text(chunk_text, max_chunk_size)

                    for sub_chunk in sub_chunks:
                        new_chunk = {
                            'id': chunk_id_counter,
                            'text': sub_chunk['text'],
                            'nickname': sub_chunk['nickname'],
                            'start_pos': sub_chunk['start_pos'],
                            'end_pos': sub_chunk['end_pos'],
                            'dirty': False,
                            'generated_audios': []  # Reset audio for rechunked pieces
                        }
                        new_chunks.append(new_chunk)
                        chunk_id_counter += 1

            chapter['chunks'] = new_chunks

        # Save updated project
        project_data['last_modified'] = datetime.now().isoformat()
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        return jsonify({
            'success': True,
            'rechunked_count': rechunked_count,
            'message': f'Successfully rechunked {rechunked_count} oversized chunks'
        })

    except Exception as e:
        import traceback
        print(f"Error auto-rechunking: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/split-chapter', methods=['POST'])
@auth_manager.require_api_key
def split_chapter():
    """Split a chapter at a specified position"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        chapter_id = data.get('chapter_id')
        split_position = data.get('split_position')  # Character position to split

        if not chapter_id or split_position is None:
            return jsonify({'error': 'chapter_id and split_position required'}), 400

        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'r') as f:
            project_data = json.load(f)

        # Find the chapter
        chapter_index = None
        for i, chapter in enumerate(project_data.get('chapters', [])):
            if chapter['id'] == chapter_id:
                chapter_index = i
                break

        if chapter_index is None:
            return jsonify({'error': 'Chapter not found'}), 404

        chapter = project_data['chapters'][chapter_index]

        # Reconstruct full chapter text from chunks
        full_text = '\n\n'.join(chunk['text'] for chunk in chapter.get('chunks', []))

        # Split the text
        text_part1 = full_text[:split_position].strip()
        text_part2 = full_text[split_position:].strip()

        # Create two new chapters
        new_chapter1 = {
            'id': str(uuid.uuid4()),
            'name': chapter.get('name', 'Chapter') + ' (Part 1)',
            'path': chapter.get('path', ''),
            'chunks': converter.smart_chunk_text(text_part1)
        }

        new_chapter2 = {
            'id': str(uuid.uuid4()),
            'name': chapter.get('name', 'Chapter') + ' (Part 2)',
            'path': '',
            'chunks': converter.smart_chunk_text(text_part2)
        }

        # Replace old chapter with new ones
        project_data['chapters'][chapter_index:chapter_index+1] = [new_chapter1, new_chapter2]

        # Save updated project
        project_data['last_modified'] = datetime.now().isoformat()
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        return jsonify({
            'success': True,
            'new_chapters': [new_chapter1['id'], new_chapter2['id']],
            'message': 'Chapter split successfully'
        })

    except Exception as e:
        import traceback
        print(f"Error splitting chapter: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/merge-chapters', methods=['POST'])
@auth_manager.require_api_key
def merge_chapters():
    """Merge two adjacent chapters"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        chapter_id1 = data.get('chapter_id1')
        chapter_id2 = data.get('chapter_id2')

        if not chapter_id1 or not chapter_id2:
            return jsonify({'error': 'chapter_id1 and chapter_id2 required'}), 400

        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        chapters = project_data.get('chapters', [])

        # Find both chapters
        chapter1_index = None
        chapter2_index = None

        for i, chapter in enumerate(chapters):
            if chapter['id'] == chapter_id1:
                chapter1_index = i
            if chapter['id'] == chapter_id2:
                chapter2_index = i

        if chapter1_index is None or chapter2_index is None:
            return jsonify({'error': 'One or both chapters not found'}), 404

        # Ensure they're adjacent
        if abs(chapter1_index - chapter2_index) != 1:
            return jsonify({'error': 'Chapters must be adjacent to merge'}), 400

        # Order them correctly
        first_index = min(chapter1_index, chapter2_index)
        second_index = max(chapter1_index, chapter2_index)

        chapter1 = chapters[first_index]
        chapter2 = chapters[second_index]

        # Merge the chunks
        merged_chunks = chapter1.get('chunks', []) + chapter2.get('chunks', [])

        # Renumber chunk IDs
        for i, chunk in enumerate(merged_chunks):
            chunk['id'] = i

        # Create merged chapter
        merged_chapter = {
            'id': str(uuid.uuid4()),
            'name': chapter1.get('name', 'Chapter') + ' + ' + chapter2.get('name', 'Chapter'),
            'path': chapter1.get('path', ''),
            'chunks': merged_chunks
        }

        # Replace the two chapters with the merged one
        chapters[first_index:second_index+1] = [merged_chapter]

        # Save updated project
        project_data['last_modified'] = datetime.now().isoformat()
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        return jsonify({
            'success': True,
            'merged_chapter_id': merged_chapter['id'],
            'message': 'Chapters merged successfully'
        })

    except Exception as e:
        import traceback
        print(f"Error merging chapters: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/save-xml', methods=['POST'])
@auth_manager.require_api_key
def save_xml_content():
    """Save XML content to project and update chunk data"""
    try:
        if converter.current_project_path is None:
            return jsonify({'error': 'No project loaded'}), 400

        data = request.json
        xml_content = data.get('xml_content')

        if not xml_content:
            return jsonify({'error': 'xml_content is required'}), 400

        # Parse XML to update chapter/chunk data
        chapters_updated = parse_xml_to_chapters(xml_content, converter.current_project_metadata.get('chapters', []))

        # Save XML content and updated chapters to metadata
        converter.current_project_metadata['content_xml'] = xml_content
        converter.current_project_metadata['chapters'] = chapters_updated
        converter.current_project_metadata['last_modified'] = datetime.now().isoformat()

        # Save to file
        project_file = os.path.join(converter.current_project_path, 'project.json')
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(converter.current_project_metadata, f, indent=2, ensure_ascii=False)

        # Invalidate lookup caches after modifying metadata
        converter._invalidate_lookup_caches()

        return jsonify({
            'success': True,
            'message': 'XML content saved successfully',
            'chapters_count': len(chapters_updated)
        })

    except Exception as e:
        import traceback
        print(f"Error saving XML content: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def parse_xml_to_chapters(xml_content, existing_chapters):
    """
    Parse XML content and update chapters with new chunk text.
    Preserves existing audio takes but updates chunk text.
    """
    import re

    # Create a map of existing chapters by title for matching
    existing_chapter_map = {}
    for chapter in existing_chapters:
        title = chapter.get('title') or chapter.get('name', '')
        existing_chapter_map[title] = chapter

    new_chapters = []

    # Parse chapter tags
    chapter_pattern = r'<chapter\s+title="([^"]*)">([\s\S]*?)</chapter>'
    non_voiced_pattern = r'<non-voiced\s+title="([^"]*)">([\s\S]*?)</non-voiced>'

    # Find all chapters and non-voiced sections
    all_sections = []

    for match in re.finditer(chapter_pattern, xml_content):
        all_sections.append({
            'type': 'chapter',
            'title': match.group(1),
            'content': match.group(2),
            'start': match.start()
        })

    for match in re.finditer(non_voiced_pattern, xml_content):
        all_sections.append({
            'type': 'non-voiced',
            'title': match.group(1),
            'content': match.group(2),
            'start': match.start()
        })

    # Sort by position in document
    all_sections.sort(key=lambda x: x['start'])

    for section in all_sections:
        title = section['title']
        content = section['content']
        is_non_voiced = section['type'] == 'non-voiced'

        # Get existing chapter if it exists
        existing_chapter = existing_chapter_map.get(title)

        # Parse chunks from content
        chunk_pattern = r'<chunk>([\s\S]*?)</chunk>'
        pause_pattern = r'<pause\s+duration="([^"]*)"\s*/>'
        common_file_pattern = r'<common_file\s+path="([^"]*)"\s*/>'

        new_chunks = []
        chunk_id = 0

        # Find all elements (chunks, pauses, common_files) and sort by position
        elements = []

        for match in re.finditer(chunk_pattern, content):
            elements.append({
                'type': 'text',
                'text': match.group(1).strip(),
                'start': match.start()
            })

        for match in re.finditer(pause_pattern, content):
            elements.append({
                'type': 'pause',
                'duration': float(match.group(1)),
                'start': match.start()
            })

        for match in re.finditer(common_file_pattern, content):
            elements.append({
                'type': 'common_file',
                'path': match.group(1),
                'start': match.start()
            })

        # Sort by position
        elements.sort(key=lambda x: x['start'])

        # Create chunk map from existing chapter
        existing_chunk_map = {}
        if existing_chapter:
            for i, chunk in enumerate(existing_chapter.get('chunks', [])):
                existing_chunk_map[i] = chunk

        # Process elements and create new chunk list
        for i, elem in enumerate(elements):
            if elem['type'] == 'text':
                # Try to find matching existing chunk
                existing_chunk = existing_chunk_map.get(i, {})

                # Check if text has changed
                old_text = existing_chunk.get('text', '')
                new_text = elem['text']

                # Create chunk with preserved audio data
                chunk = {
                    'id': existing_chunk.get('id', chunk_id),
                    'type': 'text',
                    'text': new_text,
                    'nickname': new_text[:50].strip() + ('...' if len(new_text) > 50 else ''),
                    'dirty': old_text != new_text and len(existing_chunk.get('generated_audios', [])) > 0,
                    'generated_audios': existing_chunk.get('generated_audios', [])
                }
                new_chunks.append(chunk)
                chunk_id = max(chunk_id, chunk['id']) + 1

            elif elem['type'] == 'pause':
                chunk = {
                    'id': chunk_id,
                    'type': 'pause',
                    'duration': elem['duration'],
                    'generated_audios': []
                }
                new_chunks.append(chunk)
                chunk_id += 1

            elif elem['type'] == 'common_file':
                chunk = {
                    'id': chunk_id,
                    'type': 'common_file',
                    'path': elem['path'],
                    'generated_audios': []
                }
                new_chunks.append(chunk)
                chunk_id += 1

        # Create new chapter
        new_chapter = {
            'id': existing_chapter.get('id') if existing_chapter else str(uuid.uuid4()),
            'title': title,
            'name': title,
            'non_voiced': is_non_voiced,
            'chunks': new_chunks
        }
        new_chapters.append(new_chapter)

    return new_chapters

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

@app.route('/index_old.html')
def serve_old_index():
    """Serve the old index page"""
    return send_file('index_old.html')

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
    print(f"\nReady for connections!")

    socketio.run(app, debug=config.DEBUG, port=config.PORT, host=config.HOST, allow_unsafe_werkzeug=True)
