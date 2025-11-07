from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

class TextToAudioConverter:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_dir = "generated_audio"
        self.voice_samples_dir = "voice_samples"
        self.stats_file = "generation_stats.json"
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.voice_samples_dir, exist_ok=True)

        # Generation tracking
        self.current_generation = None
        self.generation_lock = threading.Lock()

        # Load existing stats or initialize
        self.generation_stats = self.load_stats()

        # Project management
        self.current_project_path = None
        self.current_project_metadata = None

    def load_model(self):
        """Load the Chatterbox TTS model"""
        if self.model is None:
            print(f"Loading Chatterbox TTS model on {self.device}...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("Model loaded successfully!")
        return self.model

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
        """Get current GPU memory usage"""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                gpu_utilization = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else None
                return {
                    'memory_allocated_gb': round(gpu_mem_allocated, 2),
                    'memory_reserved_gb': round(gpu_mem_reserved, 2),
                    'utilization_percent': gpu_utilization
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

    def smart_chunk_text(self, text, max_chunk_size=500):
        """
        Smart text chunking that respects paragraph breaks, quotations, and sentence boundaries.
        Returns a list of dicts with chunk metadata.
        """
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
                    # Split by sentences
                    sentences = re.split(r'([.!?]+\s+|[.!?]+$)', para)
                    sentence_chunk = ""
                    sentence_start = current_pos

                    for i in range(0, len(sentences), 2):
                        sentence = sentences[i]
                        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
                        full_sentence = sentence + punctuation

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
        Returns the path to the stitched audio file.
        """
        try:
            if not audio_paths:
                raise ValueError("No audio files provided for stitching")

            # Load all audio files
            combined = None
            for audio_path in audio_paths:
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue

                audio_segment = AudioSegment.from_wav(audio_path)

                if combined is None:
                    combined = audio_segment
                else:
                    # Add a small pause between chunks (100ms)
                    silence = AudioSegment.silent(duration=100)
                    combined = combined + silence + audio_segment

            if combined is None:
                raise ValueError("No valid audio files found to stitch")

            # Export the combined audio
            combined.export(output_path, format="wav")
            print(f"Successfully stitched {len(audio_paths)} audio files to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error stitching audio files: {str(e)}")
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

    def generate_audio(self, text, output_path, audio_prompt_path=None, language_id="en", exaggeration=0.5, cfg_weight=0.5):
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

            print(f"Generating with parameters: {gen_params}")
            print(f"Text length: {char_count} characters")

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

            # Clear current generation
            with self.generation_lock:
                self.current_generation = None

            return output_path
        except Exception as e:
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

            audio_path = self.generate_audio(text, audio_path)

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
            converter.generate_audio(
                text,
                audio_path,
                audio_prompt_path=audio_prompt_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            print(f"Audio generated successfully!")

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

@app.route('/api/status')
def status():
    """Check server status"""
    return jsonify({
        'status': 'running',
        'device': converter.device,
        'model_loaded': converter.model is not None
    })

@app.route('/api/voice-samples', methods=['GET'])
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
def create_project():
    """Create a new project in a user-selected folder"""
    try:
        import json
        import shutil
        from datetime import datetime

        data = request.json
        project_path = data.get('project_path')
        project_name = data.get('project_name', 'Unnamed Project')

        if not project_path:
            return jsonify({'error': 'project_path is required'}), 400

        # Create project directory structure
        os.makedirs(project_path, exist_ok=True)
        texts_dir = os.path.join(project_path, 'texts')
        audio_dir = os.path.join(project_path, 'audio')

        os.makedirs(texts_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        # Create project metadata
        project_metadata = {
            'name': project_name,
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'version': '1.0'
        }

        # Save project metadata
        project_file = os.path.join(project_path, 'project.json')
        with open(project_file, 'w') as f:
            json.dump(project_metadata, f, indent=2)

        # Update converter paths
        converter.current_project_path = project_path
        converter.current_project_metadata = project_metadata
        converter.audio_dir = audio_dir
        # Keep voice_samples_dir pointing to main folder (not project-specific)

        return jsonify({
            'success': True,
            'project_path': project_path,
            'metadata': project_metadata
        })

    except Exception as e:
        import traceback
        print(f"Error creating project: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/load', methods=['POST'])
def load_project():
    """Load an existing project from a folder"""
    try:
        import json
        from datetime import datetime

        data = request.json
        project_path = data.get('project_path')

        if not project_path:
            return jsonify({'error': 'project_path is required'}), 400

        if not os.path.exists(project_path):
            return jsonify({'error': 'Project path does not exist'}), 404

        # Load project metadata
        project_file = os.path.join(project_path, 'project.json')
        if not os.path.exists(project_file):
            return jsonify({'error': 'Not a valid project folder (project.json not found)'}), 400

        with open(project_file, 'r') as f:
            project_metadata = json.load(f)

        # Update last modified
        project_metadata['last_modified'] = datetime.now().isoformat()
        with open(project_file, 'w') as f:
            json.dump(project_metadata, f, indent=2)

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

        return jsonify({
            'success': True,
            'project_path': project_path,
            'metadata': project_metadata,
            'text_files': text_files
        })

    except Exception as e:
        import traceback
        print(f"Error loading project: {str(e)}")
        print(traceback.format_exc())
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

        return jsonify({'success': True, 'file_path': file_path})

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
        data = request.json
        txt_filename = data.get('txt_filename')
        audio_filename = data.get('audio_filename')

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
        max_chunk_size = data.get('max_chunk_size', 500)

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
            'text_preview': chunk_text[:200],
            'is_best_take': False
        }

        # Generate audio
        print(f"Generating audio for chunk {chunk_id}...")
        converter.generate_audio(
            chunk_text,
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
                    'in_progress': False
                })

            current = converter.current_generation.copy()

        elapsed_time = time.time() - current['start_time']
        estimated = current.get('estimated_time')

        progress_data = {
            'in_progress': True,
            'char_count': current['char_count'],
            'elapsed_ms': int(elapsed_time * 1000),
            'elapsed_seconds': round(elapsed_time, 1)
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

if __name__ == '__main__':
    print(f"Starting Text to Audio Converter API...")
    print(f"Using device: {converter.device}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, host='0.0.0.0')
