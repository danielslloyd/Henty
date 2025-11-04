from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
import torch
import numpy as np
from scipy.io import wavfile

app = Flask(__name__)
CORS(app)

class TextToAudioConverter:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_dir = "generated_audio"
        self.voice_samples_dir = "voice_samples"
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.voice_samples_dir, exist_ok=True)

    def load_model(self):
        """Load the Chatterbox TTS model"""
        if self.model is None:
            print(f"Loading Chatterbox TTS model on {self.device}...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("Model loaded successfully!")
        return self.model

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

            # Use scipy.io.wavfile which is more reliable on Windows
            try:
                wavfile.write(output_path, model.sr, wav_int16)
                print(f"Successfully wrote WAV file to: {output_path}")
            except Exception as write_error:
                print(f"Error writing WAV file: {str(write_error)}")
                raise

            return output_path
        except Exception as e:
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

@app.route('/api/set-best-take', methods=['POST'])
def set_best_take():
    """Set or unset an audio file as the best take for a text file"""
    try:
        import json
        data = request.json
        txt_filename = data.get('txt_filename')
        audio_filename = data.get('audio_filename')

        if not txt_filename or not audio_filename:
            return jsonify({'error': 'txt_filename and audio_filename are required'}), 400

        base_name = os.path.splitext(txt_filename)[0]
        updated = False

        # Update all metadata files for this text file
        for filename in os.listdir(converter.audio_dir):
            if filename.startswith(base_name) and filename.endswith('.json'):
                metadata_path = os.path.join(converter.audio_dir, filename)

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

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

if __name__ == '__main__':
    print(f"Starting Text to Audio Converter API...")
    print(f"Using device: {converter.device}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, host='0.0.0.0')
