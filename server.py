from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
import torch
import soundfile as sf

app = Flask(__name__)
CORS(app)

class TextToAudioConverter:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_dir = "generated_audio"
        os.makedirs(self.audio_dir, exist_ok=True)

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

    def generate_audio(self, text, output_path):
        """Generate audio from text using Chatterbox TTS"""
        try:
            model = self.load_model()
            wav = model.generate(text)
            # Use soundfile directly to avoid torchcodec dependency
            # Convert tensor to numpy array for soundfile
            wav_numpy = wav.cpu().numpy() if wav.is_cuda else wav.numpy()
            sf.write(output_path, wav_numpy.T, model.sr)
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

        # Limit text length for demo purposes
        if len(text) > 1000:
            print(f"Truncating text from {len(text)} to 1000 characters")
            text = text[:1000] + "..."

        # Generate audio filename
        audio_filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"
        audio_path = os.path.join(converter.audio_dir, audio_filename)
        print(f"Audio will be saved to: {audio_path}")

        # Generate audio if it doesn't exist
        if not os.path.exists(audio_path):
            print(f"Generating audio for: {filename}...")
            print(f"Text preview: {text[:100]}...")
            converter.generate_audio(text, audio_path)
            print(f"Audio generated successfully!")
        else:
            print(f"Audio already exists, using cached version")

        response_data = {
            'audio_url': f'/api/audio/{audio_filename}',
            'audio_path': audio_path
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

if __name__ == '__main__':
    print(f"Starting Text to Audio Converter API...")
    print(f"Using device: {converter.device}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, host='0.0.0.0')
