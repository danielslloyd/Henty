import gradio as gr
import os
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta
import torch

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
                    txt_files.append(os.path.join(root, file))

        return sorted(txt_files)

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
            ta.save(output_path, wav, model.sr)
            return output_path
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None

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
                return None

            # Limit text length for demo purposes (adjust as needed)
            if len(text) > 1000:
                text = text[:1000] + "..."

            audio_path = self.generate_audio(text, audio_path)

        return audio_path

# Create converter instance
converter = TextToAudioConverter()

def update_file_list(directory):
    """Update the list of text files when directory changes"""
    if not directory:
        return gr.update(choices=[], value=None)

    txt_files = converter.find_txt_files(directory)
    if not txt_files:
        return gr.update(choices=[], value=None)

    # Show relative paths for cleaner display
    display_files = [os.path.relpath(f, directory) for f in txt_files]
    return gr.update(choices=list(zip(display_files, txt_files)), value=txt_files[0] if txt_files else None)

def display_file_content(file_path, directory):
    """Display the selected file's text content and audio"""
    if not file_path or not os.path.exists(file_path):
        return "No file selected", None

    # Read text content
    text_content = converter.read_text_file(file_path)

    # Get or generate audio
    audio_path = converter.get_or_generate_audio(file_path)

    return text_content, audio_path

def generate_all_audio(directory, progress=gr.Progress()):
    """Generate audio for all text files in the directory"""
    if not directory:
        return "Please select a directory first"

    txt_files = converter.find_txt_files(directory)
    if not txt_files:
        return "No text files found in the directory"

    total_files = len(txt_files)
    for i, txt_file in enumerate(txt_files):
        progress((i + 1) / total_files, desc=f"Processing {os.path.basename(txt_file)}...")
        converter.get_or_generate_audio(txt_file)

    return f"Successfully generated audio for {total_files} file(s)!"

# Create Gradio interface
with gr.Blocks(title="Text to Audio Converter - Chatterbox TTS") as demo:
    gr.Markdown("# Text to Audio Converter")
    gr.Markdown("Convert text files to audio using Chatterbox TTS from Hugging Face")

    with gr.Row():
        with gr.Column(scale=1):
            directory_input = gr.Textbox(
                label="Directory Path",
                placeholder="Enter the path to your directory containing .txt files",
                value=os.getcwd()
            )
            browse_btn = gr.Button("Browse Directory")

            file_dropdown = gr.Dropdown(
                label="Select Text File",
                choices=[],
                interactive=True
            )

            generate_all_btn = gr.Button("Generate Audio for All Files", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Text Content")
            text_display = gr.Textbox(
                label="",
                lines=20,
                max_lines=30,
                interactive=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### Generated Audio")
            audio_player = gr.Audio(
                label="",
                type="filepath"
            )

    # Event handlers
    directory_input.change(
        fn=update_file_list,
        inputs=[directory_input],
        outputs=[file_dropdown]
    )

    browse_btn.click(
        fn=update_file_list,
        inputs=[directory_input],
        outputs=[file_dropdown]
    )

    file_dropdown.change(
        fn=display_file_content,
        inputs=[file_dropdown, directory_input],
        outputs=[text_display, audio_player]
    )

    generate_all_btn.click(
        fn=generate_all_audio,
        inputs=[directory_input],
        outputs=[status_text]
    )

    gr.Markdown("---")
    gr.Markdown("**Note:** Audio files are saved in the `generated_audio` folder. The first generation may take longer as the model loads.")

if __name__ == "__main__":
    print("Starting Text to Audio Converter...")
    print(f"Using device: {converter.device}")
    demo.launch(share=False)
