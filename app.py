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

    def validate_txt_file(self, file_path):
        """Validate that the file is a .txt file"""
        if not file_path:
            return False
        return file_path.endswith('.txt') and os.path.isfile(file_path)

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

def display_file_content(file_obj):
    """Display the selected file's text content and audio"""
    if not file_obj:
        return "No file selected", None

    file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj

    if not os.path.exists(file_path):
        return "File not found", None

    # Validate it's a txt file
    if not converter.validate_txt_file(file_path):
        return "Please select a valid .txt file", None

    # Read text content
    text_content = converter.read_text_file(file_path)

    # Get or generate audio
    audio_path = converter.get_or_generate_audio(file_path)

    return text_content, audio_path

def generate_audio_from_file(file_obj, progress=gr.Progress()):
    """Generate audio for the selected text file"""
    if not file_obj:
        return "Please select a file first"

    file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj

    if not converter.validate_txt_file(file_path):
        return "Please select a valid .txt file"

    progress(0.5, desc=f"Processing {os.path.basename(file_path)}...")
    converter.get_or_generate_audio(file_path)

    return f"Successfully generated audio for {os.path.basename(file_path)}!"

# Create Gradio interface
with gr.Blocks(title="Text to Audio Converter - Chatterbox TTS") as demo:
    gr.Markdown("# Text to Audio Converter")
    gr.Markdown("Convert text files to audio using Chatterbox TTS from Hugging Face")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Select Text File",
                file_types=[".txt"],
                type="filepath"
            )

            generate_btn = gr.Button("Generate Audio", variant="primary")
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
    file_input.change(
        fn=display_file_content,
        inputs=[file_input],
        outputs=[text_display, audio_player]
    )

    generate_btn.click(
        fn=generate_audio_from_file,
        inputs=[file_input],
        outputs=[status_text]
    )

    gr.Markdown("---")
    gr.Markdown("**Note:** Audio files are saved in the `generated_audio` folder. The first generation may take longer as the model loads.")

if __name__ == "__main__":
    print("Starting Text to Audio Converter...")
    print(f"Using device: {converter.device}")
    demo.launch(share=False)
