"""
Gutenberg Text Annotator - Gradio UI

A user-friendly interface for annotating Project Gutenberg texts with AI.
Supports both local models (Ollama) and cloud models (Anthropic).
"""

import gradio as gr
import os
import sys
import json
from pathlib import Path
from typing import Optional, List

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from text_annotator import TextAnnotator, list_ollama_models
from gutenberg_processor import GutenbergProcessor


class AnnotatorApp:
    """Main application for text annotation"""

    def __init__(self):
        self.output_dir = "./annotated_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_annotator = None
        self.latest_output_file = None

    def get_available_models(self) -> List[str]:
        """Get list of available models (Ollama + Anthropic)"""
        models = []

        # Add Anthropic options
        models.append("🌐 Anthropic: Claude Sonnet 4.5")
        models.append("🌐 Anthropic: Claude Opus 4")

        # Get Ollama models
        try:
            ollama_models = list_ollama_models()
            for model in ollama_models:
                models.append(f"💻 Ollama: {model}")

            if not ollama_models:
                models.append("⚠️ No Ollama models found (run 'ollama pull llama3.2')")
        except Exception as e:
            models.append(f"⚠️ Ollama not available: {str(e)}")

        return models

    def parse_model_selection(self, selection: str) -> tuple:
        """
        Parse the model selection into (backend, model)

        Returns:
            (backend, model_name)
        """
        if "Anthropic" in selection:
            if "Sonnet" in selection:
                return ("anthropic", "claude-sonnet-4-5-20250929")
            elif "Opus" in selection:
                return ("anthropic", "claude-opus-4-20250514")
        elif "Ollama" in selection:
            # Extract model name after "Ollama: "
            model = selection.split("Ollama: ")[-1].strip()
            return ("ollama", model)

        # Default
        return ("ollama", "llama3.2")

    def process_gutenberg_url(
        self,
        url: str,
        progress=gr.Progress()
    ) -> tuple:
        """
        Download and process a Gutenberg URL

        Returns:
            (status_message, file_choices)
        """
        if not url:
            return "Please enter a Gutenberg URL", []

        try:
            progress(0.2, desc="Downloading from Project Gutenberg...")

            processor = GutenbergProcessor(self.output_dir)
            book_name, saved_files = processor.process_url(url)

            progress(1.0, desc="Complete!")

            # Return list of chapter files
            file_choices = [str(f) for f in saved_files]

            status = f"✓ Successfully processed '{book_name}'\n"
            status += f"  Created {len(saved_files)} chapter files\n"
            status += f"  Select a chapter below to annotate"

            return status, gr.update(choices=file_choices, value=file_choices[0] if file_choices else None)

        except Exception as e:
            return f"✗ Error: {str(e)}", gr.update(choices=[])

    def annotate_file(
        self,
        file_obj,
        model_selection: str,
        max_paragraphs: int,
        anthropic_api_key: str,
        progress=gr.Progress()
    ) -> tuple:
        """
        Annotate a text file

        Returns:
            (status_message, output_json_path, reader_html)
        """
        # Get file path
        if file_obj is None:
            return "Please select or upload a file", None, None

        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj

        if not os.path.exists(file_path):
            return f"File not found: {file_path}", None, None

        # Parse model selection
        backend, model = self.parse_model_selection(model_selection)

        # Check for Anthropic API key if needed
        if backend == "anthropic":
            if not anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
                return "⚠️ Anthropic API key required. Please enter it in the settings or set ANTHROPIC_API_KEY environment variable.", None, None

        try:
            progress(0.1, desc=f"Initializing {backend} ({model})...")

            # Read text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Create metadata
            filename = Path(file_path).stem
            metadata = {
                "title": filename.replace('_', ' ').title(),
                "source_file": file_path
            }

            # Create annotator
            annotator = TextAnnotator(
                backend=backend,
                model=model,
                api_key=anthropic_api_key if backend == "anthropic" else None
            )

            # Progress callback
            def update_progress(current, total, message):
                progress((0.1 + 0.8 * current / total), desc=message)

            # Annotate
            progress(0.1, desc="Starting annotation...")
            document = annotator.annotate_text(
                text,
                metadata,
                max_paragraphs=max_paragraphs if max_paragraphs > 0 else None,
                progress_callback=update_progress
            )

            # Save output
            output_filename = f"{filename}_annotated.json"
            output_path = os.path.join(self.output_dir, output_filename)

            progress(0.95, desc="Saving...")
            annotator.save_annotated_document(document, output_path)

            self.latest_output_file = output_path

            progress(1.0, desc="Complete!")

            # Create status message
            status = f"✓ Annotation complete!\n"
            status += f"  Backend: {backend}\n"
            status += f"  Model: {model}\n"
            status += f"  Paragraphs: {len(document['content'])}\n"
            status += f"  Annotations: {len(document['annotations_index'])}\n"
            status += f"  Output: {output_path}\n\n"
            status += f"Click 'Open in Reader' to view the annotated text."

            return status, output_path, self.get_reader_html()

        except Exception as e:
            import traceback
            error_msg = f"✗ Error during annotation:\n{str(e)}\n\n"
            error_msg += traceback.format_exc()
            return error_msg, None, None

    def get_reader_html(self) -> str:
        """Get the reader HTML page path"""
        reader_path = Path(__file__).parent / "reader.html"
        if reader_path.exists():
            return str(reader_path.absolute())
        return None

    def create_ui(self):
        """Create the Gradio interface"""

        with gr.Blocks(
            title="Gutenberg Text Annotator",
            theme=gr.themes.Soft()
        ) as demo:
            gr.Markdown("""
            # 📚 Gutenberg Text Annotator

            Automatically annotate historical texts with AI-powered context:
            - 📍 **Places**: Locations with coordinates and maps
            - 👤 **People**: Historical figures with biographies
            - 📖 **Topics**: Historical events and concepts
            - 🔤 **Terms**: Archaic words with definitions

            **Choose your LLM**: Use local models (Ollama) or cloud models (Anthropic)
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # Model Selection
                    gr.Markdown("### 🤖 Model Selection")
                    model_dropdown = gr.Dropdown(
                        choices=self.get_available_models(),
                        label="Select LLM Model",
                        value=self.get_available_models()[0] if self.get_available_models() else None,
                        interactive=True
                    )

                    refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")

                    # Anthropic API Key (only shown when needed)
                    anthropic_key_input = gr.Textbox(
                        label="Anthropic API Key (optional if using Anthropic)",
                        placeholder="sk-ant-...",
                        type="password",
                        value=os.environ.get("ANTHROPIC_API_KEY", "")
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    max_paragraphs_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=3,
                        step=1,
                        label="Max Paragraphs (0 = all)",
                        info="Limit paragraphs to process (for testing)"
                    )

            gr.Markdown("---")

            with gr.Tabs() as tabs:
                # Tab 1: Process from Gutenberg URL
                with gr.Tab("📥 From Gutenberg URL"):
                    gr.Markdown("""
                    Download and process a book directly from Project Gutenberg.

                    Example: `https://www.gutenberg.org/cache/epub/4932/pg4932.txt`
                    """)

                    gutenberg_url = gr.Textbox(
                        label="Gutenberg URL",
                        placeholder="https://www.gutenberg.org/cache/epub/XXXXX/pgXXXXX.txt"
                    )

                    download_btn = gr.Button("📥 Download & Process", variant="primary")

                    download_status = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False
                    )

                    chapter_dropdown = gr.Dropdown(
                        label="Select Chapter to Annotate",
                        choices=[],
                        interactive=True
                    )

                    annotate_chapter_btn = gr.Button("✨ Annotate Selected Chapter", variant="primary")

                # Tab 2: Upload Text File
                with gr.Tab("📤 Upload Text File"):
                    gr.Markdown("""
                    Upload a plain text file (.txt) to annotate.
                    """)

                    file_upload = gr.File(
                        label="Upload Text File",
                        file_types=[".txt"],
                        type="filepath"
                    )

                    annotate_upload_btn = gr.Button("✨ Annotate File", variant="primary")

            gr.Markdown("---")

            # Results Section
            gr.Markdown("### 📊 Results")

            annotation_status = gr.Textbox(
                label="Annotation Status",
                lines=8,
                interactive=False
            )

            with gr.Row():
                output_file_display = gr.Textbox(
                    label="Output File",
                    interactive=False
                )

                open_reader_btn = gr.Button("📖 Open in Reader", variant="secondary")

            # Hidden component to store reader path
            reader_html_path = gr.Textbox(visible=False)

            # Event Handlers

            # Refresh models
            def refresh_models():
                return gr.update(choices=self.get_available_models())

            refresh_models_btn.click(
                fn=refresh_models,
                outputs=[model_dropdown]
            )

            # Download from Gutenberg
            download_btn.click(
                fn=self.process_gutenberg_url,
                inputs=[gutenberg_url],
                outputs=[download_status, chapter_dropdown]
            )

            # Annotate chapter from Gutenberg
            def annotate_chapter_wrapper(chapter_path, model, max_para, api_key, progress=gr.Progress()):
                # Create a file-like object with chapter path
                class FileObj:
                    def __init__(self, path):
                        self.name = path

                return self.annotate_file(
                    FileObj(chapter_path),
                    model,
                    max_para,
                    api_key,
                    progress
                )

            annotate_chapter_btn.click(
                fn=annotate_chapter_wrapper,
                inputs=[
                    chapter_dropdown,
                    model_dropdown,
                    max_paragraphs_slider,
                    anthropic_key_input
                ],
                outputs=[annotation_status, output_file_display, reader_html_path]
            )

            # Annotate uploaded file
            annotate_upload_btn.click(
                fn=self.annotate_file,
                inputs=[
                    file_upload,
                    model_dropdown,
                    max_paragraphs_slider,
                    anthropic_key_input
                ],
                outputs=[annotation_status, output_file_display, reader_html_path]
            )

            # Open reader
            def open_reader(reader_path, output_file):
                if not reader_path or not output_file:
                    return "No file to open yet. Please annotate a file first."

                import webbrowser
                import urllib.parse

                # Open reader.html in browser
                # The user will then need to manually load the JSON file
                webbrowser.open(f"file://{reader_path}")

                return f"Opening reader.html in your browser.\n\nTo view the annotated text:\n1. Click 'Choose File' in the reader\n2. Select: {output_file}"

            open_reader_btn.click(
                fn=open_reader,
                inputs=[reader_html_path, output_file_display],
                outputs=[annotation_status]
            )

            # Footer
            gr.Markdown("""
            ---
            ### 💡 Tips
            - **Local Models**: Install Ollama and pull models with `ollama pull llama3.2`
            - **Cloud Models**: Get Anthropic API key from [console.anthropic.com](https://console.anthropic.com/)
            - **Testing**: Start with 3-5 paragraphs to test before processing full chapters
            - **Quality**: Larger models (70B+) produce better annotations but are slower

            ### 🔧 Model Recommendations
            - **Fast & Local**: `llama3.2` (3B)
            - **Balanced**: `llama3.1` (8B)
            - **Best Quality**: Anthropic Claude Sonnet 4.5
            """)

        return demo


def main():
    """Launch the UI"""
    app = AnnotatorApp()
    demo = app.create_ui()

    print("\n" + "="*70)
    print("🚀 Gutenberg Text Annotator UI")
    print("="*70)
    print("\nStarting Gradio interface...")
    print("\nMake sure:")
    print("  1. Ollama is running (if using local models): ollama serve")
    print("  2. You have models pulled: ollama pull llama3.2")
    print("  3. Or have ANTHROPIC_API_KEY set for cloud models")
    print("\nThe UI will open in your browser...")
    print("="*70 + "\n")

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
