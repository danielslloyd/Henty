#!/usr/bin/env python3
"""
Demonstration of the complete Gutenberg Reader workflow

This script demonstrates how to:
1. Download a Gutenberg text
2. Process it into chapters
3. Annotate with AI
4. View in the HTML reader

Usage:
    python demo_workflow.py <gutenberg_url>

Example:
    python demo_workflow.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gutenberg_processor import GutenbergProcessor
from text_annotator import TextAnnotator


def print_step(step_num, message):
    """Print a formatted step message"""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {message}")
    print('='*70)


def demo_full_workflow(gutenberg_url, max_chapters=2):
    """
    Run the complete workflow on a Gutenberg text

    Args:
        gutenberg_url: URL to a Project Gutenberg .txt file
        max_chapters: Limit number of chapters to process (for demo)
    """
    print_step(1, "Downloading and Processing Gutenberg Text")

    # Create output directory
    output_dir = "./demo_output"
    os.makedirs(output_dir, exist_ok=True)

    # Download and process
    processor = GutenbergProcessor(output_dir)
    try:
        book_name, saved_files = processor.process_url(gutenberg_url)
        print(f"\n✓ Successfully processed '{book_name}'")
        print(f"  Created {len(saved_files)} chapter files")
    except Exception as e:
        print(f"\n✗ Error processing URL: {e}")
        return

    # Limit chapters for demo
    chapter_files = saved_files[:max_chapters]
    print(f"\n  Processing first {len(chapter_files)} chapters for demo...")

    print_step(2, "Annotating Text with AI")

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not found in environment")
        print("   Skipping annotation step.")
        print("\n   To enable annotations:")
        print("   1. Get API key from: https://console.anthropic.com/")
        print("   2. Set environment variable:")
        print("      export ANTHROPIC_API_KEY='your-key-here'")
        print("\n   You can still view the chapter files in the reader as plain text.")
        return

    # Annotate each chapter
    annotator = TextAnnotator(api_key)
    annotated_dir = os.path.join(output_dir, book_name + "_annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    for i, chapter_file in enumerate(chapter_files, 1):
        print(f"\n--- Annotating Chapter {i}/{len(chapter_files)} ---")

        # Read chapter text
        with open(chapter_file, 'r', encoding='utf-8') as f:
            chapter_text = f.read()

        # Create metadata
        chapter_name = Path(chapter_file).stem
        metadata = {
            "title": f"{book_name} - {chapter_name}",
            "author": "Unknown",  # Could extract from Gutenberg metadata
            "source_file": chapter_file,
            "chapter_number": i
        }

        # Annotate (limit paragraphs for demo)
        print(f"  Processing text (limited to 5 paragraphs for demo)...")
        annotated_doc = annotator.annotate_text(
            chapter_text,
            metadata,
            max_paragraphs=5  # Limit for demo to save time/cost
        )

        # Save annotated document
        output_file = os.path.join(annotated_dir, f"{chapter_name}_annotated.json")
        annotator.save_annotated_document(annotated_doc, output_file)

        print(f"  ✓ Saved to: {output_file}")

    print_step(3, "Viewing in HTML Reader")

    print(f"\nTo view the annotated documents:")
    print(f"  1. Open 'reader.html' in your web browser")
    print(f"  2. Click 'Choose File' button")
    print(f"  3. Select a file from: {annotated_dir}/")
    print(f"\nOr, if you have a local web server:")
    print(f"  python -m http.server 8000")
    print(f"  Then visit: http://localhost:8000/reader.html")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  Source chapters: {output_dir}/{book_name}/")
    print(f"  Annotated JSON: {annotated_dir}/")
    print(f"\nNext steps:")
    print(f"  - View in reader.html")
    print(f"  - Increase max_paragraphs for full chapter annotation")
    print(f"  - Process additional chapters")
    print(f"  - Customize annotation prompts in text_annotator.py")


def demo_simple_annotation(text_file):
    """
    Simple demo: annotate a single text file

    Args:
        text_file: Path to a .txt file
    """
    print_step(1, f"Annotating: {text_file}")

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n✗ ANTHROPIC_API_KEY not found in environment")
        print("  Please set your API key to run this demo.")
        return

    # Read text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create metadata
    filename = Path(text_file).stem
    metadata = {
        "title": filename.replace('_', ' ').title(),
        "source_file": text_file
    }

    # Annotate
    print("\nProcessing text with AI annotations...")
    annotator = TextAnnotator(api_key)
    annotated_doc = annotator.annotate_text(
        text,
        metadata,
        max_paragraphs=3  # Small demo
    )

    # Save
    output_file = text_file.replace('.txt', '_annotated.json')
    annotator.save_annotated_document(annotated_doc, output_file)

    print(f"\n✓ Annotation complete!")
    print(f"  Input: {text_file}")
    print(f"  Output: {output_file}")
    print(f"  Annotations: {len(annotated_doc['annotations_index'])}")
    print(f"\nOpen reader.html and load {output_file} to view.")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Gutenberg Reader Workflow Demo")
        print("="*70)
        print("\nUsage:")
        print("  Full workflow:  python demo_workflow.py <gutenberg_url>")
        print("  Single file:    python demo_workflow.py <text_file.txt>")
        print("\nExamples:")
        print("  python demo_workflow.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt")
        print("  python demo_workflow.py test_sample.txt")
        print("\nNote: Annotation requires ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    arg = sys.argv[1]

    # Check if argument is a URL or file
    if arg.startswith('http'):
        # Full workflow with Gutenberg URL
        demo_full_workflow(arg, max_chapters=2)
    elif os.path.isfile(arg):
        # Simple annotation of existing file
        demo_simple_annotation(arg)
    else:
        print(f"Error: '{arg}' is not a valid URL or file path")
        sys.exit(1)


if __name__ == '__main__':
    main()
