#!/usr/bin/env python3
"""
Batch import all books from BOOKS_DIR without the UI.
Creates projects and chunks for each book.json found.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import uuid as _uuid

# Add current dir to path to import config
sys.path.insert(0, os.path.dirname(__file__))

from config import config


class SimpleTextChunker:
    """Minimal text chunker to mimic server.py's smart_chunk_text"""
    def __init__(self, max_chunk_size=500):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text):
        """Split text into chunks by sentences/words"""
        if len(text) <= self.max_chunk_size:
            return [{'text': text}]

        chunks = []
        current = ""
        sentences = text.replace('!', '.').replace('?', '.').split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            test = current + sentence + ". "
            if len(test) <= self.max_chunk_size:
                current = test
            else:
                if current:
                    chunks.append({'text': current.rstrip()})
                current = sentence + ". "

        if current:
            chunks.append({'text': current.rstrip()})

        return chunks if chunks else [{'text': text}]


def build_chapters_from_blocks(blocks, variant='original', chunker=None):
    """Convert book.json blocks into chapter/chunk structures (same logic as server.py)"""
    if chunker is None:
        chunker = SimpleTextChunker()

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
        pieces = chunker.chunk(text)
        for piece in pieces:
            ptext = piece['text']
            chunk = {
                'id': len(current['chunks']),
                'type': 'text',
                'text': ptext,
                'original_text': text,
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
                chunk['source_block_id'] = block.get('id')
            current['chunks'].append(chunk)

    for block in blocks:
        btype = block.get('type')
        if btype == 'heading':
            text = block_text(block)
            if text:
                start_chapter(text)
        elif btype in ('para', 'verse'):
            text = block_text(block)
            add_content(text, block)

    return chapters


def has_existing_audio(book_path):
    """Check if this book already has generated audio"""
    audio_dir = os.path.join(book_path, 'audio')
    if os.path.isdir(audio_dir):
        try:
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            return len(audio_files) > 0
        except:
            pass
    return False


def import_book(book_path, variant='original'):
    """Import a single book (same logic as /api/project/import-book)"""
    # Skip if already has audio
    if has_existing_audio(book_path):
        print(f"  ⊘ skipped (audio already exists)")
        return True

    book_json = os.path.join(book_path, 'book.json')

    if not os.path.isfile(book_json):
        print(f"  ✗ book.json not found")
        return False

    try:
        with open(book_json, 'r', encoding='utf-8') as f:
            book = json.load(f)

        blocks = book.get('blocks', [])
        if not blocks:
            print(f"  ✗ No blocks in book.json")
            return False

        title = book.get('title') or os.path.basename(book_path)
        chunker = SimpleTextChunker(config.MAX_CHUNK_SIZE)
        chapters = build_chapters_from_blocks(blocks, variant, chunker)

        if not chapters:
            print(f"  ✗ No chapters built from blocks")
            return False

        # Create audio/texts directories
        audio_dir = os.path.join(book_path, 'audio')
        texts_dir = os.path.join(book_path, 'texts')
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(texts_dir, exist_ok=True)

        # Create or update project.json
        project_file = os.path.join(book_path, 'project.json')

        if os.path.isfile(project_file):
            with open(project_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'name': title,
                'created_at': datetime.now().isoformat(),
                'default_audio_settings': {
                    'exaggeration': 0.6,
                    'cfg_weight': 0.4,
                    'voice_sample': 'none',
                    'seed': 0,
                    'temperature': 0.8,
                    'ref_vad_trimming': False
                },
            }

        metadata['name'] = metadata.get('name') or title
        metadata['last_modified'] = datetime.now().isoformat()
        metadata['version'] = '3.0'
        metadata['source'] = 'book_json'
        metadata['book_path'] = book_path
        metadata['text_variant'] = variant
        metadata['original_filename'] = f"{title}.json"
        metadata['chapters'] = chapters
        metadata['chapters_locked'] = True

        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Write chapters snapshot
        try:
            parts = []
            for ch in chapters:
                parts.append(f"## {ch['title']}")
                for c in ch['chunks'][1:]:
                    if c.get('type') == 'text':
                        parts.append(c['text'])
            with open(os.path.join(book_path, 'chapters_original.txt'), 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(parts))
        except Exception as e:
            print(f"  [warning] Could not write chapters snapshot: {e}")

        total_chunks = sum(
            len([c for c in ch['chunks'] if c.get('type') == 'text']) for ch in chapters
        )

        print(f"  ✓ {len(chapters)} chapters, {total_chunks} chunks")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Import book.json project(s) into Henty (build project.json)."
    )
    parser.add_argument(
        '--folder',
        help="Import only this single book folder (must contain book.json). "
             "Default: import every folder in BOOKS_DIR."
    )
    parser.add_argument(
        '--variant', default='original', choices=['original', 'rewrite'],
        help="Which block text to voice (default: original)."
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  Batch Import Books")
    print("="*70 + "\n")

    # Single-folder mode: import just the given book (one-article pipeline).
    if args.folder:
        folder_path = os.path.abspath(args.folder)
        if not os.path.isfile(os.path.join(folder_path, 'book.json')):
            print(f"ERROR: no book.json in {folder_path}")
            sys.exit(1)
        print(f"Importing single folder: {folder_path}\n")
        ok = import_book(folder_path, variant=args.variant)
        print("\n" + "="*70)
        print(f"Complete: {'1 imported' if ok else '1 failed'}")
        print("="*70 + "\n")
        sys.exit(0 if ok else 1)

    if not config.BOOKS_DIR:
        print("ERROR: BOOKS_DIR not configured in .env or config.py")
        sys.exit(1)

    if not os.path.isdir(config.BOOKS_DIR):
        print(f"ERROR: BOOKS_DIR does not exist: {config.BOOKS_DIR}")
        sys.exit(1)

    print(f"BOOKS_DIR: {config.BOOKS_DIR}\n")

    # Find all folders with book.json
    book_folders = []
    try:
        for entry in os.listdir(config.BOOKS_DIR):
            folder_path = os.path.join(config.BOOKS_DIR, entry)
            if os.path.isdir(folder_path):
                book_json = os.path.join(folder_path, 'book.json')
                if os.path.isfile(book_json):
                    book_folders.append((entry, folder_path))
    except Exception as e:
        print(f"ERROR reading BOOKS_DIR: {e}")
        sys.exit(1)

    if not book_folders:
        print("No books found (no folders with book.json)")
        sys.exit(0)

    print(f"Found {len(book_folders)} book(s):\n")

    imported = 0
    failed = 0

    for folder_name, folder_path in sorted(book_folders):
        print(f"Importing: {folder_name}")
        if import_book(folder_path, variant=args.variant):
            imported += 1
        else:
            failed += 1

    print("\n" + "="*70)
    print(f"Complete: {imported} imported, {failed} failed")
    print("="*70 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
