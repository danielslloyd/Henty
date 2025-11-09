"""
Project Gutenberg Text Processor

This script downloads and processes Project Gutenberg texts:
1. Downloads text from URLs
2. Strips Gutenberg headers/footers
3. Extracts book title from content (within *** markers)
4. Processes line breaks (removes single, keeps multiple)
5. Splits by chapter breaks (3+ line breaks)
6. Saves to sequentially numbered files in title-based subdirectories
"""

import os
import re
import requests
from typing import List, Tuple


class GutenbergProcessor:
    """Process Project Gutenberg text files"""

    # Common Gutenberg header/footer markers
    START_MARKERS = [
        r'\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG',
        r'\*{3,}\s*START OF',
        r'START OF (?:THIS|THE) PROJECT GUTENBERG'
    ]

    END_MARKERS = [
        r'\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG',
        r'\*{3,}\s*END OF',
        r'END OF (?:THIS|THE) PROJECT GUTENBERG'
    ]

    def __init__(self, output_dir: str):
        """
        Initialize the processor

        Args:
            output_dir: Directory to save processed files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_text(self, url: str) -> str:
        """
        Download text from a URL

        Args:
            url: URL to download from

        Returns:
            Downloaded text content
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def strip_gutenberg_metadata(self, text: str) -> str:
        """
        Remove Project Gutenberg headers and footers

        Args:
            text: Raw text from Gutenberg

        Returns:
            Text with metadata removed
        """
        # Find start marker
        start_pos = 0
        for marker in self.START_MARKERS:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                # Find the end of the line containing the marker
                start_pos = text.find('\n', match.end())
                if start_pos == -1:
                    start_pos = match.end()
                break

        # Find end marker
        end_pos = len(text)
        for marker in self.END_MARKERS:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                end_pos = match.start()
                break

        # Extract the content
        content = text[start_pos:end_pos].strip()
        return content

    def extract_title(self, text: str) -> str:
        """
        Extract the book title from text, typically found within *** markers

        Args:
            text: Text to search for title

        Returns:
            Sanitized title string or empty string if not found
        """
        # Look for text between *** markers (common Gutenberg title format)
        # Pattern: *** TITLE *** or ***TITLE***
        patterns = [
            r'\*{3,}\s*([A-Z][^\*\n]{3,80}?)\s*\*{3,}',  # *** TITLE ***
            r'\*{3,}\s*([^\*\n]{3,80}?)\s*\*{3,}',       # More flexible version
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text[:2000], re.MULTILINE)  # Search first 2000 chars
            if matches:
                title = matches[0].strip()
                # Clean up the title for use as directory name
                # Remove common prefixes
                title = re.sub(r'^(THE|A|AN)\s+', '', title, flags=re.IGNORECASE)
                # Remove special characters, keep alphanumeric and spaces
                title = re.sub(r'[^\w\s-]', '', title)
                # Replace spaces with underscores
                title = re.sub(r'\s+', '_', title)
                # Limit length
                title = title[:50]
                if title:
                    return title

        return ''

    def process_line_breaks(self, text: str) -> str:
        """
        Process line breaks:
        - Remove single line breaks (join paragraphs)
        - Keep multiple line breaks (paragraph/chapter separators)

        Args:
            text: Text to process

        Returns:
            Text with processed line breaks
        """
        # Replace 3+ line breaks with a placeholder to preserve chapter breaks
        text = re.sub(r'\n{3,}', '<<<CHAPTER_BREAK>>>', text)

        # Replace 2 line breaks with a placeholder to preserve paragraph breaks
        text = re.sub(r'\n{2}', '<<<PARA_BREAK>>>', text)

        # Remove single line breaks (join continued lines)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Restore paragraph breaks (2 line breaks)
        text = text.replace('<<<PARA_BREAK>>>', '\n\n')

        # Restore chapter breaks (3+ line breaks become 3)
        text = text.replace('<<<CHAPTER_BREAK>>>', '\n\n\n')

        return text

    def split_by_chapters(self, text: str) -> List[str]:
        """
        Split text into chapters based on 3+ consecutive line breaks

        Args:
            text: Text to split

        Returns:
            List of chapter texts
        """
        # Split on 3+ line breaks
        chapters = re.split(r'\n{3,}', text)

        # Filter out empty chapters and strip whitespace
        chapters = [ch.strip() for ch in chapters if ch.strip()]

        return chapters

    def save_chapters(self, chapters: List[str], book_name: str) -> List[str]:
        """
        Save chapters to numbered files

        Args:
            chapters: List of chapter texts
            book_name: Base name for the book

        Returns:
            List of saved file paths
        """
        saved_files = []

        # Create subdirectory for this book
        book_dir = os.path.join(self.output_dir, book_name)
        os.makedirs(book_dir, exist_ok=True)

        # Save each chapter
        for i, chapter in enumerate(chapters, start=1):
            filename = f"{i:03d}.txt"
            filepath = os.path.join(book_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(chapter)

            saved_files.append(filepath)

        return saved_files

    def extract_book_name(self, url: str) -> str:
        """
        Extract a book name from the URL

        Args:
            url: Gutenberg URL

        Returns:
            Book name (e.g., "pg4932")
        """
        # Extract the file name from URL (e.g., pg4932.txt)
        match = re.search(r'pg(\d+)', url)
        if match:
            return f"pg{match.group(1)}"

        # Fallback: use last part of URL
        parts = url.rstrip('/').split('/')
        name = parts[-1].replace('.txt', '')
        return name if name else 'book'

    def process_url(self, url: str) -> Tuple[str, List[str]]:
        """
        Process a single Gutenberg URL

        Args:
            url: URL to process

        Returns:
            Tuple of (book_name, list of saved file paths)
        """
        # Download text
        print(f"Downloading: {url}")
        text = self.download_text(url)

        # Extract book ID from URL
        book_id = self.extract_book_name(url)

        # Strip Gutenberg metadata
        text = self.strip_gutenberg_metadata(text)

        # Extract title from the text content
        title = self.extract_title(text)

        # Use title if found, otherwise fallback to book ID
        book_name = title if title else book_id
        print(f"Processing book: {book_name} ({book_id})")

        # Process line breaks
        text = self.process_line_breaks(text)

        # Split into chapters
        chapters = self.split_by_chapters(text)
        print(f"Found {len(chapters)} chapters")

        # Save chapters
        saved_files = self.save_chapters(chapters, book_name)
        print(f"Saved {len(saved_files)} files to {book_name}/")

        return book_name, saved_files

    def process_urls(self, urls: List[str]) -> dict:
        """
        Process multiple Gutenberg URLs

        Args:
            urls: List of URLs to process

        Returns:
            Dictionary mapping book names to their saved files
        """
        results = {}

        for url in urls:
            try:
                book_name, saved_files = self.process_url(url)
                results[book_name] = {
                    'url': url,
                    'files': saved_files,
                    'count': len(saved_files)
                }
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results[url] = {
                    'error': str(e)
                }

        return results


def main():
    """Command-line interface for testing"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python gutenberg_processor.py <output_dir> <url1> [url2] ...")
        print("\nExample:")
        print("  python gutenberg_processor.py ./books https://www.gutenberg.org/cache/epub/4932/pg4932.txt")
        sys.exit(1)

    output_dir = sys.argv[1]
    urls = sys.argv[2:]

    processor = GutenbergProcessor(output_dir)
    results = processor.process_urls(urls)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)

    for book_name, info in results.items():
        if 'error' in info:
            print(f"\n❌ {book_name}: ERROR - {info['error']}")
        else:
            print(f"\n✓ {book_name}:")
            print(f"  URL: {info['url']}")
            print(f"  Chapters: {info['count']}")
            print(f"  Directory: {output_dir}/{book_name}/")


if __name__ == '__main__':
    main()
