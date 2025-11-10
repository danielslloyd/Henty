"""
Project Gutenberg Text Processor

This script downloads and processes Project Gutenberg texts:
1. Verifies URL points to a valid .txt file (skips if not)
2. Extracts title from "*** START OF THE PROJECT GUTENBERG EBOOK [title] ***"
3. Removes everything before and including that string, replaces with [title]
4. Removes everything after the END marker
5. Removes all single carriage returns (keeping only text and newlines)
6. Splits file whenever 4+ consecutive carriage returns are found
7. Saves sections to files named with first 40 characters in folder named [title]
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
        Download text from a URL and verify it's a valid txt file

        Args:
            url: URL to download from

        Returns:
            Downloaded text content

        Raises:
            ValueError: If URL doesn't point to a valid txt file
        """
        # Verify URL points to a .txt file
        if not url.lower().endswith('.txt'):
            raise ValueError(f"URL does not point to a .txt file: {url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Verify content-type if available
        content_type = response.headers.get('content-type', '')
        if content_type and 'text' not in content_type.lower():
            raise ValueError(f"URL does not return text content: {content_type}")

        return response.text

    def strip_gutenberg_metadata(self, text: str, title: str) -> str:
        """
        Remove everything before and including the START marker, replace with title.
        Remove everything after the END marker.

        Args:
            text: Raw text from Gutenberg
            title: Extracted title to place at the beginning

        Returns:
            Text with metadata removed and title at the start
        """
        # Find the START marker
        start_pattern = r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\s+.+?\s*\*\*\*'
        start_match = re.search(start_pattern, text, re.IGNORECASE)

        if start_match:
            # Find the end of the line containing the START marker
            start_pos = text.find('\n', start_match.end())
            if start_pos == -1:
                start_pos = start_match.end()
        else:
            # Fallback to old method if START marker not found
            start_pos = 0
            for marker in self.START_MARKERS:
                match = re.search(marker, text, re.IGNORECASE)
                if match:
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

        # Extract the content after the START marker and before the END marker
        content = text[start_pos:end_pos].strip()

        # Prepend the title to the content
        if title:
            content = f"{title}\n\n{content}"

        return content

    def extract_title(self, text: str) -> str:
        """
        Extract the book title from the START OF PROJECT GUTENBERG EBOOK marker

        Args:
            text: Text to search for title

        Returns:
            Sanitized title string or empty string if not found
        """
        # Look for: *** START OF THE PROJECT GUTENBERG EBOOK [TITLE] ***
        # The title is between "EBOOK" and the closing "***"
        pattern = r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\s+(.+?)\s*\*\*\*'

        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean up the title for use as directory name
            # Remove special characters, keep alphanumeric and spaces
            title = re.sub(r'[^\w\s-]', '', title)
            # Replace spaces with underscores
            title = re.sub(r'\s+', '_', title)
            # Limit length
            title = title[:50]
            if title:
                return title

        return ''

    def process_carriage_returns(self, text: str) -> str:
        """
        Process carriage returns:
        - Remove single carriage returns (not followed by another CR)
        - Preserve 4+ consecutive carriage returns as section breaks

        Args:
            text: Text to process

        Returns:
            Processed text
        """
        # First, mark 4+ consecutive CRs with a placeholder to preserve them
        # Pattern matches: \r\r\r\r or \r\n\r\n\r\n\r\n or mixed
        text = re.sub(r'(?:\r\n?){4,}', '<<<SECTION_BREAK>>>', text)

        # Now remove all remaining single carriage returns
        # This removes \r that's not part of a 4+ sequence
        text = text.replace('\r', '')

        # Return the text (section breaks will be split later)
        return text

    def split_by_section_breaks(self, text: str) -> List[str]:
        """
        Split text into sections based on the section break markers.

        Args:
            text: Text with section break markers

        Returns:
            List of text sections
        """
        # Split on the section break markers
        sections = text.split('<<<SECTION_BREAK>>>')

        # Filter out empty sections and strip whitespace
        sections = [section.strip() for section in sections if section.strip()]

        return sections

    def save_chapters(self, chapters: List[str], book_name: str) -> List[str]:
        """
        Save chapters to files named with first 40 characters of chapter text

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

        # Track used filenames to handle duplicates
        used_filenames = set()

        # Save each chapter
        for i, chapter in enumerate(chapters, start=1):
            # Get first 40 characters of chapter text
            first_chars = chapter[:40].strip()

            # Sanitize for filename: remove special characters, keep alphanumeric and spaces
            sanitized = re.sub(r'[^\w\s-]', '', first_chars)
            # Replace spaces with underscores
            sanitized = re.sub(r'\s+', '_', sanitized)
            # Remove leading/trailing underscores
            sanitized = sanitized.strip('_')

            # If sanitized name is empty, use chapter number
            if not sanitized:
                sanitized = f"chapter_{i}"

            # Handle duplicate filenames by adding number suffix
            filename = f"{sanitized}.txt"
            counter = 1
            while filename in used_filenames:
                filename = f"{sanitized}_{counter}.txt"
                counter += 1

            used_filenames.add(filename)
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
        # Download text (verifies it's a valid .txt file)
        print(f"Downloading: {url}")
        try:
            text = self.download_text(url)
        except ValueError as e:
            print(f"Skipping {url}: {e}")
            raise

        # Extract title from the raw text (before stripping metadata)
        title = self.extract_title(text)

        if not title:
            print(f"Warning: Could not extract title from {url}")
            # Extract book ID from URL as fallback
            title = self.extract_book_name(url)

        print(f"Processing book: {title}")

        # Strip Gutenberg metadata and replace with title
        text = self.strip_gutenberg_metadata(text, title)

        # Process carriage returns (remove singles, preserve 4+ as section breaks)
        text = self.process_carriage_returns(text)

        # Split into sections by section break markers
        sections = self.split_by_section_breaks(text)
        print(f"Found {len(sections)} sections")

        # Save sections (using title as the folder name)
        saved_files = self.save_chapters(sections, title)
        print(f"Saved {len(saved_files)} files to {title}/")

        return title, saved_files

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
