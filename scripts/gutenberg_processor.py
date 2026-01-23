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
from typing import List, Tuple, Optional
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("Warning: beautifulsoup4 not found. HTML processing will be limited.")


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
        Download text from a URL (supports .txt, .htm, .html files)

        Args:
            url: URL to download from

        Returns:
            Downloaded text content

        Raises:
            ValueError: If URL doesn't point to a valid text or HTML file
        """
        # Remove fragment identifier (#...) and query parameters before checking extension
        url_for_check = url.split('#')[0].split('?')[0].lower()

        # Verify URL points to a supported file type
        if not (url_for_check.endswith('.txt') or url_for_check.endswith('.htm') or url_for_check.endswith('.html')):
            raise ValueError(f"URL does not point to a .txt, .htm, or .html file: {url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Verify content-type if available
        content_type = response.headers.get('content-type', '')
        if content_type and not any(t in content_type.lower() for t in ['text', 'html']):
            raise ValueError(f"URL does not return text/html content: {content_type}")

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

    def is_html(self, text: str) -> bool:
        """
        Detect if the text content is HTML

        Args:
            text: Text to check

        Returns:
            True if content appears to be HTML
        """
        # Look for common HTML patterns
        html_patterns = [
            r'<!DOCTYPE\s+html',
            r'<html[>\s]',
            r'<head[>\s]',
            r'<body[>\s]',
            r'<div[>\s]',
            r'<p[>\s]'
        ]

        text_lower = text[:1000].lower()  # Check first 1000 chars
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in html_patterns)

    def extract_html_chapters(self, html_content: str, title: str) -> List[Tuple[str, str]]:
        """
        Extract chapters from Gutenberg HTML format

        Args:
            html_content: HTML content
            title: Book title

        Returns:
            List of tuples (chapter_title, chapter_text)
        """
        if not HAS_BS4:
            raise ValueError("beautifulsoup4 is required for HTML processing. Install with: pip install beautifulsoup4")

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove Gutenberg header and footer
        # Look for the main content div/body
        main_content = soup.find('body')
        if not main_content:
            main_content = soup

        chapters = []

        # Strategy 1: Look for chapter divisions with h2 or h3 headers
        # Gutenberg HTML files often use h2 for chapter headers
        chapter_headers = main_content.find_all(['h2', 'h3'])

        if chapter_headers:
            # Process content by chapter headers
            for i, header in enumerate(chapter_headers):
                chapter_title = header.get_text(strip=True)

                # Skip table of contents and front matter
                if any(skip in chapter_title.lower() for skip in ['contents', 'table of contents', 'toc', 'index']):
                    continue

                # Collect all content until the next header
                content_parts = []
                current = header.find_next_sibling()

                while current:
                    # Stop if we hit another chapter header
                    if current.name in ['h2', 'h3']:
                        break

                    # Get text from paragraph-level elements
                    if current.name in ['p', 'div']:
                        text = current.get_text(separator=' ', strip=True)
                        if text:
                            content_parts.append(text)

                    current = current.find_next_sibling()

                if content_parts:
                    chapter_text = '\n\n'.join(content_parts)
                    chapters.append((chapter_title, chapter_text))

        # Strategy 2: If no clear chapter structure, split by major divs or sections
        if not chapters:
            # Look for div elements with significant content
            divs = main_content.find_all('div', recursive=False)

            for i, div in enumerate(divs):
                # Get all paragraphs in this div
                paragraphs = div.find_all('p')
                if paragraphs:
                    text_parts = [p.get_text(separator=' ', strip=True) for p in paragraphs]
                    text_parts = [t for t in text_parts if t]  # Filter empty

                    if text_parts:
                        chapter_text = '\n\n'.join(text_parts)
                        # Use first few words as chapter title if no header
                        first_words = ' '.join(chapter_text.split()[:5])
                        chapter_title = f"Section {i+1}: {first_words}..."
                        chapters.append((chapter_title, chapter_text))

        # Strategy 3: If still no chapters, get all paragraphs as one chapter
        if not chapters:
            all_paragraphs = main_content.find_all('p')
            text_parts = [p.get_text(separator=' ', strip=True) for p in all_paragraphs]
            text_parts = [t for t in text_parts if t]  # Filter empty

            if text_parts:
                chapter_text = '\n\n'.join(text_parts)
                chapters.append((title, chapter_text))

        return chapters

    def process_carriage_returns(self, text: str) -> str:
        """
        Process carriage returns:
        - Remove single line breaks (within paragraphs)
        - Preserve double line breaks (between paragraphs)
        - Preserve 4+ consecutive line breaks as section breaks

        Args:
            text: Text to process

        Returns:
            Processed text
        """
        # Normalize all line endings to \n for easier processing
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # First, mark 4+ consecutive line breaks with a placeholder to preserve them as section breaks
        text = re.sub(r'\n{4,}', '<<<SECTION_BREAK>>>', text)

        # Mark double line breaks (paragraph breaks) with a temporary placeholder
        text = text.replace('\n\n', '<<<PARAGRAPH_BREAK>>>')

        # Now remove all remaining single line breaks (these are just wrapping within paragraphs)
        # Replace them with a space to join lines
        text = text.replace('\n', ' ')

        # Restore paragraph breaks as double newlines
        text = text.replace('<<<PARAGRAPH_BREAK>>>', '\n\n')

        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)

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
        Process a single Gutenberg URL (supports both .txt and .html files)

        Args:
            url: URL to process

        Returns:
            Tuple of (book_name, list of saved file paths)
        """
        # Download content
        print(f"Downloading: {url}")
        try:
            content = self.download_text(url)
        except ValueError as e:
            print(f"Skipping {url}: {e}")
            raise

        # Check if content is HTML
        is_html = self.is_html(content)
        print(f"Content type: {'HTML' if is_html else 'Plain text'}")

        if is_html:
            # Process HTML content
            # Try to extract title from HTML
            if HAS_BS4:
                soup = BeautifulSoup(content, 'html.parser')
                title_tag = soup.find('title')
                if title_tag:
                    # Clean up title (remove "The Project Gutenberg eBook of" etc)
                    title = title_tag.get_text(strip=True)
                    title = re.sub(r'The Project Gutenberg eBook of\s+', '', title, flags=re.IGNORECASE)
                    title = re.sub(r'\s+by\s+.+$', '', title)  # Remove author
                    title = re.sub(r'[^\w\s-]', '', title)
                    title = re.sub(r'\s+', '_', title)
                    title = title[:50]
                else:
                    title = None

            if not title:
                print(f"Warning: Could not extract title from HTML")
                title = self.extract_book_name(url)

            print(f"Processing HTML book: {title}")

            # Extract chapters from HTML
            chapters_data = self.extract_html_chapters(content, title)
            print(f"Found {len(chapters_data)} chapters from HTML structure")

            # Save chapters
            saved_files = []
            book_dir = os.path.join(self.output_dir, title)
            os.makedirs(book_dir, exist_ok=True)

            for i, (chapter_title, chapter_text) in enumerate(chapters_data, start=1):
                # Sanitize chapter title for filename
                sanitized = re.sub(r'[^\w\s-]', '', chapter_title)
                sanitized = re.sub(r'\s+', '_', sanitized)
                sanitized = sanitized.strip('_')[:50]  # Limit length

                if not sanitized:
                    sanitized = f"chapter_{i}"

                filepath = os.path.join(book_dir, f"{sanitized}.txt")

                # Handle duplicate filenames
                counter = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(book_dir, f"{sanitized}_{counter}.txt")
                    counter += 1

                # Write chapter with title header
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"{chapter_title}\n\n{chapter_text}")

                saved_files.append(filepath)

            print(f"Saved {len(saved_files)} chapters to {title}/")
            return title, saved_files

        else:
            # Process plain text content (original logic)
            # Extract title from the raw text (before stripping metadata)
            title = self.extract_title(content)

            if not title:
                print(f"Warning: Could not extract title from {url}")
                # Extract book ID from URL as fallback
                title = self.extract_book_name(url)

            print(f"Processing book: {title}")

            # Strip Gutenberg metadata and replace with title
            text = self.strip_gutenberg_metadata(content, title)

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

    if len(sys.argv) < 2:
        print("Usage: python gutenberg_processor.py <url1> [url2] ... [--output-dir <dir>]")
        print("\nExamples:")
        print("  python gutenberg_processor.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt")
        print("  python gutenberg_processor.py https://www.gutenberg.org/cache/epub/4932/pg4932.txt --output-dir ./my_books")
        print("\nDefaults to ./books if no output directory is specified")
        sys.exit(1)

    # Parse arguments
    output_dir = "./books"  # Default output directory
    urls = []

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--output-dir":
            if i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --output-dir requires a directory path")
                sys.exit(1)
        else:
            urls.append(sys.argv[i])
            i += 1

    if not urls:
        print("Error: At least one URL is required")
        sys.exit(1)

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
