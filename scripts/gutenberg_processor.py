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

    def get_gutenberg_book_id(self, url: str) -> Optional[str]:
        """Extract numeric book ID from any Gutenberg URL format."""
        # Covers /ebooks/120, /epub/120/pg120.txt, pg120.txt, etc.
        for pattern in [r'/(?:ebooks|epub)/(\d+)', r'pg(\d+)']:
            m = re.search(pattern, url)
            if m:
                return m.group(1)
        return None

    def get_epub_url(self, book_id: str) -> str:
        """Canonical no-images EPUB URL."""
        return f"https://www.gutenberg.org/ebooks/{book_id}.epub.noimages"

    def get_plain_text_url(self, book_id: str) -> str:
        """Canonical plain-text cache URL."""
        return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

    def download_epub(self, book_id: str) -> Optional[bytes]:
        """Download the no-images EPUB; returns None on failure."""
        url = self.get_epub_url(book_id)
        try:
            print(f"Downloading EPUB: {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"EPUB download failed ({e})")
            return None

    def get_epub_chapter_titles(self, epub_bytes: bytes) -> List[str]:
        """
        Extract ordered chapter titles from an EPUB's NCX or nav.xhtml.
        Prefers HTML heading text (h1/h2/h3) over NCX labels since headings
        more closely match plain-text chapter headings.
        Returns a flat list of title strings (front/back matter skipped).
        """
        import zipfile, io
        if not HAS_BS4:
            print("DEBUG [EPUB]: beautifulsoup4 not available, skipping EPUB parsing")
            return []

        skip_kw = {'contents', 'table of contents', 'cover', 'copyright',
                   'title page', 'half-title', 'dedication', 'epigraph',
                   'acknowledgement', 'index', 'bibliography', 'about the'}
        titles: List[str] = []
        print(f"DEBUG [EPUB]: Processing EPUB ({len(epub_bytes):,} bytes)")

        with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
            names = set(zf.namelist())
            print(f"DEBUG [EPUB]: {len(names)} files in archive")

            # Locate OPF
            opf_path = None
            try:
                c = zf.read('META-INF/container.xml').decode('utf-8', errors='replace')
                rf = BeautifulSoup(c, 'html.parser').find('rootfile')
                if rf:
                    opf_path = rf.get('full-path')
            except Exception as e:
                print(f"DEBUG [EPUB]: container.xml error: {e}")
            opf_path = opf_path or next((n for n in names if n.endswith('.opf')), None)
            if not opf_path:
                print("DEBUG [EPUB]: No OPF file found")
                return []
            print(f"DEBUG [EPUB]: OPF at '{opf_path}'")

            opf_dir = os.path.dirname(opf_path)
            opf_soup = BeautifulSoup(
                zf.read(opf_path).decode('utf-8', errors='replace'), 'html.parser'
            )

            def abs_path(href: str) -> str:
                """Resolve href (strip fragment) relative to OPF directory."""
                href = href.split('#')[0]
                if opf_dir:
                    return opf_dir.rstrip('/') + '/' + href.lstrip('/')
                return href

            manifest = {
                item.get('id', ''): {
                    'path': abs_path(item.get('href', '')),
                    'href': item.get('href', ''),
                    'media_type': item.get('media-type', ''),
                    'properties': item.get('properties', ''),
                }
                for item in opf_soup.find_all('item')
            }

            def get_html_heading(file_path: str) -> str:
                """Return first h1/h2/h3 text from an HTML file inside the EPUB."""
                try:
                    if file_path not in names:
                        return ''
                    html = zf.read(file_path).decode('utf-8', errors='replace')
                    soup = BeautifulSoup(html, 'html.parser')
                    for tag in ['h1', 'h2', 'h3']:
                        h = soup.find(tag)
                        if h:
                            text = h.get_text(strip=True)
                            if text:
                                return text
                except Exception as e:
                    print(f"DEBUG [EPUB]: heading error in '{file_path}': {e}")
                return ''

            def should_skip(label: str) -> bool:
                return any(kw in label.lower() for kw in skip_kw)

            def add_entry(ncx_label: str, html_file: str = '') -> None:
                ncx_label = ncx_label.strip()
                if not ncx_label or should_skip(ncx_label):
                    return
                if html_file:
                    heading = get_html_heading(html_file)
                    if heading and not should_skip(heading):
                        print(f"DEBUG [EPUB]: title '{heading}' (HTML heading; NCX='{ncx_label}')")
                        titles.append(heading)
                        return
                print(f"DEBUG [EPUB]: title '{ncx_label}' (NCX label; no HTML heading found)")
                titles.append(ncx_label)

            # Try NCX (EPUB 2)
            ncx_item = opf_soup.find('item', attrs={'media-type': 'application/x-dtbncx+xml'})
            if ncx_item:
                ncx_path = manifest.get(ncx_item.get('id', ''), {}).get('path', '')
                try:
                    ncx_soup = BeautifulSoup(
                        zf.read(ncx_path).decode('utf-8', errors='replace'), 'html.parser'
                    )
                    nav_points = ncx_soup.find_all('navPoint')
                    print(f"DEBUG [EPUB]: {len(nav_points)} NCX navPoints")
                    for pt in nav_points:
                        lbl = pt.find('navLabel')
                        content = pt.find('content')
                        label_text = lbl.get_text(strip=True) if lbl else ''
                        src = content.get('src', '') if content else ''
                        html_file = abs_path(src) if src else ''
                        add_entry(label_text, html_file)
                    if titles:
                        print(f"DEBUG [EPUB]: {len(titles)} titles extracted via NCX")
                        return titles
                except Exception as e:
                    print(f"DEBUG [EPUB]: NCX parse error: {e}")

            # Try nav.xhtml (EPUB 3)
            nav_item = next(
                (v for v in manifest.values() if 'nav' in v.get('properties', '')), None
            )
            if nav_item:
                try:
                    nav_soup = BeautifulSoup(
                        zf.read(nav_item['path']).decode('utf-8', errors='replace'),
                        'html.parser'
                    )
                    nav_el = (nav_soup.find('nav', attrs={'epub:type': 'toc'})
                              or nav_soup.find('nav'))
                    if nav_el:
                        for a in nav_el.find_all('a'):
                            label_text = a.get_text(strip=True)
                            href = a.get('href', '')
                            html_file = abs_path(href) if href else ''
                            add_entry(label_text, html_file)
                    print(f"DEBUG [EPUB]: {len(titles)} titles extracted via nav.xhtml")
                except Exception as e:
                    print(f"DEBUG [EPUB]: nav.xhtml parse error: {e}")

        return titles

    def split_text_by_chapter_titles(
        self, text: str, chapter_titles: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Split stripped plain text at positions matching EPUB chapter titles.
        Tries multiple strategies per title: exact → normalized → chapter prefix.
        Returns [(title, chapter_text), ...] for titles that matched.
        """

        def normalize(s: str) -> str:
            """Lowercase, strip punctuation, collapse whitespace."""
            s = s.lower()
            s = re.sub(r'[^\w\s]', ' ', s)
            return re.sub(r'\s+', ' ', s).strip()

        def try_exact(title: str) -> Optional[re.Match]:
            """Case-insensitive standalone-line match with optional trailing punct."""
            escaped = re.escape(title)
            pattern = r'(?m)^[ \t]*' + escaped + r'[.!?]?[ \t]*$'
            return re.search(pattern, text, re.IGNORECASE)

        def try_normalized(title: str) -> Optional[re.Match]:
            """Normalize title and match with flexible whitespace between words."""
            words = normalize(title).split()
            if not words:
                return None
            pattern = r'(?m)^[ \t]*' + r'[\s.]*'.join(re.escape(w) for w in words) + r'[.!?]?[ \t]*$'
            return re.search(pattern, text, re.IGNORECASE)

        def try_chapter_prefix(title: str) -> Optional[re.Match]:
            """
            If title starts with 'Chapter/Part/Section/Book N', try matching
            just that prefix as a standalone line (handles subtitles in NCX).
            """
            m = re.match(
                r'^((?:chapter|part|section|book|volume)\s+(?:\d+|[ivxlcdmIVXLCDM]+))',
                title, re.IGNORECASE
            )
            if m:
                escaped = re.escape(m.group(1))
                pattern = r'(?m)^[ \t]*' + escaped + r'[.!?]?[ \t]*$'
                return re.search(pattern, text, re.IGNORECASE)
            return None

        def try_split_heading(title: str) -> Optional[re.Match]:
            """
            Handle EPUB titles like "Chapter I: The King Maker" where the plain text
            has the chapter number on one line and the subtitle on the next, e.g.:
                CHAPTER I.
                THE KING MAKER
            Tries to match the chapter-number line and validates the subtitle follows
            within a few lines.
            """
            # Parse "Chapter N: Subtitle" or "Chapter N—Subtitle"
            m = re.match(
                r'^((?:chapter|part|section|book|volume)\s+(?:\d+|[ivxlcdmIVXLCDM]+))'
                r'[\s:.\-—]+(.+)$',
                title, re.IGNORECASE
            )
            if not m:
                return None
            prefix, subtitle = m.group(1).strip(), m.group(2).strip()
            escaped_prefix = re.escape(prefix)
            escaped_subtitle = re.escape(subtitle)
            # Match chapter-number line followed within ~5 lines by subtitle
            pattern = (
                r'(?m)^[ \t]*' + escaped_prefix + r'[.!?]?[ \t]*\n'
                r'(?:[ \t]*\n){0,4}'                 # up to 4 blank lines
                r'[ \t]*' + escaped_subtitle + r'[.!?]?[ \t]*$'
            )
            return re.search(pattern, text, re.IGNORECASE)

        def try_subtitle_only(title: str) -> Optional[re.Match]:
            """
            For titles that look like 'Chapter I: The King Maker', try matching
            just the subtitle part as a standalone ALL-CAPS line.
            Common in G.A. Henty plain texts where subtitle is its own line.
            """
            m = re.match(
                r'^(?:chapter|part|section|book|volume)\s+(?:\d+|[ivxlcdmIVXLCDM]+)'
                r'[\s:.\-—]+(.+)$',
                title, re.IGNORECASE
            )
            if not m:
                return None
            subtitle = m.group(1).strip()
            # Only try if subtitle is reasonably short (not a whole sentence)
            if len(subtitle) > 60:
                return None
            escaped = re.escape(subtitle)
            pattern = r'(?m)^[ \t]*' + escaped + r'[.!?]?[ \t]*$'
            return re.search(pattern, text, re.IGNORECASE)

        print(f"\nDEBUG [SPLIT]: Matching {len(chapter_titles)} titles in text ({len(text):,} chars)")
        # Show first 10 lines of plain text for orientation
        first_lines = text[:500].split('\n')
        for i, line in enumerate(first_lines[:10]):
            print(f"DEBUG [SPLIT]: text line {i}: {repr(line[:80])}")

        splits: List[Tuple[int, int, str]] = []
        used_positions: set = set()

        for title in chapter_titles:
            match = None
            strategy = None

            match = try_exact(title)
            if match:
                strategy = "exact"

            if not match:
                match = try_normalized(title)
                if match:
                    strategy = "normalized"

            if not match:
                match = try_chapter_prefix(title)
                if match:
                    strategy = "chapter-prefix"

            if not match:
                match = try_split_heading(title)
                if match:
                    strategy = "split-heading"

            if not match:
                match = try_subtitle_only(title)
                if match:
                    strategy = "subtitle-only"

            if match:
                pos = match.start()
                snippet = repr(text[max(0, pos-20):pos+60])
                if pos not in used_positions:
                    used_positions.add(pos)
                    splits.append((pos, match.end(), title))
                    print(f"DEBUG [SPLIT]: ✓ '{title}' → {strategy}, pos={pos}, text={snippet}")
                else:
                    print(f"DEBUG [SPLIT]: ✗ '{title}' → pos={pos} already claimed (duplicate)")
            else:
                # Show nearby lines to diagnose mismatches
                norm_title = normalize(title)
                words = norm_title.split()
                print(f"DEBUG [SPLIT]: ✗ '{title}' → NOT FOUND (tried exact/normalized/prefix/split/subtitle)")
                # Show up to 3 lines containing the first significant word
                for word in words[:3]:
                    if len(word) < 3:
                        continue
                    for m2 in list(re.finditer(r'(?m)^.*' + re.escape(word) + r'.*$', text, re.IGNORECASE))[:2]:
                        ctx_start = max(0, m2.start() - 40)
                        ctx_end = min(len(text), m2.end() + 40)
                        print(f"DEBUG [SPLIT]:   near '{word}': {repr(text[ctx_start:ctx_end])}")
                    break

        if not splits:
            print("DEBUG [SPLIT]: 0 titles matched → caller should fall back to detect_chapters()")
            return []

        splits.sort(key=lambda x: x[0])

        chapters: List[Tuple[str, str]] = []
        for i, (_, content_start, title) in enumerate(splits):
            next_start = splits[i + 1][0] if i + 1 < len(splits) else len(text)
            chapter_text = text[content_start:next_start].strip()
            if chapter_text:
                chapters.append((title, chapter_text))

        print(f"DEBUG [SPLIT]: {len(chapters)} non-empty chapters generated")
        return chapters

    # ------------------------------------------------------------------ #
    #  Five Chapter Parsing Methods (selectable by user)                  #
    # ------------------------------------------------------------------ #

    PARSING_METHODS = {
        'epub_toc': 'EPUB Table of Contents — uses NCX/nav chapter titles from the EPUB to locate chapter boundaries in the plain text',
        'epub_spine_html': 'EPUB Spine + HTML Headings — walks the EPUB spine order reading h1-h3 headings and body text from each content file',
        'regex_headings': 'Regex Heading Detection — scans plain text for "Chapter N", Roman numerals, "Part N", ALL-CAPS headings, etc.',
        'blank_line_sections': 'Blank-Line Section Breaks — splits on 3+ consecutive blank lines (common Gutenberg section separator)',
        'hybrid_epub_regex': 'Hybrid EPUB + Regex — tries EPUB TOC first; fills gaps and validates with regex heading detection',
    }

    def parse_chapters(self, text: str, epub_bytes: Optional[bytes],
                       method: str, log: Optional[list] = None) -> List[Tuple[str, str]]:
        """
        Unified entry point.  Dispatches to one of five methods.
        *log* is an accumulator list — each call appends verbose log strings.
        Returns [(title, chapter_body), ...].
        """
        if log is None:
            log = []

        log.append(f"[PARSE] Method selected: {method}")
        log.append(f"[PARSE] Text length: {len(text):,} chars")
        log.append(f"[PARSE] EPUB available: {epub_bytes is not None} ({len(epub_bytes):,} bytes)" if epub_bytes else "[PARSE] EPUB available: False")

        dispatch = {
            'epub_toc':          self._method_epub_toc,
            'epub_spine_html':   self._method_epub_spine_html,
            'regex_headings':    self._method_regex_headings,
            'blank_line_sections': self._method_blank_line_sections,
            'hybrid_epub_regex': self._method_hybrid_epub_regex,
        }

        fn = dispatch.get(method)
        if fn is None:
            log.append(f"[PARSE] ERROR: Unknown method '{method}', falling back to regex_headings")
            fn = self._method_regex_headings

        result = fn(text, epub_bytes, log)
        log.append(f"[PARSE] Final result: {len(result)} chapters")
        for i, (title, body) in enumerate(result):
            log.append(f"[PARSE]   ch{i}: \"{title}\" — {len(body):,} chars, first 80: {repr(body[:80])}")
        return result

    # ---------- Method 1: EPUB TOC ------------------------------------ #

    def _method_epub_toc(self, text: str, epub_bytes: Optional[bytes],
                         log: list) -> List[Tuple[str, str]]:
        """Use EPUB NCX/nav titles and match them in the plain text."""
        log.append("[M1-EPUB_TOC] Starting EPUB TOC method")

        if not epub_bytes:
            log.append("[M1-EPUB_TOC] No EPUB data — returning empty (try another method)")
            return []

        titles = self.get_epub_chapter_titles(epub_bytes)
        log.append(f"[M1-EPUB_TOC] EPUB titles extracted: {len(titles)}")
        for i, t in enumerate(titles):
            log.append(f"[M1-EPUB_TOC]   title[{i}]: {repr(t)}")

        if not titles:
            log.append("[M1-EPUB_TOC] No titles found in EPUB — returning empty")
            return []

        chapters = self.split_text_by_chapter_titles(text, titles)
        log.append(f"[M1-EPUB_TOC] Matched {len(chapters)} of {len(titles)} titles in plain text")
        return chapters

    # ---------- Method 2: EPUB Spine + HTML Headings ------------------- #

    def _method_epub_spine_html(self, text: str, epub_bytes: Optional[bytes],
                                log: list) -> List[Tuple[str, str]]:
        """Walk the EPUB spine to get chapter titles/order, then cut the plain text at matching positions."""
        import zipfile, io
        log.append("[M2-EPUB_SPINE] Starting EPUB Spine + HTML method")
        log.append("[M2-EPUB_SPINE] Strategy: extract headings from EPUB HTML, use them to split plain text")

        if not epub_bytes:
            log.append("[M2-EPUB_SPINE] No EPUB data — returning empty")
            return []
        if not HAS_BS4:
            log.append("[M2-EPUB_SPINE] beautifulsoup4 not available — returning empty")
            return []

        skip_kw = {'contents', 'table of contents', 'cover', 'copyright',
                    'title page', 'half-title', 'dedication', 'epigraph',
                    'acknowledgement', 'index', 'bibliography', 'about the',
                    'project gutenberg'}

        # Step 1: Walk the EPUB spine and collect headings from each HTML file
        headings: List[str] = []

        with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
            names = set(zf.namelist())
            log.append(f"[M2-EPUB_SPINE] {len(names)} files in EPUB zip")

            # Locate OPF
            opf_path = None
            try:
                c = zf.read('META-INF/container.xml').decode('utf-8', errors='replace')
                rf = BeautifulSoup(c, 'html.parser').find('rootfile')
                if rf:
                    opf_path = rf.get('full-path')
            except Exception:
                pass
            opf_path = opf_path or next((n for n in names if n.endswith('.opf')), None)
            if not opf_path:
                log.append("[M2-EPUB_SPINE] No OPF — returning empty")
                return []

            opf_dir = os.path.dirname(opf_path)
            opf_soup = BeautifulSoup(zf.read(opf_path).decode('utf-8', errors='replace'), 'html.parser')

            def abs_path(href):
                href = href.split('#')[0]
                if opf_dir:
                    return opf_dir.rstrip('/') + '/' + href.lstrip('/')
                return href

            # Build manifest lookup
            manifest = {}
            for item in opf_soup.find_all('item'):
                manifest[item.get('id', '')] = abs_path(item.get('href', ''))

            # Read spine order
            spine_ids = [ref.get('idref', '') for ref in opf_soup.find_all('itemref')]
            log.append(f"[M2-EPUB_SPINE] Spine has {len(spine_ids)} items")

            for idx, idref in enumerate(spine_ids):
                fpath = manifest.get(idref, '')
                if not fpath or fpath not in names:
                    log.append(f"[M2-EPUB_SPINE]   spine[{idx}] id={idref} — file not found, skipping")
                    continue

                html = zf.read(fpath).decode('utf-8', errors='replace')
                soup = BeautifulSoup(html, 'html.parser')

                # Extract heading
                heading = ''
                for tag in ['h1', 'h2', 'h3']:
                    h = soup.find(tag)
                    if h:
                        heading = h.get_text(strip=True)
                        break

                # Check if this spine item has meaningful body content
                body_len = len(soup.get_text(strip=True))

                if not heading or body_len < 50:
                    log.append(f"[M2-EPUB_SPINE]   spine[{idx}] \"{heading or '(no heading)'}\" — too short ({body_len} chars), skipping")
                    continue

                if any(kw in heading.lower() for kw in skip_kw):
                    log.append(f"[M2-EPUB_SPINE]   spine[{idx}] \"{heading}\" — skip keyword match, skipping")
                    continue

                headings.append(heading)
                log.append(f"[M2-EPUB_SPINE]   spine[{idx}] heading: \"{heading}\"")

        log.append(f"[M2-EPUB_SPINE] Collected {len(headings)} headings from EPUB spine")

        if not headings:
            log.append("[M2-EPUB_SPINE] No headings found — returning empty")
            return []

        # Step 2: Use these headings to split the plain text (same logic as epub_toc)
        log.append("[M2-EPUB_SPINE] Matching headings against plain text...")
        chapters = self.split_text_by_chapter_titles(text, headings)
        log.append(f"[M2-EPUB_SPINE] Matched {len(chapters)} of {len(headings)} headings in plain text")
        return chapters

    # ---------- Method 3: Regex Heading Detection ---------------------- #

    def _method_regex_headings(self, text: str, epub_bytes: Optional[bytes],
                               log: list) -> List[Tuple[str, str]]:
        """Scan plain text for common chapter heading patterns."""
        log.append("[M3-REGEX] Starting regex heading detection")

        patterns = [
            (r'(?m)^[ \t]*(CHAPTER\s+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY)[^\n]*)', 'CHAPTER_KEYWORD'),
            (r'(?m)^[ \t]*(Chapter\s+(?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty)[^\n]*)', 'Chapter_keyword'),
            (r'(?m)^[ \t]*((?:Book|Part|Volume)\s+(?:[IVXLCDM]+|\d+|the\s+\w+)[^\n]*)', 'Book_Part_Volume'),
            (r'(?m)^[ \t]*([A-Z][A-Z ]{4,60})[ \t]*$', 'ALL_CAPS_LINE'),
            (r'(?m)^[ \t]*([IVXLCDM]{1,10})\.\s*$', 'Roman_numeral_dot'),
        ]

        all_matches = []
        for pat, label in patterns:
            matches = list(re.finditer(pat, text))
            log.append(f"[M3-REGEX] Pattern '{label}': {len(matches)} matches")
            for m in matches[:5]:  # log first 5
                log.append(f"[M3-REGEX]   pos={m.start()} text={repr(m.group(1).strip()[:60])}")
            if len(matches) > 5:
                log.append(f"[M3-REGEX]   ... and {len(matches) - 5} more")
            for m in matches:
                all_matches.append((m.start(), m.end(), m.group(1).strip(), label))

        # Deduplicate by position (keep first match at each position)
        all_matches.sort(key=lambda x: x[0])
        deduped = []
        last_pos = -1
        for pos, end, title, label in all_matches:
            if pos > last_pos + 10:  # allow 10 char gap for near-duplicates
                deduped.append((pos, end, title, label))
                last_pos = pos

        # Filter out ALL_CAPS lines that are likely not chapter headings
        # (single words, very short, or too many matches indicating body text)
        caps_count = sum(1 for _, _, _, lbl in deduped if lbl == 'ALL_CAPS_LINE')
        other_count = len(deduped) - caps_count
        if other_count > 2 and caps_count > other_count * 3:
            log.append(f"[M3-REGEX] ALL_CAPS overrepresented ({caps_count} vs {other_count} others) — removing ALL_CAPS matches")
            deduped = [(p, e, t, l) for p, e, t, l in deduped if l != 'ALL_CAPS_LINE']

        log.append(f"[M3-REGEX] {len(deduped)} heading positions after dedup")

        if not deduped:
            log.append("[M3-REGEX] No headings found — returning entire text as one chapter")
            return [('Complete Text', text)]

        chapters: List[Tuple[str, str]] = []

        # Pre-chapter text
        if deduped[0][0] > 100:
            pre = text[:deduped[0][0]].strip()
            if pre:
                chapters.append(('[Preface]', pre))
                log.append(f"[M3-REGEX] Pre-chapter text: {len(pre):,} chars")

        for i, (pos, end, title, label) in enumerate(deduped):
            # Skip to after the title line
            content_start = text.find('\n', end)
            if content_start == -1:
                content_start = end
            else:
                content_start += 1

            content_end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
            body = text[content_start:content_end].strip()

            if body:
                chapters.append((title, body))

        log.append(f"[M3-REGEX] Produced {len(chapters)} chapters")
        return chapters

    # ---------- Method 4: Blank-Line Section Breaks -------------------- #

    def _method_blank_line_sections(self, text: str, epub_bytes: Optional[bytes],
                                    log: list) -> List[Tuple[str, str]]:
        """Split text wherever 3+ consecutive blank lines appear."""
        log.append("[M4-BLANK_LINES] Starting blank-line section splitting")

        # Try 4+ newlines first (Gutenberg standard), fall back to 3+
        sections_4 = re.split(r'\n{4,}', text)
        sections_4 = [s.strip() for s in sections_4 if s.strip()]
        log.append(f"[M4-BLANK_LINES] Split on 4+ newlines: {len(sections_4)} sections")

        sections_3 = re.split(r'\n{3,}', text)
        sections_3 = [s.strip() for s in sections_3 if s.strip()]
        log.append(f"[M4-BLANK_LINES] Split on 3+ newlines: {len(sections_3)} sections")

        # Use 4+ unless it produces only 1 section
        sections = sections_4 if len(sections_4) > 1 else sections_3
        threshold = '4+' if len(sections_4) > 1 else '3+'
        log.append(f"[M4-BLANK_LINES] Using {threshold} threshold: {len(sections)} sections")

        if len(sections) <= 1:
            log.append("[M4-BLANK_LINES] Only 1 section — returning as single chapter")
            return [('Complete Text', text)]

        chapters: List[Tuple[str, str]] = []
        for i, section in enumerate(sections):
            # Use first line as title (up to 80 chars)
            lines = section.split('\n', 1)
            first_line = lines[0].strip()[:80]
            # If first line looks like a heading (short, possibly caps), use it
            if len(first_line) < 80 and len(first_line) > 0:
                title = first_line
            else:
                title = f"Section {i + 1}"
            chapters.append((title, section))
            log.append(f"[M4-BLANK_LINES]   section[{i}]: \"{title[:50]}\" — {len(section):,} chars")

        log.append(f"[M4-BLANK_LINES] Produced {len(chapters)} chapters")
        return chapters

    # ---------- Method 5: Hybrid EPUB + Regex -------------------------- #

    def _method_hybrid_epub_regex(self, text: str, epub_bytes: Optional[bytes],
                                  log: list) -> List[Tuple[str, str]]:
        """
        Best-effort hybrid: try EPUB TOC first, validate/supplement with regex.
        If EPUB produces chapters, verify that regex headings roughly align.
        If EPUB misses chapters, use regex to fill gaps.
        """
        log.append("[M5-HYBRID] Starting hybrid EPUB + Regex method")

        # Step 1: try EPUB TOC
        epub_chapters = []
        if epub_bytes:
            epub_log = []
            epub_chapters = self._method_epub_toc(text, epub_bytes, epub_log)
            for line in epub_log:
                log.append(line.replace('[M1-EPUB_TOC]', '[M5-HYBRID/epub]'))

        # Step 2: get regex headings
        regex_log = []
        regex_chapters = self._method_regex_headings(text, epub_bytes, regex_log)
        for line in regex_log:
            log.append(line.replace('[M3-REGEX]', '[M5-HYBRID/regex]'))

        log.append(f"[M5-HYBRID] EPUB produced {len(epub_chapters)} chapters, regex produced {len(regex_chapters)} chapters")

        # Decision logic
        if not epub_chapters and not regex_chapters:
            log.append("[M5-HYBRID] Both methods returned nothing — single chapter fallback")
            return [('Complete Text', text)]

        if not epub_chapters:
            log.append("[M5-HYBRID] EPUB returned nothing — using regex results")
            return regex_chapters

        if not regex_chapters or len(regex_chapters) <= 1:
            log.append("[M5-HYBRID] Regex returned nothing useful — using EPUB results")
            return epub_chapters

        # Both returned results — pick the one with more granularity,
        # but prefer EPUB if counts are close (EPUB titles are usually better)
        epub_count = len(epub_chapters)
        regex_count = len(regex_chapters)

        if regex_count > epub_count * 1.5:
            log.append(f"[M5-HYBRID] Regex found significantly more chapters ({regex_count} vs {epub_count}) — using regex")
            return regex_chapters
        else:
            log.append(f"[M5-HYBRID] Using EPUB results ({epub_count} chapters; regex had {regex_count})")
            return epub_chapters

    def download_text(self, url: str) -> str:
        """Download text from a URL (plain-text or HTML)."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
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
