"""
Text Annotator using LLM (Anthropic or Ollama)

This script processes plain text and generates rich annotations for:
- Places (with coordinates and descriptions)
- Historical figures and topics (with bios/summaries)
- Archaic terminology (with definitions)

All annotations include sources and links.

Supports both cloud (Anthropic) and local (Ollama) LLMs.
"""

import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class TextAnnotator:
    """Annotate text using LLM for educational annotations"""

    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize the annotator

        Args:
            backend: "anthropic" or "ollama"
            model: Model name (e.g., "llama3.2", "claude-sonnet-4-5-20250929")
            api_key: Anthropic API key (only for anthropic backend)
            ollama_host: Ollama server URL
        """
        self.backend = backend.lower()
        self.ollama_host = ollama_host

        if self.backend == "anthropic":
            import anthropic
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set for Anthropic backend")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = model or "claude-sonnet-4-5-20250929"

        elif self.backend == "ollama":
            try:
                import ollama
                self.ollama = ollama
            except ImportError:
                raise ImportError("Please install ollama: pip install ollama")

            # Set the model
            self.model = model or "llama3.2"

            # Test connection
            try:
                self.ollama.list(host=ollama_host)
            except Exception as e:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {ollama_host}. "
                    f"Make sure Ollama is running: ollama serve\n"
                    f"Error: {e}"
                )

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'anthropic' or 'ollama'")

    def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """
        Call the LLM backend (Anthropic or Ollama)

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        if self.backend == "anthropic":
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        elif self.backend == "ollama":
            response = self.ollama.chat(
                model=self.model,
                host=self.ollama_host,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            )
            return response['message']['content']

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines
        paragraphs = text.split('\n\n')
        # Clean and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def identify_entities(self, paragraph: str) -> Dict[str, Any]:
        """
        Use LLM to identify places, people, topics, and archaic terms in a paragraph

        Returns a structured dictionary with all identified entities
        """
        prompt = f"""Analyze this historical text paragraph and identify:

1. PLACES: Any geographic locations (cities, regions, countries, landmarks)
2. PEOPLE: Historical figures, notable persons
3. TOPICS: Historical events, periods, or concepts that need explanation
4. ARCHAIC TERMS: Old-fashioned words, phrases, or terms that modern readers might not understand

For each entity, provide:
- The exact text as it appears in the paragraph
- The entity type (place/person/topic/term)
- A brief one-sentence summary/definition
- For places: estimated latitude/longitude coordinates
- For people: birth/death years if known
- For archaic terms: modern equivalent or definition

Return ONLY a JSON object with this structure:
{{
  "entities": [
    {{
      "text": "exact text from paragraph",
      "type": "place|person|topic|term",
      "summary": "brief one-sentence description",
      "lat": 12.34,
      "lon": 56.78,
      "born": "year",
      "died": "year",
      "modern_equivalent": "modern term",
      "sources": ["Wikipedia: Article Name"]
    }}
  ]
}}

Paragraph to analyze:
{paragraph}

Return only the JSON object, no other text."""

        try:
            response_text = self._call_llm(prompt)

            # Try to extract JSON if wrapped in markdown
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            result = json.loads(response_text)
            return result

        except Exception as e:
            print(f"Error identifying entities: {e}")
            print(f"Response was: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
            return {"entities": []}

    def identify_entities_batch(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """
        Batch version: Identify entities in multiple paragraphs with a single LLM call

        Returns a list of results, one for each paragraph (in the same order)
        """
        if not paragraphs:
            return []

        # Build batch prompt
        prompt = f"""Analyze these {len(paragraphs)} historical text paragraphs and identify entities in each.

For EACH paragraph, identify:
1. PLACES: Any geographic locations (cities, regions, countries, landmarks)
2. PEOPLE: Historical figures, notable persons
3. TOPICS: Historical events, periods, or concepts that need explanation
4. ARCHAIC TERMS: Old-fashioned words, phrases, or terms that modern readers might not understand

For each entity, provide:
-- The exact text as it appears in the paragraph
-- The entity type (place/person/topic/term)
-- A brief one-sentence summary/definition
-- For places: estimated latitude/longitude coordinates
-- For people: birth/death years if known
-- For archaic terms: modern equivalent or definition

Return ONLY a JSON array with one object per paragraph:
[
  {{
    "paragraph_index": 0,
    "entities": [
      {{
        "text": "exact text from paragraph",
        "type": "place|person|topic|term",
        "summary": "brief one-sentence description",
        "lat": 12.34,
        "lon": 56.78,
        "born": "year",
        "died": "year",
        "modern_equivalent": "modern term",
        "sources": ["Wikipedia: Article Name"]
      }}
    ]
  }},
  ...
]

Paragraphs to analyze:
"""

        for i, paragraph in enumerate(paragraphs):
            # Truncate very long paragraphs to avoid token limits
            para_text = paragraph[:500] if len(paragraph) > 500 else paragraph
            prompt += f"\n\nPARAGRAPH {i}:\n{para_text}\n"

        prompt += "\n\nReturn only the JSON array, no other text."

        try:
            response_text = self._call_llm(prompt, max_tokens=4000)

            # Try to extract JSON if wrapped in markdown
            json_match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            results = json.loads(response_text)

            # Ensure we have a result for each paragraph
            results_by_index = {r.get("paragraph_index", i): r for i, r in enumerate(results)}
            ordered_results = []
            for i in range(len(paragraphs)):
                ordered_results.append(results_by_index.get(i, {"entities": []}))

            return ordered_results

        except Exception as e:
            print(f"Error in batch entity identification: {e}")
            print(f"Response was: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
            # Return empty results for all paragraphs on error
            return [{"entities": []} for _ in paragraphs]

    def enrich_entity_with_sources(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to find authoritative sources and additional details for an entity
        """
        entity_text = entity.get("text", "")
        entity_type = entity.get("type", "")
        entity_summary = entity.get("summary", "")

        prompt = f"""For this {entity_type} mentioned in historical text: "{entity_text}"

Summary: {entity_summary}

Provide:
1. 2-3 authoritative online sources (Wikipedia, Britannica, academic sites)
2. For PLACES: exact coordinates and modern name if changed
3. For PEOPLE: exact birth/death years, key accomplishments
4. For ARCHAIC TERMS: etymology and modern equivalent
5. Relevant image URL from Wikimedia Commons if available

Return ONLY a JSON object:
{{
  "sources": [
    {{
      "title": "Source Title",
      "url": "https://...",
      "type": "wikipedia|britannica|academic|other"
    }}
  ],
  "coordinates": {{"lat": 12.34, "lon": 56.78}},
  "born": "year",
  "died": "year",
  "etymology": "word origin",
  "modern_equivalent": "current term",
  "image_url": "https://upload.wikimedia.org/...",
  "enhanced_summary": "2-3 sentence detailed explanation"
}}

Include only fields that are relevant for this {entity_type}. Return only the JSON, no other text."""

        try:
            response_text = self._call_llm(prompt, max_tokens=2048)

            # Try to extract JSON if wrapped in markdown
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            enrichment = json.loads(response_text)

            # Merge enrichment data into entity
            entity.update(enrichment)

            return entity

        except Exception as e:
            print(f"Error enriching entity '{entity_text}': {e}")
            return entity

    def create_annotation(self, entity: Dict[str, Any], annotation_id: str) -> Dict[str, Any]:
        """
        Convert an enriched entity into the annotation format
        """
        entity_type = entity.get("type", "topic")

        annotation = {
            "id": annotation_id,
            "type": entity_type,
            "summary": entity.get("enhanced_summary") or entity.get("summary", ""),
            "details": {},
            "sources": entity.get("sources", [])
        }

        # Add type-specific details
        if entity_type == "place":
            if "coordinates" in entity and entity["coordinates"]:
                annotation["details"]["coordinates"] = entity["coordinates"]
            if "modern_equivalent" in entity:
                annotation["details"]["modern_equivalent"] = entity["modern_equivalent"]

        elif entity_type == "person":
            if "born" in entity:
                annotation["details"]["born"] = entity["born"]
            if "died" in entity:
                annotation["details"]["died"] = entity["died"]

        elif entity_type == "term":
            if "definition" in entity:
                annotation["details"]["definition"] = entity.get("definition", entity.get("summary", ""))
            if "etymology" in entity:
                annotation["details"]["etymology"] = entity["etymology"]
            if "modern_equivalent" in entity:
                annotation["details"]["modern_equivalent"] = entity["modern_equivalent"]

        # Add image if available
        if "image_url" in entity:
            annotation["image_url"] = entity["image_url"]

        return annotation

    def segment_paragraph_with_annotations(
        self,
        paragraph: str,
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Split paragraph into segments, marking which parts have annotations

        Returns:
            Tuple of (segments, annotations_index)
        """
        segments = []
        annotations_index = {}

        # Sort entities by their position in the paragraph (longest first to avoid overlaps)
        entities_with_pos = []
        for entity in entities:
            text = entity.get("text", "")
            pos = paragraph.find(text)
            if pos >= 0:
                entities_with_pos.append((pos, len(text), entity))

        # Sort by position
        entities_with_pos.sort(key=lambda x: x[0])

        # Create segments
        current_pos = 0
        annotation_counter = 0

        for pos, length, entity in entities_with_pos:
            # Add text before this entity (if any)
            if pos > current_pos:
                segments.append({
                    "text": paragraph[current_pos:pos],
                    "annotations": []
                })

            # Create annotation for this entity
            annotation_id = f"ann_{annotation_counter}"
            annotation_counter += 1

            # Enrich the entity with detailed sources
            print(f"    Enriching: {entity.get('text', '')[:50]}...")
            enriched_entity = self.enrich_entity_with_sources(entity)

            # Create annotation
            annotation = self.create_annotation(enriched_entity, annotation_id)
            annotations_index[annotation_id] = annotation

            # Add segment with annotation
            segments.append({
                "text": paragraph[pos:pos + length],
                "annotations": [annotation]
            })

            current_pos = pos + length

        # Add remaining text (if any)
        if current_pos < len(paragraph):
            segments.append({
                "text": paragraph[current_pos:],
                "annotations": []
            })

        return segments, annotations_index

    def annotate_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        batch_size: int = 5,
        max_paragraphs: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Annotate an entire text document

        Args:
            text: Plain text to annotate
            metadata: Document metadata (title, author, etc.)
            batch_size: Process paragraphs in batches (to avoid rate limits)
            max_paragraphs: Limit number of paragraphs to process (for testing)
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            Complete annotated document in the schema format
        """
        paragraphs = self.split_into_paragraphs(text)

        if max_paragraphs:
            paragraphs = paragraphs[:max_paragraphs]

        print(f"Processing {len(paragraphs)} paragraphs with {self.backend} ({self.model})...")

        content = []
        annotations_index = {}

        # Detect title (usually first paragraph or heading)
        if paragraphs:
            first_para = paragraphs[0]
            # If first paragraph is short and in title case, treat as heading
            if len(first_para) < 100 and first_para[0].isupper():
                content.append({
                    "type": "heading",
                    "level": 1,
                    "text": first_para
                })
                paragraphs = paragraphs[1:]

        # Process paragraphs in batches for better performance
        total = len(paragraphs)
        processed_count = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_paragraphs = paragraphs[batch_start:batch_end]

            msg = f"Processing paragraphs {batch_start + 1}-{batch_end}/{total} (batch)"
            print(f"\n{msg}...")

            if progress_callback:
                progress_callback(batch_end, total, msg)

            # Identify entities in batch of paragraphs with single LLM call
            batch_results = self.identify_entities_batch(batch_paragraphs)

            # Process each paragraph in the batch
            for i, (paragraph, result) in enumerate(zip(batch_paragraphs, batch_results)):
                entities = result.get("entities", [])

                if entities:
                    print(f"  Paragraph {batch_start + i + 1}: Found {len(entities)} entities")
                    # Create segments with annotations
                    segments, para_annotations = self.segment_paragraph_with_annotations(
                        paragraph,
                        entities
                    )
                    annotations_index.update(para_annotations)
                else:
                    # No annotations, just plain text
                    segments = [{"text": paragraph, "annotations": []}]

                content.append({
                    "type": "paragraph",
                    "text": paragraph,
                    "segments": segments
                })

            processed_count = batch_end

        # Build final document
        document = {
            "metadata": {
                **metadata,
                "processed_date": datetime.now().isoformat(),
                "annotator_backend": self.backend,
                "annotator_model": self.model
            },
            "content": content,
            "annotations_index": annotations_index
        }

        return document

    def save_annotated_document(
        self,
        document: Dict[str, Any],
        output_path: str
    ):
        """Save annotated document to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        print(f"\nSaved annotated document to: {output_path}")


def list_ollama_models(host: str = "http://localhost:11434") -> List[str]:
    """
    List available Ollama models

    Returns:
        List of model names
    """
    try:
        import ollama
        models = ollama.list(host=host)
        return [model['name'] for model in models.get('models', [])]
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        return []


def main():
    """Command-line interface"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_annotator.py <input.txt> [output.json] [--backend anthropic|ollama] [--model MODEL]")
        print("\nExamples:")
        print("  # Use Ollama (default)")
        print("  python text_annotator.py sample.txt annotated.json")
        print("\n  # Use specific Ollama model")
        print("  python text_annotator.py sample.txt annotated.json --model llama3.2")
        print("\n  # Use Anthropic")
        print("  python text_annotator.py sample.txt annotated.json --backend anthropic")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else input_file.replace('.txt', '_annotated.json')

    # Parse optional arguments
    backend = "ollama"
    model = None

    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
        elif arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]

    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract metadata from filename
    filename = os.path.basename(input_file)
    title = filename.replace('.txt', '').replace('_', ' ').title()

    metadata = {
        "title": title,
        "source_file": input_file
    }

    # Create annotator and process
    print(f"Initializing annotator with {backend} backend...")
    annotator = TextAnnotator(backend=backend, model=model)

    # Process (limit to first 3 paragraphs for testing)
    document = annotator.annotate_text(text, metadata, max_paragraphs=3)

    # Save result
    annotator.save_annotated_document(document, output_file)

    print(f"\n✓ Processing complete!")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Paragraphs: {len(document['content'])}")
    print(f"  Annotations: {len(document['annotations_index'])}")


if __name__ == '__main__':
    main()
