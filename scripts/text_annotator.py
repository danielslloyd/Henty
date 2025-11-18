"""
Text Annotator using LLM

This script processes plain text and generates rich annotations for:
- Places (with coordinates and descriptions)
- Historical figures and topics (with bios/summaries)
- Archaic terminology (with definitions)

All annotations include sources and links.
"""

import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import anthropic


class TextAnnotator:
    """Annotate text using Claude AI for educational annotations"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the annotator

        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set or provided")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines
        paragraphs = text.split('\n\n')
        # Clean and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def identify_entities(self, paragraph: str) -> Dict[str, Any]:
        """
        Use Claude to identify places, people, topics, and archaic terms in a paragraph

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
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response
            response_text = message.content[0].text

            # Try to extract JSON if wrapped in markdown
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            result = json.loads(response_text)
            return result

        except Exception as e:
            print(f"Error identifying entities: {e}")
            return {"entities": []}

    def enrich_entity_with_sources(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Claude to find authoritative sources and additional details for an entity
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
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

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
        max_paragraphs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Annotate an entire text document

        Args:
            text: Plain text to annotate
            metadata: Document metadata (title, author, etc.)
            batch_size: Process paragraphs in batches (to avoid rate limits)
            max_paragraphs: Limit number of paragraphs to process (for testing)

        Returns:
            Complete annotated document in the schema format
        """
        paragraphs = self.split_into_paragraphs(text)

        if max_paragraphs:
            paragraphs = paragraphs[:max_paragraphs]

        print(f"Processing {len(paragraphs)} paragraphs...")

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

        # Process each paragraph
        for i, paragraph in enumerate(paragraphs):
            print(f"Processing paragraph {i+1}/{len(paragraphs)}...")

            # Identify entities in this paragraph
            result = self.identify_entities(paragraph)
            entities = result.get("entities", [])

            if entities:
                print(f"  Found {len(entities)} entities")
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

        # Build final document
        document = {
            "metadata": {
                **metadata,
                "processed_date": datetime.now().isoformat()
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
        print(f"Saved annotated document to: {output_path}")


def main():
    """Command-line interface"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_annotator.py <input.txt> [output.json]")
        print("\nExample:")
        print("  python text_annotator.py sample.txt annotated_sample.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.txt', '_annotated.json')

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
    annotator = TextAnnotator()

    # Process (limit to first 10 paragraphs for testing)
    document = annotator.annotate_text(text, metadata, max_paragraphs=10)

    # Save result
    annotator.save_annotated_document(document, output_file)

    print(f"\n✓ Processing complete!")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Paragraphs: {len(document['content'])}")
    print(f"  Annotations: {len(document['annotations_index'])}")


if __name__ == '__main__':
    main()
