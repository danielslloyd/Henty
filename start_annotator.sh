#!/bin/bash
# Quick start script for Gutenberg Text Annotator UI

echo "=========================================="
echo "Gutenberg Text Annotator UI"
echo "=========================================="
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama found"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is running"

        # List available models
        echo ""
        echo "Available Ollama models:"
        ollama list
    else
        echo "⚠ Ollama is not running"
        echo ""
        echo "To start Ollama, run in another terminal:"
        echo "  ollama serve"
        echo ""
        echo "Then pull a model:"
        echo "  ollama pull llama3.2       # Fast (3B)"
        echo "  ollama pull llama3.1:8b    # Better (8B)"
        echo "  ollama pull qwen2.5:14b    # Best (14B)"
        echo ""
    fi
else
    echo "⚠ Ollama not found"
    echo ""
    echo "To use local models (free), install Ollama:"
    echo "  https://ollama.ai/download"
    echo ""
    echo "Or use Anthropic (cloud, paid):"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    echo ""
fi

# Check for Anthropic API key
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✓ Anthropic API key found"
else
    echo "ℹ No Anthropic API key (optional)"
fi

echo ""
echo "=========================================="
echo "Starting UI..."
echo "=========================================="
echo ""
echo "The interface will open at:"
echo "  http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch the UI
python annotator_ui.py
