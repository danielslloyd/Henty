#!/bin/bash
# Henty Launcher Script for Linux/Mac
# Starts the server and opens the landing page in your default browser

echo "🎤 Starting Henty Audiobook Creation Suite..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if server.py exists
if [ ! -f "server.py" ]; then
    echo "❌ server.py not found. Please run this script from the Henty directory."
    exit 1
fi

# Start the server in the background
echo "🚀 Starting server on http://localhost:5000..."
python3 server.py &
SERVER_PID=$!

# Save PID for later cleanup
echo $SERVER_PID > .server.pid

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo ""

    # Open browser based on OS
    URL="http://localhost:5000/index.html"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v xdg-open &> /dev/null; then
            xdg-open "$URL" &> /dev/null &
        elif command -v gnome-open &> /dev/null; then
            gnome-open "$URL" &> /dev/null &
        fi
        echo "🌐 Opening browser at $URL"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac
        open "$URL"
        echo "🌐 Opening browser at $URL"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✨ Henty is ready!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Landing Page: http://localhost:5000/index.html"
    echo "Server URL:   http://localhost:5000"
    echo "API Docs:     http://localhost:5000/api/status"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Wait for Ctrl+C
    trap "echo ''; echo '🛑 Stopping server...'; kill $SERVER_PID 2>/dev/null; rm -f .server.pid; echo '✅ Server stopped'; exit 0" INT TERM

    # Keep script running
    wait $SERVER_PID
else
    echo "❌ Failed to start server"
    rm -f .server.pid
    exit 1
fi
