#!/usr/bin/env python3
"""
Henty Launcher Script (Cross-Platform)
Starts the server and opens the landing page in your default browser
"""

import os
import sys
import time
import webbrowser
import subprocess
import signal
import platform

def print_banner():
    print()
    print("=" * 60)
    print("  🎤 Henty Audiobook Creation Suite")
    print("=" * 60)
    print()

def check_python():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def check_server_file():
    """Check if server.py exists"""
    if not os.path.exists('server.py'):
        print("❌ server.py not found")
        print("   Please run this script from the Henty directory")
        return False
    return True

def start_server():
    """Start the Flask server"""
    print("🚀 Starting server on http://localhost:5000...")

    # Determine the Python command
    python_cmd = sys.executable

    # Start server as subprocess
    if platform.system() == 'Windows':
        # On Windows, create a new console window
        process = subprocess.Popen(
            [python_cmd, 'server.py'],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # On Unix-like systems, run in background
        process = subprocess.Popen(
            [python_cmd, 'server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    return process

def open_browser(url, delay=3):
    """Open browser after a delay"""
    print(f"⏳ Waiting {delay} seconds for server to initialize...")
    time.sleep(delay)

    print(f"🌐 Opening browser at {url}")
    webbrowser.open(url)

def main():
    print_banner()

    # Check prerequisites
    if not check_python():
        input("\nPress Enter to exit...")
        return 1

    if not check_server_file():
        input("\nPress Enter to exit...")
        return 1

    # Start server
    try:
        server_process = start_server()

        # Open browser
        url = "http://localhost:5000/index.html"
        open_browser(url)

        # Print success message
        print()
        print("=" * 60)
        print("✨ Henty is ready!")
        print("=" * 60)
        print()
        print("Landing Page: http://localhost:5000/index.html")
        print("Server URL:   http://localhost:5000")
        print("API Status:   http://localhost:5000/api/status")
        print()

        if platform.system() == 'Windows':
            print("The server is running in a separate window.")
            print("Close that window to stop the server.")
            print()
            input("Press Enter to close this launcher window...")
        else:
            print("Press Ctrl+C to stop the server")
            print("=" * 60)
            print()

            # Set up signal handler for graceful shutdown
            def signal_handler(sig, frame):
                print("\n")
                print("🛑 Stopping server...")
                server_process.terminate()
                server_process.wait()
                print("✅ Server stopped")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Keep script running
            server_process.wait()

        return 0

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
