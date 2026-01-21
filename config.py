"""
Server Configuration for Remote Access
Allows secure remote collaboration on audio generation
"""
import os
import secrets
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class ServerConfig:
    """Configuration for remote server access"""

    def __init__(self):
        # Server settings
        self.HOST = os.getenv('SERVER_HOST', '0.0.0.0')  # Bind to all interfaces for remote access
        self.PORT = int(os.getenv('SERVER_PORT', '5000'))
        self.DEBUG = os.getenv('SERVER_DEBUG', 'False').lower() == 'true'

        # Authentication
        self.REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'True').lower() == 'true'
        self.API_KEY = os.getenv('API_KEY', None)

        # CORS settings
        self.ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')

        # WebSocket settings
        self.ENABLE_WEBSOCKET = os.getenv('ENABLE_WEBSOCKET', 'True').lower() == 'true'

        # Max file sizes
        self.MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', str(100 * 1024 * 1024)))  # 100MB default

        # Directory settings
        self.COMMON_FILES_DIR = os.getenv('COMMON_FILES_DIR', 'common_files')
        self.DEFAULT_PROJECT_DIR = os.getenv('DEFAULT_PROJECT_DIR', 'projects')

        # Default Gutenberg URL
        self.DEFAULT_GUTENBERG_URL = os.getenv('DEFAULT_GUTENBERG_URL', 'https://www.gutenberg.org/cache/epub/4932/pg4932.txt')

        # Default Voice Sample (file extension agnostic)
        self.DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', 'Stoker Extended')

        # Text chunking settings
        self.MAX_CHUNK_SIZE = int(os.getenv('MAX_CHUNK_SIZE', '500'))

        # Generation settings
        self.MAX_PARALLEL_GENERATIONS = int(os.getenv('MAX_PARALLEL_GENERATIONS', '3'))
        self.DEVICE_PREFERENCE = os.getenv('DEVICE_PREFERENCE', 'auto')  # auto, cuda, cpu

        # Initialize API key if needed
        if self.REQUIRE_AUTH and not self.API_KEY:
            self._generate_api_key()

    def _generate_api_key(self):
        """Generate a secure API key if none is set"""
        self.API_KEY = secrets.token_urlsafe(32)
        print("\n" + "="*80)
        print("🔐 NEW API KEY GENERATED")
        print("="*80)
        print(f"\nAPI Key: {self.API_KEY}")
        print("\nTo reuse this key on server restart, set the environment variable:")
        print(f"  export API_KEY='{self.API_KEY}'")
        print("\nOr add to a .env file:")
        print(f"  API_KEY={self.API_KEY}")
        print("\nShare this key securely with your collaborator.")
        print("="*80 + "\n")

    def get_client_config(self) -> dict:
        """Get configuration that should be shared with clients"""
        return {
            'require_auth': self.REQUIRE_AUTH,
            'websocket_enabled': self.ENABLE_WEBSOCKET,
            'max_upload_size': self.MAX_UPLOAD_SIZE,
            'default_project_dir': self.DEFAULT_PROJECT_DIR,
            'common_files_dir': self.COMMON_FILES_DIR,
            'default_gutenberg_url': self.DEFAULT_GUTENBERG_URL,
            'default_voice': self.DEFAULT_VOICE
        }

    @staticmethod
    def create_env_template():
        """Create a .env.template file with example configuration"""
        template = """# Henty Server Configuration
# Copy this file to .env and configure for your setup

# Server Network Settings
SERVER_HOST=0.0.0.0  # 0.0.0.0 for remote access, 127.0.0.1 for local only
SERVER_PORT=5000
SERVER_DEBUG=False

# Authentication (REQUIRED for remote access)
REQUIRE_AUTH=True
API_KEY=your-secret-api-key-here  # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"

# CORS - Allowed Origins (comma-separated)
# Use * for any origin (less secure) or specify like: http://192.168.1.10:8080,http://collaborator-pc:8080
ALLOWED_ORIGINS=*

# WebSocket Support (for real-time progress updates)
ENABLE_WEBSOCKET=True

# Max Upload Size (bytes)
MAX_UPLOAD_SIZE=104857600  # 100MB

# Directory Settings
COMMON_FILES_DIR=common_files  # Directory for shared audio files (intros, outros, etc.)
DEFAULT_PROJECT_DIR=projects   # Default directory for all projects
"""
        return template

# Global config instance
config = ServerConfig()
