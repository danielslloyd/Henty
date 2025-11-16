"""
Authentication middleware for remote server access
"""
from functools import wraps
from flask import request, jsonify
from typing import Optional

class AuthManager:
    """Manages API key authentication for the server"""

    def __init__(self, api_key: Optional[str] = None, require_auth: bool = True):
        self.api_key = api_key
        self.require_auth = require_auth

        # Public endpoints that don't require authentication
        self.public_endpoints = {
            '/',
            '/api/status',
            '/api/config'
        }

    def require_api_key(self, f):
        """Decorator to require API key authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Skip auth if not required or endpoint is public
            if not self.require_auth or request.path in self.public_endpoints:
                return f(*args, **kwargs)

            # Get API key from header or query parameter
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

            # Check if API key is valid
            if not api_key:
                return jsonify({
                    'error': 'API key required',
                    'message': 'Please provide API key in X-API-Key header or api_key query parameter'
                }), 401

            if api_key != self.api_key:
                return jsonify({
                    'error': 'Invalid API key',
                    'message': 'The provided API key is not valid'
                }), 403

            # API key is valid, proceed with request
            return f(*args, **kwargs)

        return decorated_function

    def update_api_key(self, new_key: str):
        """Update the API key"""
        self.api_key = new_key

    def set_require_auth(self, require: bool):
        """Enable or disable authentication"""
        self.require_auth = require
