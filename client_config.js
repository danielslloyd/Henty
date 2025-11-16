/**
 * Client Configuration for Remote Server Connection
 *
 * This file allows the web UI to connect to a remote Henty server.
 *
 * USAGE:
 * 1. Copy this file to the same directory as index.html
 * 2. Update the SERVER_URL to point to your remote server
 * 3. Set the API_KEY if authentication is enabled on the server
 * 4. Include this file in index.html BEFORE the main script:
 *    <script src="client_config.js"></script>
 */

const HentyConfig = {
    // Server URL - change this to your remote server address
    // Examples:
    //   - Local: 'http://localhost:5000'
    //   - LAN: 'http://192.168.1.100:5000'
    //   - Remote: 'http://your-server.com:5000'
    SERVER_URL: 'http://localhost:5000',

    // API Key for authentication (if server has REQUIRE_AUTH=True)
    // Get this from the server administrator
    API_KEY: '',

    // WebSocket URL (usually same as server but with /socket.io)
    // Auto-generated from SERVER_URL if not set
    WEBSOCKET_URL: null,

    // Request timeout in milliseconds
    TIMEOUT: 300000,  // 5 minutes for long audio generation

    // Enable debug logging
    DEBUG: false,

    /**
     * Get the full API URL for an endpoint
     * @param {string} endpoint - API endpoint path (e.g., '/api/status')
     * @returns {string} Full URL
     */
    getApiUrl(endpoint) {
        // Remove leading slash if present to avoid double slashes
        const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
        return `${this.SERVER_URL}/${cleanEndpoint}`;
    },

    /**
     * Get WebSocket URL
     * @returns {string} WebSocket URL
     */
    getWebSocketUrl() {
        if (this.WEBSOCKET_URL) {
            return this.WEBSOCKET_URL;
        }
        // Auto-generate from SERVER_URL
        return this.SERVER_URL;
    },

    /**
     * Get headers for authenticated requests
     * @returns {Object} Headers object
     */
    getAuthHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.API_KEY) {
            headers['X-API-Key'] = this.API_KEY;
        }

        return headers;
    },

    /**
     * Make an authenticated fetch request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise} Fetch promise
     */
    async fetch(endpoint, options = {}) {
        const url = this.getApiUrl(endpoint);

        // Merge auth headers with any provided headers
        const headers = {
            ...this.getAuthHeaders(),
            ...(options.headers || {})
        };

        // Set timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.TIMEOUT);

        try {
            const response = await fetch(url, {
                ...options,
                headers,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (this.DEBUG) {
                console.log(`[HentyConfig] ${options.method || 'GET'} ${endpoint}:`, response.status);
            }

            return response;
        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new Error(`Request timeout after ${this.TIMEOUT}ms`);
            }

            throw error;
        }
    },

    /**
     * Test connection to server
     * @returns {Promise<Object>} Server status
     */
    async testConnection() {
        try {
            const response = await this.fetch('/api/status');
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }
            const data = await response.json();
            console.log('✅ Successfully connected to server:', data);
            return data;
        } catch (error) {
            console.error('❌ Failed to connect to server:', error);
            throw error;
        }
    },

    /**
     * Get server configuration
     * @returns {Promise<Object>} Server configuration
     */
    async getServerConfig() {
        try {
            const response = await this.fetch('/api/config');
            if (!response.ok) {
                throw new Error(`Failed to get config: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Failed to get server config:', error);
            return {};
        }
    }
};

// Auto-detect local vs remote based on current page URL
if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    console.log('🌐 Remote mode detected. Please configure SERVER_URL in client_config.js');
}

// Log current configuration
console.log('Henty Client Configuration:', {
    serverUrl: HentyConfig.SERVER_URL,
    hasApiKey: !!HentyConfig.API_KEY,
    debug: HentyConfig.DEBUG
});

// Make available globally
window.HentyConfig = HentyConfig;
