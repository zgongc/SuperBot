"""
IPC Server - JSON-RPC over Unix Socket
=======================================

Provides IPC/RPC server for daemon-client communication.
Supports Unix sockets (primary) and TCP (fallback).

Author: SuperBot Team
Date: 2025-11-07
"""

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import traceback


class IPCServer:
    """
    IPC Server using JSON-RPC 2.0 protocol over Unix socket

    Features:
    - Async request handling
    - Method registration
    - Error handling
    - Authentication support
    """

    def __init__(self, socket_path: str, logger, daemon=None, auth_token: Optional[str] = None):
        """
        Initialize IPC Server

        Args:
            socket_path: Path to Unix socket
            logger: Logger instance
            daemon: Reference to daemon (for RPC methods)
            auth_token: Optional authentication token
        """
        self.socket_path = socket_path
        self.logger = logger
        self.daemon = daemon
        self.auth_token = auth_token or os.getenv('DAEMON_AUTH_TOKEN')

        self.server: Optional[asyncio.Server] = None
        self.handlers: Dict[str, Callable] = {}
        self.running = False

    def register_handler(self, method: str, handler: Callable):
        """Register RPC method handler"""
        self.handlers[method] = handler

    async def start(self, tcp_host: str = '127.0.0.1', tcp_port: int = 9999):
        """Start IPC server

        Args:
            tcp_host: TCP host for Windows (default: 127.0.0.1)
            tcp_port: TCP port for Windows (default: 9999, read from config)
        """
        import sys

        # Windows doesn't support Unix sockets, use TCP instead
        if sys.platform == 'win32':
            self.logger.info("Windows detected, using TCP server instead of Unix socket")

            self.server = await asyncio.start_server(
                self._handle_client,
                host=tcp_host,
                port=tcp_port
            )

            self.running = True
            self.logger.info(f"IPC Server started on {tcp_host}:{tcp_port}")

            # Update socket_path to indicate TCP
            self.socket_path = f"{tcp_host}:{tcp_port}"

        else:
            # Unix/Linux/Mac: use Unix socket
            # Remove existing socket
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            # Ensure directory exists
            socket_dir = Path(self.socket_path).parent
            socket_dir.mkdir(parents=True, exist_ok=True)

            # Start Unix socket server
            self.server = await asyncio.start_unix_server(
                self._handle_client,
                path=self.socket_path
            )

            # Set socket permissions (owner only)
            os.chmod(self.socket_path, 0o600)

            self.running = True
            self.logger.info(f"IPC Server started on {self.socket_path}")

    async def stop(self):
        """Stop IPC server"""
        if not self.running:
            return

        self.running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Remove socket (only if Unix socket, not TCP)
        import sys
        if sys.platform != 'win32' and os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.logger.info("IPC Server stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection"""
        peer_name = writer.get_extra_info('peername', 'unknown')

        try:
            # Read request (newline-delimited JSON)
            data = await reader.readuntil(b'\n')
            request_str = data.decode('utf-8').strip()

            # Parse JSON-RPC request
            try:
                request = json.loads(request_str)
            except json.JSONDecodeError as e:
                response = self._error_response(None, -32700, f"Parse error: {e}")
                await self._send_response(writer, response)
                return

            # Validate JSON-RPC format
            if not self._validate_request(request):
                response = self._error_response(
                    request.get('id'),
                    -32600,
                    "Invalid Request"
                )
                await self._send_response(writer, response)
                return

            # Authenticate (if auth enabled)
            if self.auth_token:
                auth_header = request.get('auth')
                if auth_header != self.auth_token:
                    response = self._error_response(
                        request['id'],
                        -32001,
                        "Authentication failed"
                    )
                    await self._send_response(writer, response)
                    return

            # Handle request
            response = await self._handle_request(request)
            await self._send_response(writer, response)

        except asyncio.IncompleteReadError:
            self.logger.warning(f"Client {peer_name} disconnected abruptly")
        except Exception as e:
            self.logger.error(f"Error handling client {peer_name}: {e}")
            error_response = self._error_response(None, -32603, f"Internal error: {e}")
            try:
                await self._send_response(writer, error_response)
            except:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

    def _validate_request(self, request: Dict) -> bool:
        """Validate JSON-RPC 2.0 request format"""
        if not isinstance(request, dict):
            return False

        if request.get('jsonrpc') != '2.0':
            return False

        if 'method' not in request:
            return False

        if 'id' not in request:
            return False

        return True

    async def _handle_request(self, request: Dict) -> Dict:
        """Handle JSON-RPC request"""
        method = request['method']
        params = request.get('params', {})
        request_id = request['id']

        # Check if method exists
        if method not in self.handlers:
            return self._error_response(
                request_id,
                -32601,
                f"Method not found: {method}"
            )

        # Execute handler
        try:
            handler = self.handlers[method]
            result = await handler(params)

            return {
                'jsonrpc': '2.0',
                'result': result,
                'id': request_id
            }

        except Exception as e:
            self.logger.error(f"Error executing method '{method}': {e}")
            self.logger.error(traceback.format_exc())

            return self._error_response(
                request_id,
                -32603,
                f"Internal error: {str(e)}"
            )

    def _error_response(self, request_id: Any, code: int, message: str) -> Dict:
        """Create JSON-RPC error response"""
        return {
            'jsonrpc': '2.0',
            'error': {
                'code': code,
                'message': message
            },
            'id': request_id
        }

    async def _send_response(self, writer: asyncio.StreamWriter, response: Dict):
        """Send JSON-RPC response"""
        response_str = json.dumps(response) + '\n'
        writer.write(response_str.encode('utf-8'))
        await writer.drain()


class TCPIPCServer(IPCServer):
    """
    TCP-based IPC Server (for remote access)

    Uses TCP socket instead of Unix socket.
    Should be used with TLS for production.
    """

    def __init__(self, host: str, port: int, logger, daemon=None, auth_token: Optional[str] = None):
        """Initialize TCP IPC Server"""
        super().__init__(socket_path=None, logger=logger, daemon=daemon, auth_token=auth_token)
        self.host = host
        self.port = port

    async def start(self):
        """Start TCP server"""
        self.server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port
        )

        self.running = True
        self.logger.info(f"TCP IPC Server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop TCP server"""
        if not self.running:
            return

        self.running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self.logger.info("TCP IPC Server stopped")
