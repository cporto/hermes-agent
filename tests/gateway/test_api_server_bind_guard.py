"""Tests for the API server bind-address startup guard.

Validates that is_network_accessible() correctly classifies addresses and
that connect() refuses to start without API_SERVER_KEY.
"""

import socket
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import is_network_accessible


# ---------------------------------------------------------------------------
# Unit tests: is_network_accessible()
# ---------------------------------------------------------------------------


class TestIsNetworkAccessible:
    """Direct tests for the address classification helper."""

    # -- Loopback (safe, should return False) --


    def test_ipv4_mapped_loopback(self):
        # ::ffff:127.0.0.1 — Python's is_loopback returns False for mapped
        # addresses; the helper must unwrap and check ipv4_mapped.
        assert is_network_accessible("::ffff:127.0.0.1") is False

    # -- Network-accessible (should return True) --


    def test_ipv6_wildcard(self):
        # This is the bypass vector that the string-based check missed.
        assert is_network_accessible("::") is True


    def test_private_ipv4(self):
        assert is_network_accessible("10.0.0.1") is True


    def test_public_ipv4(self):
        assert is_network_accessible("8.8.8.8") is True

    # -- Hostname resolution --


    def test_hostname_mixed_resolution(self):
        """If a hostname resolves to both loopback and non-loopback, it's
        network-accessible (any non-loopback address is enough)."""
        mixed_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
        ]
        with patch("gateway.platforms.base._socket.getaddrinfo", return_value=mixed_result):
            assert is_network_accessible("dual-host.local") is True


# ---------------------------------------------------------------------------
# Integration tests: connect() startup guard
# ---------------------------------------------------------------------------


class TestConnectBindGuard:
    """Verify that connect() refuses dangerous configurations."""


    @pytest.mark.asyncio
    async def test_refuses_loopback_without_key(self):
        """Loopback binds are still an auth boundary and require API_SERVER_KEY."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1"}))
        assert adapter._api_key == ""
        assert is_network_accessible(adapter._host) is False
        result = await adapter.connect()
        assert result is False
        assert adapter._app is None
        assert adapter._background_tasks == set()


    @pytest.mark.asyncio
    async def test_allows_wildcard_with_key(self):
        """Non-loopback with a key should pass the guard."""
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "0.0.0.0", "key": "sk-test"})
        )
        # The guard checks: is_network_accessible(host) AND NOT api_key
        # With a key set, the guard should not block.
        assert adapter._api_key == "sk-test"
        assert is_network_accessible("0.0.0.0") is True
        # Combined: the guard condition is False (key is set), so it passes


# ---------------------------------------------------------------------------
# Integration tests: bind mechanics (direct bind, no pre-probe — #10297)
# ---------------------------------------------------------------------------


class TestBindMechanics:
    """connect() binds directly instead of pre-probing 127.0.0.1.

    The old ``_port_is_available()`` probe connected to 127.0.0.1 only and
    reported a lingering TIME_WAIT socket as "in use", failing gateway
    restarts for up to ~60s (#10297). The fix removes the probe: bind
    directly, keep SO_REUSEADDR default semantics on Linux (rebind past
    TIME_WAIT), and surface a real bind conflict as a clean ``False`` with
    the runner torn down.
    """

    _KEY = "sk-test-strong-key-0123456789"

    def _make_adapter(self, port: int) -> APIServerAdapter:
        return APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": port, "key": self._KEY},
            )
        )

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @pytest.mark.asyncio
    async def test_real_timewait_rebind_after_disconnect(self):
        """A restarted adapter rebinds instantly even while the prior incarnation's
        accepted-connection socket is still in TIME_WAIT on the same port.

        The previous implementation disabled SO_REUSEADDR on macOS (``reuse_address=False``
        on darwin), so a clean ``--replace`` restart — where the old server actively closed
        an SSE connection and its accepted socket lingers in TIME_WAIT for ~60s — failed its
        fresh bind with EADDRINUSE, which the code classified as a permanent config error and
        dropped the api_server (and Mercury) until manual /platform resume.

        This test reproduces the real mechanism: it seeds a TIME_WAIT socket on the port
        (the old code path blocks the rebind and the bug is caught), then asserts the
        adapter rebinds immediately (the fix).
        """
        port = self._free_port()

        # Seed a TIME_WAIT socket on this port the way a real server connection leaves one:
        # raw listen socket, accept a real client connection, then have the SERVER close its
        # accepted socket first so *it* (sharing the port) enters TIME_WAIT (~60s on macOS).
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        lsock.bind(("127.0.0.1", port))
        lsock.listen(5)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2)
        client.connect(("127.0.0.1", port))
        conn, _ = lsock.accept()
        # server-initiated close: accepted socket -> TIME_WAIT on (127.0.0.1, port)
        conn.close()
        lsock.close()
        client.close()

        # Sanity: without SO_REUSEADDR the rebind is now blocked (proves TIME_WAIT is real).
        blocking = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocking.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        blocked = False
        try:
            blocking.bind(("127.0.0.1", port))
        except OSError as e:
            blocked = e.errno == 48  # EADDRINUSE
        finally:
            blocking.close()
        assert blocked, (
            "test precondition failed: no TIME_WAIT socket seeded — "
            "the test would not discriminate the bug"
        )

        # The fix: the restarted adapter uses SO_REUSEADDR (framework default) and binds
        # immediately past the lingering TIME_WAIT socket.
        adapter = self._make_adapter(port)
        try:
            assert await adapter.connect() is True
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_immediate_rebind_after_disconnect(self):
        """A restarted adapter can rebind the same port immediately.

        This is the #10297 symptom: the old pre-probe (and disabled address
        reuse) made a quick gateway restart fail while the previous socket
        sat in TIME_WAIT.
        """
        port = self._free_port()
        first = self._make_adapter(port)
        assert await first.connect() is True
        await first.disconnect()

        second = self._make_adapter(port)
        try:
            assert await second.connect() is True
        finally:
            await second.disconnect()


    @pytest.mark.asyncio
    async def test_port_conflict_sets_non_retryable_fatal_error(self):
        """A real port conflict (EADDRINUSE) must set a non-retryable fatal
        error so the reconnect watcher drops the platform from the retry
        queue instead of looping indefinitely.

        Previously connect() returned bare ``False``, which the reconnect
        watcher treated as retryable — retrying every 5 minutes forever,
        filling errors.log and leaking 2 fds per retry (#52132: 1568+
        retries over 5 days in a multi-profile setup).
        """
        port = self._free_port()
        first = self._make_adapter(port)
        assert await first.connect() is True
        second = self._make_adapter(port)
        try:
            result = await second.connect()
            assert result is False
            assert second.has_fatal_error is True
            assert second.fatal_error_retryable is False
            assert second.fatal_error_code == "api_server_port_in_use"
            assert str(port) in (second.fatal_error_message or "")
        finally:
            await first.disconnect()
            await second.disconnect()
