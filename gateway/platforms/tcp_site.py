"""Bind an aiohttp ``TCPSite`` for the HTTP-serving adapters (webhook, api_server): exclusive on macOS,
yet able to rebind over a lingering TIME_WAIT socket right after a gateway restart."""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aiohttp import web

from gateway.platforms.shared_ingress import is_wildcard_host

logger = logging.getLogger(__name__)

def has_live_listener(host: str, port: int) -> bool:
    """Blocking probe: True when something accepts connections on ``host:port``. Refused = nobody listens;
    any other failure (timeout, unroutable) is treated as live so the caller stays exclusive."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        return True


def _local_ipv4_addresses() -> list[str]:
    """Every local IPv4 a wildcard bind on this host would cover, fresh each call (never cached).

    Includes loopback, the LAN/primary interface and VPN/Tailscale (utun) addresses, and 0.0.0.0
    (a connect to 0.0.0.0 routes to localhost and so detects a wildcard listener). A wildcard
    listener is only re-enabled with SO_REUSEADDR when every one of these refuses a connection, so
    a foreign specific-interface listener (e.g. on a Tailscale/LAN IP) cannot be silently split.
    Enumeration order: socket.getifaddrs() when the runtime provides it, otherwise a bounded
    `ifconfig` parse on macOS (captures utun/Tailscale and all interfaces), otherwise hostname
    resolution. Parsing ifconfig only happens on the rare EADDRINUSE-with-wildcard path and is
    failed-safe (an empty set just narrows the probe, keeping the caller strict)."""
    addrs = {"0.0.0.0", "127.0.0.1"}
    got = False
    try:
        for _name, entries in socket.getifaddrs():
            for entry in entries:
                if entry.family == socket.AF_INET:
                    addrs.add(entry.address)
                    got = True
    except (OSError, AttributeError):
        pass
    if not got:
        try:
            import re as _re
            import subprocess as _sub
            out = _sub.run(
                ["ifconfig"], check=False, capture_output=True, text=True, timeout=2
            ).stdout or ""
            for _m in _re.finditer(r"inet (\d+\.\d+\.\d+\.\d+)", out):
                addrs.add(_m.group(1))
        except Exception:
            pass
    if not got:
        try:
            for _af, _st, _pr, _cn, sockaddr in socket.getaddrinfo(
                socket.gethostname(), None, socket.AF_INET
            ):
                addrs.add(sockaddr[0])
        except OSError:
            pass
    return sorted(addrs)


async def _any_local_live_listener(port: int) -> bool:
    """Fail-closed: True when ANY local IPv4 accepts a connection on ``port``, or when a probe is
    uncertain (has_live_listener returns True for anything that is not a clean refusal). A wildcard
    bind must only be re-enabled with SO_REUSEADDR when this returns False (proving nothing is live;
    only a TIME_WAIT socket lingers)."""
    for addr in _local_ipv4_addresses():
        if await asyncio.to_thread(has_live_listener, addr, port):
            return True
    return False


async def start_tcp_site(runner: web.BaseRunner, host: Optional[str], port: int, *, log_tag: str) -> web.TCPSite:
    """Bind ``host:port`` on ``runner`` and return the started site; raises OSError when unavailable.

    SO_REUSEADDR: on macOS (BSD) two wildcard/specific sockets can silently split traffic while
    both report success → disable. On Linux it only permits rebinding past TIME_WAIT (a quick
    restart would otherwise fail to bind for ~60s) → keep the default.

    The macOS exclusive bind also refuses the port while a server-side TIME_WAIT socket lingers
    (2*MSL = 30s after the previous gateway closed a connection first — its shutdown, or any
    ``Connection: close`` request), so a ``/restart`` re-binding within seconds failed with
    EADDRINUSE although nobody was listening. Preventing the TIME_WAIT at close time (SO_LINGER 0)
    would reset in-flight senders and misses the per-request case, hence the bind-side retry: for an
    explicit host, ``has_live_listener`` refused proves the address is free (the kernel still
    rejects an exact duplicate even with SO_REUSEADDR, and a foreign wildcard listener answers the
    probe), so one retry with reuse_address=True is safe. A wildcard host widens the probe to every
    local address (loopback + real interfaces + 0.0.0.0): only when ALL of them refuse is it known to
    be a lone TIME_WAIT socket, so the single conditional rebind with reuse_address=True is safe;
    any live or uncertain probe keeps the strict raise (never always-on SO_REUSEADDR, which would let
    a foreign specific-interface listener silently split traffic)."""
    from aiohttp import web

    exclusive = sys.platform == "darwin"
    site = web.TCPSite(runner, host, port, reuse_address=False if exclusive else None)
    try:
        await site.start()
    except OSError as exc:
        if not exclusive or exc.errno != errno.EADDRINUSE:
            raise
        live = (
            await _any_local_live_listener(port)
            if is_wildcard_host(host)
            else await asyncio.to_thread(has_live_listener, host, port)
        )
        if live:
            raise
        await site.stop()  # aiohttp registers a site before binding: drop the dead one from the runner
        logger.info("[%s] %s:%d busy without a live listener (TIME_WAIT from the previous gateway); "
                    "rebinding with SO_REUSEADDR", log_tag, host, port)
        site = web.TCPSite(runner, host, port, reuse_address=True)
        await site.start()
    return site
