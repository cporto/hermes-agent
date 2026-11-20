"""Session chat/stream -> replayable run stream bridge (Mercury sync fix).

GET /v1/runs/{id}/events must keep receiving a session-chat run's frames AFTER the POST client
disconnects (the run keeps going by design), and every attached subscriber must receive the
terminal sentinel (a clean ``: stream closed``) instead of keepalives forever.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import ClientSession, TCPConnector, web
from aiohttp.test_utils import TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


class _FakeAgent:
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    def __init__(self, session_id, cb, started, proceed):
        self.session_id = session_id
        self._cb = cb
        self._started = started
        self._proceed = proceed

    def interrupt(self, _message=None):
        pass

    def run_conversation(self, user_message, conversation_history, task_id):
        del user_message, conversation_history, task_id
        self._started.set()
        self._cb("one ")
        self._proceed.wait(timeout=5)
        self._cb("two")
        return {"final_response": "one two", "session_id": self.session_id}


class _Response:
    """Fake aiohttp StreamResponse; optionally 'disconnects' after N writes."""

    def __init__(self, disconnect_after=None):
        self.writes = 0
        self._disconnect_after = disconnect_after

    async def prepare(self, request):
        del request

    async def write(self, payload):
        del payload
        self.writes += 1
        if self._disconnect_after is not None and self.writes >= self._disconnect_after:
            raise ConnectionResetError("simulated client disconnect")


def _deltas(stream):
    return [
        ev.get("delta")
        for _seq, ev in stream.backlog
        if ev is not None and ev.get("event") == "assistant.delta"
    ]


async def _run(adapter, session_db, response, *, disconnect_expected):
    session_id = session_db.create_session("mirror-session", "api_server")
    started, proceed = threading.Event(), threading.Event()
    request = MagicMock()
    request.headers = {}
    request.match_info = {"session_id": session_id}

    with patch.object(
        adapter, "_get_existing_session_or_404", return_value=({"id": session_id}, None)
    ), patch.object(
        adapter, "_read_json_body", return_value=({"message": "go"}, None)
    ), patch.object(
        adapter,
        "_create_agent",
        side_effect=lambda **kw: _FakeAgent(session_id, kw["stream_delta_callback"], started, proceed),
    ), patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
        handler = asyncio.create_task(adapter._handle_session_chat_stream(request))
        for _ in range(100):
            if started.is_set() and adapter._run_streams:
                break
            await asyncio.sleep(0.02)
        assert started.is_set()
        run_id = next(iter(adapter._run_streams))
        stream = adapter._run_streams[run_id]
        sub, _replay = stream.attach(-1)  # a second device attached mid-run

        if disconnect_expected:
            await asyncio.wait_for(handler, timeout=5)  # POST client is gone, run is not
            assert run_id in adapter._active_run_agents
        proceed.set()
        for _ in range(200):
            if run_id not in adapter._active_run_agents and stream.terminal:
                break
            await asyncio.sleep(0.02)
        if not disconnect_expected:
            await asyncio.wait_for(handler, timeout=5)
    return run_id, stream, sub


@pytest.mark.asyncio
async def test_run_stream_keeps_receiving_frames_after_post_client_disconnects(adapter, session_db):
    # run.started, message.started, then the first delta write raises -> POST consumer exits.
    _run_id, stream, _sub = await _run(
        adapter, session_db, _Response(disconnect_after=3), disconnect_expected=True
    )
    assert "two" in "".join(d or "" for d in _deltas(stream)), (
        "frames produced after the POST client disconnected never reached the replayable stream"
    )


@pytest.mark.asyncio
async def test_attached_subscriber_receives_terminal_sentinel_before_stream_is_dropped(adapter, session_db):
    run_id, stream, sub = await _run(adapter, session_db, _Response(), disconnect_expected=False)
    seen = []
    while not sub.empty():
        seen.append(sub.get_nowait())
    assert seen and seen[-1][1] is None, "attached subscriber never got the terminal sentinel"
    assert stream.terminal
    # contract preserved: once finished the run stream is released (late joiners 404)
    assert run_id not in adapter._run_streams


@pytest.mark.asyncio
async def test_disconnected_client_still_gets_terminal_close_for_attached_subscriber(adapter, session_db):
    _run_id, stream, sub = await _run(
        adapter, session_db, _Response(disconnect_after=3), disconnect_expected=True
    )
    seen = []
    while not sub.empty():
        seen.append(sub.get_nowait())
    assert seen and seen[-1][1] is None
    assert stream.terminal


# ---------------------------------------------------------------------------------------------
# Real TCP disconnect. aiohttp CANCELS the handler when the client goes away while the handler is
# idle between frames (the usual case for a backgrounded phone); that path used to interrupt the
# agent, contradicting "a clean client disconnect keeps the run going".
# ---------------------------------------------------------------------------------------------
class _SlowAgent:
    session_prompt_tokens = session_completion_tokens = session_total_tokens = 0

    def __init__(self, session_id, cb, log):
        self.session_id, self._cb, self.log = session_id, cb, log

    def interrupt(self, message=None):
        self.log.append("interrupt")

    def run_conversation(self, user_message, conversation_history, task_id):
        del user_message, conversation_history, task_id
        for i in range(12):
            self._cb(f"w{i} ")
            time.sleep(0.25)
        self.log.append("agent_finished")
        return {"final_response": "all done", "session_id": self.session_id}


async def _real_disconnect(adapter, session_db):
    session_id = session_db.create_session("real-disconnect", "api_server")
    log: list[str] = []
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    with patch.object(
        adapter, "_get_existing_session_or_404", return_value=({"id": session_id}, None)
    ), patch.object(
        adapter, "_read_json_body", return_value=({"message": "go"}, None)
    ), patch.object(
        adapter, "_create_agent",
        side_effect=lambda **kw: _SlowAgent(session_id, kw["stream_delta_callback"], log),
    ):
        server = TestServer(app)
        await server.start_server()
        try:
            async with ClientSession(connector=TCPConnector(force_close=True)) as cs:
                resp = await cs.post(
                    server.make_url(f"/api/sessions/{session_id}/chat/stream"), json={"message": "go"}
                )
                seen = 0
                async for line in resp.content:
                    if line.startswith(b"event: assistant.delta"):
                        seen += 1
                    if seen >= 2:
                        break
                resp.close()  # the phone goes away mid-reply
            run_id = next(iter(adapter._run_streams))
            stream = adapter._run_streams[run_id]
            for _ in range(150):
                await asyncio.sleep(0.1)
                if stream.terminal:
                    break
            await asyncio.sleep(0.3)
            return log, stream
        finally:
            await server.close()


@pytest.mark.asyncio
async def test_real_client_disconnect_does_not_interrupt_the_run(adapter, session_db):
    log, _stream = await _real_disconnect(adapter, session_db)
    assert "interrupt" not in log, "an idle-time client disconnect interrupted a run that should keep going"
    assert "agent_finished" in log


@pytest.mark.asyncio
async def test_real_client_disconnect_replay_gets_every_frame_and_terminal(adapter, session_db):
    _log, stream = await _real_disconnect(adapter, session_db)
    names = [(ev or {}).get("event") for _seq, ev in stream.backlog]
    assert len([n for n in names if n == "assistant.delta"]) == 12
    assert names[-2:] == ["done", None] or (names[-1] is None and "run.completed" in names)
    assert stream.terminal
