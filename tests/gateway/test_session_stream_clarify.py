"""Session chat streams surface interactive clarify questions keyed by run id.

Mirrors test_session_chat_approval: an opted-in api_server /chat/stream must emit a
``clarify.request`` frame carrying the pending control's id as ``clarify_id`` -- the wire
key the Mercury ClarifyCard POSTs back to resolve the prompt. Regression for the missing
clarify-notify bridge on the api_server surface (questions previously waited out the full
clarify_timeout because the frame was never delivered).
"""
import asyncio

from gateway.platforms.api_server import APIServerAdapter, _SessionEventQueue


def test_session_stream_clarify_enqueued_with_clarify_id():
    """The bridge enqueues ``clarify.request`` with run/message ids + the wire clarify_id."""

    async def _go():
        host = APIServerAdapter.__new__(APIServerAdapter)
        events = _SessionEventQueue("sess-1", "run_1")
        notify = host._register_session_stream_clarify("run_1", events, "msg_1")
        await asyncio.to_thread(
            notify,
            {"question": "Which environment?", "choices": ["staging", "prod"],
             "multi_select": False, "clarify_id": "clr_test_1"})
        name, event = await asyncio.wait_for(events.queue.get(), 2)
        assert name == "clarify.request"
        assert event["run_id"] == "run_1" and event["message_id"] == "msg_1"
        assert event["clarify_id"] == "clr_test_1"
        assert event["question"] == "Which environment?"
        assert event["choices"] == ["staging", "prod"]

    asyncio.run(_go())


def test_session_stream_clarify_prompt_id_fallback():
    """If the source carries ``prompt_id`` (the pending-controls key) rather than
    ``clarify_id``, the bridge must still expose a non-empty wire ``clarify_id``."""

    async def _go():
        host = APIServerAdapter.__new__(APIServerAdapter)
        events = _SessionEventQueue("sess-1", "run_1")
        notify = host._register_session_stream_clarify("run_1", events, "msg_1")
        await asyncio.to_thread(
            notify, {"question": "Pick one", "choices": ["a", "b"], "prompt_id": "q0"})
        name, event = await asyncio.wait_for(events.queue.get(), 2)
        assert name == "clarify.request"
        assert event["clarify_id"] == "q0"

    asyncio.run(_go())
