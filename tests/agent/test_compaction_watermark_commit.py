"""Regression test for #21301: the in-place compaction watermark/fallback commit
raised ``UnboundLocalError`` because the A2 steer filter referenced ``tail_ids``
before it was assigned (``archive_and_compact`` in the sqlite conversation store). The
result was that EVERY in-place hygiene compaction rolled back and the transcript
was bounded to the last 400 messages forever.

Ported onto the persistence layer: ``store.messages.compact`` with no watermark is the
fallback commit path.
"""

import sqlite3

from persistence.conversation.messages import NewMessage
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import storage_handle


def _seed(store, sid, n=8):
    seed_session(store, sid, "gateway", model="test/model")
    seed_messages(store, sid, *({"role": "user" if i % 2 == 0 else "assistant", "content": f"seed msg {i}"}
                                for i in range(n)))


def test_watermark_path_commit_does_not_raise_and_archives(tmp_path):
    store = storage_handle(tmp_path)
    sid = "20261022_wm_commit"
    _seed(store, sid, n=8)

    compiled = [NewMessage.from_live({"role": "assistant", "content": "[CONTEXT COMPACTION] summary"})]
    result = store.messages.compact(sid, compiled, watermark=None)

    assert result.active_count == 1, f"expected 1 active row after commit, got {result.active_count}"
    con = sqlite3.connect(tmp_path / "state.db")
    try:
        swept = con.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=0 AND compacted=1", (sid,)
        ).fetchone()[0]
    finally:
        con.close()
    assert swept == 8, f"expected 8 soft-archived rows, got {swept}"
