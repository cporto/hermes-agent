"""Regression test for #21301: the in-place compaction watermark/fallback commit
raised ``UnboundLocalError`` because the A2 steer filter referenced ``tail_ids``
before it was assigned (hermes_state_messages._archive_and_compact_on_conn). The
result was that EVERY in-place hygiene compaction rolled back and the transcript
was bounded to the last 400 messages forever.
"""

import contextlib
import tempfile
from pathlib import Path


@contextlib.contextmanager
def _session_db():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "wm.db")
        try:
            yield db
        finally:
            db.close()


def _seed(db, sid, n=8):
    db.create_session(sid, "gateway", model="test/model")
    for i in range(n):
        db.append_message(
            session_id=sid,
            role="user" if i % 2 == 0 else "assistant",
            content=f"seed msg {i}",
        )


def test_watermark_path_commit_does_not_raise_and_archives():
    with _session_db() as db:
        sid = "20261022_wm_commit"
        _seed(db, sid, n=8)

        compiled = [{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}]
        new_active = db.archive_and_compact(
            session_id=sid, compacted_messages=compiled,
            watermark=None, covered_ids=None, unresolved_held=None)

        assert new_active == 1, f"expected 1 active row after commit, got {new_active}"
        swept = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=0 AND compacted=1",
            (sid,),
        ).fetchone()[0]
        assert swept == 8, f"expected 8 soft-archived rows, got {swept}"
