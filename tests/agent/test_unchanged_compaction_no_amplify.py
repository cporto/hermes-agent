"""Regression tests for compaction write amplification (#82956).

A compaction pass whose output still starts with rows that are already the leading live rows
(the no-op / ``method:none`` case, N -> N) must keep those rows in place: same row id, same
``message_uid``, still active. Before the fix both ``archive_and_compact`` paths (watermark and
covered-ids) archived every active row and re-inserted a fresh copy, so each pass doubled the
stored rows (measured 7-47x on long sessions).
"""

import contextlib
import sqlite3
import tempfile
from pathlib import Path


@contextlib.contextmanager
def _session_db():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "amplify.db"
        db = SessionDB(db_path=path)
        try:
            yield db, path
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


def _counts(path, sid):
    con = sqlite3.connect(path)
    try:
        total, active, archived = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(active = 1), 0), COALESCE(SUM(active = 0 AND compacted = 1), 0) "
            "FROM messages WHERE session_id = ?", (sid,)).fetchone()
        dup_uids = con.execute(
            "SELECT COUNT(*) FROM (SELECT message_uid FROM messages WHERE session_id = ? AND message_uid IS NOT NULL "
            "GROUP BY message_uid HAVING COUNT(*) > 1)", (sid,)).fetchone()[0]
        active_ids = [r[0] for r in con.execute(
            "SELECT id FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,))]
        return {"total": total, "active": active, "archived": archived, "dup_uids": dup_uids, "active_ids": active_ids}
    finally:
        con.close()


def _live_as_messages(db, sid):
    return [{"role": m["role"], "content": m["content"]} for m in db.get_messages(sid)]


def test_noop_compaction_watermark_path_keeps_rows_in_place():
    with _session_db() as (db, path):
        sid = "amp_wm_noop"
        _seed(db, sid, 8)
        before = _counts(path, sid)
        assert (before["total"], before["active"], before["archived"]) == (8, 8, 0)

        wm = db.get_active_message_watermark(sid)
        for _ in range(3):  # repeated passes must not accumulate rows
            db.archive_and_compact(sid, _live_as_messages(db, sid), watermark=wm)

        after = _counts(path, sid)
        assert after["total"] == 8, "no-op compaction re-inserted rows (write amplification)"
        assert after["active"] == 8
        assert after["archived"] == 0
        assert after["dup_uids"] == 0
        assert after["active_ids"] == before["active_ids"], "row ids must be preserved in place"


def test_noop_compaction_covered_ids_path_keeps_rows_in_place():
    with _session_db() as (db, path):
        sid = "amp_cov_noop"
        _seed(db, sid, 8)
        before = _counts(path, sid)

        for _ in range(3):
            db.archive_and_compact(
                sid, _live_as_messages(db, sid), covered_ids=_counts(path, sid)["active_ids"])

        after = _counts(path, sid)
        assert after["total"] == 8
        assert after["active"] == 8
        assert after["archived"] == 0
        assert after["dup_uids"] == 0
        assert after["active_ids"] == before["active_ids"]


def test_partial_prefix_keeps_leading_rows_and_replaces_the_rest():
    with _session_db() as (db, path):
        sid = "amp_partial"
        _seed(db, sid, 8)
        before = _counts(path, sid)
        kept_ids = before["active_ids"][:3]

        msgs = _live_as_messages(db, sid)[:3] + [{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}]
        count = db.archive_and_compact(sid, msgs, watermark=db.get_active_message_watermark(sid))

        after = _counts(path, sid)
        assert count == 4 == after["active"], "returned ACTIVE count must include the kept prefix"
        assert after["active_ids"][:3] == kept_ids, "leading rows must keep their ids"
        assert after["archived"] == 5, "only the replaced tail is archived"
        assert after["total"] == 8 + 1  # the single new summary row
        assert after["dup_uids"] == 0
        assert [m["content"] for m in db.get_messages(sid)][-1] == "[CONTEXT COMPACTION] summary"


def test_real_compaction_with_changed_head_still_replaces_everything():
    with _session_db() as (db, path):
        sid = "amp_real"
        _seed(db, sid, 8)

        msgs = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "recent reply"},
        ]
        count = db.archive_and_compact(sid, msgs, watermark=db.get_active_message_watermark(sid))

        after = _counts(path, sid)
        assert count == 2 == after["active"]
        assert after["archived"] == 8
        assert after["dup_uids"] == 0
