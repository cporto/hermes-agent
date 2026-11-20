"""Regression tests for compaction write amplification (#82956).

A compaction pass whose output still starts with rows that are already the leading live rows
(the no-op / ``method:none`` case, N -> N) must keep those rows in place: same row id, same
``message_uid``, still active. Before the fix both compaction paths (watermark and covered-ids)
archived every active row and re-inserted a fresh copy, so each pass doubled the stored rows
(measured 7-47x on long sessions).
"""

import sqlite3

from persistence.conversation.messages import HeldCoverage, NewMessage
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _seed(store, sid, n=8):
    seed_session(store, sid, "gateway", model="test/model")
    seed_messages(store, sid, *({"role": "user" if i % 2 == 0 else "assistant", "content": f"seed msg {i}"}
                                for i in range(n)))


def _counts(home, sid):
    con = sqlite3.connect(home / "state.db")
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


def _live(store, sid):
    return [NewMessage.from_live({"role": m["role"], "content": m["content"]}) for m in transcript_rows(store, sid)]


def test_noop_compaction_watermark_path_keeps_rows_in_place(tmp_path):
    store = storage_handle(tmp_path)
    sid = "amp_wm_noop"
    _seed(store, sid, 8)
    before = _counts(tmp_path, sid)
    assert (before["total"], before["active"], before["archived"]) == (8, 8, 0)

    wm = store.messages.active_watermark(sid)
    for _ in range(3):  # repeated passes must not accumulate rows
        store.messages.compact(sid, _live(store, sid), watermark=wm)

    after = _counts(tmp_path, sid)
    assert after["total"] == 8, "no-op compaction re-inserted rows (write amplification)"
    assert after["active"] == 8
    assert after["archived"] == 0
    assert after["dup_uids"] == 0
    assert after["active_ids"] == before["active_ids"], "row ids must be preserved in place"


def test_noop_compaction_covered_ids_path_keeps_rows_in_place(tmp_path):
    store = storage_handle(tmp_path)
    sid = "amp_cov_noop"
    _seed(store, sid, 8)
    before = _counts(tmp_path, sid)

    for _ in range(3):
        store.messages.compact(sid, _live(store, sid),
                               coverage=HeldCoverage(covered_ids=tuple(_counts(tmp_path, sid)["active_ids"])))

    after = _counts(tmp_path, sid)
    assert after["total"] == 8
    assert after["active"] == 8
    assert after["archived"] == 0
    assert after["dup_uids"] == 0
    assert after["active_ids"] == before["active_ids"]


def test_partial_prefix_keeps_leading_rows_and_replaces_the_rest(tmp_path):
    store = storage_handle(tmp_path)
    sid = "amp_partial"
    _seed(store, sid, 8)
    before = _counts(tmp_path, sid)
    kept_ids = before["active_ids"][:3]

    msgs = _live(store, sid)[:3] + [NewMessage.from_live({"role": "assistant", "content": "[CONTEXT COMPACTION] summary"})]
    result = store.messages.compact(sid, msgs, watermark=store.messages.active_watermark(sid))

    after = _counts(tmp_path, sid)
    assert result.active_count == 4 == after["active"], "returned ACTIVE count must include the kept prefix"
    assert after["active_ids"][:3] == kept_ids, "leading rows must keep their ids"
    assert after["archived"] == 5, "only the replaced tail is archived"
    assert after["total"] == 8 + 1  # the single new summary row
    assert after["dup_uids"] == 0
    assert transcript_rows(store, sid)[-1]["content"] == "[CONTEXT COMPACTION] summary"


def test_real_compaction_with_changed_head_still_replaces_everything(tmp_path):
    store = storage_handle(tmp_path)
    sid = "amp_real"
    _seed(store, sid, 8)

    msgs = [NewMessage.from_live({"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"}),
            NewMessage.from_live({"role": "assistant", "content": "recent reply"})]
    result = store.messages.compact(sid, msgs, watermark=store.messages.active_watermark(sid))

    after = _counts(tmp_path, sid)
    assert result.active_count == 2 == after["active"]
    assert after["archived"] == 8
    assert after["dup_uids"] == 0
