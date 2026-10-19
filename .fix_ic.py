#!/usr/bin/env python3
path = "/Users/hermes/.hermes/cache/scratch/merge-sandbox/agent/interrupt_control.py"
lines = open(path).read().split("\n")
# lines is 1-indexed in display; find the _drain_pending_steer def line (0-indexed).
start = None
end = None
for i, ln in enumerate(lines):
    if ln.startswith("    def _drain_pending_steer(self):"):
        start = i
    if start is not None and i > start and ln.startswith("    def _drain_leftover_steer"):
        end = i  # exclusive
        break
assert start is not None and end is not None, (start, end)
new_block = '''    def _drain_pending_steer(self):
        """Return the joined pending-steer text (None when nothing is pending). Also settles upstream's
        ticketed parts (marks their fates and caches the drain for ``requeue_drained_steer``) and clears
        the stable ``_pending_steer_uid`` slot, so both the uid-keyed durable persist and the ticket
        transfer model stay coherent."""
        with _ic_lock(self, "_pending_steer_lock"):
            text = _ic_slot(self, "_pending_steer_lock", "_pending_steer")
            parts = _steer_parts(self)
            _set_steer_parts(self, [])
            for ticket, _ in parts:
                if ticket is not None:
                    _steer_fates(self)[ticket] = True
            self._steer_last_drain = (text, parts)  # requeue_drained_steer restores its tickets
            self._pending_steer_uid = None
        return text
'''
lines[start:end] = new_block.split("\n")
open(path, "w").write("\n".join(lines))
import ast
ast.parse("\n".join(lines))
print("interrupt_control resolved; markers:", sum(l.count("<<<<<<<")+l.count(">>>>>>>") for l in lines))
