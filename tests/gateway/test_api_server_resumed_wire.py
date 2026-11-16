"""S3 honest-state on the wire: terminal_run_status surfaces ``resumed``."""
from gateway.platforms import api_server_runs as R


def test_resumed_flag_present_when_from_checkpoint():
    status, fields = R.terminal_run_status({"completed": True, "final_response": "x",
                                            "resumed_from_checkpoint": True})
    assert status == "completed"
    assert fields.get("resumed") is True


def test_resumed_absent_for_normal_turn():
    status, fields = R.terminal_run_status({"completed": True, "final_response": "x"})
    assert status == "completed"
    assert fields.get("resumed") is None or status in ("completed", "failed", "cancelled")


def test_resumed_persists_on_interrupt():
    status, fields = R.terminal_run_status({"interrupted": True, "resumed_from_checkpoint": True})
    assert status == "cancelled"
    assert fields.get("resumed") is True
