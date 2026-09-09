import time
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, _Upstream


def req(group, index, size=2):
    return SimpleNamespace(
        rid=f"{group}-{index}",
        _moss_score_group=group,
        _moss_score_group_index=index,
        _moss_score_group_size=size,
        _moss_score_group_created=time.perf_counter(),
    )


def test_wait_for_complete_group_and_preserve_other_groups():
    a, b, c = req("a", 0), req("a", 1), req("b", 0)
    scheduler = SimpleNamespace(waiting_queue=[a, c])
    with patch.object(_Upstream, "get_new_batch_prefill") as upstream:
        assert OmniScheduler.get_new_batch_prefill(scheduler, None).batch_to_run is None
        upstream.assert_not_called()
    scheduler.waiting_queue = [b, c, a]

    def admit(scheduler, running):
        assert [r.rid for r in scheduler.waiting_queue] == ["a-0", "a-1"]
        scheduler.waiting_queue = []
        return NextBatchPlan(batch_to_run="batch", running_batch=running)

    with patch.object(_Upstream, "get_new_batch_prefill", side_effect=admit):
        assert (
            OmniScheduler.get_new_batch_prefill(scheduler, None).batch_to_run == "batch"
        )
    assert scheduler.waiting_queue == [c]


def test_aborted_group_cannot_block_queue_forever():
    a = req("a", 0)
    a._moss_score_group_created = time.perf_counter() - 31
    errors = []
    scheduler = SimpleNamespace(waiting_queue=[a])
    scheduler._emit_request_error = lambda rid, error: errors.append((rid, str(error)))
    scheduler.abort = lambda rid: scheduler.waiting_queue.clear()
    assert OmniScheduler.get_new_batch_prefill(scheduler, None).batch_to_run is None
    assert errors and scheduler.waiting_queue == []
